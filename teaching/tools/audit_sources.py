#!/usr/bin/env python3
"""Validate the source registry; optionally recheck public upstream metadata.

Default execution is offline and never downloads source archives or model weights.
--refresh checks the existing immutable pins. --repin explicitly moves only records
whose pin_strategy is default_branch. Historical pins, including RealNVP, stay fixed.
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

TEACHING = Path(__file__).resolve().parents[1]
REQUIRED = {
    "model_id", "num", "title", "paper_url", "atlas_code_url", "repo_url",
    "source_kind", "commit", "license_spdx", "license_url", "redistribution",
    "evidence_checked", "notes",
}
POLICIES = {"permitted_with_notice", "restricted", "unknown"}
KINDS = {"original", "canonical_library", "paper"}
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0"}


def validate(data: dict, catalog: dict) -> list[str]:
    errors = []
    rows = data.get("models", [])
    expected = {m["id"]: m for m in catalog["models"]}
    actual = [m.get("model_id") for m in rows]
    if len(actual) != len(set(actual)):
        errors.append("Duplicate model_id values")
    if set(actual) != set(expected):
        errors.append("Registry IDs do not match the Atlas catalog exactly")
    if len({m.get("num") for m in rows}) != len(rows):
        errors.append("Duplicate model numbers")
    for row in rows:
        label = row.get("model_id", "<missing>")
        if REQUIRED - row.keys():
            errors.append(f"{label}: missing {sorted(REQUIRED - row.keys())}")
            continue
        original = expected.get(label)
        if original and (row["num"] != original["num"] or row["title"] != original["title"]):
            errors.append(f"{label}: Atlas number/title changed")
        if row["source_kind"] not in KINDS or row["redistribution"] not in POLICIES:
            errors.append(f"{label}: invalid source kind or permission category")
        if row["commit"] and not re.fullmatch(r"[0-9a-f]{40}", row["commit"]):
            errors.append(f"{label}: commit must be a complete immutable 40-character SHA")
        if not isinstance(row["notes"], list):
            errors.append(f"{label}: notes must be a list")
        if row["redistribution"] == "permitted_with_notice":
            if row["license_spdx"] == "NOASSERTION" or not row["license_url"]:
                errors.append(f"{label}: permissive classification lacks license evidence")
        for key in ("paper_url", "repo_url", "source_url", "license_url"):
            value = row.get(key)
            if value and not value.startswith("https://"):
                errors.append(f"{label}: {key} must use HTTPS")
    return errors


class PublicAPI:
    def __init__(self, timeout: float, token: str | None = None):
        self.timeout, self.token = timeout, token

    def get(self, url: str):
        host = urllib.parse.urlparse(url).hostname
        if host not in {"api.github.com", "huggingface.co"}:
            raise ValueError("Only public GitHub and Hugging Face metadata hosts are supported")
        headers = {"User-Agent": "Biomedical-AI-Atlas-source-audit", "Accept": "application/json"}
        if self.token and host == "api.github.com":
            headers["Authorization"] = "Bearer " + self.token
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Never print request headers, credentials, or full signed URLs.
            raise RuntimeError(f"HTTP {exc.code} from {host}; upstream access/rate limit must be resolved") from None


def identify_license(text: str, reported: str | None, repo: str) -> tuple[str, str]:
    """Conservative source-code license summary; not a weight/data license check."""
    custom = {
        "MrGiovanni/ModelsGenesis": "LicenseRef-ASU-ModelsGenesis-NonCommercial",
        "MrGiovanni/UNetPlusPlus": "LicenseRef-ASU-UNetPlusPlus-NonCommercial",
        "graphdeco-inria/gaussian-splatting": "LicenseRef-Gaussian-Splatting-Research",
        "NVlabs/SegFormer": "LicenseRef-NVIDIA-SegFormer-NonCommercial",
        "NVlabs/instant-ngp": "LicenseRef-NVIDIA-InstantNGP-NonCommercial",
        "microsoft/LLaVA-Med": "LicenseRef-Microsoft-Research",
    }
    if not text:
        return "NOASSERTION", "unknown"
    if re.search(r"Attribution-NonCommercial-NoDerivatives 4.0", text, re.I):
        return "CC-BY-NC-ND-4.0", "restricted"
    if re.search(r"Attribution-NonCommercial-ShareAlike 4.0", text, re.I):
        return "CC-BY-NC-SA-4.0", "restricted"
    if re.search(r"Attribution-NonCommercial 4.0", text, re.I):
        return "CC-BY-NC-4.0", "restricted"
    if repo in custom and re.search(r"non.?commercial|research purposes", text, re.I):
        return custom[repo], "restricted"
    if reported and re.match(r"(?:A?GPL|LGPL)-", reported):
        return reported, "restricted"
    if repo in {"phillipi/pix2pix", "junyanz/CycleGAN"} and "LICENSE FOR DCGAN" in text:
        return "BSD-2-Clause AND BSD-3-Clause", "permitted_with_notice"
    if re.search(r"\bMIT License\b", text, re.I):
        return "MIT", "permitted_with_notice"
    if re.search(r"Apache License[\s,]+(?:Version )?2\.0", text, re.I):
        return "Apache-2.0", "permitted_with_notice"
    if "Redistribution and use in source and binary forms" in text:
        if re.search(r"Neither the name|Neither the names", text):
            return "BSD-3-Clause", "permitted_with_notice"
        if repo == "BVLC/caffe" or reported == "BSD-2-Clause":
            return "BSD-2-Clause", "permitted_with_notice"
    if reported in PERMISSIVE:
        return reported, "permitted_with_notice"
    return reported or "NOASSERTION", "unknown"


def github_snapshot(api: PublicAPI, row: dict, repin: bool) -> dict:
    parsed = urllib.parse.urlparse(row["repo_url"])
    repo = "/".join(parsed.path.strip("/").split("/")[:2])
    endpoint = "https://api.github.com/repos/" + repo
    meta = api.get(endpoint)
    branch = meta["default_branch"]
    latest = api.get(endpoint + "/git/ref/heads/" + urllib.parse.quote(branch, safe=""))["object"]["sha"]
    pin = row.get("commit")
    if not pin or (repin and row.get("pin_strategy") == "default_branch"):
        pin = latest
    api.get(endpoint + "/git/commits/" + pin)
    paths = api.get(endpoint + "/contents?ref=" + pin)
    candidates = [p for p in paths if p.get("type") == "file" and re.match(r"^(licen[sc]e|copying|copyright)", p["name"], re.I)]
    candidates.sort(key=lambda p: (p["name"].lower() != "license", not p["name"].lower().startswith("license"), p["name"]))
    license_path = candidates[0]["path"] if candidates else None
    text = ""
    if license_path:
        blob = api.get(endpoint + "/contents/" + urllib.parse.quote(license_path) + "?ref=" + pin)
        if blob.get("encoding") != "base64":
            raise ValueError("Unexpected GitHub contents encoding")
        text = base64.b64decode(blob["content"]).decode("utf-8")
    reported = (meta.get("license") or {}).get("spdx_id")
    spdx, policy = identify_license(text, reported, repo)
    url = "https://github.com/" + meta["full_name"]
    return {
        "repo_url": url, "commit": pin, "latest_commit": latest,
        "default_branch": branch, "archived": meta["archived"],
        "github_license_spdx": reported, "license_spdx": spdx,
        "license_path": license_path,
        "license_url": f"{url}/blob/{pin}/{license_path}" if license_path else None,
        "redistribution": policy,
        "license_evidence": "license_text_read" if text else "no_root_license_found",
        "metadata_status": "metadata_and_license_verified" if text else "metadata_verified_license_unknown",
    }


def hf_snapshot(api: PublicAPI, row: dict, repin: bool) -> dict:
    model = "/".join(urllib.parse.urlparse(row["repo_url"]).path.strip("/").split("/")[:2])
    meta = api.get("https://huggingface.co/api/models/" + model)
    pin = meta["sha"] if (repin or not row.get("commit")) else row["commit"]
    # Revision metadata, not gated model files, is queried.
    pinned = api.get("https://huggingface.co/api/models/" + model + "/revision/" + pin)
    card = pinned.get("cardData") or {}
    names = {"mit": "MIT", "apache-2.0": "Apache-2.0", "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0", "cc-by-nc-4.0": "CC-BY-NC-4.0"}
    reported = card.get("license", "NOASSERTION")
    spdx = names.get(reported, "LicenseRef-Microsoft-Research" if card.get("license_name") == "msrla" else "NOASSERTION")
    policy = "permitted_with_notice" if spdx in PERMISSIVE else ("restricted" if spdx != "NOASSERTION" else "unknown")
    files = [f["rfilename"] for f in pinned.get("siblings", [])]
    path = next((p for p in files if p.lower().startswith("license")), "README.md")
    url = "https://huggingface.co/" + model
    return {
        "commit": pin, "latest_commit": meta["sha"], "default_branch": "main",
        "gated": meta.get("gated"), "license_spdx": spdx, "redistribution": policy,
        "license_path": path, "license_url": f"{url}/blob/{pin}/{path}",
        "model_card_url": f"{url}/blob/{pin}/README.md",
        "metadata_status": "model_card_metadata_verified",
        "license_evidence": "public_Hugging_Face_card_metadata_and_gate_terms",
    }


def summary(data: dict) -> dict:
    rows = data["models"]
    return {
        "models": len(rows), "immutable_pins": sum(bool(m.get("commit")) for m in rows),
        "redistribution": dict(collections.Counter(m["redistribution"] for m in rows)),
        "source_kinds": dict(collections.Counter(m["source_kind"] for m in rows)),
        "refresh_errors": sum(m.get("refresh_status") == "error" for m in rows),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=TEACHING / "sources.json")
    p.add_argument("--catalog", type=Path, default=TEACHING.parent / "biomed-ai-atlas" / "catalog.json")
    p.add_argument("--refresh", action="store_true", help="Read public upstream metadata; keep existing pins")
    p.add_argument("--repin", action="store_true", help="With --refresh, update default-branch pins only")
    p.add_argument("--write", action="store_true", help="With --refresh, write the checked registry")
    p.add_argument("--model", action="append", help="Refresh one number, exact title, or model ID; repeatable")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=20)
    p.add_argument("--github-token-env", help="Optional name of an explicitly supplied token environment variable")
    args = p.parse_args(argv)
    if (args.repin or args.write or args.model) and not args.refresh:
        p.error("--repin, --write, and --model require --refresh")
    if not 1 <= args.workers <= 8:
        p.error("--workers must be between 1 and 8")
    data = json.loads(args.sources.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    errors = validate(data, catalog)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    if args.refresh:
        token = None
        if args.github_token_env:
            token = os.environ.get(args.github_token_env)
            if not token:
                p.error("The explicitly selected token environment variable is empty")
        api = PublicAPI(args.timeout, token)
        output = copy.deepcopy(data)
        selectors = set(args.model or [])
        selected = [r for r in output["models"] if not selectors or selectors.intersection({str(r["num"]), r["title"], r["model_id"]})]
        if selectors and not selected:
            p.error("No selected model matches the registry")
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()

        def refresh(row):
            host = urllib.parse.urlparse(row.get("repo_url") or "").hostname
            try:
                if host == "github.com":
                    result = github_snapshot(api, row, args.repin)
                elif host == "huggingface.co":
                    result = hf_snapshot(api, row, args.repin)
                else:
                    row["refresh_status"] = "manual_review_required"
                    return
                old_pin = row.get("commit")
                row.update(result)
                if row["source_kind"] == "paper" and row.get("repo_url"):
                    row["metadata_status"] = "evaluation_data_only"
                if old_pin and old_pin != row["commit"]:
                    for key in ("source_url", "model_card_url"):
                        if row.get(key):
                            row[key] = row[key].replace(old_pin, row["commit"])
                row["evidence_checked"] = today
                row["refresh_status"] = "ok"
                row.pop("refresh_error", None)
            except Exception as exc:
                row["refresh_status"] = "error"
                row["refresh_error"] = str(exc)
                # Keep the last successfully verified date and pin on failure.

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(refresh, selected))
        data = output
        if args.write:
            args.sources.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": True, **summary(data)}, indent=2))
    if args.refresh:
        failures = [{"num": r["num"], "title": r["title"], "error": r.get("refresh_error")} for r in data["models"] if r.get("refresh_status") == "error"]
        if failures:
            print(json.dumps({"upstream_checks_failed": failures}, indent=2))
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
