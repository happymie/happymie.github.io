#!/usr/bin/env python3
"""Offline, dependency-free release checks; no model training or network access."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = "https://happymie.github.io"
ERRORS: list[str] = []
JSON: dict[str, object] = {}
CALLS: dict[str, str] = {}
SKIP = {".git", ".venv", "node_modules", "__pycache__"}
KIB = 1024
# The legacy reference is explicitly budgeted. Audit manifests are not initial
# browser payloads; the compact catalog and individual lesson shards are.
BUDGETS = {
    "biomed-ai-atlas/reference.html": 2560 * KIB,
    "biomed-ai-atlas/reference-data.js": 1280 * KIB,
    "teaching/manifest.json": 1280 * KIB,
    "teaching/web/catalog.json": 160 * KIB,
}


def error(message: str) -> None:
    ERRORS.append(message)


def files(pattern: str):
    return sorted(p for p in ROOT.rglob(pattern) if not SKIP.intersection(p.relative_to(ROOT).parts))


def budget(path: Path, default: int) -> None:
    rel = path.relative_to(ROOT).as_posix()
    limit = BUDGETS.get(rel, default)
    if path.stat().st_size > limit:
        error(f"{rel}: {path.stat().st_size:,} bytes exceeds the {limit:,}-byte budget")


def check_url(raw: str, base: Path, location: str, *, asset: bool = False) -> None:
    if not raw or raw.startswith("#"):
        return  # Includes legitimate local SVG symbol references.
    if re.search(r"[\x00-\x20\x7f\\]", raw):
        error(f"{location}: URL contains whitespace, controls, or a backslash")
        return
    try:
        url = urlsplit(raw)
        if url.username or url.password or (url.port and url.port != 443):
            raise ValueError("credentials or nonstandard port")
        if url.scheme == "mailto" and not asset:
            return
        if url.scheme and url.scheme != "https":
            raise ValueError("only HTTPS or local URLs are permitted")
        if url.netloc:
            if url.scheme != "https":
                raise ValueError("protocol-relative URL")
            if url.netloc != "happymie.github.io":
                if asset:
                    raise ValueError("asset must be hosted on this site")
                return
        target = unquote(url.path)
        if not target:
            return
        candidate = (ROOT / target.lstrip("/") if target.startswith("/") else base.parent / target).resolve()
        candidate.relative_to(ROOT)
        if candidate.is_dir():
            candidate /= "index.html"
        if not candidate.is_file():
            raise ValueError(f"missing local target {target}")
    except ValueError as exc:
        error(f"{location}: unsafe or broken URL {raw!r}: {exc}")


class DocumentCheck(HTMLParser):
    def __init__(self, path: Path, fragment: bool = False):
        super().__init__(convert_charrefs=True)
        self.path, self.fragment = path, fragment
        self.base = ROOT / "biomed-ai-atlas/index.html" if fragment else path
        self.in_head = self.in_script = self.has_html = self.closed_html = False
        self.csp_count = 0

    def issue(self, message: str) -> None:
        error(f"{self.path.relative_to(ROOT)}:{self.getpos()[0]}: {message}")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "html":
            self.has_html = True
        if tag == "head":
            self.in_head = True
        location = f"{self.path.relative_to(ROOT)}:{self.getpos()[0]}"
        if tag in {"iframe", "object", "embed", "base"}:
            self.issue(f"forbidden active element <{tag}>")
        if self.fragment and tag in {"script", "style", "link", "meta", "form", "foreignobject", "animate", "animatetransform", "set"}:
            self.issue(f"forbidden model-fragment element <{tag}>")
        if tag == "meta" and (a.get("http-equiv") or "").lower() == "content-security-policy":
            self.csp_count += 1
            if not self.in_head:
                self.issue("CSP must be declared in <head>")
            directives = {}
            for part in (a.get("content") or "").split(";"):
                fields = part.split()
                if fields:
                    if fields[0] in directives:
                        self.issue(f"duplicate CSP directive {fields[0]}")
                    directives[fields[0]] = fields[1:]
            required = {"default-src": ["'none'"], "script-src": ["'self'"], "script-src-attr": ["'none'"],
                        "object-src": ["'none'"], "base-uri": ["'none'"], "form-action": ["'none'"]}
            for key, expected in required.items():
                if directives.get(key) != expected:
                    self.issue(f"CSP requires {key} {' '.join(expected)}")
            if "script-src-elem" in directives and directives["script-src-elem"] != ["'self'"]:
                self.issue("CSP script-src-elem must not weaken script-src")
            if {"frame-ancestors", "report-uri", "sandbox"}.intersection(directives):
                self.issue("CSP meta contains a directive that requires an HTTP response header")
        if not self.fragment and (tag in {"script", "style"} or (tag == "link" and a.get("rel") == "stylesheet")) and not self.csp_count:
            self.issue("CSP must precede executable or style resources")
        if tag == "script":
            self.in_script = True
            if not a.get("src"):
                self.issue("inline scripts are forbidden; use a local external script")
        for key, value in attrs:
            value = value or ""
            if key.startswith("on") or key in {"srcdoc", "formaction", "action", "ping", "srcset"}:
                self.issue(f"forbidden execution or unvalidated URL attribute {key}")
            if key.startswith("data-call-"):
                if key[10:] not in {"click", "input", "change", "mouseenter", "mouseleave"}:
                    self.issue(f"unsupported delegated event {key}")
                CALLS.setdefault(value, location)
            if key in {"href", "src", "xlink:href", "poster"}:
                navigation = (tag == "a" and key == "href") or (tag == "link" and a.get("rel") == "canonical")
                check_url(value, self.base, location, asset=not navigation)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self.in_head = False
        if tag == "script":
            self.in_script = False
        if tag == "html":
            self.closed_html = True

    def handle_data(self, data: str) -> None:
        if self.in_script and data.strip():
            self.issue("inline script content is forbidden")

    def finish(self, source: str) -> None:
        self.feed(source)
        self.close()
        if not self.fragment:
            if not (self.has_html and self.closed_html):
                self.issue("missing HTML root element")
            if self.csp_count != 1:
                self.issue("exactly one CSP meta declaration is required")


def check_documents() -> None:
    for path in files("*.json"):
        rel = path.relative_to(ROOT).as_posix()
        default = 64 * KIB if rel.startswith("teaching/web/lessons/") else 512 * KIB
        if rel.startswith("biomed-ai-atlas/models/"):
            default = 128 * KIB
        budget(path, default)
        try:
            JSON[rel] = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            error(f"{rel}: invalid JSON: {exc}")
    for path in files("*.html"):
        budget(path, 128 * KIB)
        DocumentCheck(path).finish(path.read_text(encoding="utf-8"))
    for path in files("biomed-ai-atlas/models/*.json"):
        model = JSON.get(path.relative_to(ROOT).as_posix(), {})
        if not isinstance(model, dict) or not isinstance(model.get("html"), str):
            error(f"{path.relative_to(ROOT)}: model needs an HTML fragment")
        else:
            DocumentCheck(path, fragment=True).finish(model["html"])


def check_javascript() -> None:
    for path in files("*.js") + files("*.cjs"):
        budget(path, 256 * KIB)
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            error(f"{path.relative_to(ROOT)}: JavaScript syntax error: {result.stderr.strip()}")
    # Reuse the browser's parser, feeding values as JSON data, never as code.
    script = """
const fs = require('node:fs');
const {parseCalls} = require('./site-assets/security.js');
const failures = [];
for (const [source, location] of JSON.parse(fs.readFileSync(0, 'utf8'))) {
  try { parseCalls(source); } catch (e) { failures.push(location + ': rejected data-call: ' + e.message); }
}
process.stdout.write(JSON.stringify(failures));
"""
    result = subprocess.run(["node", "-e", script], input=json.dumps(list(CALLS.items())),
                            capture_output=True, text=True, cwd=ROOT, timeout=30)
    if result.returncode:
        error(f"Action parser check failed: {result.stderr.strip()}")
    else:
        ERRORS.extend(json.loads(result.stdout))


def model_rows(name: str, id_key: str = "id") -> dict[int, dict]:
    document = JSON.get(name)
    rows = document.get("models") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        error(f"{name}: missing model list")
        return {}
    result, ids = {}, set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get("num")) is not int or row["num"] < 1:
            error(f"{name}: invalid model number")
            continue
        number = row["num"]
        identifier = row.get(id_key)
        if number in result or (id_key and identifier in ids):
            error(f"{name}: duplicate model {number}")
        if id_key and (not isinstance(identifier, str) or not re.fullmatch(r"architecture_model_\d{3,}_[a-z0-9_]+", identifier)):
            error(f"{name}: invalid model ID for {number}")
        if "num_str" in row and row["num_str"] != f"{number:03d}":
            error(f"{name}: inconsistent num_str for {number}")
        result[number] = row
        ids.add(identifier)
    return result


def check_data() -> None:
    atlas = model_rows("biomed-ai-atlas/catalog.json")
    catalog = model_rows("teaching/catalog.json")
    manifest = model_rows("teaching/manifest.json")
    compact = model_rows("teaching/web/catalog.json")
    sources = model_rows("teaching/sources.json", "model_id")
    datasets = model_rows("teaching/data/manifest.json", "")
    for name, table in [("teaching catalog", catalog), ("manifest", manifest), ("web catalog", compact), ("source registry", sources), ("datasets", datasets)]:
        if set(table) != set(atlas):
            error(f"{name}: model numbers differ from the atlas catalog")
    expected = {
        "biomed-ai-atlas/models/*.json": {f"biomed-ai-atlas/models/{row['id']}.json" for row in atlas.values()},
        "teaching/web/lessons/*.json": {f"teaching/web/lessons/{num:03d}.json" for num in atlas},
        "teaching/lessons/*.md": {f"teaching/lessons/{num:03d}.md" for num in atlas},
        "teaching/data/*.sqlite3": {f"teaching/data/{num:03d}.sqlite3" for num in atlas},
    }
    for pattern, wanted in expected.items():
        actual = {p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern)}
        if actual != wanted:
            error(f"{pattern}: missing {sorted(wanted - actual)}, unexpected {sorted(actual - wanted)}")
    total = 0
    for num, row in manifest.items():
        for label, other in [("atlas", atlas.get(num, {})), ("catalog", catalog.get(num, {})), ("compact", compact.get(num, {}))]:
            if any(other.get(key) != row.get(key) for key in ("id", "num", "title")):
                error(f"model {num}: identity differs in {label}")
        if sources.get(num, {}).get("model_id") != row.get("id"):
            error(f"model {num}: source registry identity mismatch")
        shard = f"teaching/web/lessons/{num:03d}.json"
        if JSON.get(shard) != row:
            error(f"{shard}: differs from the full manifest")
        short = compact.get(num, {})
        data = row.get("dataset", {})
        if short.get("implementation") != row.get("implementation") or short.get("dataset") != {key: data.get(key) for key in ("bytes", "kind")}:
            error(f"model {num}: compact implementation or dataset metadata mismatch")
        if any(datasets.get(num, {}).get(key) != data.get(key) for key in ("bytes", "kind", "counts")):
            error(f"model {num}: dataset manifest mismatch")
        db = ROOT / f"teaching/data/{num:03d}.sqlite3"
        if db.is_file():
            size = db.stat().st_size
            total += size
            if size != data.get("bytes") or hashlib.sha256(db.read_bytes()).hexdigest() != data.get("sha256"):
                error(f"model {num}: dataset bytes or SHA-256 differs from the manifest")
    counts = dict(Counter(row.get("implementation") for row in manifest.values()))
    for name in ("teaching/manifest.json", "teaching/web/catalog.json"):
        document = JSON.get(name, {})
        if document.get("counts") != counts or document.get("dataset_total_bytes") != total:
            error(f"{name}: implementation counts or dataset total do not match files")


def check_metadata() -> None:
    robots = ROOT / "robots.txt"
    if not robots.exists() or f"Sitemap: {ORIGIN}/sitemap.xml" not in robots.read_text(encoding="utf-8"):
        error("robots.txt: missing canonical sitemap declaration")
    try:
        tree = ET.parse(ROOT / "sitemap.xml")
        urls = [node.text for node in tree.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
        if not urls or any(not value or not value.startswith(ORIGIN + "/") for value in urls):
            raise ValueError("URLs must use the canonical origin")
        for value in urls:
            check_url(value, ROOT / "index.html", "sitemap.xml")
    except (ValueError, OSError, ET.ParseError) as exc:
        error(f"sitemap.xml: {exc}")
    try:
        text = (ROOT / ".well-known/security.txt").read_text(encoding="utf-8")
        if not re.search(r"^Contact: https://", text, re.M):
            raise ValueError("missing HTTPS Contact")
        expiry = re.search(r"^Expires: (.+)$", text, re.M)
        if not expiry or datetime.fromisoformat(expiry[1].replace("Z", "+00:00")) <= datetime.now(timezone.utc):
            raise ValueError("missing or expired Expires date")
    except (ValueError, OSError) as exc:
        error(f"security.txt: {exc}")


def main() -> int:
    for check in (check_documents, check_javascript, check_data, check_metadata):
        try:
            check()
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            error(f"{check.__name__}: {exc}")
    for item in ERRORS[:40]:
        print(f"ERROR: {item}")
    if len(ERRORS) > 40:
        print(f"... {len(ERRORS) - 40} additional errors")
    print(f"Site check: {len(ERRORS)} error(s); {len(JSON)} JSON documents; {len(CALLS)} distinct delegated actions")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
