"""Offline security checks for the public-metadata updater; no real credentials."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.request


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "audit_sources.py"
SPEC = importlib.util.spec_from_file_location("source_audit_security_target", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def registry():
    row = {
        "model_id": "architecture_model_001_example", "num": 1, "title": "Example",
        "paper_url": "https://example.org/paper", "atlas_code_url": "https://github.com/example/model",
        "repo_url": "https://github.com/example/model", "source_kind": "original",
        "commit": "a" * 40, "license_spdx": "MIT",
        "license_url": "https://github.com/example/model/blob/" + "a" * 40 + "/LICENSE",
        "redistribution": "permitted_with_notice", "evidence_checked": "2026-01-01",
        "notes": [],
    }
    return {"models": [row]}, {"models": [{"id": row["model_id"], "num": 1, "title": "Example"}]}


class Response(io.BytesIO):
    def __init__(self, payload, url="https://api.github.com/repos/example/model", headers=None):
        super().__init__(payload)
        self.url = url
        self.headers = headers or {}

    def geturl(self):
        return self.url


class SourceAuditSecurityTests(unittest.TestCase):
    def test_registry_validation_rejects_credential_urls_and_bad_schemas(self):
        data, catalog = registry()
        self.assertEqual(audit.validate(data, catalog), [])
        mutations = [
            ("repo_url", "https://api.github.com@elsewhere.example/model"),
            ("repo_url", "https://github.com/example/model?redirect=elsewhere"),
            ("repo_url", "https://github.com/../model"),
            ("paper_url", "javascript:alert(1)"),
            ("source_url", "https://example.org:8080/source"),
            ("source_url", "https://example.org/with\na-newline"),
            ("atlas_code_url", {"url": "https://example.org"}),
            ("latest_commit", "not-an-immutable-revision"),
            ("notes", [{"html": "unvalidated"}]),
            ("evidence_urls", ["http://example.org/source"]),
            ("num", True),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                candidate = copy.deepcopy(data)
                candidate["models"][0][key] = value
                self.assertTrue(audit.validate(candidate, catalog))

    def test_cross_host_and_https_downgrade_redirects_are_rejected(self):
        handler = audit.MetadataRedirectHandler()
        req = urllib.request.Request("https://api.github.com/repos/example/model",
                                     headers={"Authorization": "Bearer synthetic-test-value"})
        for destination in (
            "https://elsewhere.example/capture", "https://huggingface.co/api/models/example/model",
            "http://api.github.com/repos/example/model", "https://api.github.com:8080/other",
            "https://user:password@api.github.com/repos/example/model",
        ):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                handler.redirect_request(req, None, 302, "Found", {}, destination)

    def test_same_host_repository_rename_can_keep_authenticated_metadata_access(self):
        req = urllib.request.Request("https://api.github.com/repos/example/model",
                                     headers={"Authorization": "Bearer synthetic-test-value"})
        redirected = audit.MetadataRedirectHandler().redirect_request(
            req, None, 301, "Moved", {}, "https://api.github.com/repositories/123")
        self.assertEqual(redirected.full_url, "https://api.github.com/repositories/123")
        self.assertEqual(redirected.get_header("Authorization"), req.get_header("Authorization"))

    def test_github_credential_is_never_added_to_hugging_face_requests(self):
        api = audit.PublicAPI(1, token="synthetic-test-value")
        for url, expected in (
            ("https://api.github.com/repos/example/model", "Bearer synthetic-test-value"),
            ("https://huggingface.co/api/models/example/model", None),
        ):
            with self.subTest(url=url):
                with mock.patch.object(api.opener, "open", return_value=Response(b"{}", url)) as opened:
                    self.assertEqual(api.get(url), {})
                    self.assertEqual(opened.call_args.args[0].get_header("Authorization"), expected)

    def test_invalid_credential_cannot_leak_through_a_header_error(self):
        for token in ("synthetic\nvalue", "synthetic\x00value", "synthetic非ASCIIvalue"):
            with self.subTest(), self.assertRaises(ValueError) as raised:
                audit.PublicAPI(1, token=token)
            self.assertNotIn(token, str(raised.exception))

    def test_streamed_and_declared_response_sizes_are_bounded(self):
        api = audit.PublicAPI(1, max_bytes=8)
        for response in (Response(b" " * 9), Response(b"{}", headers={"Content-Length": "900000000"})):
            with self.subTest(headers=response.headers):
                with mock.patch.object(api.opener, "open", return_value=response), self.assertRaises(ValueError):
                    api.get("https://api.github.com/repos/example/model")

    def test_response_origin_is_checked_again_before_parsing(self):
        api = audit.PublicAPI(1)
        with mock.patch.object(api.opener, "open", return_value=Response(b"{}", "https://huggingface.co/api/models/x/y")):
            with self.assertRaises(ValueError):
                api.get("https://api.github.com/repos/example/model")

    def test_invalid_upstream_metadata_keeps_the_last_successful_pin_and_date(self):
        data, catalog = registry()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.json"
            catalog_path = Path(folder) / "catalog.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            with mock.patch.object(audit, "github_snapshot", return_value={"commit": "b" * 40, "latest_commit": "invalid"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = audit.main(["--sources", str(path), "--catalog", str(catalog_path), "--refresh", "--write"])
            written = json.loads(path.read_text())["models"][0]
            self.assertEqual(result, 2)
            self.assertEqual(written["commit"], data["models"][0]["commit"])
            self.assertEqual(written["evidence_checked"], "2026-01-01")
            self.assertNotIn("latest_commit", written)
            self.assertEqual(written["refresh_status"], "error")

    def test_invalid_input_is_never_replaced(self):
        data, catalog = registry()
        data["models"][0]["repo_url"] = "http://github.com/example/model"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.json"
            catalog_path = Path(folder) / "catalog.json"
            original = json.dumps(data)
            path.write_text(original)
            catalog_path.write_text(json.dumps(catalog))
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit.main(["--sources", str(path), "--catalog", str(catalog_path), "--refresh", "--write"])
            self.assertEqual(result, 1)
            self.assertEqual(path.read_text(), original)

    def test_failed_atomic_replace_preserves_existing_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.json"
            path.write_text('{"keep": "previous"}\n')
            with mock.patch.object(audit.os, "replace", side_effect=OSError("simulated interrupted replacement")):
                with self.assertRaises(OSError):
                    audit.atomic_write_json(path, {"replacement": "unpublished"})
            self.assertEqual(path.read_text(), '{"keep": "previous"}\n')
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_atomic_write_produces_complete_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sources.json"
            audit.atomic_write_json(path, {"title": "医学模型", "checked": True})
            self.assertEqual(json.loads(path.read_text()), {"title": "医学模型", "checked": True})
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
