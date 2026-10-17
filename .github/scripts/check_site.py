#!/usr/bin/env python3
"""Fast, dependency-free integrity checks for this static GitHub Pages site."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ERRORS: list[str] = []
WARNINGS: list[str] = []
MAX_HTML_BYTES = 9 * 1024 * 1024
MAX_JS_BYTES = 1 * 1024 * 1024


def error(message: str) -> None:
    ERRORS.append(message)


def warning(message: str) -> None:
    WARNINGS.append(message)


def check_json() -> None:
    for path in ROOT.rglob("*.json"):
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            error(f"Invalid JSON: {path.relative_to(ROOT)} ({exc})")


def check_html_and_links() -> None:
    href_re = re.compile(r'''(?:href|src)\s*=\s*["']([^"'#]+)["']''', re.I)
    for path in ROOT.rglob("*.html"):
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        size = path.stat().st_size
        rel = path.relative_to(ROOT)
        if size > MAX_HTML_BYTES:
            warning(f"Large HTML file ({size:,} bytes): {rel}; consider splitting or lazy-loading content")
        source = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"<html\b", source, re.I) or not re.search(r"</html\s*>", source, re.I):
            error(f"Missing HTML root element: {rel}")
        for raw in href_re.findall(source):
            target = unquote(raw.strip())
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith(("data:", "blob:", "javascript:")):
                continue
            target_path = parsed.path
            if not target_path:
                continue
            if target_path.startswith("/"):
                candidate = (ROOT / target_path.lstrip("/")).resolve()
            else:
                candidate = (path.parent / target_path).resolve()
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                error(f"Local link escapes repository: {rel} -> {raw}")
                continue
            if candidate.is_dir():
                candidate = candidate / "index.html"
            if not candidate.exists():
                error(f"Broken local link: {rel} -> {raw}")


def check_assets_and_metadata() -> None:
    for path in ROOT.rglob("*.js"):
        if path.stat().st_size > MAX_JS_BYTES:
            warning(f"Large JavaScript file ({path.stat().st_size:,} bytes): {path.relative_to(ROOT)}")
    robots = ROOT / "robots.txt"
    if not robots.exists() or "Sitemap: https://happymie.github.io/sitemap.xml" not in robots.read_text(encoding="utf-8"):
        error("robots.txt is missing the canonical sitemap declaration")
    sitemap = ROOT / "sitemap.xml"
    try:
        tree = ET.parse(sitemap)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [node.text for node in tree.findall(".//sm:loc", ns)]
        if not urls or any(not value.startswith("https://happymie.github.io/") for value in urls if value):
            error("sitemap.xml contains no URLs or a URL outside the canonical site")
    except Exception as exc:
        error(f"Invalid sitemap.xml: {exc}")
    security_txt = ROOT / ".well-known" / "security.txt"
    if not security_txt.exists() or "Contact:" not in security_txt.read_text(encoding="utf-8"):
        error(".well-known/security.txt is missing or has no Contact field")


def main() -> int:
    check_json()
    check_html_and_links()
    check_assets_and_metadata()
    for item in WARNINGS:
        print(f"WARNING: {item}")
    for item in ERRORS:
        print(f"ERROR: {item}")
    print(f"Site check: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
