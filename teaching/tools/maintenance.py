#!/usr/bin/env python3
"""Bounded public discovery, offline checks, and honest maintenance records.

Uses only Python's standard library. It never installs or executes downloaded
code, downloads weights, changes Git refs, or promotes a candidate to a verified
adapter. The daily assistant reviews sources and uses `record` after validation.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE = 2 * 1024 * 1024
ALLOWED_HOSTS = {'export.arxiv.org', 'api.github.com'}
SPLITS = {'train': 16, 'validation': 4, 'test': 4}
ID_PATTERN = re.compile(r'^\d{8}T\d{6}Z(?:-[a-z0-9-]{1,40})?$')


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamps need an explicit UTC offset')
    return result


def https_url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Evidence links must be public HTTPS URLs without credentials')
    return value


def read_json(path):
    if path.stat().st_size > MAX_RESPONSE:
        raise ValueError(f'{path.name}: JSON exceeds the 2 MiB maintenance budget')
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if len(payload.encode()) > MAX_RESPONSE:
        raise ValueError('Maintenance record exceeds 2 MiB')
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Public metadata redirect blocked; review the new URL separately')


def public_get(url, timeout=10):
    https_url(url)
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname not in ALLOWED_HOSTS or parsed.port not in (None, 443):
        raise ValueError('Only the declared public arXiv/GitHub metadata hosts are allowed')
    request = urllib.request.Request(url, headers={'User-Agent': 'Biomedical-AI-Atlas-maintenance/1.0',
                                                 'Accept': 'application/json, application/atom+xml'})
    # No credentials are read from the environment or forwarded. Reject redirects
    # instead of trusting an upstream Location target.
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        body = response.read(MAX_RESPONSE + 1)
    if len(body) > MAX_RESPONSE:
        raise ValueError('Upstream response exceeds 2 MiB')
    return body


def scan(days=30, max_items=20, timeout=10):
    """Return discovery evidence. This command does not write success state."""
    if not 1 <= days <= 30 or not 1 <= max_items <= 50 or not 1 <= timeout <= 15:
        raise ValueError('Allowed: 1..30 days, 1..50 items, 1..15 seconds per request')
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days)
    query = ('(all:"medical imaging" OR all:"computed tomography" OR all:"EEG") '
             f'AND submittedDate:[{start:%Y%m%d}0000 TO {today:%Y%m%d}2359]')
    endpoints = [('arxiv', 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({
        'search_query': query, 'start': 0, 'max_results': max_items,
        'sortBy': 'submittedDate', 'sortOrder': 'descending'}))]
    # Selected maintained upstream feeds complement paper discovery; this is
    # deliberately a bounded watchlist, not a claim to cover the entire field.
    for repository in ('Project-MONAI/MONAI', 'bowang-lab/MedSAM', 'facebookresearch/sam2'):
        endpoints.append(('github_releases', f'https://api.github.com/repos/{repository}/releases?per_page=10'))
    checks, candidates = [], []
    for kind, url in endpoints:
        try:
            raw = public_get(url, timeout)
            found = []
            if kind == 'arxiv':
                ns = {'a': 'http://www.w3.org/2005/Atom'}
                for entry in ET.fromstring(raw).findall('a:entry', ns):
                    paper_url = entry.findtext('a:id', namespaces=ns) or ''
                    paper_url = paper_url.replace('http://arxiv.org/', 'https://arxiv.org/')
                    https_url(paper_url)
                    found.append({'title': ' '.join((entry.findtext('a:title', namespaces=ns) or '').split()),
                                  'url': paper_url, 'published_utc': entry.findtext('a:published', namespaces=ns),
                                  'kind': 'paper_candidate', 'decision': 'needs_primary_source_review'})
            else:
                releases = json.loads(raw)
                if not isinstance(releases, list):
                    raise ValueError('GitHub release response must be an array')
                for release in releases:
                    published = release.get('published_at')
                    if published and start <= timestamp(published).date() <= today and not release.get('draft'):
                        found.append({'title': release.get('name') or release['tag_name'],
                                      'url': https_url(release['html_url']), 'published_utc': published,
                                      'kind': 'upstream_release_candidate', 'decision': 'needs_primary_source_review'})
            candidates.extend(found)
            checks.append({'url': url, 'kind': kind, 'status': 'success', 'required': True,
                           'checked_utc': utc_now(), 'items': len(found), 'response_sha256': hashlib.sha256(raw).hexdigest()})
        except Exception as exc:
            # Exceptions can contain proxy URLs; persist only class and HTTP code.
            code = getattr(exc, 'code', None)
            checks.append({'url': url, 'kind': kind, 'status': 'failed', 'required': True,
                           'checked_utc': utc_now(), 'error': type(exc).__name__ + (f' HTTP {code}' if code else '')})
    successful = sum(check['status'] == 'success' for check in checks)
    return {'schema_version': 1, 'generated_utc': utc_now(), 'window_start': start.isoformat(),
            'window_end': today.isoformat(), 'query': query,
            'status': 'success' if successful == len(checks) else 'partial' if successful else 'failed',
            'scope': 'One recent arXiv query and three declared upstream release feeds; not exhaustive.',
            'source_checks': checks, 'candidates': candidates,
            'note': 'Discovery only. Review original papers, source and licenses before admission; no model execution occurred.'}


def validate_run(run):
    if run.get('schema_version') != 1 or not ID_PATTERN.fullmatch(run.get('run_id', '')):
        raise ValueError('Unsupported run schema or unsafe run ID')
    if run.get('status') not in {'success', 'partial', 'failed'}:
        raise ValueError('Run status must be success, partial or failed')
    if timestamp(run['finished_utc']) < timestamp(run['started_utc']):
        raise ValueError('Run finished before it started')
    if date.fromisoformat(run['window_start']) > date.fromisoformat(run['window_end']):
        raise ValueError('Invalid scan window')
    checks = run.get('source_checks', [])
    searches = run.get('searches', [])
    if not checks or not searches:
        raise ValueError('A run needs both recorded searches and primary-source checks')
    for check in checks:
        https_url(check['url'])
        if check['status'] not in {'success', 'partial', 'failed'}:
            raise ValueError('Invalid source-check status')
    for search in searches:
        if not search.get('query') or not search.get('provider') or search.get('status') not in {'success', 'partial', 'failed'}:
            raise ValueError('A search needs its actual query, provider and result status')
    for candidate in run.get('candidates', []):
        for url in candidate.get('source_urls', []):
            https_url(url)
    if run['status'] == 'success':
        if any(c['status'] != 'success' and c.get('required', True) for c in checks):
            raise ValueError('A successful scan cannot hide a required failed source')
        if any(s['status'] != 'success' for s in searches):
            raise ValueError('A successful scan cannot hide a failed search')
    changes = run.get('changes', {})
    changed = []
    for key in ('added_model_ids', 'upgraded_model_ids'):
        values = changes.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
            raise ValueError('Model changes must be distinct stable-ID arrays')
        changed.extend(values)
    if changed and (not run.get('validation') or any(c.get('status') != 'passed' for c in run['validation'])):
        raise ValueError('Content changes require successful recorded validation')
    for name in ('summary_zh', 'summary_en'):
        if not isinstance(run.get(name), str) or not 1 <= len(run[name]) <= 1500:
            raise ValueError('Provide a concise bilingual result summary')
    return run


def record(run, destination=None):
    """Write immutable evidence first, then derive small public UI snapshots."""
    validate_run(run)
    destination = destination or ROOT / 'maintenance'
    path = destination / 'runs' / (run['run_id'] + '.json')
    if path.exists():
        if read_json(path) != run:
            raise ValueError('Run IDs are immutable; use a new run ID for a corrected record')
    previous_path = destination / 'status.json'
    previous = read_json(previous_path) if previous_path.exists() else {}
    if previous and timestamp(run['finished_utc']) < timestamp(previous['last_attempt_utc']):
        raise ValueError('A late result must not roll the current state backwards')
    history_path = destination / 'history.json'
    history = read_json(history_path) if history_path.exists() else {'schema_version': 1, 'runs': []}
    changes = run.get('changes', {})
    latest = {key: run[key] for key in ('run_id', 'status', 'window_start', 'window_end', 'finished_utc', 'summary_zh', 'summary_en')}
    latest.update(sources_checked=sum(c['status'] == 'success' for c in run['source_checks']),
                  sources_attempted=len(run['source_checks']), candidates_reviewed=len(run.get('candidates', [])),
                  added_models=len(changes.get('added_model_ids', [])), upgraded_models=len(changes.get('upgraded_model_ids', [])),
                  record_url='runs/' + run['run_id'] + '.json')
    content_changed = bool(latest['added_models'] or latest['upgraded_models'] or changes.get('updated_content_paths'))
    status = {'schema_version': 1, 'status': run['status'], 'last_attempt_utc': run['finished_utc'],
              'last_successful_scan_utc': run['finished_utc'] if run['status'] == 'success' else previous.get('last_successful_scan_utc'),
              'last_content_change_utc': run['finished_utc'] if content_changed else previous.get('last_content_change_utc'),
              'latest_run_id': run['run_id'], 'latest_run': latest,
              'schedule': {'timezone': 'Asia/Shanghai', 'local_time': '09:00', 'executor': 'ChatGPT scheduled task'},
              'coverage_note': 'Bounded public-source checks; timestamps are evidence records, not a guarantee that every new model was found.'}
    history['runs'] = [item for item in history['runs'] if item['run_id'] != run['run_id']]
    history['runs'].insert(0, latest)
    history['runs'] = history['runs'][:90]  # Full immutable evidence stays in runs/.
    atomic_json(path, run)
    atomic_json(history_path, history)
    atomic_json(previous_path, status)
    return status


def validate_catalog(root=ROOT):
    """Validate source-route additions without importing numpy or torch."""
    teaching = json.loads((root / 'catalog.json').read_text(encoding='utf-8'))
    atlas = json.loads((root.parent / 'biomed-ai-atlas/catalog.json').read_text(encoding='utf-8'))
    sources = json.loads((root / 'sources.json').read_text(encoding='utf-8'))
    models = teaching['models']
    keys = lambda rows, field: {m[field] for m in rows}
    if len(keys(models, 'id')) != len(models) or keys(models, 'num') != set(range(1, len(models) + 1)):
        raise ValueError('Catalog has duplicate IDs or a noncontiguous model number')
    if keys(models, 'id') != keys(atlas['models'], 'id') or keys(models, 'id') != keys(sources['models'], 'model_id'):
        raise ValueError('Atlas, teaching and source registry IDs differ')
    seen_seeds, seen_payloads = set(), set()
    total = 0
    for model in models:
        path = root / 'data' / f"{model['num']:03d}.sqlite3"
        if not 1 <= path.stat().st_size <= 4 * 1024 * 1024:
            raise ValueError('Database missing or exceeds the per-model budget')
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError(f'{path.name}: SQLite integrity failure')
            metadata = {k: json.loads(v) for k, v in connection.execute('SELECT key,value FROM metadata')}
            if metadata['model_id'] != model['id'] or metadata['model_num'] != model['num']:
                raise ValueError(f'{path.name}: model identity mismatch')
            if dict(connection.execute('SELECT split, COUNT(*) FROM samples GROUP BY split')) != SPLITS:
                raise ValueError(f'{path.name}: sample split mismatch')
            for seed, sha, blob in connection.execute('SELECT seed,sha256,arrays FROM samples'):
                if seed in seen_seeds or sha in seen_payloads or hashlib.sha256(blob).hexdigest() != sha:
                    raise ValueError(f'{path.name}: duplicate seed/payload or bad hash')
                seen_seeds.add(seed)
                seen_payloads.add(sha)
            total += path.stat().st_size
    return {'status': 'passed', 'models': len(models), 'databases': len(models),
            'distinct_specimens': len(seen_seeds), 'dataset_bytes': total,
            'scope': 'Offline catalog identity, SQLite integrity, specimen hashes and cross-database distinct seeds; no model inference.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    discovery = sub.add_parser('scan', help='Fetch bounded public metadata; never mark a candidate as implemented')
    discovery.add_argument('--days', type=int, default=30)
    discovery.add_argument('--max-items', type=int, default=20)
    discovery.add_argument('--timeout', type=int, default=10)
    discovery.add_argument('--output', type=Path, required=True)
    recorder = sub.add_parser('record', help='Validate and commit an actual reviewed evidence record')
    recorder.add_argument('--input', type=Path, required=True)
    sub.add_parser('validate', help='Validate catalogs and lightweight databases offline')
    args = parser.parse_args(argv)
    if args.command == 'scan':
        result = scan(args.days, args.max_items, args.timeout)
        atomic_json(args.output, result)
        print(json.dumps({'status': result['status'], 'candidates': len(result['candidates']), 'output': str(args.output)}))
        return 0 if result['status'] == 'success' else 2
    if args.command == 'record':
        result = record(read_json(args.input))
    else:
        result = validate_catalog()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
