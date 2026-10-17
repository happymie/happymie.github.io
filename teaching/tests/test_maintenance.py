"""Tests for evidence integrity and a dependency-free source-route update.

Run independently of the optional ML stack:
python -m unittest discover -s teaching/tests -p test_maintenance.py -v
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


maintenance = module('biomed_maintenance_test_target', ROOT / 'tools/maintenance.py')
builder = module('biomed_site_builder_test_target', ROOT / 'tools/build_site.py')


def example_run(run_id='20261017T090000Z-test', status='success'):
    return {'schema_version': 1, 'run_id': run_id, 'status': status,
            'started_utc': '2026-10-17T08:59:00Z', 'finished_utc': '2026-10-17T09:00:00Z',
            'window_start': '2026-09-17', 'window_end': '2026-10-17',
            'searches': [{'query': 'medical reconstruction', 'provider': 'mock provider', 'status': 'success'}],
            'source_checks': [{'url': 'https://example.org/paper', 'status': 'success', 'required': True}],
            'candidates': [], 'changes': {'added_model_ids': [], 'upgraded_model_ids': []},
            'validation': [], 'summary_zh': '仅测试夹具，不是实际网络检查。',
            'summary_en': 'Test fixture only; not a real network check.'}


class MaintenanceRecords(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name)

    def test_no_change_success_updates_scan_date_without_claiming_content_change(self):
        run = example_run()
        result = maintenance.record(run, self.destination)
        self.assertEqual(result['last_successful_scan_utc'], run['finished_utc'])
        self.assertIsNone(result['last_content_change_utc'])
        self.assertEqual(result['latest_run']['added_models'], 0)

    def test_partial_failure_retains_last_success_and_can_record_validated_content(self):
        previous = maintenance.record(example_run(), self.destination)
        run = example_run('20261018T090000Z-partial', 'partial')
        run['started_utc'], run['finished_utc'] = '2026-10-18T08:00:00Z', '2026-10-18T09:00:00Z'
        run['source_checks'].append({'url': 'https://example.org/blocked', 'status': 'failed', 'required': True})
        run['changes']['added_model_ids'] = ['architecture_model_138_phydict']
        run['validation'] = [{'name': 'synthetic test fixture', 'status': 'passed'}]
        result = maintenance.record(run, self.destination)
        self.assertEqual(result['last_successful_scan_utc'], previous['last_successful_scan_utc'])
        self.assertEqual(result['last_content_change_utc'], run['finished_utc'])
        self.assertEqual(result['latest_run']['added_models'], 1)

    def test_success_cannot_hide_failed_primary_source_or_search(self):
        run = example_run()
        run['source_checks'][0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'cannot hide'):
            maintenance.record(run, self.destination)
        run = example_run()
        run['searches'][0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'cannot hide'):
            maintenance.record(run, self.destination)

    def test_unvalidated_model_admission_is_rejected(self):
        run = example_run()
        run['changes']['added_model_ids'] = ['architecture_model_138_phydict']
        with self.assertRaisesRegex(ValueError, 'validation'):
            maintenance.record(run, self.destination)

    def test_immutable_record_and_exact_replay(self):
        run = example_run()
        first = maintenance.record(run, self.destination)
        second = maintenance.record(copy.deepcopy(run), self.destination)
        self.assertEqual(first, second)
        self.assertEqual(len(json.loads((self.destination / 'history.json').read_text())['runs']), 1)
        run['summary_en'] = 'Changed fixture'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            maintenance.record(run, self.destination)

    def test_replay_recovers_snapshots_after_interrupted_write(self):
        run = example_run()
        maintenance.atomic_json(self.destination / 'runs' / (run['run_id'] + '.json'), run)
        result = maintenance.record(run, self.destination)
        self.assertEqual(result['latest_run_id'], run['run_id'])

    def test_safe_paths_timestamps_and_public_urls(self):
        for field, value in [('run_id', '../../outside'), ('finished_utc', '2020-01-01T01:00:00Z'), ('started_utc', '2026-10-17T00:00:00')]:
            run = example_run()
            run[field] = value
            with self.assertRaises(ValueError):
                maintenance.record(run, self.destination)
        for url in ('javascript:alert(1)', 'http://example.org', 'https://private:secret@example.org'):
            with self.assertRaises(ValueError):
                maintenance.https_url(url)

    def test_stale_late_result_cannot_roll_state_back(self):
        maintenance.record(example_run(), self.destination)
        run = example_run('20261016T090000Z-late')
        run['started_utc'], run['finished_utc'] = '2026-10-16T08:00:00Z', '2026-10-16T09:00:00Z'
        with self.assertRaisesRegex(ValueError, 'backwards'):
            maintenance.record(run, self.destination)


class BoundedDiscovery(unittest.TestCase):
    def test_failed_requests_are_not_reported_as_success_and_hide_proxy_details(self):
        with mock.patch.object(maintenance, 'public_get', side_effect=OSError('https://secret:password@proxy.invalid')):
            result = maintenance.scan()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['source_checks']), 4)
        self.assertNotIn('password', json.dumps(result))

    def test_allowed_hosts_and_redirect_refusal(self):
        for url in ('https://127.0.0.1/metadata', 'https://api.github.com:8443/repos/a/b', 'https://api.github.com.evil.test/'):
            with self.assertRaises(ValueError):
                maintenance.public_get(url)
        with self.assertRaisesRegex(ValueError, 'redirect blocked'):
            maintenance.NoRedirect().redirect_request(urllib.request.Request('https://api.github.com/'), None, 302, '', {}, 'https://example.org/')

    def test_budget_limits_fail_before_network(self):
        for kwargs in ({'days': 31}, {'max_items': 51}, {'timeout': 16}):
            with self.assertRaises(ValueError):
                maintenance.scan(**kwargs)


class RealSourceRouteFiles(unittest.TestCase):
    def test_catalog_database_and_existing_adapters(self):
        result = maintenance.validate_catalog()
        self.assertEqual(result['models'], result['databases'])
        self.assertGreaterEqual(result['models'], 138)
        adapters, snapshot = builder.adapters()
        self.assertIn(6, adapters)  # The existing U-Net adapter must remain visible.
        self.assertNotIn(138, adapters)
        self.assertEqual(snapshot['runtime_sha256'], builder.digest_code(include_sources=False))

    def test_changed_runtime_cannot_silently_reuse_snapshot(self):
        with mock.patch.object(builder, 'digest_code', return_value='changed-runtime'):
            with self.assertRaisesRegex(ValueError, 'Runtime changed'):
                builder.adapters()

    def test_compact_catalog_and_new_source_route_remain_truthful(self):
        catalog = json.loads((ROOT / 'web/catalog.json').read_text())
        detail = json.loads((ROOT / 'web/lessons/138.json').read_text())
        self.assertEqual(detail['id'], 'architecture_model_138_phydict')
        self.assertEqual(detail['implementation'], 'source_route')
        self.assertIsNone(detail['adapter'])
        self.assertIsNone(detail['verification'])
        self.assertLess((ROOT / 'web/catalog.json').stat().st_size, (ROOT / 'manifest.json').stat().st_size / 5)
        self.assertTrue(all('source' not in model and 'adapter' not in model for model in catalog['models']))
        evidence = json.loads((ROOT / 'results/verification.json').read_text())
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for record in evidence:
            item = next(model for model in manifest['models'] if model['num'] == record['model_num'])
            self.assertEqual(item['verification']['run_id'], record['run_id'])
            self.assertEqual(item['verification']['source_code_sha256'], record['source_code_sha256'])
            self.assertEqual(item['verification']['status'], 'historical')


if __name__ == '__main__':
    unittest.main()
