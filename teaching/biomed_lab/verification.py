"""Fresh-process CPU checks; these are not paper-performance reproductions."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from . import registry


def _worker(num, steps, out_dir, result_file):
    try:
        from .runner import run
        result, folder = run(num, steps=steps, out_dir=out_dir,
                             max_seconds=180, max_rss_mb=3072)
        result['run_path'] = str(folder.relative_to(registry.ROOT)) if folder.is_relative_to(registry.ROOT) else folder.name
    except Exception as exc:
        result = {'model_num': num, 'status': 'failed',
                  'error': f'{type(exc).__name__}: {exc}',
                  'traceback': traceback.format_exc()}
    Path(result_file).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def verify(query='all', out_dir=None, steps=2):
    """Run each model sequentially in a new interpreter, never reusing RSS peaks."""
    available = registry.specs()
    nums = sorted(available) if query == 'all' else [registry.resolve(query)['num']]
    destination = Path(out_dir or registry.ROOT/'runs'/'verification').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for num in nums:
        if num not in available:
            results.append({'model_num': num, 'status': 'source_route',
                            'note': 'No CPU adapter; not counted as verified'})
            continue
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='biomed-check-') as temporary:
            result_file = Path(temporary)/'result.json'
            log_file = destination/f'{num:03d}-worker.log'
            env = dict(os.environ)
            env.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'biomed-mpl'))
            try:
                with log_file.open('w', encoding='utf-8') as log:
                    process = subprocess.run(
                        [sys.executable, '-m', 'biomed_lab.verification', '--worker', str(num),
                         '--steps', str(steps), '--out-dir', str(destination),
                         '--result-file', str(result_file)],
                        cwd=registry.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        timeout=240, check=False)
                if not result_file.exists():
                    raise RuntimeError(f'Worker exited {process.returncode}; inspect {log_file.name}')
                result = json.loads(result_file.read_text(encoding='utf-8'))
                if process.returncode and result.get('status') == 'completed':
                    result.update(status='failed', error=f'Worker exited {process.returncode}')
            except subprocess.TimeoutExpired:
                result = {'model_num': num, 'status': 'budget_exceeded',
                          'error': 'Fresh worker exceeded the external 240-second timeout'}
            except Exception as exc:
                result = {'model_num': num, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        result['isolation'] = 'One fresh Python process per model; sequential execution'
        result['process_wall_seconds'] = time.perf_counter()-started
        results.append(result)
        print(f"{num:03d} {result['status']}", flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--result-file', type=Path, required=True)
    args = parser.parse_args()
    _worker(args.worker, args.steps, args.out_dir, args.result_file)
