import hashlib
import json
import sqlite3
import numpy as np
from biomed_lab import data,registry


def test_all_models_have_valid_independent_databases():
    rows=registry.entries();assert len(rows)==137
    for model in rows:
        path=data.ROOT/'data'/f"{model['num']:03d}.sqlite3"
        result=data.inspect_database(path)
        assert result['integrity']=='ok'
        assert result['counts']==data.SPLITS
        assert result['metadata']['model_id']==model['id']
        assert result['bytes']<1024*1024
        with sqlite3.connect(path) as con:
            seeds=con.execute('SELECT seed,sha256,arrays FROM samples').fetchall()
        assert len({x[0] for x in seeds})==24
        assert len({x[1] for x in seeds})==24
        for _,sha,blob in seeds:assert hashlib.sha256(blob).hexdigest()==sha


def test_noise_views_have_independent_realizations():
    a=data.make_sample(105,'noisy_shapes')
    assert not np.array_equal(a['noise_a'],a['noise_b'])
    many=[data.make_sample(s,'noisy_shapes') for s in range(100)]
    x=np.concatenate([s['noise_a'].ravel() for s in many]);y=np.concatenate([s['noise_b'].ravel() for s in many])
    assert abs(np.corrcoef(x,y)[0,1])<.02
    assert abs(x.mean())<.002 and abs(y.mean())<.002


def test_no_dataset_contains_pickle_payloads():
    # Every archive is loaded with allow_pickle=False by the production reader.
    for n in [1,7,30,75,78]:
        samples=data.load_samples(data.ROOT/'data'/f'{n:03d}.sqlite3','train')
        assert all(v.dtype!=object for s in samples for v in s.values())


def test_unknown_model_refuses_substitution():
    import pytest
    with pytest.raises(ValueError):registry.resolve('unrecognized-model')


def _budget_probe(monkeypatch, tmp_path, case):
    """Use controlled clock/memory observations, without allocating excess RAM."""
    import torch
    from torch import nn
    from biomed_lab import runner

    clock = [0.]

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.phase = 'init'

        def training_loss(self, batch):
            self.phase = 'train' if self.training else 'validation'
            clock[0] = (2.1 if case == 'final_step_time' else .1) if self.training else .2
            return self.weight.square()

    model = Probe()
    model_info = {'num': 1, 'id': 'budget-test-probe', 'title': 'budget-test-probe', 'family_code': '01'}
    spec = {'task': 'classification', 'implementation': 'mechanism_demo', 'config': {}}
    samples = [data.make_sample(20+i, 'shapes2d') for i in range(4)]
    source = tmp_path / 'dataset-probe.bin'
    source.write_bytes(b'budget-test')

    def memory():
        value = 500 if case == 'validation_memory' and model.phase == 'validation' else 100
        return value, 'deterministic test observation'

    monkeypatch.setattr(registry, 'build', lambda *args, **kwargs: (model, model_info, spec, {}))
    monkeypatch.setattr(registry, 'source', lambda *args: {})
    monkeypatch.setattr(data, 'create_database', lambda *args: source)
    monkeypatch.setattr(data, 'load_samples', lambda *args: samples)
    monkeypatch.setattr(runner, 'digest_code', lambda: 'budget-test')
    monkeypatch.setattr(runner, 'memory_usage', memory)
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: clock[0])
    return runner.run(1, steps=1, max_seconds=1, max_rss_mb=200, out_dir=tmp_path / 'runs')


def test_final_training_step_cannot_report_completed_after_time_budget(monkeypatch, tmp_path):
    result, folder = _budget_probe(monkeypatch, tmp_path, 'final_step_time')
    assert result['steps_completed'] == 1
    assert result['elapsed_seconds'] == 2.1
    assert result['status'] == 'budget_exceeded'
    assert json.loads((folder / 'result.json').read_text())['status'] == 'budget_exceeded'


def test_validation_memory_updates_peak_and_enforces_budget(monkeypatch, tmp_path):
    result, folder = _budget_probe(monkeypatch, tmp_path, 'validation_memory')
    validation = [x for x in result['losses'] if x['split'] == 'validation']
    assert validation and validation[0]['rss_mb'] == 500
    assert result['observed_peak_rss_mb'] == 500
    assert result['status'] == 'budget_exceeded'
    with sqlite3.connect(folder / 'experiment.sqlite3') as con:
        assert con.execute("SELECT rss_mb FROM metrics WHERE split='validation'").fetchone()[0] == 500
        assert json.loads(con.execute("SELECT value FROM run WHERE key='status'").fetchone()[0]) == 'budget_exceeded'
