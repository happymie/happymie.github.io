"""Budgeted local CPU training with measured results and immutable run folders."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data, registry


def loss_for(model,batch,spec):
    if hasattr(model,"training_loss"):
        loss=model.training_loss(batch)
    else:
        task=spec["task"]
        if task in {"classification","classification2d","image_classification"}:
            pred=model(batch['image']); target=batch['label']
            if isinstance(pred,dict): pred=pred['out']
            loss=F.cross_entropy(pred,target)
        elif task in {"segmentation","segmentation2d","segmentation_2d"}:
            pred=model(batch['image'])
            if isinstance(pred,dict): pred=pred['out']
            if pred.shape[1]==1: loss=F.binary_cross_entropy_with_logits(pred[:,0],batch['mask'].float())
            else: loss=F.cross_entropy(pred,batch['mask'])
        elif task in {"segmentation3d","segmentation_3d"}:
            pred=model(batch['volume'])
            loss=F.cross_entropy(pred,batch['volume_mask'])
        elif task in {"restoration","denoising"}:
            loss=F.mse_loss(model(batch['noisy']),batch['clean'])
        elif task in {"sequence","sequence_classification"}:
            loss=F.cross_entropy(model(batch['sequence']),batch['sequence_label'])
        else:
            raise ValueError(f"Task {task!r} must implement training_loss(batch)")
    if not isinstance(loss,torch.Tensor) or loss.ndim!=0:
        raise ValueError("An experiment must return a scalar torch loss")
    if not torch.isfinite(loss): raise FloatingPointError("Loss is not finite")
    return loss


def digest_code():
    h=hashlib.sha256()
    files=list((registry.ROOT/"biomed_lab").rglob('*.py'))
    files += [p for p in (registry.ROOT/'vendor').rglob('*') if p.is_file() and (p.suffix in {'.py','.json'} or 'LICENSE' in p.name)]
    files += [registry.ROOT/'sources.json',registry.ROOT/'pyproject.toml']
    for p in sorted(files):
        h.update(str(p.relative_to(registry.ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def memory_usage():
    """Use this process's kernel high-water mark on POSIX.

    psutil/procfs can refer to a synthetic PID in nested container namespaces;
    resource.RUSAGE_SELF is independent of that procfs PID mapping.
    """
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale=2**20 if platform.system()=='Darwin' else 1024
        return value/scale,'resource.RUSAGE_SELF.ru_maxrss (process lifetime high-water mark)'
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss/2**20,'psutil resident memory (sampled; Windows fallback)'


def run(query,steps=20,batch_size=2,seed=2026,threads=2,device='cpu',overrides=None,
        out_dir=None,max_seconds=120,max_rss_mb=3072,learning_rate=1e-3,save_checkpoint=False):
    if not (1<=steps<=10000 and 2<=batch_size<=64 and 1<=threads<=8):
        raise ValueError('Allowed: steps 1..10000; batch size 2..64; CPU threads 1..8')
    if max_seconds<=0 or max_rss_mb<=0: raise ValueError('Budgets must be positive')
    if device not in {'cpu','cuda','mps'}: raise ValueError('Choose cpu, cuda or mps')
    if device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; use --device cpu')
    if device=='mps' and not torch.backends.mps.is_available(): raise ValueError('MPS unavailable; use --device cpu')
    torch.set_num_threads(threads)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    started=time.perf_counter()
    model,m,spec,config=registry.build(query,overrides)
    model=model.to(device)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+f"-{m['num']:03d}"
    folder=(Path(out_dir) if out_dir else registry.ROOT/'runs')/run_id
    folder.mkdir(parents=True,exist_ok=False)
    db=data.create_database(m)
    train=data.load_samples(db,'train'); val=data.load_samples(db,'validation')
    image_size=int(config.get('image_size',config.get('img_size',32)))
    # Square input size only; volumetric dimensions are handled by each adapter.
    if not 16<=image_size<=512: raise ValueError('CPU teaching image_size must be 16..512')
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    metadata={"run_id":run_id,"model_num":m['num'],"title":m['title'],"implementation":spec['implementation'],
              "scope":"Synthetic CPU training check; not an original-paper or clinical benchmark reproduction",
              "config":config,"requested_steps":steps,"batch_size":batch_size,"seed":seed,"threads":threads,"device":device,
              "learning_rate":learning_rate,"parameters":parameters,"torch":torch.__version__,"python":platform.python_version(),
              "platform":platform.platform(),"source_code_sha256":digest_code(),"dataset_sha256":hashlib.sha256(db.read_bytes()).hexdigest(),
              "source":registry.source(m['num']),"adapter_source":spec.get('source_url'),
              "dependencies":{name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','monai','timm','torch-geometric','fastmri','einops','ml-collections','transformers','huggingface-hub','psutil','scipy','scikit-image','tifffile','h5py','torchmetrics','pytorch-lightning')},
              "max_seconds":max_seconds,"max_rss_mb":max_rss_mb,
              "memory_measurement":memory_usage()[1],"budget_enforcement":"cooperative boundary checks; not an OS allocation limit",
              "started_utc":datetime.now(timezone.utc).isoformat()}
    (folder/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    connection=sqlite3.connect(folder/'experiment.sqlite3')
    connection.execute('CREATE TABLE run (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    connection.execute('CREATE TABLE metrics (step INTEGER, split TEXT, loss REAL, elapsed_seconds REAL, rss_mb REAL, PRIMARY KEY(step,split))')
    for k,v in metadata.items(): connection.execute('INSERT INTO run VALUES (?,?)',(k,json.dumps(v,ensure_ascii=False)))
    peak=memory_usage()[0]
    history=[]; status='completed'; error=None
    try:
        for step in range(1,steps+1):
            elapsed=time.perf_counter()-started
            rss=memory_usage()[0];peak=max(peak,rss)
            if elapsed>max_seconds or rss>max_rss_mb:
                status='budget_exceeded';break
            same_scene=spec.get('evaluation_scope')=='same_scene'
            idx=([0]*batch_size if same_scene else [(step*batch_size+i)%len(train) for i in range(batch_size)])
            batch=data.batch_from_samples([train[i] for i in idx],device,image_size)
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model,batch,spec);loss.backward()
            grad=[p.grad for p in model.parameters() if p.grad is not None]
            if not grad or not all(torch.isfinite(g).all() for g in grad):
                raise FloatingPointError('Missing or non-finite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if hasattr(model,'after_optimizer_step'): model.after_optimizer_step()
            item={"step":step,"split":"train","loss":float(loss.detach()),"elapsed_seconds":time.perf_counter()-started,"rss_mb":memory_usage()[0]}
            peak=max(peak,item['rss_mb']);history.append(item)
            connection.execute('INSERT INTO metrics VALUES (?,?,?,?,?)',tuple(item.values()))
            if item['elapsed_seconds']>max_seconds or peak>max_rss_mb:
                status='budget_exceeded';break
            if step in {1,steps}:
                # Validation never enters the optimizer. State is restored because
                # some paper losses maintain queues/teacher centers on forward.
                state={k:v.detach().clone() for k,v in model.state_dict().items()}
                rng_state=torch.random.get_rng_state()
                py_state=random.getstate();np_state=np.random.get_state()
                accelerator_rng=(torch.cuda.get_rng_state_all() if device=='cuda' else torch.mps.get_rng_state() if device=='mps' else None)
                model.eval()
                try:
                    with torch.no_grad():
                        vb=data.batch_from_samples(([train[0]]*batch_size if same_scene else val[:batch_size]),device,image_size)
                        vloss=float(loss_for(model,vb,spec))
                    vi={"step":step,"split":"same_scene_diagnostic" if same_scene else "validation","loss":vloss,"elapsed_seconds":time.perf_counter()-started,"rss_mb":memory_usage()[0]}
                    peak=max(peak,vi['rss_mb'])
                    history.append(vi);connection.execute('INSERT INTO metrics VALUES (?,?,?,?,?)',tuple(vi.values()))
                finally:
                    model.load_state_dict(state);torch.random.set_rng_state(rng_state)
                    random.setstate(py_state);np.random.set_state(np_state)
                    if device=='cuda':torch.cuda.set_rng_state_all(accelerator_rng)
                    elif device=='mps':torch.mps.set_rng_state(accelerator_rng)
                    del state
                if time.perf_counter()-started>max_seconds or peak>max_rss_mb:
                    status='budget_exceeded';break
            connection.commit()
        if save_checkpoint:
            torch.save({'model':model.state_dict(),'config':config,'model_num':m['num'],'seed':seed},folder/'model.pt')
    except Exception as exc:
        status='failed';error=f'{type(exc).__name__}: {exc}'
    finally:
        peak=max(peak,memory_usage()[0])
        if status=='completed' and (time.perf_counter()-started>max_seconds or peak>max_rss_mb):
            status='budget_exceeded'
        summary={"model_num":m['num'],"title":m['title'],"status":status,"error":error,"steps_completed":len([x for x in history if x['split']=='train']),
                 "parameters":parameters,"elapsed_seconds":time.perf_counter()-started,"observed_peak_rss_mb":peak,
                 "memory_measurement":metadata['memory_measurement'],"budget_enforcement":metadata['budget_enforcement'],
                 "implementation":spec['implementation'],"seed":seed,"batch_size":batch_size,"threads":threads,"device":device,
                 "run_id":run_id,"source_code_sha256":metadata['source_code_sha256'],"dataset_sha256":metadata['dataset_sha256'],
                 "config":config,"dependencies":metadata['dependencies'],"adapter_source":metadata['adapter_source'],
                 "losses":history,"evaluation_scope":spec.get('evaluation_scope','held_out_synthetic_specimens'),"limitations":spec.get('limitations',[])}
        for k in ('status','error'): connection.execute('INSERT INTO run VALUES (?,?)',(k,json.dumps(summary[k])))
        connection.commit();connection.close()
        (folder/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary,folder
