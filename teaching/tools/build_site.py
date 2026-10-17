"""Build the static curriculum from code, source evidence and real SQLite data."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLITS = {'train': 16, 'validation': 4, 'test': 4}


def digest_code(include_sources=True):
    """Match the runner's digest without importing the machine-learning stack."""
    files = list((ROOT / 'biomed_lab').rglob('*.py'))
    files += [p for p in (ROOT / 'vendor').rglob('*') if p.is_file()
              and (p.suffix in {'.py', '.json'} or 'LICENSE' in p.name)]
    files += [ROOT / 'pyproject.toml']
    if include_sources:
        files.append(ROOT / 'sources.json')
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def adapters(refresh=False):
    """Metadata-only rebuilds require an unchanged executable source tree.

    A changed runtime must refresh this snapshot in the real ML environment;
    an absent dependency never silently converts an adapter to a source route.
    This snapshot records constructor metadata, never new execution evidence.
    """
    path = ROOT / 'adapter_snapshot.json'
    runtime_hash = digest_code(include_sources=False)
    if refresh:
        from biomed_lab import registry
        snapshot = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'provenance': 'Constructor metadata imported from the actual registry; not an execution check.',
                    'source_code_sha256': digest_code(), 'runtime_sha256': runtime_hash,
                    'adapters': {str(n): spec for n, spec in registry.specs().items()}}
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    if snapshot.get('runtime_sha256') != runtime_hash:
        raise ValueError('Runtime changed after adapter snapshot: run build_site.py --refresh-adapters in the pinned ML environment, then verify affected models. Offline metadata must not invent adapters.')
    return {int(n): spec for n, spec in snapshot['adapters'].items()}, snapshot


def inspect_database(path):
    """Standard-library integrity check; read-only and bounded to teaching data."""
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError(f'{path.name}: database exceeds the 4 MiB teaching budget')
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        metadata = {k: json.loads(v) for k, v in connection.execute('SELECT key,value FROM metadata')}
        counts = dict(connection.execute('SELECT split,COUNT(*) FROM samples GROUP BY split'))
        integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
        for sample_id, expected, payload in connection.execute('SELECT sample_id,sha256,arrays FROM samples'):
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'{path.name}: corrupt specimen {sample_id}')
    return {'metadata': metadata, 'counts': counts, 'bytes': path.stat().st_size, 'integrity': integrity}

STATUS = {
    'upstream': '复用上游实现 / Upstream adapter',
    'paper_port': '独立论文结构或方程实现 / Paper port',
    'mechanism_demo': '部分机制演示 / Mechanism demo',
    'source_route': '原版学习路径，尚无 CPU 适配器 / Source route',
}
GAPS = {
    9: 'nnU-Net includes dataset fingerprinting, automatic planning, preprocessing, training and inference. A generic U-Net is not the system; the present upstream default is v2, so the paper-matched version and full pipeline must be selected.',
    14: 'The original VISTA3D point/class-conditioned 3D architecture and pretrained workflow have not been integrated or checked here.',
    31: 'The complete BLIP-2 two-stage recipe and pretrained frozen encoders/language model are not reproduced.',
    34: 'SAM2 requires its hierarchical image encoder, memory encoder/attention and video prompt/state protocol; these are not provided by the SAM1 adapter.',
    35: 'The identified author release contains evaluation material, not a complete verified ConVIRT training implementation. The image/text encoders, pair sampling and full pretraining recipe still need a paper-based implementation.',
    36: 'The defining local/global radiology image–text alignment and complete report preprocessing/training workflow are not implemented here.',
    37: 'The official BioViL implementation is in Microsoft hi-ml. Its image encoder, clinical language model and original pretraining stages have not been integrated into this CPU lab.',
    38: 'The published BiomedCLIP architecture, biomedical tokenizer and pretraining workflow are not supplied by the generic CLIP lesson. No BiomedCLIP weights are bundled.',
    39: 'The UNI pretrained pathology representation and data/training recipe are not reproduced. The upstream CC BY-NC-ND terms are not replaced by this repository license.',
    40: 'CONCH requires its pathology vision–language architecture and objectives; neither the CLIP lesson nor a generic pathology classifier is a CONCH reproduction.',
    41: 'The complete tile encoder plus slide-level long-context aggregation and original pretraining stages are not implemented in the present CPU lab.',
    42: 'The Merlin 3D imaging and clinical-text pretraining pipeline has not been integrated; random 3D classification is not Merlin.',
    58: 'A faithful latent-diffusion workflow needs its learned first-stage representation, latent scaling and conditional diffusion model. The pixel-space DDPM lesson is not substituted.',
    61: 'The pretrained score prior, 3D/2.5D reconstruction workflow and full DiffusionMBIR data-consistency procedure have not been implemented or checked.',
    69: 'The uniGradICON foundation registration model and training distribution are not supplied by the GradICON mechanism demonstration.',
    95: 'The original large-scale ALIGN noisy-pair pretraining recipe and weights are not reproduced.',
    96: 'BLIP requires the image–text contrastive/matching/language objectives and captioning/filtering workflow; the full original method is not implemented here.',
    97: 'Florence-2 uses a task-conditioned vision/language architecture and task-specific token/output processing. Its released checkpoint and full workflow are not integrated here.',
    98: 'The source audit did not establish a root license grant for MedCLIP. No permission to copy was inferred; its semantic-matching objective and encoders are not reimplemented here.',
    99: 'Med-Flamingo requires its visual resampler, cross-attention language bridge and original adaptation protocol. A generic image–text contrastive network is not substituted.',
    100: 'The LLaVA-Med visual projector, language backbone and two-stage medical instruction tuning are not implemented or validated here.',
    101: 'RadFM requires a volumetric vision encoder, language bridge and multimodal instruction pipeline. This is not replaced by a 3D classifier.',
    102: 'An accessible complete MAIRA-1 release was not established in the source audit. MAIRA-2 is a different model. Paper-only implementation details remain unresolved.',
    103: 'The MAIRA-2 image/report grounding and language generation pipeline is not integrated; the published model-card/access terms remain upstream.',
    104: 'BiomedGPT requires its unified multimodal token/sequence tasks and training mixture. A small generic Transformer does not reproduce these stages.',
    105: 'Virchow pretrained pathology representations and their data/training protocol are not reproduced by the generic DINOv2 mechanism lesson.',
    106: 'The CHIEF pathology tile/slide representation and clinical-context workflow are not integrated into this CPU teaching runner.',
    107: 'TITAN uses a complete slide-level visual/text representation pipeline; original training stages and pretrained capabilities are not reproduced here.',
    108: 'PRISM2 public metadata is available, but its model access is gated and its terms restrict redistribution. No gated model files are mirrored and no local PRISM2 implementation is claimed.',
    109: 'The original BME-X implementation is linked and pinned; its complete multimodal learning workflow has not been converted into a tested low-resource adapter.',
    110: 'The original Prima implementation is linked and pinned; its scan/series representation and complete training/inference protocol are not implemented locally.',
    111: 'The original MARS implementation is linked and pinned; its defining pretraining and downstream workflow have not been integrated and tested here.',
    112: 'No complete public Decipher-MR release was verified. The named DINOv2/BERT/OpenCLIP components are not themselves Decipher-MR, and the paper states pretrained weights may not be shared.',
    127: 'DeepDRR includes calibrated geometry, material decomposition/scatter and physical image formation. The differentiable toy ray projectors here are not a DeepDRR replacement.',
    130: 'MedNeRF requires its radiance-field adversarial training and original conditioning pipeline. The generic NeRF mechanism demo is not renamed as MedNeRF.',
    134: 'Voxel2Mesh needs its deformable mesh initialization, graph-convolution refinement, adaptive unpooling and mesh losses. The graph classification lessons are not substitutes; the original GPL source remains linked.',
    135: 'Vox2Cortex requires cortical surface topology, volumetric feature sampling and coupled mesh refinement/losses. These have not been implemented in this CPU lab; original GPL source remains linked.',
    138: 'The official PhyDiCT inference source is available and pinned. Its pretrained MedSyn 3D diffusion prior, report embedding, calibrated projection geometry, LPIPS guidance, and CUDA-oriented sampling have not been adapted or executed locally. The synthetic database supports source study only; it is not a PhyDiCT reconstruction experiment.',
}
TASK_EXERCISES = {
    'seg': ('画出每一尺度的空间/通道尺寸并追踪跳连。', '把宽度或输入尺寸减半，记录参数量、内存和边界误差；说明配置改变了什么。', '检查 mask 重采样使用最近邻，解释 Dice/交叉熵所监督的对象。'),
    'sequence': ('逐步说明序列的时间轴、特征轴与状态更新。', '修改未来输入，检查声称因果的模块是否改变过去输出。', '比较较短状态/通道配置；将结构限制与训练结果分别记录。'),
    'inverse': ('写出观测 y、未知量 x、前向算子 A 与噪声假设。', '区分先验训练与重建/采样步骤，确认数据一致性实际进入哪一条路径。', '检查伴随、坐标单位或逆映射；用模型文档中的方程测试解释为何输出 shape 不足以验收。'),
    'representation': ('列出两个视图/模态的配对关系、正负样本或 mask 目标。', '检查教师、目标分支、队列与冻结编码器的梯度/状态更新规则。', '打乱配对或改变 mask/温度，记录固定输入下目标函数如何变化；讨论小批次局限。'),
    'graph': ('说明节点、边和图的统计单位；验证完整图的训练/验证种子不同。', '一致地置换节点及其邻接/标签，检查输出同步置换。', '指出该层如何聚合邻居，以及哪些原论文边特征或图级目标尚未实现。'),
    'default': ('根据实际代码画出输入到损失的计算图，标注每个张量维度。', '用默认配置记录基线，再只改变一个支持的模型参数，比较参数量与 RSS。', '阅读逐项差异并选择一个结构或梯度性质做反例检查；解释两步训练不能证明什么。'),
}


def group(task):
    if 'seg' in task: return 'seg'
    if any(x in task for x in ('registration','reconstruction','field','generative','denois','restoration','transform','gan','diffusion')): return 'inverse'
    if any(x in task for x in ('self_supervised','image_text','masked','representation','contrastive')): return 'representation'
    if 'graph' in task: return 'graph'
    if 'sequence' in task: return 'sequence'
    return 'default'


def safe_cell(v):
    return str(v).replace('|', '\\|').replace('\n', ' ')


def lesson(m):
    n=f"{m['num']:03d}"; s=m['source']; a=m['adapter']; v=m.get('verification'); ds=m['dataset']
    lines=[f"# {n} · {m['title']}", '',
           f"[教学网站](https://happymie.github.io/teaching/#model={m['num']}) · [返回覆盖表](../docs/COVERAGE.md) · [原图谱结构与数学](https://happymie.github.io/biomed-ai-atlas/#/model/{m['id']})", '',
           f"**实现状态：{STATUS[m['implementation']]}**", '',
           f"{m['core_mech']}", '', m['core_mech_en'], '',
           '## 1. 输入、输出与学习目标', '',
           f"- 原模型输入：{m.get('input_desc','参见原论文')} / {m.get('input_en','')}",
           f"- 原模型输出：{m.get('output_desc','参见原论文')} / {m.get('output_en','')}",
           '- 分辨原模型的必要计算机制、本地缩小配置和没有覆盖的原训练阶段。',
           '- 从数据、损失、梯度到日志完成可检查的学习记录，保留失败与不确定性。', '',
           '## 2. 原始证据与实际实现来源', '',
           f"- 论文：[{m.get('paper_title') or m['title']}]({m['paper_url']})",
           f"- 原版来源：[{s.get('source_kind','source')}]({s.get('source_url') or s.get('repo_url') or m['paper_url']})",
           f"- 来源档案固定版本：`{s.get('commit') or '未固定 / unpinned'}`；检查日期：{s.get('evidence_checked')}",
           f"- 来源许可证：`{s.get('license_spdx','NOASSERTION')}`"+(f" · [原文]({s['license_url']})" if s.get('license_url') else ''),
           '- 原始仓库的档案版本不自动等于本适配器执行的库版本，也不自动等于原论文历史版本。', '']
    if a:
        lines += [f"- 实际执行来源：{a.get('source_url') or m['paper_url']}",
                  f"- 本地代码：[biomed_lab/models/{a['module']}.py](../biomed_lab/models/{a['module']}.py)",
                  '- 依赖：[CPU 固定版本](../requirements-cpu.txt) / [Apple Silicon 固定版本](../requirements-macos.txt)；每次运行另外记录安装版本。', '',
                  '## 3. 本地实现与 CPU 配置', '', a['mechanism'], '',
                  '```json',json.dumps(a['config'],ensure_ascii=False,indent=2),'```','',
                  '配置直接来自实际注册表。宽度、深度、头数或网格尺寸的改变必须满足结构约束；不要把缩小配置的结果写成原模型结果。', '',
                  '**已知差异 / Limitations**', '']
        lines += [f'- {x}' for x in a['limitations']]
    else:
        lines += ['## 3. 尚未完成的实现', '', GAPS.get(m['num'],'This original method has no implemented and tested local CPU adapter.'), '',
                  '这里提供原版路径和教学数据，**没有可运行的本地模型，不计入已验证适配器**。需要先按论文/作者代码补齐上述组成，再完成结构、目标函数、梯度与真实任务数据验证。不能把同家族网络改名作为替代。']
    lines += ['', '## 4. 轻量数据库', '',
              f"[下载 {n}.sqlite3](../data/{n}.sqlite3) · `{ds['kind']}` · {ds['bytes']/1024:.0f} KiB · CC0-1.0", '',
              '每库 24 个独立合成样本：16 train / 4 validation / 4 test。数据库含 metadata 与 samples 表、真实数组、标签、种子和 SHA-256。噪声视图按原样本种子确定性生成；不跨样本划分。', '',
              f"数据 SHA-256：`{ds['sha256']}`。数据类型可能经过适配器的明确尺寸/模态转换；这些无量纲合成观察不是原论文或临床数据。", '',
              f"评价范围：{'同场景/同图像对拟合诊断，不是新场景泛化。' if a and a.get('evaluation_scope')=='same_scene' else '独立合成验证样本；保留测试划分不参与默认训练。' if a else '尚无本地模型评价。'}", '',
              '## 5. 运行或获取原版', '',
              '先按 [README](../README.md) 安装，在 `teaching/` 目录执行。以下是 Linux/Apple Silicon 路径；Windows 将 `.venv/bin/python` 换为 `.venv\\Scripts\\python.exe`。', '',
              '```bash',f'.venv/bin/python -m biomed_lab info {n}',f'.venv/bin/python -m biomed_lab data {n} --inspect']
    if a:
        lines += [f'.venv/bin/python -m biomed_lab run {n} --steps 20 --threads 2 --batch-size 2 --device cpu']
    elif (s.get('repo_url') or '').startswith('https://github.com/') and s.get('commit'):
        lines += [f'.venv/bin/python -m biomed_lab fetch-upstream {n}']
    lines += ['```', '', '运行日志在新的 `runs/` 子目录中；验证状态、预算和失败均需保留。获取原版命令只下载固定源码，不安装、执行原脚本或下载权重。部分原版流程所需数据、访问与计算资源仍由作者文档规定。', '',
              '## 6. 当前执行证据', '']
    if v and v.get('status') in {'completed', 'historical'}:
        label = '已有版本的 CPU 执行记录（本次维护未重跑）' if v['status'] == 'historical' else '代码/数据摘要匹配的 CPU 执行记录（生成网页不会重跑模型）'
        lines += [f"{label}：**{v['steps_completed']} 步**，{v['parameters']:,} 参数，模型构造起耗时 {v['elapsed_seconds']:.3f} 秒，进程峰值 RSS {v['observed_peak_rss_mb']:.1f} MiB。", '',
                  f"原始运行编号：`{v.get('run_id', 'not recorded')}`；显示状态：`{v['status']}`。", '',
                  f"方法：{v['memory_measurement']}。{v['isolation']}", '',
                  f"源代码 SHA-256：`{v['source_code_sha256']}`。", '',
                  '这是发布环境的短程执行检查，不是笔记本速度保证、收敛证明、预训练复现或临床/原基准结果。[完整记录](../results/verification.json)。']
    else:
        lines += ['尚无与当前代码/数据匹配的成功 CPU 检查；不计入当前已验证实验。'+(f" 记录状态：`{v.get('status')}`。" if v else '')]
    lines += ['', '## 7. 练习与验收', '']
    exercises=TASK_EXERCISES[group(a['task'])] if a else ('列出原论文不可缺少的模块与目标函数，对照当前来源确认哪些细节尚不完整。','逐项核查源码、依赖、数据和权重的许可及可用性；在独立目录获取允许使用的材料。','为尚缺实现写出明确的最小验收：结构、方程、真实梯度/状态行为和资源记录；在通过前保留 source_route 状态。')
    lines += [f'{i}. {x}' for i,x in enumerate(exercises,1)]
    lines += ['', '提交一份包含配置、数据/代码摘要、实验现象和限制的记录。完整论文复现的进一步要求见 [复现标准](../docs/REPRODUCTION_STANDARD.md)。', '', '## 来源附注', '']
    lines += [f'- {x}' for x in s.get('notes',[])] or ['- 固定代码版本不构成原论文结果复现的证明。']
    if m['num'] == 138:
        lines += ['', '## 专项来源学习', '', '[PhyDiCT 输入条件、来源读取范围与 CPU 适配验收](../docs/PHYDICT_SOURCE_ROUTE.md)。本库中的二维形状观察不是三维椭球的标定投影，不得当作原方法输入对。']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-adapters', action='store_true', help='Import real model constructors in the pinned ML environment; does not run training')
    args = parser.parse_args(argv)
    catalog=json.loads((ROOT/'catalog.json').read_text()); rows=catalog['models']
    sources=json.loads((ROOT/'sources.json').read_text())['models']
    source_by={s['num']:s for s in sources}; available,snapshot=adapters(args.refresh_adapters)
    assert len(rows)==len({m['id'] for m in rows})==len({m['num'] for m in rows})
    assert {m['num'] for m in rows}==set(range(1,len(rows)+1))
    assert len(source_by)==len(rows) and set(available).issubset(source_by)
    code_hash=digest_code()
    records=json.loads((ROOT/'results/verification.json').read_text()) if (ROOT/'results/verification.json').exists() else []
    evidence={v['model_num']:v for v in records}
    models=[]
    for original in sorted(rows,key=lambda x:x['num']):
        m=dict(original);n=m['num'];s=source_by[n]
        assert s['model_id']==m['id'] and s['title']==m['title']
        path=ROOT/'data'/f'{n:03d}.sqlite3'; info=inspect_database(path)
        assert info['integrity']=='ok' and info['counts']==SPLITS
        assert info['metadata']['model_num']==n and info['metadata']['model_id']==m['id']
        assert info['metadata']['generator_sha256']==hashlib.sha256((ROOT/'biomed_lab/data.py').read_bytes()).hexdigest()
        ds={'bytes':info['bytes'],'kind':info['metadata']['kind'],'counts':info['counts'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        a=available.get(n);v=dict(evidence[n]) if n in evidence else None
        if v and v.get('status')=='completed' and (v.get('source_code_sha256')!=code_hash or v.get('dataset_sha256')!=ds['sha256']):
            if v.get('source_code_sha256')==snapshot.get('source_code_sha256') and v.get('dataset_sha256')==ds['sha256']:
                v.update(status='historical', evidence_note='Original successful execution retained at its original version and date. Executable code and this dataset match the captured release; source registry metadata changed. No model was rerun by this build.')
            else:
                v.update(status='stale',error='Runtime or dataset cannot be matched to this record; rerun verification.')
        m.update(source=s,adapter=a,implementation=a['implementation'] if a else 'source_route',dataset=ds,
                 verification=v,gaps=[GAPS.get(n,'Original method has no implemented and tested local CPU adapter.')])
        models.append(m)
    counts=collections.Counter(m['implementation'] for m in models)
    checked=sum(m.get('verification',{}).get('status')=='completed' for m in models if m.get('verification'))
    historical=sum(m.get('verification',{}).get('status')=='historical' for m in models if m.get('verification'))
    total_bytes=sum(m['dataset']['bytes'] for m in models)
    manifest={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'source_code_sha256':code_hash,
              'runtime_sha256':snapshot['runtime_sha256'], 'counts':dict(counts),'verified':checked,'historical_verified':historical,
              'verification_note':'Building the website does not rerun experiments. Historical records retain the original run ID and digest.',
              'dataset_total_bytes':total_bytes,'families':catalog['families'],'models':models}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    # The browser list needs compact cards, not all source/adapter/run records.
    # Full per-lesson records are loaded only when a learner opens that lesson.
    web = ROOT / 'web'
    (web / 'lessons').mkdir(parents=True, exist_ok=True)
    card_keys = ('num','num_str','id','title','year','family_code','family_short','family_en','core_mech','core_mech_en','implementation')
    cards = []
    for model in models:
        card = {key: model[key] for key in card_keys}
        card.update(has_adapter=bool(model['adapter']),
                    dataset={key: model['dataset'][key] for key in ('bytes','kind')},
                    verification={'status': (model.get('verification') or {}).get('status', 'not_run')})
        cards.append(card)
        (web / 'lessons' / f"{model['num']:03d}.json").write_text(json.dumps(model,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    index = {key: manifest[key] for key in ('schema_version','generated_utc','counts','verified','historical_verified','families','dataset_total_bytes','verification_note')}
    index['models'] = cards
    (web / 'catalog.json').write_text(json.dumps(index,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    (ROOT/'lessons').mkdir(exist_ok=True)
    for m in models:(ROOT/'lessons'/f"{m['num']:03d}.md").write_text(lesson(m),encoding='utf-8')
    summary=f"{len(rows)} 项课程与 {len(rows)} 个独立数据库（合计 **{total_bytes/2**20:.2f} MiB**，每库最大 {max(m['dataset']['bytes'] for m in models)/1024:.0f} KiB）；**{len(available)} 项本地适配器**：{counts['upstream']} 复用上游、{counts['paper_port']} 论文结构/方程实现、{counts['mechanism_demo']} 部分机制演示；**{counts['source_route']} 项原版路径尚无 CPU 适配器**。完整摘要匹配的成功 CPU 记录 **{checked} 项**；保留原版本的历史成功记录 **{historical} 项**。网页构建不等于模型重跑。"
    readme=ROOT/'README.md'
    text=readme.read_text();text=re.sub(r'<!-- release-summary:start -->.*?<!-- release-summary:end -->','<!-- release-summary:start -->\n'+summary+'\n<!-- release-summary:end -->',text,flags=re.S);readme.write_text(text)
    coverage=['# 覆盖与实际验证 / Coverage', '',summary,'','生成自真实注册表和数据；状态含义见 [复现标准](REPRODUCTION_STANDARD.md)。两步执行成功不等于完整论文复现。','',
              '| # | 模型 / 课程 | 实现范围 | CPU 检查 | 参数量 | 峰值 MiB | 数据 KiB |','| --- | --- | --- | --- | ---: | ---: | ---: |']
    for m in models:
        v=m.get('verification') or {};ok=v.get('status') in {'completed','historical'};n=f"{m['num']:03d}"
        coverage.append(f"| {n} | [{safe_cell(m['title'])}](../lessons/{n}.md) | `{m['implementation']}` | {v.get('status','—')} | {v.get('parameters','—') if ok else '—'} | {v['observed_peak_rss_mb']:.1f} | {m['dataset']['bytes']/1024:.0f} |" if ok else f"| {n} | [{safe_cell(m['title'])}](../lessons/{n}.md) | `{m['implementation']}` | {v.get('status','—')} | — | — | {m['dataset']['bytes']/1024:.0f} |")
    coverage += ['','## 尚无 CPU 适配器的明确缺口','']
    for m in models:
        if not m['adapter']:coverage += [f"### {m['num']:03d} {m['title']}",'',m['gaps'][0],'']
    (ROOT/'docs/COVERAGE.md').write_text('\n'.join(coverage)+'\n')
    notices=['# Third-party notices / 第三方来源声明','','Original teaching code is MIT; synthetic data is CC0. Third-party code and libraries retain their own licenses. A source catalog pin can differ from the library actually executed. No pretrained weights are bundled.','','## Vendored source files','','Each directory preserves its license text and a PROVENANCE.json with source URLs, revisions and modifications. Noncommercial and other restrictions continue to apply.','']
    for p in sorted((ROOT/'vendor').glob('*/PROVENANCE.json')):
        obj=json.loads(p.read_text());licenses=sorted(x for x in p.parent.iterdir() if x.is_file() and 'LICENSE' in x.name.upper())
        notices += [f'### {p.parent.name}','',f"[Provenance]({p.relative_to(ROOT).as_posix()}) · "+' · '.join(f'[{q.name}]({q.relative_to(ROOT).as_posix()})' for q in licenses),'','```json',json.dumps(obj,ensure_ascii=False,indent=2),'```','']
    notices += ['## Installed libraries','','The exact direct dependencies are in [requirements-cpu.txt](requirements-cpu.txt), [requirements-macos.txt](requirements-macos.txt) and [pyproject.toml](pyproject.toml); each execution record captures installed versions. Library code is installed from its upstream distribution, not relicensed here.','',
                '| Library | Actual execution version/source |','| --- | --- |',
                '| PyTorch / torchvision | [2.5.1](https://github.com/pytorch/pytorch/tree/v2.5.1) / [0.20.1](https://github.com/pytorch/vision/tree/v0.20.1) |',
                '| MONAI | [1.4.0](https://github.com/Project-MONAI/MONAI/tree/1.4.0) |',
                '| timm | [1.0.11](https://github.com/huggingface/pytorch-image-models/tree/v1.0.11) |',
                '| PyTorch Geometric | [2.6.1](https://github.com/pyg-team/pytorch_geometric/tree/2.6.1) |',
                '| Transformers | [4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3) |',
                '| fastMRI | [0.3.0](https://github.com/facebookresearch/fastMRI/tree/v0.3.0) |','',
                f'The complete {len(rows)}-item source/license audit, including repositories not copied here, is [sources.json](sources.json). NOASSERTION means that copying permission has not been established, not that a work has no owner. GPL is an open-source copyleft license; its obligations are different from noncommercial restrictions. HeartBEiT includes only the individual architecture file carrying an explicit inherited Microsoft MIT grant, not the whole unlicensed repository.']
    (ROOT/'THIRD_PARTY_NOTICES.md').write_text('\n'.join(notices)+'\n')
    print(json.dumps({'models':len(models),'adapters':len(available),'verified':checked,'historical_verified':historical,'counts':dict(counts),'dataset_MiB':total_bytes/2**20},ensure_ascii=False))


if __name__=='__main__':main()
