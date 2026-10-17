# Biomedical AI Teaching Lab / 生物医学 AI 教学实验室

[打开教学网站](https://happymie.github.io/teaching/) · [原架构图谱](https://happymie.github.io/biomed-ai-atlas/) · [137 项覆盖清单](docs/COVERAGE.md) · [来源与许可证审计](docs/SOURCES.md)

这是 Biomedical AI Atlas 的配套教学子项目。137 个条目与原图谱一一对应，每项都有独立、可检查的 SQLite 合成数据库、论文/代码来源和逐模型课程。可执行部分支持本地 CPU、小批次与缩小配置，并记录实际运行证据。

**覆盖 137 个课程条目，不等于完成 137 个原模型的完整复现。** 页面和课程分别标记：`upstream`（复用上游实现）、`paper_port`（独立实现所列论文结构/方程）、`mechanism_demo`（只实现明确列出的机制）、`source_route`（有原始来源与数据，尚无本地 CPU 适配器）。缩小结构、随机初始化和两步执行检查不能证明原论文性能、预训练能力或临床价值。具体差异逐项列出，不把别的网络冒充为目标模型。

<!-- release-summary:start -->
137 项课程与 137 个独立数据库（合计 **13.03 MiB**，每库最大 128 KiB）；**102 项本地适配器**：45 复用上游、27 论文结构/方程实现、30 部分机制演示；**35 项原版路径尚无 CPU 适配器**。当前摘要匹配的成功 CPU 执行检查：**102 项**。
<!-- release-summary:end -->

## 1. 安装并完成第一个实验

建议从 Python 3.10–3.12、8 GB 内存的笔记本开始；这是目标使用配置，不是对每台机器的性能保证。默认 CPU 两线程、批次 2、32×32 图像；个别结构需要更大空间分辨率。首次安装会下载依赖，数据库已经随仓库提供，不会自动下载预训练权重。使用新的虚拟环境，避免改变其他项目。

### Linux x86-64

```bash
git clone https://github.com/happymie/happymie.github.io.git
cd happymie.github.io/teaching
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m biomed_lab run 006 --steps 20
```

### Windows x86-64 / PowerShell

安装 Python 3.12 与 Git 后运行以下命令。直接调用虚拟环境解释器，不需要更改 PowerShell 执行策略。

```powershell
git clone https://github.com/happymie/happymie.github.io.git
cd happymie.github.io/teaching
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m biomed_lab run 006 --steps 20
```

### macOS / Apple Silicon

```bash
git clone https://github.com/happymie/happymie.github.io.git
cd happymie.github.io/teaching
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m biomed_lab run 006 --steps 20 --device cpu
```

CPU 安装清单的 `+cpu` wheel 适用于 Linux/Windows；Apple Silicon 使用另一个清单。**此固定 PyTorch 版本不支持 Intel Mac**；可使用 [Colab notebook](https://colab.research.google.com/github/happymie/happymie.github.io/blob/main/teaching/notebooks/quickstart.ipynb)。依据：[PyTorch 官方 Intel macOS 构建终止说明](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)、[官方 wheel 索引](https://download.pytorch.org/whl/cpu/torch/)。本仓库实测环境为 Linux CPU；Windows/Apple Silicon 命令按平台 wheel 配置，未声称已在这些硬件上逐模型验证。CUDA/MPS 是实验性选项，发布证据只覆盖 CPU。

本项目目前按**克隆仓库 + editable install**使用，代码通过仓库相对路径读取来源与数据；不提供脱离仓库的独立 wheel 使用保证。以下命令用 Linux/macOS 路径，Windows 请将 `.venv/bin/python` 换为 `.venv\Scripts\python.exe`。

## 2. 查阅、调整与检查

```bash
# 看清哪个条目真正可执行，以及实现属于哪一层
.venv/bin/python -m biomed_lab list
.venv/bin/python -m biomed_lab info 006
.venv/bin/python -m biomed_lab data 006 --inspect

# 一个更窄的 U-Net；保持编码/解码与跳连机制
.venv/bin/python -m biomed_lab run 006 --steps 10 --threads 1 --set width=4

# 查看原作者固定版本源码：只获取源码，不安装或执行上游脚本
.venv/bin/python -m biomed_lab fetch-upstream 009

# 为一个模型重新生成独立进程执行记录
.venv/bin/python -m biomed_lab verify 006 --steps 2 --output runs/unet-check.json
```

每项课程的默认配置来自实际构造器，可以在网页生成命令或用 `--set KEY=JSON` 调整。未知键直接报错；头数与宽度、尺度与 patch size 等结构约束仍必须满足。缩小 `width`、`depth`、`image_size` 等参数会改变容量，不能直接与原论文数值比较。不是每个模型都支持所有参数，也不是每个原结构都能任意缩窄。

默认 `--max-seconds 120 --max-rss-mb 3072` 是在训练/验证边界执行的**软预算**。它会把超限状态记录为 `budget_exceeded`，不是操作系统的硬内存分配上限。低内存机器先用单线程、短步数和较窄配置；查看该模型的实测 RSS 后再扩大。`--max-rss-mb 1024` 可降低停止阈值，但无法拦截一次算子内部的分配。训练日志和可选 checkpoint 都只写入本地 `runs/`；它已在 `.gitignore` 中排除。

## 3. 每个实验留下什么

`runs/<时间戳>-<模型号>/` 包含：

| 文件 | 可检查内容 |
| --- | --- |
| `config.json` | 模型参数、种子、步数、设备、依赖版本、实际适配器来源、论文来源档案、数据和代码 SHA-256 |
| `experiment.sqlite3` | `run` 配置表与 `metrics` 表；每步损失、训练/验证范围、耗时、RSS |
| `result.json` | `completed` / `budget_exceeded` / `failed`，完成步数、参数量、测量方法、已知差异 |
| `model.pt`（仅 `--save-checkpoint` 时） | 本次随机初始化实验训练得到的权重与配置 |

`verify` 为每个模型顺序启动一个新 Python 进程。Linux/macOS 使用系统 `ru_maxrss` 峰值；Windows 回退到采样 RSS 并注明方法。已发布的测量包含依赖导入与验证的内存开销。`elapsed_seconds` 从模型构造前开始；`process_wall_seconds` 还包含解释器启动/导入。两步检查证明相应代码路径能完成有限损失、反向传播和参数更新；不证明已经学会任务或收敛。对 DPS/DDRM 等方法，训练先验和后验重建是不同路径；独立的方程测试与模块文档说明覆盖了哪条路径。

## 4. 数据为什么足够轻量

`data/001.sqlite3` 至 `data/137.sqlite3` 各含 **24 个独立合成样本：16 训练、4 验证、4 测试**，共有 3,288 个不同种子。每个数据库都含实际数组与目标、来源/许可证、坐标约定、生成器摘要和逐样本 SHA-256。每个样本使用压缩 NPZ blob；读取时禁用 pickle 并校验哈希。

数据覆盖形状分割、三维椭球、波形、图文短序列、图结构和场采样。噪声视图根据同一基础样本的种子确定性生成，两个噪声实现相互独立；划分单位始终是基础样本或完整图，增强视图不跨划分。图像提示框与点用 `[0,1]` 的 `(x,y)` 坐标，场点用 `[-1,1]` 的 `(x,y,z)`，体素数组顺序为 `C,D,H,W`。这些数据没有患者信息，也不代表任何疾病分布；强度无量纲，不是 HU。

NeRF、NAF、Gaussian Splatting、IDIR、Instant-NGP、EndoNeRF 等当前条目是单场景/单对优化实验，它们的诊断使用同一个训练场景并明确标为 `same_scene_diagnostic`，不能当成独立样本验证。其他适配器使用独立合成验证样本；保留测试划分，不在训练循环中使用。查看数据库：

```python
import json, sqlite3
from pathlib import Path
from biomed_lab.data import inspect_database, load_samples
print(inspect_database(Path('data/006.sqlite3')))
sample = load_samples(Path('data/006.sqlite3'), 'train')[0]
print({k: v.shape for k, v in sample.items()})
with sqlite3.connect('file:data/006.sqlite3?mode=ro', uri=True) as db:
    print(db.execute('SELECT split, COUNT(*) FROM samples GROUP BY split').fetchall())
```

## 5. 建议的学习顺序

| 阶段 | 课程 | 需要回答的问题 |
| --- | --- | --- |
| 第一个完整循环 | [006 U-Net](lessons/006.md)、[002 ResNet](lessons/002.md) | 输入、输出、损失、梯度与优化器各做什么？缩窄后参数量如何变化？ |
| 表示与序列 | [020 Transformer](lessons/020.md)、[015 S4](lessons/015.md)、[016 Mamba](lessons/016.md) | 信息如何沿序列传播？因果性与状态更新怎样检查？ |
| 无监督与图文 | [024 SimCLR](lessons/024.md)、[026 MAE](lessons/026.md)、[030 CLIP](lessons/030.md) | 监督信号来自哪里？哪些目标必须停止梯度？小批次有哪些局限？ |
| 逆问题 | [048 LPD](lessons/048.md)、[050 MoDL](lessons/050.md)、[060 DPS](lessons/060.md) | 前向算子、伴随、数据一致性与学习先验分别承担什么？ |
| 医学与基础模型 | [033 MedSAM](lessons/033.md)、[125 TransMorph](lessons/125.md)、[137 ECGFounder](lessons/137.md) | 结构复用、模态适配与预训练能力有什么区别？ |
| 原版流程 | 从覆盖表选择 `source_route` | 哪些权重、数据、授权和训练阶段仍缺失？什么证据才足以升级复现状态？ |

每项课程都要求记录至少一次基线、一次有解释的参数变化和一次失败/限制分析。不要仅凭 16 个训练样本上的下降曲线宣称某个模型优于另一个模型。

## 6. 维护、测试和贡献

```bash
.venv/bin/python -m pytest
.venv/bin/python -m biomed_lab verify all --steps 2 --output results/verification.json
.venv/bin/python tools/build_site.py
python3 -m http.server 8080 --directory ..
# 浏览 http://localhost:8080/teaching/
```

`tools/build_site.py` 将模型目录、来源档案、真实注册表、数据库和验证 JSON 合并为网页 `manifest.json`，并生成 137 项课程、覆盖表和第三方说明。修改运行代码或数据后，应更新相应测试及执行证据；生成器会拒绝把代码/数据摘要不匹配的旧结果标为当前有效验证。

新增适配器必须提交明确的结构/方程范围、参数缩小方式、实际来源及许可证、任务相关的正确性测试和实测记录；不能只换名称或在模型缺失时默默执行通用网络。尚未解决的条目保留在覆盖表里。

## 7. 来源与许可证

原创教学代码遵循仓库 [MIT 许可证](../LICENSE)，合成数据库按 [CC0-1.0](data/LICENSE.txt) 提供。第三方代码保留各自许可证、固定版本与修改记录，**并不全部受本仓库 MIT 许可覆盖**。例如 MAE/DiT/SegFormer 的非商业条款、GPL 组件的源码义务、UNI/PRISM2 的额外限制各不相同。未确认授权的代码不会因为能在网页上看到就被当作可复制代码；门控权重不会自动下载或重新发布。

完整记录见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[sources.json](sources.json)、`vendor/*/PROVENANCE.json` 及各模块文档：[图像/序列](docs/IMAGE_MODELS.md)、[逆问题与生成](docs/INVERSE_MODELS.md)、[基础模型](docs/FOUNDATION_MODELS.md)、[扩展生理信号模型](docs/EXTENDED_MODELS.md)、[自监督/图学习](docs/RESEARCH_MODELS.md)。

## English overview

This is a source-backed teaching companion to the 137-entry Biomedical AI Atlas. Each entry has an independent SQLite synthetic dataset and a lesson. The coverage table distinguishes adapted upstream code, independent paper-equation/architecture implementations, explicitly incomplete mechanism demos, and source-only routes. Random initialization, reduced capacity and short CPU checks do not establish reproduction of pretrained capabilities or paper metrics. Linux CPU execution is measured; other platforms are documented separately. All differences, sources, pinned dependencies, test scopes and third-party licenses remain inspectable.
