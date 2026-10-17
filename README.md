# Biomedical AI Atlas / 生物医学 AI 架构图谱

Xiaoyang Li 的生物医学 AI 学习网站：从原论文架构与数学原理，到可检查的代码、数据与实验记录。

## 学习入口

| 内容 | 网站 | 仓库目录 |
| --- | --- | --- |
| 交互架构图谱 | [探索 137 个模型](https://happymie.github.io/biomed-ai-atlas/) | [biomed-ai-atlas](biomed-ai-atlas/) |
| 中英双语完整专著 | [阅读完整专著](https://happymie.github.io/biomed-ai-atlas/reference.html) | [reference.html](biomed-ai-atlas/reference.html) |
| 配套教学实验室 | [进入教学实验室](https://happymie.github.io/teaching/) | [安装与使用说明](teaching/README.md) |

教学实验室提供与图谱逐项对应的 **137 门课程**。每个模型配备独立的 SQLite 合成数据库；已实现的模型支持可调 CPU 配置，让学习者从小批次、短步数和缩小结构开始。课程逐项标明上游实现复用、论文结构/方程移植、机制演示或原作路线，并提供来源、许可证、已知差异和运行证据。

**课程覆盖不等于 137 个原模型全部完成复现。** 请以教学站的逐模型状态和[覆盖清单](teaching/docs/COVERAGE.md)为准。合成数据、缩小配置和短步数检查用于教学，不证明原论文性能、预训练能力或临床效果。

## 本地浏览

在仓库根目录运行：

```bash
python3 -m http.server 8080
```

打开 `http://localhost:8080/`，会进入原 Atlas 首页；教学站位于 `http://localhost:8080/teaching/`。模型运行环境的安装和平台说明见[教学实验室 README](teaching/README.md)。

## English

An interactive atlas, a bilingual compendium, and a teaching companion for 137 biomedical AI architectures. Each teaching entry has its own SQLite synthetic dataset. Implemented models offer adjustable CPU configurations, while every course states its source, implementation scope, and reproduction limits. The published course count is not a claim that all original models or their reported performance have been reproduced.

[Open the atlas](https://happymie.github.io/biomed-ai-atlas/) · [Start learning](https://happymie.github.io/teaching/)
