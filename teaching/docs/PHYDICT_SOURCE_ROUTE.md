# 138 · PhyDiCT：先核输入条件，再研究重建机制

**状态：source_route。** 原始源码和根许可证已核；本地尚无 CPU 适配器，也没有运行或复现论文数值。该页在 2026-10-17 的实际维护中纳入，模型本身不是当天发布。

## 已核实的材料

| 材料 | 本次实际检查 | 结论 |
| --- | --- | --- |
| [MICCAI 2026 官方页](https://papers.miccai.org/miccai-2026/0773-Paper1942.html) | 题目、作者、摘要、代码链接、年份 | 纳入近期来源路径；PDF 本次无法取回，没有声称阅读全文。 |
| [作者仓库](https://github.com/batmanlab/PhyDiCT/tree/a5366fd19cc4a44e52445ce6385259f00cf770bb) | 固定 commit、README、src 目录、环境说明与推理代码 | 源码已经存在，不能继续把摘要中的将来时当作当前代码状态。 |
| [根许可证](https://github.com/batmanlab/PhyDiCT/blob/a5366fd19cc4a44e52445ce6385259f00cf770bb/LICENSE.txt) | Apache-2.0 全文 | 是根源码许可；模型权重、数据和各依赖的条款仍独立。 |
| [推理入口](https://github.com/batmanlab/PhyDiCT/blob/a5366fd19cc4a44e52445ce6385259f00cf770bb/src/eval_low_PhyDiCT.py) | 默认观测角度、显式 CUDA 路径、投影误差和细化代码 | 不能只缩小参数就声称已适配普通 CPU 笔记本。 |

日期应分开记录：官方页标识 MICCAI 2026；固定仓库 commit 为 **2026-09-25**；检索到的 [arXiv 记录](https://arxiv.org/abs/2610.09253) 为 **2026-10-07**。README 的 BibTeX 年份写作 2027，与官方会议页不一致，本目录按官方页记为 2026。

## 需要理解的结构

论文摘要描述一个冻结的文本条件三维扩散先验，通过可微投影引导采样；作者实现还提供额外计算细化。这里的“无需训练”针对该重建阶段的先验使用方式，已有预训练权重仍是必要条件。学习时把 **观测 X 光、采集几何、文本报告、预训练先验** 四类信息逐项列出来，不能把后三项悄悄删掉后仍称同一实验。

本地配套数据库含 24 个独立三维椭球合成样本及二维形状观察，体素数组为 `(1,16,16,16)`，各项强度无量纲。它用于检查数据读取、种子和划分以及设计后续机制实验。**当前库没有 PhyDiCT 所需的标定稀疏锥束投影或真实报告；二维形状图像不是库中三维椭球的标定 X 光。** 因而不能直接送入原实现就声称完成重建实验。检查数据库的命令见 [课程 138](../lessons/138.md)。

## CPU 适配前的最小验收

1. 在本次固定版本上明确哪些代码、库和权重可以获取和使用。先做依赖清单与内存预算，不运行来源中的安装脚本。
2. 先对一个明确标定的合成体建立对应投影，并核对体素坐标、采样顺序、像素单位和 pose。双向数值梯度检查应对着前向渲染计算，而非只检查输出 shape。
3. 区分无文本、通用文本和病例对应文本的条件，说明每项在真实部署时怎样获得。不得使用测试目标报告或 GT 派生位姿、裁剪、尺度而未披露。
4. 缩小扩散网络后核对原机制保留范围。随机初始化的小网络不具有 MedSyn 的预训练知识；运行两个优化步也不能重建这些知识。
5. 分别记录投影一致性、三维结构误差与资源消耗。单个体拟合只能是同场景诊断；真实病例泛化和临床使用需要独立研究设计。

这些是后续实现的验收条件，本轮没有把它们标为已完成。当前能够复核的是目录纳入、一手来源、许可证和独立合成数据库的完整性。

## English scope

PhyDiCT was added as a source route, with an immutable author-source pin and a small synthetic database. The official conference page and selected source files were inspected. The paper PDF was unavailable in this check. No pretrained weights were downloaded and no model inference was run. The database contains synthetic volumes and separate 2D shape observations; it does not contain calibrated projections of those volumes or clinical reports. The historical CPU records of other lessons were not rerun by this update.
