# CLIP、BLIP-2、SAM、MedSAM 与 ALIGN 的 CPU 架构教学

这里有五个可执行的模型实验：**#030 CLIP、#031 BLIP-2、#032 SAM、#033 MedSAM、#095 ALIGN**。
CLIP、SAM、MedSAM 实际调用固定版本的作者模型代码；BLIP-2 与 ALIGN 使用固定版本的 Transformers 规范库实现。教学层负责缩小构造参数、连接合成数据和明确训练范围。
五个实验都从随机参数开始。它们用于检查架构、提示和损失怎样工作，**不代表已经复现基础模型的预训练能力、原论文指标或临床效果**。BLIP-2 只运行第 2 阶段桥接结构和语言建模目标。

完整 Atlas 的 137 项来源审计和固定版本见 [SOURCES.md](SOURCES.md) 与 [sources.json](../sources.json)。
本文件只解释这五个运行适配器；其余基础模型仍有各自的来源入口与独立教学数据库。

## 1. 原始代码与本地文件

| 编号 | 固定来源 | 本地原始模块 | 原始许可证 |
|---|---|---|---|
| 030 CLIP | [OpenAI CLIP，d05afc43](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6) | [model.py](../vendor/foundation_clip/model.py)，以及仅供阅读的 [clip_reference.py](../vendor/foundation_clip/clip_reference.py) | [MIT](../vendor/foundation_clip/LICENSE) |
| 032 SAM | [Meta Segment Anything，dca509fe](https://github.com/facebookresearch/segment-anything/tree/dca509fe793f601edb92606367a655c15ac00fdf) | [foundation_sam](../vendor/foundation_sam/) 中的 image encoder、prompt encoder、mask decoder、two-way transformer、Sam 类及公共模块 | [Apache-2.0](../vendor/foundation_sam/LICENSE) |
| 033 MedSAM | [bowang-lab MedSAM，d71e8a1a](https://github.com/bowang-lab/MedSAM/tree/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488) | [foundation_medsam](../vendor/foundation_medsam/) 中的作者模型模块和仅供阅读的 [train_one_gpu.py](../vendor/foundation_medsam/train_one_gpu.py) | [Apache-2.0](../vendor/foundation_medsam/LICENSE) |
| 031 BLIP-2 / 095 ALIGN | [Transformers 4.46.3，052e652d](https://github.com/huggingface/transformers/tree/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models) | 从固定 `transformers==4.46.3` 依赖导入；[PROVENANCE.json](../vendor/foundation_transformers/PROVENANCE.json) 记录六个实际库文件的哈希 | [Apache-2.0](../vendor/foundation_transformers/LICENSE) |

每个 `foundation_*` 目录都有 `PROVENANCE.json`：列出完整 40 位 commit、原始路径、固定版本 URL 和文件 SHA-256。Transformers 的六个已安装架构/配置文件已与该 commit 的原始字节独立比较，全部一致；文件保留在安装包中，仓库附许可证和可验证清单。
下载的作者文件没有修改字节；`clip/clip.py` 仅改了本地文件名 `clip_reference.py`。
教学包装全部位于 [foundation_models.py](../biomed_lab/models/foundation_models.py)，不会执行两个参考训练/加载脚本，也不会下载检查点。

这些文件继续使用各自的 MIT / Apache-2.0 许可证。上层教学代码的许可证不替换第三方代码的许可与署名。

## 2. 一次运行需要多少资源

默认设置如下。可调整参数通过统一 CLI 的 `--set KEY=JSON` 传入。

| 项目 | CLIP | SAM | MedSAM |
|---|---:|---:|---:|
| 输入图像 | 32 × 32，灰度复制为 RGB | 32 × 32，灰度复制为 RGB | 32 × 32，灰度复制为 RGB |
| 图像 patch | 8 × 8 | 8 × 8 | 8 × 8 |
| 图像编码器 | 宽 64、2 层 ViT | 宽 64、2 层 ViT，局部/全局注意力 | 宽 64、2 层 ViT，局部/全局注意力 |
| 文本 / 提示维度 | 文本宽 64，2 层，4 头 | 提示宽 32 | 提示宽 32 |
| 解码器 | 对比投影维度 32 | 2 层双向交叉注意力 | 2 层双向交叉注意力 |
| 总参数量 | 220,929 | 166,246 | 166,246 |
| 可训练参数 | 220,929 | 166,246 | 165,664 |
| 特有默认设置 | 词表 32，上下文长度 16 | 2 轮交互提示 | 冻结提示编码器，框提示 |

在此次验证环境 Python 3.12.14、PyTorch 2.5.1+cpu、2 个线程中，固定两个合成样本、种子 2026、AdamW 学习率 0.001，运行 10 次参数更新得到：

| 模型 | 首次损失 → 第 10 次损失 | 构造/首次导入/10 步总时间 | 到此为止的进程峰值 RSS |
|---|---:|---:|---:|
| CLIP | 0.693266 → 0.693259 | 2.172 秒 | 281.5 MiB |
| SAM | 2.983744 → 2.339019 | 1.243 秒 | 292.3 MiB |
| MedSAM | 1.274062 → 1.144578 | 5.608 秒 | 444.6 MiB |

这是同一个进程按上述顺序做的有限训练检查，RSS 是累计进程峰值，包含框架及已导入模块；不是每个模型的独占内存，也不是任意笔记本的速度保证。
MedSAM 首次运行还包含 MONAI 导入。表中使用固定训练批次，不能作为独立测试集评估。
这里 CLIP 的两个样本描述重复，损失接近 `log(2)`，并没有显示有意义的语义检索学习；这也提示了微型对比学习数据的限制。

从 `teaching/` 目录运行：

```bash
python -m biomed_lab info 30
python -m biomed_lab data 30 --inspect
python -m biomed_lab run 30 --steps 10 --batch-size 2 --threads 2
python -m biomed_lab run 32 --steps 10 --batch-size 2 --threads 2
python -m biomed_lab run 33 --steps 10 --batch-size 2 --threads 2
```

每个模型有独立的 SQLite 数据库；实验产物与训练、验证指标由统一 runner 记录。数据库内是从独立种子生成的形状、掩膜、描述或提示，均不是医学影像或医学报告。
数据制作与拆分规则见 [data.py](../biomed_lab/data.py)，运行与保存规则见 [runner.py](../biomed_lab/runner.py)。

例如减少计算，或增大结构供课堂比较：

```bash
# CLIP 的图像注意力头数由作者代码按 vision_width // 64 决定，图像宽度下限为 64。
python -m biomed_lab run 30 --steps 10 --set vision_layers=1 --set text_layers=1 --set context_length=4

# 1 轮只训练初次点提示；默认 2 轮会额外训练纠错点与上一轮掩膜提示。
python -m biomed_lab run 32 --steps 10 --set interactive_rounds=1
python -m biomed_lab run 32 --steps 10 --set interactive_rounds=3 --set image_size=64

# 调整 MedSAM 图像编码器的容量，仍保留真正的提示编码与掩膜解码。
python -m biomed_lab run 33 --steps 10 --set width=32 --set depth=1
```

输入尺寸必须能被 patch 大小整除；图像宽度必须能被注意力头数整除。
SAM/MedSAM 的提示维度必须能被 8 整除，且其一半必须能被解码器头数整除。
非法配置会直接报错，不会自动换成另一种模型。更大的宽度、图像或批次仍可能超过所用电脑的资源预算。

## 3. CLIP：保留双编码器和 EOT 读出

作者 `CLIP` 类包含 ViT 图像编码器、带因果掩码的文本 Transformer、末尾标记向量读出、两个投影、特征归一化和可学习的 logit scale。
适配器直接构造这个类，训练时对图片到文本和文本到图片的交叉熵求平均，保持成对索引为正样本的对称目标。
图像使用作者 `clip.py` 中的均值/标准差；优化后对 logit scale 做论文描述的最大尺度 100 截断。[作者模型](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py)、[论文](https://arxiv.org/abs/2103.00020)

合成词表为 `PAD=0, BOS=1, EOS=2, A=3, class=4..6`。原始 CLIP 用 `text.argmax()` 找到 EOT，因为原始分词器的 EOT ID 最大。
如果直接传入这里的 `EOS=2`，读出的将是类别标记而非 EOS。
`prepare_tokens()` 因此在进入原始模型前把 EOS 映射成 `vocab_size-1`，并保留唯一 EOS 检查。
缩短上下文时只允许去掉尾部零填充；有内容的标记不被静默截断。

这里没有实现或冒充 OpenAI BPE 分词器，也没有加载 CLIP 权重。
几个合成形状的短描述仅用于理解双编码器训练。重复描述仍采用作者的成对对比目标，可能成为假负样本；改变为多正样本损失会是另一种目标，不能无说明地代替 CLIP。

适合做的课堂实验包括：观察图片/文本 logits 互为转置、检查改变 EOS 后的填充不影响因果文本读出，以及比较不同形状描述与重复描述对对比损失的影响。

## 4. SAM：真实的点、框、掩膜提示和交互纠错

这里复用作者的三个主要模块：

1. `ImageEncoderViT` 保留 patch 嵌入、窗口/全局自注意力、相对位置编码与卷积 neck。
2. `PromptEncoder` 保留随机 Fourier 位置编码、正/负点和两个框角的不同可学习嵌入，以及卷积掩膜提示编码。
3. `MaskDecoder` 保留 prompt 自注意力、两个方向的图像/提示交叉注意力、动态 mask MLP、三个歧义候选掩膜与 IoU 质量头。

原始 SAM 的解码器假设一次接收一张图像及其若干提示集合；它会在内部重复图像嵌入。
将一整个 `B` 张图像和 `B` 组提示直接交给该接口会形成错误的 `B²` 展开。
教学包装先批量编码图像，再逐图调用未修改的解码器，从而保持正确的图像—提示配对。
测试还把包装输出与原始 `Sam.forward()` 的输出逐项比较。[作者解码器](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/mask_decoder.py)

默认训练有两轮。第一轮使用明确提供的前景点，输出三个候选掩膜；第二轮从第一轮预测的错误区域采样纠错点，同时输入上一轮的低分辨率掩膜 logits。
纠错点的正/负标签取决于对应像素是真前景还是背景；没有错误时用 `-1` 填充点，不把正确位置标成负点。
候选中用于下一轮交互的掩膜由预测 IoU 选择，上一轮 logits 停止梯度。

掩膜损失是 `20 × focal + Dice`，候选掩膜只对损失最小的一个反向传播；另有预测 IoU 与真实 IoU 的均方误差。
此处对每个返回候选都计算 IoU 回归，focal 使用 `alpha=0.25, gamma=2`。这些训练函数位于教学层，作者公开的 SAM 架构文件保持原样。[SAM 论文附录 A](https://arxiv.org/html/2304.02643v1#A1)

默认两轮保留了交互纠错机制，但不是论文完整的 11 轮采样日程。即使把 `interactive_rounds` 改为 11，仍未恢复论文的点/框混合、框扰动、随机插入不加新点的迭代、MAE 初始化、数据规模与学习率策略。
框和点在本例中是显式输入：来自生成时的真值，不是模型自动发现目标的位置。运行结果只能说明有提示的合成分割路径。

## 5. MedSAM：保留作者的不同训练选择

MedSAM 使用其自身仓库的原始模块，并没有把 SAM 的一次普通运行换个名称。
相对上一节，教学适配器保留了三个可检验的变化：

- **框提示与冻结提示编码器。** 模型只接受框；提示编码器参数 `requires_grad=False`，其前向不构建梯度图。
- **作者的输入约定。** 训练脚本直接把已归一化到 `[0,1]` 的图像交给图像编码器，不调用 SAM 的像素均值/标准差预处理。这里遵循这一约定。
- **作者的训练目标。** 使用 `monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")` 加 `BCEWithLogitsLoss`。IoU 头会返回数值，但这套目标不训练该头，因此不能把其随机分数称为可信置信度。

MedSAM 的原始 mask decoder 已有成对图像批次的分支，因此这里直接使用该分支。
冻结规则和损失配置都可在固定版本 [train_one_gpu.py](https://github.com/bowang-lab/MedSAM/blob/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/train_one_gpu.py) 查看。

教学框是合成掩膜的紧框；没有照搬作者医疗数据处理和随机扩框过程，也没有从 SAM 检查点初始化。
因此更准确的实验范围是“缩小的作者架构 + MedSAM 训练机制”，不是复现已训练的 MedSAM。

## 6. BLIP-2：真实 Q-Former，第 2 阶段桥接结构

这里使用 `transformers==4.46.3` 的 `Blip2ForConditionalGeneration`，组合真正的 `Blip2VisionModel`、可学习 query、`Blip2QFormerModel`、语言投影层和 `OPTForCausalLM`。
这与来源登记中的 Salesforce LAVIS 作者仓库是两条明确区分的路径：登记保留作者来源，当前适配器链接具体执行的规范库版本。[固定库实现](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/blip_2/modeling_blip_2.py)、[作者论文](https://proceedings.mlr.press/v202/li23q.html)

默认图像为 32 × 32，patch 为 8，ViT、Q-Former 与 OPT 均宽 32、2 层、4 个注意力头，4 个 query，32 个词表项和 16 个文本位置。总参数 **65,408**，其中 **22,624** 个可训练。
图像编码器和语言模型被明确冻结，切换到训练模式也保持这两部分的评估模式；梯度通过冻结语言模型的运算流回投影、query 和 Q-Former。
测试检查两部分参数无梯度，query、投影、Q-Former 图像交叉注意力则确实收到非零梯度。

为调用该固定版本支持的输入接口，包装先放入 4 个保留的图像占位标记，再接教学文本；原始库把占位符替换成投影后的 query 嵌入。
视觉前缀、BOS 和 PAD 不作为监督目标，普通文本及 EOS 使用带因果移位的交叉熵。
这保留了“图像 → query → 语言前缀 → 下一词预测”的计算路径，测试还确认未来文本不能改变更早的语言输出，以及图像变化可以影响输出。

**该实验只覆盖第 2 阶段的桥接结构与损失。** 论文第 1 阶段的图文对比、图文匹配和图像条件文本生成联合预训练没有实现，作者预训练视觉/语言检查点也没有加载。
两套冻结骨干本身是随机函数，不能承担论文中的视觉知识或语言知识；生成 logits 可供检查计算，却不能称为能看图描述或能回答医学问题的基础模型。
教学词表不实现原始自然语言分词器。

随机初始化还需要明确处理库的检查点默认值：ViT 的初始化标准差设为 0.02，query 从标准差 0.02 的正态分布开始，避免直接使用近零视觉参数和全零 query。
这只是教学初始化选择，架构类的源码字节没有修改，不表示复现作者预训练初始化。

```bash
python -m biomed_lab run 31 --steps 10 --batch-size 2 --threads 2
python -m biomed_lab run 31 --steps 10 --set query_tokens=2 --set depth=1
```

## 7. ALIGN：EfficientNet 与 BERT 双塔

ALIGN 使用固定库中的 `AlignModel`。图像分支保留 EfficientNet 的七个阶段、深度可分离卷积、MBConv 扩展/投影、SE 通道门控和跳连；文本分支保留 BERT 自注意力和首标记表示。
库内对两个分支的嵌入归一化，计算带可学习温度的相似度，对图文两个方向的标签平滑交叉熵求平均，平滑系数为 0.1。[固定库实现](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/align/modeling_align.py)、[原始论文](https://proceedings.mlr.press/v139/jia21b.html)

默认宽度与深度系数均为 0.25，每个阶段至少保留一个 block；BERT 宽 32、2 层、4 头，图像为 32 × 32，教学词表 32、上下文 16。
最后一层图像通道按库中的 `round_filters` 规则计算，文本投影严格匹配这个通道数；没有另加一个虚构的图像投影层。
默认总参数与可训练参数均为 **119,821**。

为保证未校准 BatchNorm 的随机模型也能直接进行评估，视觉权重初始化标准差使用 0.1。
库默认 0.02 在这个深层缩小随机网络中会产生极小特征，后续单位长度归一化可能因浮点下溢产生无效结果；此初始化变化记录在模型限制中。
图像使用合成数据的 `[-1,1]` 归一化。原始网页图文数据、增强、优化日程和词表没有复现；重复描述还可能成为成对对比目标中的假负样本。

这里的规范库沿用 Kakao Brain 的 ALIGN 复现路径，**不是 Google 未发布的原始训练代码，也不是原始 Google 检查点**。[Kakao 模型卡](https://huggingface.co/kakaobrain/align-base)
实际路径只用于教学构造、反向传播、温度和双向目标；10 步有限训练及后续评估已检查，不能据此推断检索准确率。

```bash
python -m biomed_lab run 95 --steps 10 --batch-size 2 --threads 2
python -m biomed_lab run 95 --steps 10 --set width_coefficient=0.125 --set text_depth=1
```

## 8. 验证内容与基础模型后续路线

[test_foundation_models.py](../tests/test_foundation_models.py) 检查有限反向传播与参数更新、CLIP EOS/因果读出和对称目标、SAM 对原始推理的数值一致性、批次配对、点/掩膜提示敏感性、交互纠错梯度、最优候选梯度、MedSAM 冻结规则、BLIP-2 桥接梯度/因果性/前缀标签、ALIGN 七阶段模块与真实对称目标，以及原始文件和固定库文件的哈希与许可证。

```bash
python -m pytest tests/test_foundation_models.py
```

这些验证证明运行路径与上述模块级约定成立，不能证明预训练表征质量或论文指标重现。

以下相关 Atlas 项仍保留明确的来源路线，没有被本模块伪装为可执行适配器：

| 范围 | 项目 |
|---|---|
| 视觉语言 / 提示基础模型 | #034 SAM 2、#035 ConVIRT、#036 GLoRIA、#037 BioViL、#038 BiomedCLIP |
| 病理 / 体积基础模型 | #039 UNI、#040 CONCH、#041 Prov-GigaPath、#042 Merlin |
| 视觉语言与报告扩展 | #096 BLIP、#097 Florence-2、#098 MedCLIP、#099 Med-Flamingo、#100 LLaVA-Med、#101 RadFM、#102 MAIRA-1、#103 MAIRA-2、#104 BiomedGPT |
| 病理与体积扩展 | #105 Virchow、#106 CHIEF、#107 TITAN、#108 PRISM2、#109 BME-X、#110 Prima、#111 MARS、#112 Decipher-MR |

例如，SAM 2 必须保留视频/对象记忆机制；把当前二维 SAM 改名不能重现 SAM 2。
病理整片模型的 patch 聚合与预训练特征、视觉语言生成模型的桥接模块与训练阶段、体积模型的三维数据/多模态目标都需要独立适配与验证。
ConVIRT 已找到的作者仓库只包含检索评估数据；MAIRA-1 与 Decipher-MR 的完整发布实现也未在本次审计中确认。没有以其他模型冒充这些原始实现。
UNI、CONCH、PRISM2 等有非商业/禁止演绎或其他专门条件；模型卡或代码能公开阅读并不自动表示允许修改后重新发布。各项具体状态、证据与固定来源均在 [SOURCES.md](SOURCES.md)。
MedCLIP 固定仓库没有找到根许可证，已检查的模型文件开头也没有授权声明；当前不能据“可以下载”推定允许复制。这不等于宣称所有剩余模型都无法在 CPU 教学：有些有合法实现，仍需要完成独立的结构、训练阶段、输入和验证工作，才应从来源路线升级为可运行适配器。
