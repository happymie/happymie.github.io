# 生成、恢复、重建与配准：可检查的 CPU 教学实验

这组课程覆盖图谱中的 **43–74、113–131**。目前 **46 项有可执行 CPU 实验**：9 项直接调用保留原样的上游网络代码，23 项实现论文中的训练或推理公式，14 项明确标注为局部机制演示。另有 5 项保留原项目入口，没有把其他网络改名充数。

**“能运行一个教学实验”与“复现论文的完整模型和结果”是两种不同的状态。** 默认实验使用小型合成图像、体积、噪声、相机和线圈。它们用于检查计算机制，不能证明临床有效性、真实影像泛化、原论文精度或大规模预训练能力。每一项的实际实现、参数、限制和一手来源都在 `biomed_lab/models/inverse_models.py` 的 `SPECS` 中。

## 1. 从一个完整的小实验开始

以下命令在 `teaching/` 目录运行。先按主 README 安装固定版本依赖。

```bash
# 看清该项实际运行的内容、原始来源和独立 SQLite 数据库。
python -m biomed_lab info 52
python -m biomed_lab data 52 --inspect

# 20 次有预算限制的 CPU 更新；记录配置、损失、代码和数据指纹。
python -m biomed_lab run 52 --steps 20 --batch-size 2 --threads 2 --max-seconds 120 --max-rss-mb 1500 --save-checkpoint

# 保留 17 层 DnCNN，将通道数再减半。配置改变会记录在运行结果中。
python -m biomed_lab run 52 --steps 20 --set width=4 --threads 2

# 学习物理算子与展开优化。
python -m biomed_lab run 48 --steps 20 --set stages=3
python -m biomed_lab run 50 --steps 20 --set stages=3

# 检查本组独立数值测试。
python -m pytest tests/test_inverse_models.py
```

先学习一个模型的输入、输出、损失和测试，再改变一个参数。减小通道数通常改变模型容量；减小展开级数改变优化迭代次数；减小射线采样数改变数值积分误差。这些设置不能混写成“同一完整模型的速度优化”。

### 配置如何对应计算量

| 设置 | 实际改变 | 使用边界 |
|---|---|---|
| `width` | CNN 通道数或 MLP 隐藏维度 | SwinIR 的 embedding dimension 是 `3 × width`；DiT/SiT 是 `8 × width`；Restormer 的 width 必须为偶数 |
| `stages` | CT/MRI 展开级数或 VarNet cascade 数 | MoDL 的网络仍在不同级之间共享；LPD/LEARN 的每级网络仍各不相同 |
| `size` | 小图像/离散测量维数 | 只在该项 `info` 列出的配置中可改；Glow 的 squeeze 要求偶数尺寸 |
| `ray_size` | 每个相机的射线网格边长 | 总射线数为 `ray_size²` |
| `ray_samples` | 每条射线的积分采样数 | 改变渲染数值近似，不能直接跨设置比较重建精度 |
| `timesteps` | 离散扩散训练时间表长度 | 默认 1000；缩短固定端点的 beta 表会留下大量信号，不能直接从纯高斯分布开始生成。短时间表的这种调用会被拒绝；应保留训练时间表并减少 DDIM 采样步数 |

原样导入的 9 个网络都通过了 **width=4 比 width=8 参数更少，并且仍可反向传播**的检查。小模型并不意味着整个 Python 环境只有几 MB：PyTorch、优化器状态和中间激活也占内存。以真实运行的资源日志判断你的笔记本是否适合，不把参数文件大小当作峰值内存。

## 2. 先区分观测与目标

本组不是“给同一个 CNN 换 45 个名字”。不同问题允许的信息不同。

| 问题 | 训练时观察到什么 | 学习什么 | 推理/优化时输入什么 |
|---|---|---|---|
| DnCNN | 噪声图与干净图 | 加性噪声残差 | 一张噪声图 |
| Noise2Noise | 同一潜在信号的两次独立噪声观测 | 条件均值 | 一次噪声观测 |
| Noise2Self / Noise2Void | 一次噪声观测 | 被遮住像素的邻域预测 | 遮挡后的观测与完整拼合策略 |
| DDPM / Score-SDE / Flow | 合成训练图与自行采样的噪声/时间 | 噪声、score 或速度场 | 随机初始状态；不同模型采用不同更新式 |
| DDRM / DPS / DDNM | 先训练图像先验 | 本体是条件采样算法 | 测量、已知算子及已训练先验 |
| LPD / LEARN / MoDL | 由已知算子生成的测量和目标 | 结合测量一致性的展开更新 | 测量及相同定义的算子 |
| VoxelMorph / GradICON | moving/fixed 图像对 | 跨样本预测的形变场 | 一对新图像 |
| IDIR | 一对给定图像 | 该对图像专属的连续位移函数 | 同一图像对的连续坐标 |
| NeRF / NAF / 3DGS / Instant-NGP | 同一场景的图像/投影及几何 | 该场景的表示 | 新相机射线或坐标 |
| pixelNeRF | 多个合成场景的参考图及目标视角 | 由参考图条件化的跨场景表示 | 参考图、相机及查询射线 |

IDIR、NeRF、NAF、3DGS、Instant-NGP、EndoNeRF 的小实验只拟合第一个合成场景或图像对，并将它保存到 checkpoint。它们的后续检查是 **same-scene diagnostics**，不是独立患者验证。把一个没有场景条件的隐式函数轮流拟合不同患者，再称为泛化测试，会改变任务本身。

## 3. 去噪：监督条件比网络名字更关键

### DnCNN、Noise2Noise 和盲点方法

DnCNN 计算噪声残差 `R(y)`，输出 `y − R(y)`。代码直接调用作者推荐的 [KAIR DnCNN](https://github.com/cszn/KAIR/tree/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e)。默认保留 17 层，将宽度缩为 8。

Noise2Noise 的 MSE 目标是 `f(y1) ≈ y2`。对给定干净图像 `x`，需要 `E[y2 | x] = x`，并需要目标噪声与输入噪声适当独立。默认数据把两次独立高斯噪声直接加到图像上，**不裁剪**到 `[0,1]`：裁剪会改变条件均值，尤其影响黑色背景。其 `training_loss` 根本不读取 `clean`。测试把 `clean` 替换成 NaN，损失保持不变，并拒绝把同一 tensor 同时当作两次观测。[Noise2Noise 论文](https://arxiv.org/abs/1803.04189)

Noise2Self 把像素分成九个子集。预测集合 `J` 时，所有 `J` 位置都替换为常数，损失只计算 `J`。改变任何被遮住的测量值，都不能改变这些位置的预测。完整预测需拼合九次前向结果。Noise2Void 使用邻域替换；本实验采用随机选择的规则分区，并保证替换来源不包含任何被遮挡的目标像素。这样不会通过另一个替换位置把目标值送回网络。[Noise2Self 论文](https://proceedings.mlr.press/v97/batson19a.html) · [Noise2Void 论文](https://arxiv.org/abs/1811.10980)

### 恢复网络的机制差异

| 模型 | 实际保留的区别 | 当前实验 |
|---|---|---|
| RED-Net | 镜像 convolution/deconvolution 栈与相加 skip | 十层小型机制演示，不称为完整 RED30 |
| SwinIR | RSTB、shifted-window attention、relative-position bias | 直接调用官方源码；2×2 个块、默认 embedding 24 |
| Restormer | MDTA 在通道间计算 attention，GDFN 使用深度卷积和门控 | 直接调用官方源码；四尺度、每尺度一个块 |

SwinIR 当前配置是去噪/恢复，不能把输出尺寸相同的实验报告成已运行其超分辨率训练方案。[SwinIR 源码](https://github.com/JingyunLiang/SwinIR/tree/6545850fbf8df298df73d81f3e8cba638787c8bd) · [Restormer 源码](https://github.com/swz30/Restormer/tree/68dc6ac472db26f16361150cb7a96a1bc87da93f)

## 4. 扩散与流：训练目标和采样器需要分开看

DDPM 使用

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon`

并训练 `epsilon_theta(x_t,t)` 预测加入的噪声。默认时间表是 1000 项 `beta=0.0001…0.02`。DDIM **没有单独的新训练损失**；它复用这个先验，改变逆向轨迹。`eta=0` 时，固定初始噪声即可获得确定性输出。测试固定网络、输入和时间，验证 DDPM/DDIM 训练损失相同，并验证 DDIM 不受额外逆向噪声的影响。[DDPM](https://arxiv.org/abs/2006.11239) · [DDIM](https://arxiv.org/abs/2010.02502)

```python
import torch
from biomed_lab.models.inverse_models import build_model

model = build_model(56, {})
# 先训练或加载该教学模型自己的 checkpoint。
initial = torch.randn(1, 1, 16, 16)
sample = model.sample(initial=initial, steps=20, eta=0.0)
```

这段代码只证明采样接口；随机初始化或训练几步的先验通常不能生成有用图像。训练损失有限，不能代替生成质量评估。

Score-SDE 演示 variance-preserving SDE，训练的是随连续时间变化的 score。逆 SDE 的 score 项系数是 `g²`，probability-flow ODE 的对应系数是 `g²/2`，不能直接互换。积分使用 `steps+1` 个时间边界，每一步的 score 时间和实际步长严格匹配；另外用已知 Gaussian 解析轨迹检查细化网格后的收敛。[Score-SDE 论文](https://arxiv.org/abs/2011.13456)

Flow Matching 的条件 OT 路径是

`x_t = [1 − (1 − sigma_min)t] z + t x`，`u_t = x − (1 − sigma_min)z`。

Rectified Flow 的第一轮使用 `sigma_min=0` 的直线路径。它另有 `reflow_pairs()`，将训练后 ODE 的起点和终点重新配对。默认短实验只运行第一轮，不能声称已经进行了多轮 reflow。[Flow Matching](https://arxiv.org/abs/2210.02747) · [Rectified Flow](https://arxiv.org/abs/2209.03003)

Consistency Models 使用同一噪声构造相邻尺度的两个输入，较低尺度由 stop-gradient EMA target 给出目标。边界系数保证 `f(x,sigma_min)=x`；EMA 在优化器更新后更新。这个边界有精确测试。当前固定 16 个尺度，没有复制论文的增长日程和长周期训练。[Consistency Models](https://proceedings.mlr.press/v202/song23a.html)

DiT 和 SiT 都直接调用对应上游 Transformer。DiT 训练 epsilon；当前 SiT 训练线性 interpolant 的 velocity，因此 **SiT 使用 ODE 速度积分，不能误用 DDIM 的 epsilon 更新式**。两者在小像素图上训练，没有附带预训练 VAE，不能声称已完成原始 latent-space 生成流程。[DiT](https://github.com/facebookresearch/DiT/tree/ed81ce2229091fd4ecc9a223645f95cf379d582b) · [SiT](https://github.com/willisma/SiT/tree/cbde832a40b153ccc79603412409da9c9b0c568c)

PFGM++ 的有限增广维度不能被普通高斯噪声代替。实现采样 `b ~ Beta(N/2,D/2)`，半径 `R = sigma sqrt(D) sqrt[b/(1-b)]`，再乘随机单位方向。数值测试检查 `E[R²] = N sigma² D/(D−2)`（`D>2`）。这是分布检查，不是生成质量证明。原 PFGM 对应 `D=1`，不是 `D→0`。[PFGM++ 论文](https://proceedings.mlr.press/v202/xu23m.html) · [核对的原始损失源码](https://github.com/Newbeeer/pfgmpp/blob/d57c1ee4488d8e4064a5b9c8548792f00395aa8b/training/loss.py)

## 5. 逆问题：A 与 A* 必须真正配对

### CT 与 MRI 的教学测量

`ParallelBeam` 明确构建一个二维离散投影矩阵 `A`。它采用像素到探测器的线性分配，坐标位于边长 1 的正方形；`adjoint()` 使用同一个矩阵的转置。FBP 再使用 ramp filter 和角度积分尺度。测试同时检查 `<Ax,y>=<x,A*y>`，以及单位衰减圆盘经 FBP 后内部幅度接近 1。**转置是离散伴随，不是逆算子。**

`FourierOperator` 使用归一化的两个合成复数 coil sensitivity maps、unitary FFT 和 Cartesian mask。其伴随先逆 FFT，再乘共轭线圈灵敏度并求和。复内积测试验证伴随；采样全部频率时测试 `A*A x=x`。这些合成 maps 不是任何真实扫描仪的估计结果。

| 模型 | 每一级实际执行什么 | 关键不同 |
|---|---|---|
| FBPConvNet | 一次 FBP 后做残差图像修复 | 没有逐级 data consistency |
| Learned Primal-Dual | measurement-domain dual CNN 与 image-domain primal CNN 交替，显式调用 A/A* | 保留五通道 primal/dual memory |
| LEARN | `x ← x − step A*(Ax−y) − CNN_k(x)` | 每级 CNN 独立 |
| Variational Network | `K* phi(Kx)` 加 MRI 数据梯度 | 可学习滤波器和 RBF influence function |
| MoDL | 共享 denoiser 后求解 `(A*A+lambda I)x=A*y+lambda D(x)` | 通过可微 CG 求解，不用一次梯度步冒充 |
| VarNet | 原始灵敏度估计网络与 normalized-U-Net k-space cascade | 直接导入 fastMRI 实现；小实验使用 MSE |
| AUTOMAP | 全连接 sensor-to-image 映射，再卷积修复 | 显式学习全局域变换；小型机制演示 |

MoDL 的 CG 还有独立对照：将随机正定小矩阵的解与 `torch.linalg.solve` 比较，并检查解对右端项可反向传播。[LPD](https://arxiv.org/abs/1707.06474) · [Variational Network](https://doi.org/10.1002/mrm.26977) · [MoDL](https://arxiv.org/abs/1712.02862) · [VarNet 源码](https://github.com/facebookresearch/fastMRI/tree/91f2df4711adbb6d643df1810f234e4abcf5881b) · [LEARN](https://pmc.ncbi.nlm.nih.gov/articles/PMC6019143/)

### 用扩散先验求逆：三个算法不是一回事

这三项的 `training_loss` 只训练先验。先验训练完成后，显式调用 `reconstruct()` 才执行条件采样。默认算子是带缺失方块的二值 inpainting mask，因此奇异值和伪逆有精确定义。

| 方法 | 测量怎样进入逆过程 | 检查 |
|---|---|---|
| DDRM | 在 SVD 坐标里分别处理零奇异值、当前噪声高于测量噪声、低于测量噪声三种分量 | 三个分支及无噪声极限 |
| DPS | 通过 Tweedie `x0` 估计，求测量误差相对于当前 `x_t` 的梯度 | 指导项确实沿测量路径传播 |
| DDNM | `x0_hat ← A†y + (I−A†A)x0_hat` | range 满足测量；null-space 保持原预测 |

DDNM 当前只实现无噪声分支，不包含 DDNM+。DPS 当前使用连续相邻的 DDPM 时间步，主动拒绝把任意跳步包装为原算法。[DDRM](https://arxiv.org/abs/2201.11793) · [DPS](https://arxiv.org/abs/2209.14687) · [DDNM](https://arxiv.org/abs/2212.00490)

## 6. 配准：坐标、组合和“每个病例重新优化”

体积数组使用 `(B,C,D,H,W)`，采样坐标最后一维使用 `(x,y,z)`，对应数组 `(W,H,D)`。形变分量是归一化的 `[-1,1]` 坐标位移。`warp(moving,u)` 计算 pull-back：固定坐标 `x` 去 moving 图像的 `x+u(x)` 处采样。不要把这个位移直接当作毫米，也不要把它与 push-forward 混用。

STN 从 identity affine 开始，测试初始输出保持输入。VoxelMorph 演示学习式 dense flow 和平滑约束。GradICON 演示前后向形变的组合误差 **空间导数**，并非仅惩罚位移差：两个不互逆的常量平移，其组合位移可非零，但导数惩罚仍为零。测试保留了这个区别，它也说明不能仅从一个损失值宣称严格拓扑保证。[STN](https://arxiv.org/abs/1506.02025) · [VoxelMorph](https://arxiv.org/abs/1809.05231) · [GradICON](https://arxiv.org/abs/2206.05897)

LapIRN 演示三个尺度的 residual displacement composition。位移采用归一化单位，扩大分辨率时不额外乘 2。该实现没有宣称复制原文的完整 residual-feature decoder。

TransMorph 直接调用官方四层级三维 Swin、原始 convolution decoder、两类 skip 和 SpatialTransformer。原作者的 baseline 训练代码使用 MSE 加 `0.02 × Grad3d(L2)`，本实验也调用原始 Grad3d。输入重采样为 `32³`，不是获得新的解剖细节；patch size4 和三次 merging 保留四层级。每层仍有普通与 shifted 两个 attention block，window 缩为 `2³`，默认 embedding width8。width4 与 width8 分别有 **78,543** 和 **306,393** 个参数。原始网络返回的 flow 是 **D/H/W 体素单位**，与前述自写归一化配准模块不同；`normalized_xyz_flow()` 提供显式转换，数值测试检查转换后与原始 sampler 一致。当前只关闭可配置 stochastic depth，未改写架构；双向损失合并成一次更新。[官方网络与配置](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/6357a1d7fc44c36db9b1d1ccaa372409253142cf/TransMorph/models) · [官方 baseline 训练脚本](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/TransMorph/train_TransMorph.py)

IDIR 用 SIREN 表示位移函数，并通过 autograd 计算空间导数。它是对**指定图像对**做 test-time optimization。为了保持这个任务定义，当前实例固定第一对合成图像，checkpoint 也保存这对图像；其他病例不被混进同一个无条件位移场。[IDIR 论文](https://proceedings.mlr.press/v172/wolterink22a.html)

DiffPose 演示“姿态初始化器 + 可微投影 + 梯度修正”的闭环。旋转由反对称矩阵的 matrix exponential 构造，保证合法旋转；默认投影是归一化平行束线积分。均匀体积的积分应为路径长度 2，测试检查这个量纲。它没有复制原版 cone-beam DiffDRR、病例预训练或位姿 geodesic loss。[DiffPose 论文](https://arxiv.org/abs/2312.06358)

## 7. 从二维图像到三维表示

X2CT-GAN 演示两个正交视图分别编码、沿缺失轴展开、融合为三维体积，并使用 adversarial、voxel、projection 三类损失。**它需要两幅视图**；不能把这个实验的结果当作单 AP 输入实验。投影当前是归一化均值积分，不是临床射线强度。[X2CT-GAN](https://openaccess.thecvf.com/content_CVPR_2019/html/Ying_X2CT-GAN_Reconstructing_CT_From_Biplanar_X-Rays_With_Generative_Adversarial_Networks_CVPR_2019_paper.html)

NeRF 的颜色是发射—吸收过程。沿射线的第 `i` 个样本：

`alpha_i = 1 − exp(−sigma_i delta_i)`，

`T_i = exp(−sum_{j<i} sigma_j delta_j)`，

`color = sum_i T_i alpha_i color_i`。

`T_i` 必须是 **exclusive** 累积。对于常密度、常颜色介质，计算应匹配解析解 `1−exp(−sigma L)`；测试验证这一点。这里使用有限积分区间，不在末尾塞入任意巨大的采样间隔。

NAF 的 X-ray attenuation field 与 NeRF 的颜色过程不同：它拟合 `integral mu(x) ds`，对应 transmission 为 `exp(−integral)`。衰减密度不依赖观察方向，测试单独验证这一不变性。Instant-NGP 演示多分辨率 hash table 与八个角点的 trilinear interpolation；没有 CUDA 融合、occupancy skipping 或实时速度承诺。[NeRF](https://arxiv.org/abs/2003.08934) · [NAF](https://github.com/ruyi-zha/naf_cbct) · [Instant-NGP](https://nvlabs.github.io/instant-ngp/)

pixelNeRF 根据已知参考相机将三维点投影到像素位置，再采样 CNN 特征作为场函数的条件。这里参考视角和目标视角不同，目标来自合成体积，不能把 GT 体积直接喂给查询网络。

3D Gaussian Splatting 演示真正的三维各向异性 covariance，经 perspective projection 的 Jacobian 投影到二维，再按深度排列 alpha-compose。它是 CPU dense rasterizer，保留 16 个 Gaussian；没有原版 tile rasterization、densification、pruning 或高阶 spherical harmonics。EndoNeRF 演示 canonical field、随时间变化的 deformation 和遮挡掩膜监督；“深度”和“器械遮挡”都由合成设置明确给出，不冒充真实内镜标注。[3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) · [pixelNeRF](https://arxiv.org/abs/2012.02190) · [EndoNeRF](https://github.com/med-air/EndoNeRF)

## 8. 可逆流：为什么不仅看“重建出来了”

RealNVP 的 affine coupling 保留一部分坐标，用它们预测另一部分坐标的 scale/translation，因此逆映射和 log determinant 都可精确计算。Glow 再加入 squeeze、数据初始化 ActNorm 和可逆 1×1 mixing。当前 Glow 直接计算小矩阵的 determinant/inverse，没有使用 LU 加速或多尺度 latent split。

对两种流，测试同时检查：

1. `inverse(forward(x)) ≈ x`。
2. 模型给出的 `log|det J|` 与 autograd 构建的完整小 Jacobian 的 determinant 一致。
3. 数据相关的 ActNorm 在第一次数据初始化后保持固定变换，避免把初始化本身的统计依赖混入 Jacobian 比较。

输入使用 8-bit uniform dequantization，损失包括 bin-width 项。这与直接对离散像素点计算连续密度的解释不同。[RealNVP](https://arxiv.org/abs/1605.08803) · [Glow](https://arxiv.org/abs/1807.03039)

## 9. 暂无 CPU 适配器的五项

| 编号 | 原项目 | 没有用其他网络替代的原因 |
|---|---|---|
| 058 | [Latent Diffusion](https://github.com/CompVis/latent-diffusion) | 完整流程需要明确训练/加载的 perceptual AE 与冻结 latent diffusion 阶段；随机编码器加 DDPM 不等于论文复现 |
| 061 | [DiffusionMBIR](https://github.com/hyungjin-chung/DiffusionMBIR) | 原版 2D score prior、三维物理算子与 TV/MBIR solver 要联合接通；一个去噪器不能替代整个方法 |
| 069 | [uniGradICON](https://github.com/uncbiag/uniGradICON) | 多域预训练和模型权重是 foundation model 主张的一部分；小 GradICON 网络不具有这些能力 |
| 127 | [DeepDRR](https://github.com/arcadelab/deepdrr) | 完整流程包含材料分解、能谱、散射、噪声和几何；单色线积分不是 DeepDRR |
| 130 | [MedNeRF](https://github.com/abrilcf/mednerf) | 需要完整 adversarial radiance-field training 与 latent inversion；普通 NeRF 不代表该生成先验 |

可以通过 `python -m biomed_lab fetch-upstream 编号` 获取登记的固定 commit；该操作不自动安装原项目、不下载权重、不执行项目脚本。需要具体 checkpoint、数据权限、CUDA 或旧运行时的步骤，按原项目 README 和许可证处理。

## 10. 来源、许可证与验收

直接调用的源码保存在 `vendor/inverse_*`。每个目录包含原许可证和 `PROVENANCE.json`，列出仓库、40 位 commit、原文件路径、下载地址和 SHA-256。DnCNN 的磁盘文件保持原样；运行时只桥接一个绝对 import，避免它的 `models.basicblock` 遮蔽其他 Python 包。fastMRI 的 VarNet/Unet 来自保留的原文件，Fourier/data helpers 使用固定的 `fastmri==0.3.0`。

| 目录 | 对应教学模型 | 许可证 |
|---|---|---|
| `inverse_cyclegan` | pix2pix、CycleGAN | BSD-2，并保留附带 DCGAN BSD-3 notices |
| `inverse_dncnn` | DnCNN | MIT |
| `inverse_fastmri` | VarNet | MIT |
| `inverse_swinir` | SwinIR | Apache-2.0 |
| `inverse_restormer` | Restormer | MIT |
| `inverse_dit` | DiT | **CC BY-NC 4.0：仅非商业使用范围，不被本仓库许可证覆盖** |
| `inverse_sit` | SiT | MIT |
| `inverse_transmorph` | TransMorph | MIT |

测试覆盖前向/损失/优化器更新、独立数学对照、信息泄漏边界、坐标与量纲、动态场景 checkpoint、配置是否真正降低参数量，以及复制源码的指纹。验收不包括原始 benchmark 数据、真实 CT/MRI/内镜结果、长期生成质量或临床测量可信度。

SwinIR 原始代码在固定依赖环境中可能输出 `timm.models.layers` 和 `torch.meshgrid` 的弃用警告。源码为保持 provenance 而未改写；这两条警告不表示上述检查失败。
