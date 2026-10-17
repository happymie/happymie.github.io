# 自监督、图学习与循环网络

实际代码：[research_models.py](../biomed_lab/models/research_models.py)。测试：[test_research_models.py](../tests/test_research_models.py)。本模块所有训练从随机权重开始，使用明确的合成数据；不复制任何基础模型预训练效果。

| 条目 | 实现与主要边界 |
| --- | --- |
| SimCLR | 2N 对称 InfoNCE，屏蔽自身相似度，保留正确跨增强正样本；较小的 patch Transformer 代替原 ResNet，不是完整 SimCLR 训练复现 |
| MoCo | 独立动量编码器、停止梯度 key 和更新后队列；小编码器/队列，没有原数据与调度 |
| MAE | 固定版本作者 encoder/decoder、masking 与像素重建；灰度复制 RGB，32×32、patch4，64 个 patch token 按 mask 比例只将可见部分送入编码器；去除新版 timm 已移除的默认参数，不替换注意力 |
| DINO / iBOT / DINOv2 | EMA 教师、中心化/温度目标；iBOT 增加 mask patch 目标。DINOv2 只教这些共享机制，缺少原数据清理、完整损失/头与规模化流程，均标为机制演示 |
| BEiT | 学习 mask token 的离散目标机制，使用固定强度量化器，未实现/下载原 dVAE tokenizer |
| BYOL | online predictor 与停止梯度 target，EMA 更新，不复制原训练规模 |
| SwAV | 多原型与数值稳定 Sinkhorn 平衡分配，交换预测；小批次和短增强不提供原版效果 |
| GCN / GAT / GraphSAGE | 固定 PyTorch Geometric 的实际对应层；完整图为数据划分单位，图内节点分类。原代码档案与这里执行的 PyG 来源分开记录 |
| Graphormer | 只包含 degree/centrality 与最短路径 attention bias，不含原 multi-hop edge-feature 编码和完整 graph-token 目标 |
| LSTM / GRU | PyTorch 的实际循环层；LSTM 使用现代遗忘门版本，GRU reset-gate 计算位置与 Cho 原论文有差异 |
| TCN | 作者 causal/chomp/dilated residual block；通道数和层数縮小，保留因果性 |

MAE 对 timm 兼容性的补丁、原始文件和修改后文件都在 `vendor/research_mae/`；作者 CC BY-NC 4.0 条款保留。TCN 保留作者 MIT 源码、许可证与固定 commit。详见各自 `PROVENANCE.json` 和全局 [第三方说明](../THIRD_PARTY_NOTICES.md)。

## 可做的实验

对 InfoNCE，将正确图文/视图配对打乱，与固定输入的损失比较；不要把两步下降当成训练成功。对 MoCo，检查 optimizer 更新前后的队列与 teacher 变化，确认 teacher 不在反向传播图中。对 MAE，改变 mask ratio，检查进入 encoder 的 token 数量；decoder 重建完整 patch grid。对 SwAV，检查非负分配、行归一化与近似平衡列边际。对图层，对完整图做一致节点置换，输出应同步置换。对 TCN，只修改未来输入，过去输出应保持不变。

默认实验不使用测试划分。教师队列/中心等 forward 状态在验证后恢复，验证不会污染下次训练，随机数状态也恢复。所有结构差异在逐模型课程中重复列明。
