# 覆盖与实际验证 / Coverage

137 项课程与 137 个独立数据库（合计 **13.03 MiB**，每库最大 128 KiB）；**102 项本地适配器**：45 复用上游、27 论文结构/方程实现、30 部分机制演示；**35 项原版路径尚无 CPU 适配器**。当前摘要匹配的成功 CPU 执行检查：**102 项**。

生成自真实注册表和数据；状态含义见 [复现标准](REPRODUCTION_STANDARD.md)。两步执行成功不等于完整论文复现。

| # | 模型 / 课程 | 实现范围 | CPU 检查 | 参数量 | 峰值 MiB | 数据 KiB |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 001 | [AlexNet](../lessons/001.md) | `upstream` | completed | 2519491 | 477.2 | 60 |
| 002 | [ResNet](../lessons/002.md) | `upstream` | completed | 11178051 | 534.3 | 60 |
| 003 | [DenseNet](../lessons/003.md) | `upstream` | completed | 16627 | 291.2 | 60 |
| 004 | [ConvNeXt](../lessons/004.md) | `upstream` | completed | 305875 | 304.2 | 56 |
| 005 | [FCN](../lessons/005.md) | `paper_port` | completed | 436974 | 304.4 | 128 |
| 006 | [U-Net](../lessons/006.md) | `paper_port` | completed | 120690 | 283.0 | 128 |
| 007 | [3D U-Net](../lessons/007.md) | `paper_port` | completed | 88130 | 288.4 | 128 |
| 008 | [V-Net](../lessons/008.md) | `upstream` | completed | 45597898 | 1544.6 | 128 |
| 009 | [nnU-Net](../lessons/009.md) | `source_route` | — | — | — | 128 |
| 010 | [UNETR](../lessons/010.md) | `upstream` | completed | 712170 | 513.3 | 128 |
| 011 | [Swin UNETR (SSL)](../lessons/011.md) | `upstream` | completed | 4080392 | 973.1 | 128 |
| 012 | [MedNeXt](../lessons/012.md) | `upstream` | completed | 69395 | 357.0 | 128 |
| 013 | [Models Genesis](../lessons/013.md) | `mechanism_demo` | completed | 21517 | 289.2 | 128 |
| 014 | [VISTA3D](../lessons/014.md) | `source_route` | — | — | — | 128 |
| 015 | [S4](../lessons/015.md) | `upstream` | completed | 1763 | 298.9 | 128 |
| 016 | [Mamba](../lessons/016.md) | `upstream` | completed | 2755 | 284.0 | 128 |
| 017 | [VMamba](../lessons/017.md) | `mechanism_demo` | completed | 3283 | 287.9 | 128 |
| 018 | [U-Mamba](../lessons/018.md) | `mechanism_demo` | completed | 24126 | 295.6 | 128 |
| 019 | [SegMamba](../lessons/019.md) | `mechanism_demo` | completed | 43070 | 301.8 | 128 |
| 020 | [Transformer](../lessons/020.md) | `upstream` | completed | 44960 | 273.5 | 60 |
| 021 | [Vision Transformer (ViT)](../lessons/021.md) | `upstream` | completed | 20931 | 291.4 | 60 |
| 022 | [Swin Transformer](../lessons/022.md) | `upstream` | completed | 574737 | 305.6 | 56 |
| 023 | [Mask2Former](../lessons/023.md) | `mechanism_demo` | completed | 50403 | 315.5 | 128 |
| 024 | [SimCLR](../lessons/024.md) | `mechanism_demo` | completed | 21952 | 281.4 | 56 |
| 025 | [MoCo](../lessons/025.md) | `mechanism_demo` | completed | 43904 | 280.9 | 56 |
| 026 | [MAE](../lessons/026.md) | `upstream` | completed | 88816 | 323.1 | 56 |
| 027 | [DINO](../lessons/027.md) | `mechanism_demo` | completed | 42848 | 280.7 | 56 |
| 028 | [iBOT](../lessons/028.md) | `mechanism_demo` | completed | 42848 | 281.1 | 60 |
| 029 | [DINOv2](../lessons/029.md) | `mechanism_demo` | completed | 42848 | 282.1 | 56 |
| 030 | [CLIP](../lessons/030.md) | `upstream` | completed | 220929 | 284.9 | 128 |
| 031 | [BLIP-2](../lessons/031.md) | `upstream` | completed | 65408 | 350.2 | 128 |
| 032 | [Segment Anything (SAM)](../lessons/032.md) | `upstream` | completed | 166246 | 292.7 | 80 |
| 033 | [MedSAM](../lessons/033.md) | `upstream` | completed | 166246 | 459.3 | 80 |
| 034 | [SAM 2](../lessons/034.md) | `source_route` | — | — | — | 80 |
| 035 | [ConVIRT](../lessons/035.md) | `source_route` | — | — | — | 128 |
| 036 | [GLoRIA](../lessons/036.md) | `source_route` | — | — | — | 128 |
| 037 | [BioViL](../lessons/037.md) | `source_route` | — | — | — | 128 |
| 038 | [BiomedCLIP](../lessons/038.md) | `source_route` | — | — | — | 128 |
| 039 | [UNI](../lessons/039.md) | `source_route` | — | — | — | 128 |
| 040 | [CONCH](../lessons/040.md) | `source_route` | — | — | — | 128 |
| 041 | [Prov-GigaPath](../lessons/041.md) | `source_route` | — | — | — | 128 |
| 042 | [Merlin](../lessons/042.md) | `source_route` | — | — | — | 128 |
| 043 | [Deep Autoencoder](../lessons/043.md) | `paper_port` | completed | 26000 | 268.8 | 60 |
| 044 | [GAN](../lessons/044.md) | `paper_port` | completed | 17249 | 268.9 | 56 |
| 045 | [pix2pix](../lessons/045.md) | `upstream` | completed | 305194 | 290.9 | 56 |
| 046 | [CycleGAN](../lessons/046.md) | `upstream` | completed | 224308 | 315.6 | 56 |
| 047 | [FBPConvNet](../lessons/047.md) | `mechanism_demo` | completed | 9449 | 282.6 | 56 |
| 048 | [Learned Primal-Dual](../lessons/048.md) | `paper_port` | completed | 8550 | 282.0 | 56 |
| 049 | [Variational Network](../lessons/049.md) | `paper_port` | completed | 699 | 280.2 | 56 |
| 050 | [MoDL](../lessons/050.md) | `paper_port` | completed | 2051 | 279.9 | 60 |
| 051 | [VarNet](../lessons/051.md) | `upstream` | completed | 65784 | 334.6 | 56 |
| 052 | [DnCNN](../lessons/052.md) | `upstream` | completed | 9153 | 279.2 | 60 |
| 053 | [Noise2Noise](../lessons/053.md) | `paper_port` | completed | 9449 | 277.5 | 56 |
| 054 | [Noise2Self](../lessons/054.md) | `paper_port` | completed | 9449 | 279.1 | 56 |
| 055 | [DDPM](../lessons/055.md) | `paper_port` | completed | 10241 | 279.3 | 56 |
| 056 | [DDIM](../lessons/056.md) | `paper_port` | completed | 10241 | 279.2 | 56 |
| 057 | [Score-SDE](../lessons/057.md) | `paper_port` | completed | 10241 | 278.9 | 56 |
| 058 | [Latent Diffusion](../lessons/058.md) | `source_route` | — | — | — | 56 |
| 059 | [DDRM](../lessons/059.md) | `paper_port` | completed | 10241 | 279.2 | 56 |
| 060 | [DPS](../lessons/060.md) | `paper_port` | completed | 10241 | 279.2 | 56 |
| 061 | [DiffusionMBIR](../lessons/061.md) | `source_route` | — | — | — | 56 |
| 062 | [Flow Matching](../lessons/062.md) | `paper_port` | completed | 10241 | 279.0 | 60 |
| 063 | [Rectified Flow](../lessons/063.md) | `paper_port` | completed | 10241 | 278.9 | 56 |
| 064 | [Consistency Models](../lessons/064.md) | `paper_port` | completed | 20482 | 279.5 | 56 |
| 065 | [DiT](../lessons/065.md) | `upstream` | completed | 181712 | 322.8 | 56 |
| 066 | [Spatial Transformer Network](../lessons/066.md) | `paper_port` | completed | 1374 | 277.5 | 128 |
| 067 | [VoxelMorph](../lessons/067.md) | `mechanism_demo` | completed | 7499 | 284.6 | 128 |
| 068 | [GradICON](../lessons/068.md) | `mechanism_demo` | completed | 7499 | 288.0 | 128 |
| 069 | [uniGradICON](../lessons/069.md) | `source_route` | — | — | — | 128 |
| 070 | [DiffPose](../lessons/070.md) | `mechanism_demo` | completed | 1302 | 279.0 | 128 |
| 071 | [X2CT-GAN](../lessons/071.md) | `mechanism_demo` | completed | 9590 | 287.3 | 128 |
| 072 | [NeRF](../lessons/072.md) | `mechanism_demo` | completed | 4676 | 272.1 | 128 |
| 073 | [NAF](../lessons/073.md) | `mechanism_demo` | completed | 6116 | 272.2 | 128 |
| 074 | [3D Gaussian Splatting](../lessons/074.md) | `mechanism_demo` | completed | 208 | 271.5 | 128 |
| 075 | [GCN](../lessons/075.md) | `upstream` | completed | 195 | 306.0 | 128 |
| 076 | [GAT](../lessons/076.md) | `upstream` | completed | 233 | 306.5 | 128 |
| 077 | [LSTM](../lessons/077.md) | `upstream` | completed | 1459 | 278.3 | 128 |
| 078 | [Temporal Convolutional Network (TCN)](../lessons/078.md) | `upstream` | completed | 4355 | 277.1 | 128 |
| 079 | [HeartBEiT](../lessons/079.md) | `upstream` | completed | 81635 | 322.5 | 128 |
| 080 | [LaBraM](../lessons/080.md) | `upstream` | completed | 177895 | 328.9 | 128 |
| 081 | [VGG-16/19](../lessons/081.md) | `upstream` | completed | 239083 | 294.6 | 56 |
| 082 | [GoogLeNet / Inception-v1](../lessons/082.md) | `upstream` | completed | 5602979 | 416.2 | 56 |
| 083 | [ResNeXt](../lessons/083.md) | `upstream` | completed | 3942211 | 394.7 | 56 |
| 084 | [SENet](../lessons/084.md) | `upstream` | completed | 11261891 | 573.8 | 56 |
| 085 | [EfficientNet](../lessons/085.md) | `upstream` | completed | 199785 | 304.1 | 56 |
| 086 | [U-Net++](../lessons/086.md) | `paper_port` | completed | 140766 | 285.1 | 128 |
| 087 | [Attention U-Net](../lessons/087.md) | `upstream` | completed | 30310 | 453.9 | 128 |
| 088 | [H-DenseUNet](../lessons/088.md) | `mechanism_demo` | completed | 12126 | 297.5 | 128 |
| 089 | [DeiT](../lessons/089.md) | `mechanism_demo` | completed | 20070 | 321.0 | 56 |
| 090 | [PVT](../lessons/090.md) | `upstream` | completed | 69587 | 324.1 | 56 |
| 091 | [SegFormer](../lessons/091.md) | `upstream` | completed | 87954 | 328.4 | 128 |
| 092 | [BEiT](../lessons/092.md) | `mechanism_demo` | completed | 20104 | 280.7 | 60 |
| 093 | [BYOL](../lessons/093.md) | `mechanism_demo` | completed | 46016 | 281.1 | 56 |
| 094 | [SwAV](../lessons/094.md) | `mechanism_demo` | completed | 22208 | 281.1 | 56 |
| 095 | [ALIGN](../lessons/095.md) | `upstream` | completed | 119821 | 360.9 | 128 |
| 096 | [BLIP](../lessons/096.md) | `source_route` | — | — | — | 128 |
| 097 | [Florence-2](../lessons/097.md) | `source_route` | — | — | — | 128 |
| 098 | [MedCLIP](../lessons/098.md) | `source_route` | — | — | — | 128 |
| 099 | [Med-Flamingo](../lessons/099.md) | `source_route` | — | — | — | 128 |
| 100 | [LLaVA-Med](../lessons/100.md) | `source_route` | — | — | — | 128 |
| 101 | [RadFM](../lessons/101.md) | `source_route` | — | — | — | 128 |
| 102 | [MAIRA-1](../lessons/102.md) | `source_route` | — | — | — | 128 |
| 103 | [MAIRA-2](../lessons/103.md) | `source_route` | — | — | — | 128 |
| 104 | [BiomedGPT](../lessons/104.md) | `source_route` | — | — | — | 128 |
| 105 | [Virchow](../lessons/105.md) | `source_route` | — | — | — | 128 |
| 106 | [CHIEF](../lessons/106.md) | `source_route` | — | — | — | 128 |
| 107 | [TITAN](../lessons/107.md) | `source_route` | — | — | — | 128 |
| 108 | [PRISM2](../lessons/108.md) | `source_route` | — | — | — | 128 |
| 109 | [BME-X](../lessons/109.md) | `source_route` | — | — | — | 128 |
| 110 | [Prima](../lessons/110.md) | `source_route` | — | — | — | 128 |
| 111 | [MARS](../lessons/111.md) | `source_route` | — | — | — | 128 |
| 112 | [Decipher-MR](../lessons/112.md) | `source_route` | — | — | — | 128 |
| 113 | [Noise2Void](../lessons/113.md) | `paper_port` | completed | 9449 | 278.9 | 60 |
| 114 | [RED-Net](../lessons/114.md) | `mechanism_demo` | completed | 4825 | 276.9 | 56 |
| 115 | [SwinIR](../lessons/115.md) | `upstream` | completed | 36253 | 348.7 | 56 |
| 116 | [Restormer](../lessons/116.md) | `upstream` | completed | 191335 | 305.4 | 60 |
| 117 | [AUTOMAP](../lessons/117.md) | `mechanism_demo` | completed | 197857 | 283.3 | 56 |
| 118 | [LEARN](../lessons/118.md) | `paper_port` | completed | 2214 | 281.3 | 56 |
| 119 | [DDNM](../lessons/119.md) | `paper_port` | completed | 10241 | 279.2 | 60 |
| 120 | [RealNVP](../lessons/120.md) | `paper_port` | completed | 25216 | 269.1 | 56 |
| 121 | [Glow](../lessons/121.md) | `paper_port` | completed | 3156 | 279.6 | 56 |
| 122 | [PFGM++](../lessons/122.md) | `paper_port` | completed | 10241 | 279.5 | 56 |
| 123 | [SiT](../lessons/123.md) | `upstream` | completed | 181712 | 322.5 | 56 |
| 124 | [LapIRN](../lessons/124.md) | `mechanism_demo` | completed | 23145 | 286.4 | 128 |
| 125 | [TransMorph](../lessons/125.md) | `upstream` | completed | 306393 | 397.7 | 128 |
| 126 | [IDIR](../lessons/126.md) | `paper_port` | completed | 1283 | 292.9 | 128 |
| 127 | [DeepDRR](../lessons/127.md) | `source_route` | — | — | — | 128 |
| 128 | [pixelNeRF](../lessons/128.md) | `mechanism_demo` | completed | 6324 | 283.2 | 128 |
| 129 | [Instant-NGP](../lessons/129.md) | `mechanism_demo` | completed | 6116 | 273.4 | 128 |
| 130 | [MedNeRF](../lessons/130.md) | `source_route` | — | — | — | 128 |
| 131 | [EndoNeRF](../lessons/131.md) | `mechanism_demo` | completed | 4935 | 272.9 | 128 |
| 132 | [GraphSAGE](../lessons/132.md) | `upstream` | completed | 371 | 305.9 | 128 |
| 133 | [Graphormer](../lessons/133.md) | `mechanism_demo` | completed | 9959 | 272.9 | 128 |
| 134 | [Voxel2Mesh](../lessons/134.md) | `source_route` | — | — | — | 128 |
| 135 | [Vox2Cortex](../lessons/135.md) | `source_route` | — | — | — | 128 |
| 136 | [GRU](../lessons/136.md) | `upstream` | completed | 1107 | 270.5 | 128 |
| 137 | [ECGFounder](../lessons/137.md) | `upstream` | completed | 73787 | 312.3 | 128 |

## 尚无 CPU 适配器的明确缺口

### 009 nnU-Net

nnU-Net includes dataset fingerprinting, automatic planning, preprocessing, training and inference. A generic U-Net is not the system; the present upstream default is v2, so the paper-matched version and full pipeline must be selected.

### 014 VISTA3D

The original VISTA3D point/class-conditioned 3D architecture and pretrained workflow have not been integrated or checked here.

### 034 SAM 2

SAM2 requires its hierarchical image encoder, memory encoder/attention and video prompt/state protocol; these are not provided by the SAM1 adapter.

### 035 ConVIRT

The identified author release contains evaluation material, not a complete verified ConVIRT training implementation. The image/text encoders, pair sampling and full pretraining recipe still need a paper-based implementation.

### 036 GLoRIA

The defining local/global radiology image–text alignment and complete report preprocessing/training workflow are not implemented here.

### 037 BioViL

The official BioViL implementation is in Microsoft hi-ml. Its image encoder, clinical language model and original pretraining stages have not been integrated into this CPU lab.

### 038 BiomedCLIP

The published BiomedCLIP architecture, biomedical tokenizer and pretraining workflow are not supplied by the generic CLIP lesson. No BiomedCLIP weights are bundled.

### 039 UNI

The UNI pretrained pathology representation and data/training recipe are not reproduced. The upstream CC BY-NC-ND terms are not replaced by this repository license.

### 040 CONCH

CONCH requires its pathology vision–language architecture and objectives; neither the CLIP lesson nor a generic pathology classifier is a CONCH reproduction.

### 041 Prov-GigaPath

The complete tile encoder plus slide-level long-context aggregation and original pretraining stages are not implemented in the present CPU lab.

### 042 Merlin

The Merlin 3D imaging and clinical-text pretraining pipeline has not been integrated; random 3D classification is not Merlin.

### 058 Latent Diffusion

A faithful latent-diffusion workflow needs its learned first-stage representation, latent scaling and conditional diffusion model. The pixel-space DDPM lesson is not substituted.

### 061 DiffusionMBIR

The pretrained score prior, 3D/2.5D reconstruction workflow and full DiffusionMBIR data-consistency procedure have not been implemented or checked.

### 069 uniGradICON

The uniGradICON foundation registration model and training distribution are not supplied by the GradICON mechanism demonstration.

### 096 BLIP

BLIP requires the image–text contrastive/matching/language objectives and captioning/filtering workflow; the full original method is not implemented here.

### 097 Florence-2

Florence-2 uses a task-conditioned vision/language architecture and task-specific token/output processing. Its released checkpoint and full workflow are not integrated here.

### 098 MedCLIP

The source audit did not establish a root license grant for MedCLIP. No permission to copy was inferred; its semantic-matching objective and encoders are not reimplemented here.

### 099 Med-Flamingo

Med-Flamingo requires its visual resampler, cross-attention language bridge and original adaptation protocol. A generic image–text contrastive network is not substituted.

### 100 LLaVA-Med

The LLaVA-Med visual projector, language backbone and two-stage medical instruction tuning are not implemented or validated here.

### 101 RadFM

RadFM requires a volumetric vision encoder, language bridge and multimodal instruction pipeline. This is not replaced by a 3D classifier.

### 102 MAIRA-1

An accessible complete MAIRA-1 release was not established in the source audit. MAIRA-2 is a different model. Paper-only implementation details remain unresolved.

### 103 MAIRA-2

The MAIRA-2 image/report grounding and language generation pipeline is not integrated; the published model-card/access terms remain upstream.

### 104 BiomedGPT

BiomedGPT requires its unified multimodal token/sequence tasks and training mixture. A small generic Transformer does not reproduce these stages.

### 105 Virchow

Virchow pretrained pathology representations and their data/training protocol are not reproduced by the generic DINOv2 mechanism lesson.

### 106 CHIEF

The CHIEF pathology tile/slide representation and clinical-context workflow are not integrated into this CPU teaching runner.

### 107 TITAN

TITAN uses a complete slide-level visual/text representation pipeline; original training stages and pretrained capabilities are not reproduced here.

### 108 PRISM2

PRISM2 public metadata is available, but its model access is gated and its terms restrict redistribution. No gated model files are mirrored and no local PRISM2 implementation is claimed.

### 109 BME-X

The original BME-X implementation is linked and pinned; its complete multimodal learning workflow has not been converted into a tested low-resource adapter.

### 110 Prima

The original Prima implementation is linked and pinned; its scan/series representation and complete training/inference protocol are not implemented locally.

### 111 MARS

The original MARS implementation is linked and pinned; its defining pretraining and downstream workflow have not been integrated and tested here.

### 112 Decipher-MR

No complete public Decipher-MR release was verified. The named DINOv2/BERT/OpenCLIP components are not themselves Decipher-MR, and the paper states pretrained weights may not be shared.

### 127 DeepDRR

DeepDRR includes calibrated geometry, material decomposition/scatter and physical image formation. The differentiable toy ray projectors here are not a DeepDRR replacement.

### 130 MedNeRF

MedNeRF requires its radiance-field adversarial training and original conditioning pipeline. The generic NeRF mechanism demo is not renamed as MedNeRF.

### 134 Voxel2Mesh

Voxel2Mesh needs its deformable mesh initialization, graph-convolution refinement, adaptive unpooling and mesh losses. The graph classification lessons are not substitutes; the original GPL source remains linked.

### 135 Vox2Cortex

Vox2Cortex requires cortical surface topology, volumetric feature sampling and coupled mesh refinement/losses. These have not been implemented in this CPU lab; original GPL source remains linked.

