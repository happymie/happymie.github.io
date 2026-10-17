# Image, segmentation, Transformer and state-space lessons

This collection contains **32 executable lessons: 21 upstream implementations, four independent paper ports and seven bounded mechanism demonstrations**. The two remaining items in this part of the atlas, nnU-Net and VISTA3D, have explicit source routes. They are not replaced with an unrelated small network.

The purpose is to make a forward pass, objective and gradient inspectable on a CPU. **None of the lessons reproduces a published benchmark score, clinical result or foundation-model capability.** All models start with random weights. Training uses the declared synthetic database. A smaller network preserves only the mechanisms and topology stated below; it does not inherit the performance of a larger pretrained model.

## Start a lesson

Run these commands from the `teaching` directory after installing its pinned environment:

```bash
python -m biomed_lab.cli info 006
python -m biomed_lab.cli data 006 --inspect
python -m biomed_lab.cli run 006 --steps 20 --threads 2 --max-seconds 120
python -m biomed_lab.cli run 006 --steps 20 --set width=4
```

The last command changes U-Net's channel width while keeping its declared skip topology. Each model has its own `SPECS[num]["config"]`; change only settings exposed by that model. Use `info` to inspect the actual defaults and limitations. For FCN, PVT, Swin UNETR and SegFormer, use spatial sizes compatible with their downsampling hierarchy; the defaults are tested. Batch size two supports the BatchNorm layers at the smallest spatial scales.

For direct Python inspection:

```python
import torch
from biomed_lab.data import demo_batch
from biomed_lab.models.image_models import SPECS, build_model

torch.manual_seed(2026)
torch.set_num_threads(2)
model = build_model(16, {"d_model": 16, "d_state": 8})
batch = demo_batch(kind="waves")
loss = model.training_loss(batch)
loss.backward()
print(float(loss.detach()))
print(sum(p.numel() for p in model.parameters()))
```

The generic interface always exposes `training_loss(batch)`. Custom losses are part of the model lesson. In particular, V-Net uses soft Dice, Models Genesis reconstructs corrupted 3D inputs, Transformer uses a causal decoder, Mask2Former uses matched set losses and DeiT trains separate class and distillation heads. Segmentation adapters explicitly interpolate output logits back to the original synthetic target grid when a minimum model input size is needed. This interpolation does not provide new measurements or increase the information in the phantom.

## Implementation levels

| Label | What it means here | What it does not establish |
|---|---|---|
| `upstream` | A published open implementation supplies the central architecture or model block. Import/channel/head changes are recorded. | Identical training configuration, pretrained capability or original experimental results. |
| `paper_port` | An independent implementation preserves the specified equations/topology, with explicit dimensional and boundary changes. | Exact agreement with original weights, full pipeline or untested settings. |
| `mechanism_demo` | The named contribution has a working, differentiable teaching exercise. Missing components are listed. | A faithful reproduction of the complete named method. |
| Source route | The atlas links the actual upstream resources; this module raises an explicit unsupported error. | A runnable local implementation. |

The implementation label describes code provenance and scope, not scientific validity. For example, an upstream Swin UNETR architecture without its SSL pretraining is not an SSL result; a real Mamba block with a tiny classification head is not a pretrained Mamba language model.

## Coverage and CPU model size

Parameter counts below were calculated from the actual default constructors with PyTorch 2.5.1. These are trainable plus non-trainable `Parameter` elements. FP32 parameter storage is `count × 4 bytes`; it excludes gradients, optimizer states, activations, framework overhead and the operating system. It is not a measured peak RAM estimate.

| ID | Lesson | Implementation | Default parameters | Key adaptation or boundary |
|---|---|---|---:|---|
| 001 | AlexNet | Upstream torchvision | 2,519,491 | Original torchvision convolution trunk; adaptive 1×1 pooling and two 128-wide hidden FC layers. |
| 002 | ResNet | Upstream torchvision | 11,178,051 | ResNet-18 stages; 32×32 input repeated to RGB; three classes. |
| 003 | DenseNet | Upstream torchvision | 16,627 | Growth rate 8; four dense stages with two layers each; initial width 16. |
| 004 | ConvNeXt | Upstream torchvision | 305,875 | Widths 16/32/64/128; depths 1/1/3/1; original depthwise block mechanism. |
| 005 | FCN-8s | Paper port + torchvision VGG layer builder | 436,974 | Pool3/pool4 skips, bilinear-initialized learned upsampling; narrow, same-padded features. |
| 006 | U-Net | Paper port | 120,690 | Width 8; three pooling levels; same-padding rather than original valid-convolution cropping. |
| 007 | 3D U-Net | Paper port | 88,130 | Width 4; 3D layers and BatchNorm; no sparse-label clinical experiment. |
| 008 | V-Net | Upstream MONAI | 45,597,898 | Volumetric residual stages and squared-denominator Dice loss; 32³ model grid; dropout disabled. |
| 009 | nnU-Net | Source route | — | Self-configuration/planning/training pipeline has no local substitute. |
| 010 | UNETR | Upstream MONAI | 712,170 | 32³ input; hidden width 48, MLP width 96, four heads; all 12 encoder layers retained. |
| 011 | Swin UNETR | Upstream MONAI architecture | 4,080,392 | Feature width 12, depths 2/2/2/2, 32³ input; **SSL pretraining is absent**. |
| 012 | MedNeXt | Vendored authors' code | 69,395 | Width 4; expansion 2; nine stages, one block each; kernel 3. |
| 013 | Models Genesis | Mechanism demo | 21,517 | Four corruption families and 3D restoration; no pretrained encoder or transfer result. |
| 014 | VISTA3D | Source route | — | No replacement for the actual automatic/promptable foundation system and weights. |
| 015 | S4 | Vendored authors' code | 1,763 | **Full DPLR S4**, state size 16 and feature width 16; naive CPU Cauchy kernel. |
| 016 | Mamba | Vendored authors' code | 2,755 | State size 8, feature width 16; original differentiable reference selective scan. |
| 017 | VMamba | SS2D mechanism demo | 3,283 | Row/column forward/reverse scans; small patch stem/head; full VSSM hierarchy absent. |
| 018 | U-Mamba | Bottleneck mechanism demo | 24,126 | Residual 3D U-shaped network and real Mamba bottleneck; nnU-Net machinery absent. |
| 019 | SegMamba | Spatial-scan mechanism demo | 43,070 | Spatial convolution and three scan orders; **not the authors' BiMamba-v3 or fast-slice implementation**. |
| 020 | Transformer | Upstream PyTorch layers | 44,960 | Encoder-decoder, sinusoidal positions and masked teacher forcing; synthetic token reversal. |
| 021 | ViT | Upstream torchvision | 20,931 | 32×32 image; 4×4 patches; hidden width 32; depth two; class token retained. |
| 022 | Swin Transformer | Upstream torchvision | 574,737 | Width 16; two blocks per stage retain alternation of regular and shifted windows. |
| 023 | Mask2Former | Masked-attention mechanism demo | 50,403 | Four queries, three decoder blocks; real Hungarian matching, class/BCE/Dice losses; simple pixel decoder. |
| 081 | VGG-16/19 | Upstream torchvision feature builder | 239,083 | Default VGG-16 conv topology at width 8; three small FC layers; `variant=19` changes depth. |
| 082 | GoogLeNet | Upstream torchvision | 5,602,979 | Inception-v1 family, auxiliary classifiers disabled; three output classes. |
| 083 | ResNeXt | Upstream torchvision | 3,942,211 | One bottleneck per stage; cardinality 8, base width 4. |
| 084 | SENet | Upstream timm | 11,261,891 | SE-ResNet-18, not SENet-154; one-channel input. |
| 085 | EfficientNet | Upstream torchvision | 199,785 | B0-family MBConv configuration with width 0.25 and depth 0.5. |
| 086 | U-Net++ | Paper port | 140,766 | Nested skip grid at depth three; each top-row decoder head has a supervised loss. |
| 087 | Attention U-Net | Upstream MONAI | 30,310 | Additive attention gates, channels 8/16/32. |
| 088 | H-DenseUNet | Hybrid fusion mechanism demo | 12,126 | 2D dense features stacked in correct slice order, 3D dense branch and feature fusion. |
| 089 | DeiT | Distillation mechanism demo + timm | 20,070 | Two real tokens/heads; default teacher is an explicit synthetic geometry classifier. |
| 090 | PVT | Vendored authors' PVT v1 code | 69,587 | Four stages, widths 8/16/32/64, spatial-reduction attention and final class token. |
| 091 | SegFormer | Vendored authors' encoder/head | 87,954 | Narrow four-stage Mix Transformer; original MLP head via CPU shims; noncommercial license. |

V-Net is the largest default at approximately **173.94 MiB for FP32 parameters alone**. ResNet and SE-ResNet need about 43 MiB each. Most other lessons are much smaller. Start with U-Net, DenseNet, Mamba or S4 on a low-memory machine. Use the runner's bounded steps and memory budget, then examine its actual measurements before increasing width, spatial size or batch size. Reducing image/volume size reduces activation cost; reducing channels/state width reduces model cost. These two interventions change different parts of the computation.

The CPU setup exercises learning on small arrays; it is not a throughput claim for a particular laptop. S4 and Mamba use transparent reference kernels, whose runtime can grow substantially for long sequences. GPU kernels are not silently installed.

## What to inspect in each mechanism

### CNNs and dense prediction

Compare `ResNet` addition with `DenseNet` concatenation while holding the synthetic split fixed. Then inspect ConvNeXt's channel permutation around LayerNorm, ResNeXt's grouped convolution, SE global pooling and EfficientNet's expanded depthwise bottleneck. The default sizes differ, so their raw synthetic accuracies are not a controlled architecture ranking.

FCN-8s predicts at coarse resolution and combines score maps from pool4 and pool3. U-Net combines feature tensors at each decoder scale. U-Net++ builds additional nodes between those scales; `NestedUNet.all_outputs` makes every supervised depth visible. Attention U-Net computes spatial gates on skip features. The tests verify that all U-Net++ supervised heads actually receive gradients.

The 2D U-Net and 3D U-Net ports use same-padding to support small phantoms. Therefore their boundary behavior differs from the original valid-convolution networks. The 3D lesson's labels are dense synthetic masks, so it does not demonstrate the original sparse-annotation claim. UNETR and Swin UNETR are genuine MONAI architectures; resized phantoms provide a compatible tensor grid but no additional anatomical information.

### S4, Mamba and image scans

S4's vendored `SSMKernelDPLR` is the full diagonal-plus-low-rank system from the authors' standalone implementation. It is **not a diagonal S4D layer relabelled as S4**. The only runtime dependency adapter replaces a single-process logging decorator. CPU construction uses the original HiPPO initialization and the original naive Cauchy fallback.

Mamba retains the authors' `Mamba` class, including input-dependent time steps and B/C projections, causal depthwise convolution, gating and state initialization. Its scan calls the authors' unchanged `selective_scan_ref` function. The test compares the full-sequence forward path with a separate token-by-token `step` path and checks numerical agreement.

The VMamba lesson applies four independently parameterized scans, restores their row/column/reverse orders and fuses aligned spatial features. The non-square-grid test catches a common transpose or reverse error. It is a compact SS2D exercise, not a full VSSM reproduction. U-Mamba and SegMamba are similarly scoped demonstrations; in particular, the local three-order SegMamba analogue must not be reported as reproducing BiMamba-v3 or the published model.

### Masked attention and distillation

The Mask2Former lesson computes each query's current mask before cross-attention. Positions outside the predicted mask are blocked. A fully blocked row is released to avoid undefined attention softmax. A Hungarian assignment matches unordered query predictions to foreground/background semantic masks. The resulting class, binary-cross-entropy and Dice losses are real, and the test checks query-permutation invariance. Its convolution pixel decoder replaces the full multi-scale deformable encoder; this is a substantive missing component, not an implementation detail to hide.

The DeiT lesson uses timm's distilled ViT class. A class token learns from the provided label; a separate distillation token learns from a teacher's hard predicted class. For the synthetic circle/square/diamond data, the default teacher measures foreground fill ratio from the **input image** and compares it with analytic shape prototypes. It does not read a target mask or a label. This provides a visible non-neural teacher for the exercise; it is not the paper's RegNetY teacher or evidence for data-efficient ImageNet training. A real detached `batch["teacher_logits"]` tensor can replace it. The test verifies gradients in both tokens and both heads.

### Restoration pretraining and hybrid 2D/3D fusion

The Models Genesis exercise uses a Bezier intensity map, local voxel shuffling, inpainting and outpainting before a 3D restoration loss. It uses an independently written PyTorch implementation with a much smaller corruption budget and network. The clean target remains unchanged, as checked by a dedicated test. This can teach why a reconstruction pretext objective differs from a segmentation objective. It cannot establish that a learned representation transfers to real disease tasks.

H-DenseUNet's teaching version encodes each axial slice with shared 2D dense layers, restores the batch/depth ordering, concatenates those features with the input volume, computes a 3D dense branch and fuses both feature sets. It does not include the original staged initialization, full DenseNet depths or LiTS preprocessing/training.

## Sources and exact code provenance

Primary implementation references:

- [torchvision v0.20.1](https://github.com/pytorch/vision/tree/v0.20.1/torchvision/models): AlexNet, ResNet, DenseNet, ConvNeXt, VGG, GoogLeNet, EfficientNet, ViT and Swin.
- [MONAI 1.4.0](https://github.com/Project-MONAI/MONAI/tree/1.4.0/monai/networks/nets): VNet, UNETR, SwinUNETR and AttentionUnet.
- [timm 1.0.11](https://github.com/huggingface/pytorch-image-models/tree/v1.0.11/timm/models): SE-ResNet and distilled ViT.
- [FCN-8s original network definition](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/train.prototxt), [U-Net](https://arxiv.org/abs/1505.04597), [3D U-Net](https://arxiv.org/abs/1606.06650) and [U-Net++](https://arxiv.org/abs/1807.10165): topology references for independent ports.
- [Models Genesis](https://arxiv.org/abs/1908.06912), [H-DenseUNet](https://arxiv.org/abs/1709.07330), [VMamba](https://github.com/MzeroMiko/VMamba), [U-Mamba](https://github.com/bowang-lab/U-Mamba), [SegMamba](https://github.com/ge-xing/SegMamba), [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [DeiT](https://github.com/facebookresearch/deit): primary sources for the explicitly partial teaching mechanisms.

Vendored files include `original/` copies and per-directory `PROVENANCE.json` records. The `git_blob_sha` field identifies the exact upstream file bytes, including line endings. The hash test recomputes Git's blob digest independently; a file that merely has the same name is insufficient. Modified files are separate from the originals.

| Vendor directory | Upstream source | Runtime changes | License |
|---|---|---|---|
| `image_mednext` | [MedNeXt, commit 0b78ed8](https://github.com/MIC-DKFZ/MedNeXt/tree/0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba) | One absolute import becomes a local relative import. `blocks.py` is copied unchanged. | Apache-2.0 |
| `image_s4` | [S4, commit e757cef](https://github.com/state-spaces/s4/tree/e757cef57d89e448c413de7325ed5601aceaac13) | Lightning logging decorator becomes an identity decorator for one CPU process. Original DPLR and FFT equations retained. | Apache-2.0 |
| `image_mamba` | [Mamba](https://github.com/state-spaces/mamba) | Backend imports route to an unchanged extracted reference scan; CUDA/Triton backend calls disabled. Exact revision and file identifiers are in its manifest. | Apache-2.0 |
| `image_pvt` | [PVT v1 source in the v2 branch](https://github.com/whai362/PVT/blob/v2/classification/pvt.py) | Global timm registration decorators removed; constructor and forward code retained. | Apache-2.0 |
| `image_segformer` | [SegFormer](https://github.com/NVlabs/SegFormer) | Remove MMCV/MMSeg registration/checkpoint dependencies; minimal decode-head interfaces; single-process BatchNorm replaces SyncBatchNorm. | NVIDIA Source Code License; noncommercial research/evaluation only |

The original teaching module has its own repository license. That license does **not** override vendor licenses. SegFormer's isolated vendor directory retains NVIDIA's copyright notice and complete license. The Models Genesis and H-DenseUNet lessons contain independent code, not copied source files from their repositories. External checkpoint licensing is a separate matter; no checkpoint is bundled or downloaded by these constructors.

## Validation and its limits

The implementation was checked with Python 3.12, PyTorch 2.5.1 CPU, torchvision 0.20.1, MONAI 1.4.0, timm 1.0.11, einops 0.8.0, NumPy 1.26.4 and SciPy 1.14.1. Use the repository's complete dependency constraints because unrelated package upgrades can break this combination.

The test file performs 42 checks in total:

1. A real finite loss, backward pass, nonzero finite gradient and AdamW update for each of the 32 declared lessons.
2. Explicit rejection of unavailable frameworks and JSON-serializable metadata.
3. Mamba batch scan versus independent streaming recurrence.
4. Full DPLR S4 type and causality.
5. Four-direction spatial scan alignment on a non-square grid.
6. Transformer decoder masking against future-token changes.
7. Mask-set loss permutation invariance and fully blocked attention behavior.
8. Gradient delivery to all U-Net++ deep-supervision heads.
9. Clean-target preservation under both Genesis painting modes.
10. Gradient delivery to both DeiT tokens and heads.
11. Exact upstream Git blob hashes for the preserved vendor sources and licenses.

Run them with:

```bash
python -m pytest tests/test_image_models.py
```

A finite training step verifies that the declared path actually executes and differentiates. It does not show optimization convergence, generalization, clinical utility or original-paper equivalence. Learned synthetic results should be interpreted within the specimen-level split and data-generation assumptions recorded by the database and runner. They must not be inserted into a clinical comparison table as published-method reproductions.
