# HeartBEiT, LaBraM and ECGFounder architecture lessons

These three lessons run **licensed upstream downstream architectures with random weights**. Their forward computation can be inspected and trained on the small synthetic databases. They do not recreate the training corpus, pretrained knowledge or clinical results of the named foundation models.

The architectures accept different representations. HeartBEiT operates on images of traces. LaBraM operates on an electrode-by-time-patch tensor. ECGFounder operates directly on waveform channels. These differences are implemented in the adapters, not hidden behind a generic CNN.

## Run

```bash
python -m biomed_lab.cli info 079
python -m biomed_lab.cli run 079 --steps 20 --threads 2
python -m biomed_lab.cli run 080 --steps 20 --threads 2
python -m biomed_lab.cli run 137 --steps 20 --threads 2
python -m pytest tests/test_extended_models.py
```

Use the repository's pinned PyTorch 2.5.1 CPU, torchvision 0.20.1, timm 1.0.11 and einops 0.8.0 environment. No checkpoint is downloaded at model construction. There is no assumption that a randomly initialized small encoder has useful medical knowledge.

The actual default constructors contain the following parameter counts. FP32 storage below includes parameters only; gradients, optimizer states, activations and Python/framework memory are additional.

| Lesson | Parameters | FP32 parameter storage |
|---|---:|---:|
| HeartBEiT architecture | 81,635 | 0.3114 MiB |
| LaBraM architecture | 177,895 | 0.6786 MiB |
| ECGFounder architecture | 73,787 | 0.2815 MiB |

## 079 — HeartBEiT

The original [HeartBEiT model file](https://github.com/akhilvaid/HeartBEiT/blob/e065cca3c41d6bc5e852a44c6f7fc1fb51c4d45e/modeling_finetune.py) contains a BEiT `VisionTransformer`. The adapter instantiates that exact class with a 64×64 image, 8×8 patches, hidden width 64, two layers and four attention heads. Relative position bias, QKV bias, layer scale and mean token pooling remain active. The three-class head is new for the synthetic task.

The input is a declared synthetic waveform tensor `(batch, time, channels)`. A deterministic renderer places its four traces in fixed rows, with a fixed dimensionless amplitude range and fixed stroke width. It accepts only the observed waveform. It does not read labels or target masks. The resulting image is repeated into RGB channels for the original encoder interface.

This is a waveform-to-image classification exercise. The generated curves are not physiological ECGs, calibrated paper ECGs or disease evidence. The adapter does not reproduce HeartBEiT's discrete DALL-E tokenizer, masked pretraining, pretrained checkpoint, clinical fine-tuning cohorts or comparison results. Its small architecture does not inherit the capabilities reported by the [HeartBEiT paper](https://www.nature.com/articles/s41746-023-00840-9).

**File-level license evidence:** the author repository has no root license, but `modeling_finetune.py` explicitly states Microsoft's copyright and MIT license and points to the upstream [Microsoft BEiT implementation](https://github.com/microsoft/unilm/tree/master/beit). The copied file retains that notice. The complete inherited [Microsoft MIT license](https://github.com/microsoft/unilm/blob/master/LICENSE) is preserved beside it. This evidence applies to this individual file; it does not grant a general right to copy the repository's other scripts. Only the model file was copied. Global timm registration decorators were removed to avoid collisions with installed models; the architecture code is unchanged.

## 080 — LaBraM

The [authors' LaBraM implementation](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/modeling_finetune.py) supplies both `TemporalConv` and `NeuralTransformer`.

The adapter converts the four synthetic channels into a tensor with shape `(B, 4, 2, 200)`: four channels, two time patches, 200 samples per patch. Linear interpolation only provides the required tensor shape. It does not add information, establish an EEG sampling rate or assign microvolt units.

The original TemporalConv uses a stride-eight temporal convolution followed by two additional temporal convolutions and GroupNorm. With a 200-sample patch, there are 25 temporal positions. The teaching preset uses four temporal feature channels, giving an embedding width of `25 × 4 = 100`. The transformer retains separate channel-position and time-position embeddings, Q/K normalization, layer scale and mean token pooling. Depth is reduced to two; the number of heads is four.

The channel IDs are explicitly synthetic. They use positions 1 through C in the source model's embedding table, with index 0 reserved for the class token. They do not claim correspondence with Fp1, Cz or any other electrode. Real EEG use requires a declared electrode mapping, montage, channel order, reference scheme, units and sampling process. The montage test permutes both channel data and the corresponding channel IDs and verifies that the pooled representation stays consistent.

This lesson covers the downstream architecture. It does not include LaBraM's vector-quantized neural tokenizer, masked code-prediction pretraining, pretrained weights or cross-dataset EEG results. The code is copied under the [authors' MIT license](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/LICENSE). Only global timm registration decorators were removed. The CPU constructor supplies an explicit positive layer-scale initialization because the source block compares this value numerically and its `None` default is not runnable directly.

## 137 — ECGFounder

The lesson copies the actual [ECGFounder Net1D implementation](https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/net1d.py) unchanged. It uses grouped temporal bottlenecks, squeeze-excitation, residual downsampling and global time pooling. SAME padding supports odd sequence lengths, which is checked separately.

The original seven-stage depth schedule is retained: `2, 2, 2, 3, 3, 4, 4` blocks. Kernel size 16 and ratio one are retained. Channels are reduced to `8, 16, 16, 24, 24, 32, 32`, with group width four. BatchNorm and dropout are disabled as in the repository's supplied fine-tuning constructors. The output head has three synthetic classes.

The default selects the first observed synthetic channel, matching a one-channel network interface. The adapter can accept a configured number of observed channels, but it raises an error if the input has too few. It never repeats channels to fabricate a 12-lead recording.

This preserves the architecture's computational mechanisms. It does not reproduce ten-million-recording pretraining, the original diagnostic ontology, lead-dropout training, published weights or single-lead/12-lead clinical performance. The real source is the authors' `PKUDigitalHealth/ECGFounder` repository; the older atlas URL under `bdsp-core` is not used. The complete [MIT license](https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/LICENSE) is preserved.

## Verification

All three training paths were checked with actual PyTorch loss computation, backward passes, finite nonzero gradients, an AdamW update and evaluation under `torch.no_grad()`. Additional tests cover the renderer's fixed-scale input behavior, LaBraM channel/time patch layout and consistent montage permutation, ECGFounder SAME padding for odd lengths, explicit rejection of missing input channels, and exact upstream Git blob identifiers for every preserved model source and license.

The ten tests validate executable architecture paths and specific representation contracts. They do not establish optimization convergence, transfer learning, clinical performance or parity with the full published methods. `SPECS` and each vendor directory's `PROVENANCE.json` retain the implementation boundaries and exact source revisions so these distinctions stay attached to the code.
