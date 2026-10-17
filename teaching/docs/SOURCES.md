# Source register and reuse rules

This register covers **all 138 model cards** in the Biomedical AI Atlas. Its purpose is to identify the actual source, retain an immutable revision, and make reuse conditions visible before a learner copies or executes code.

**Checked: 2026-10-17.** The machine-readable authority is [sources.json](../sources.json). It contains 132 immutable upstream revisions, including historical code and Hugging Face releases. These are source-reference counts; they are not counts of fully reproduced models.

## What was checked

For the reachable GitHub sources, the audit read current repository metadata, the default-branch commit, and the root license at that immutable revision where available. For Hugging Face sources, it read the public model-card API, revision, declared license and access-gate terms. Targeted author pages and paper availability sections resolved missing or incorrect Atlas links.

The original paper URLs are retained from the Atlas for traceability. This source audit did not reread every full paper or verify every training script, dependency, dataset, checkpoint, or numerical result. An author/institution source label identifies the attributed upstream project; it does not assert that every file reproduces the linked paper. Per-model implementation notes and executed checks must establish that separately.

No upstream model weights or gated data were downloaded for this register. Public access to a repository is not, by itself, permission to redistribute its contents.

## Permission categories

| Value | Meaning |
| --- | --- |
| `permitted_with_notice` | A permissive source license was identified. Preserve all required copyright, full license and NOTICE material. Check any separately licensed files, data and weights. |
| `restricted` | Conditions go beyond retaining a notice: for example, noncommercial use, no derivatives, custom research terms, or copyleft obligations. Read the actual license before reuse. |
| `unknown` | Sufficient reuse permission was not established. Do not infer permission from a public URL or an absent license file. |

Copyleft licenses such as GPL, LGPL and AGPL are open-source licenses. Their `restricted` label here concerns this repository's simple permissive-vendoring workflow; it does not describe them as proprietary or noncommercial. The original teaching code's MIT license does not replace any upstream license.

## Corrections that affect teaching fidelity

- **ConVIRT:** the author repository supplies retrieval evaluation CSVs, not the model/training code. It is retained as a related source, while `source_kind` is `paper`.
- **3D U-Net:** Freiburg supplies the original 3D Caffe patch, architecture definitions and solver settings. Its archive's own license still needs inspection before copying. The author site also covers 2D U-Net.
- **Deep Autoencoder:** Hinton and Salakhutdinov supply original MATLAB code and a permissive notice. RBM pretraining plus fine-tuning is part of the 2006 method; a plain autoencoder omits it.
- **DnCNN:** the original README points to the same author's **KAIR** implementation. That repository has an MIT license and an explicit DnCNN implementation.
- **Florence-2:** the Atlas GitHub URL returned 404. Microsoft's public Hugging Face release is the verified source.
- **ECGFounder:** the working official source is **PKUDigitalHealth/ECGFounder**, not the Atlas's obsolete `bdsp-core` URL.
- **RealNVP:** the original TensorFlow Models directory was removed. The registry pins the last repository state immediately before removal, **5eb294f84bd3f415b548980e69fee63db1f6f1df**, where the original Dinh implementation and license remain available. It requires a legacy TensorFlow 1 environment.
- **nnU-Net and PVT:** current default branches contain newer method versions. A latest repository pin is not proof of fidelity to the earlier paper.
- **LSTM and GRU:** library cells may differ from the historical paper. PyTorch LSTM is the modern forget-gate version; PyTorch GRU places the reset operation differently from the original formulation.
- **UNI and CONCH:** the source repositories carry CC BY-NC-ND terms. A modified copy must not be presented as MIT teaching code.
- **PRISM2:** the public model card has manual access approval and additional terms against model redistribution. Each learner must obtain any required access from the author.
- **MAIRA-1, Decipher-MR and RED-Net:** a complete usable original release was not established by this audit. Their entries remain explicit paper routes. This is a bounded search result, not a claim that no implementation exists.
- **Decipher-MR:** its paper names constituent open-source frameworks and states that pretrained weights may not be directly shared. Those constituent frameworks are not a release of Decipher-MR.
- **NAF / U-Net++ / PFGM++:** stable Atlas IDs are preserved even where filenames omit punctuation. NAF is Neural Attenuation Fields, not NAFNet.

## Recheck the register

The validator has no third-party dependency and is offline by default:

```bash
python teaching/tools/audit_sources.py
```

Check one public upstream entry without modifying any files:

```bash
python teaching/tools/audit_sources.py --refresh --model 38
```

To write refreshed evidence, add `--write`. Existing immutable pins remain fixed. Add `--repin` only when you intentionally want to move records that use the default branch. Historical pins remain fixed even with `--repin`.

```bash
python teaching/tools/audit_sources.py --refresh --model ResNet --write
```

GitHub can limit unauthenticated API reads. If you have explicitly supplied a token in an environment variable, the optional `--github-token-env VARIABLE_NAME` flag uses it only for GitHub requests. The tool does not print tokens, fetch gated model files, accept model agreements, install code, or download checkpoints.

A refresh failure keeps the last successfully checked date and pin and reports the failed check. Source metadata and model implementation validation are separate: passing this validator does not establish architectural or numerical reproduction.

## Complete model-to-source mapping

Each immutable commit is in [sources.json](../sources.json); source and license links below point to the recorded upstream material. “Paper” rows can include an author dataset repository that is not executable model code.

| ID | Model | Source route | License | Reuse category |
| --- | --- | --- | --- | --- |
| 001 | AlexNet | [Canonical library](https://github.com/pytorch/vision/tree/main/torchvision/models/alexnet.py) | [BSD-3-Clause](https://github.com/pytorch/vision/blob/268ddfce8e7938226c4ec8ed23fc7b9c46b97316/LICENSE) | `permitted_with_notice` |
| 002 | ResNet | [Author / institution](https://github.com/KaimingHe/deep-residual-networks) | [MIT](https://github.com/KaimingHe/deep-residual-networks/blob/a7026cb6d478e131b765b898c312e25f9f6dc031/LICENSE) | `permitted_with_notice` |
| 003 | DenseNet | [Author / institution](https://github.com/liuzhuang13/DenseNet) | [BSD-3-Clause](https://github.com/liuzhuang13/DenseNet/blob/e3d24fb44997875f6eddc90dfd605e028d052bad/LICENSE) | `permitted_with_notice` |
| 004 | ConvNeXt | [Author / institution](https://github.com/facebookresearch/ConvNeXt) | [MIT](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/LICENSE) | `permitted_with_notice` |
| 005 | FCN | [Author / institution](https://github.com/shelhamer/fcn.berkeleyvision.org) | NOASSERTION | `unknown` |
| 006 | U-Net | [Author / institution](https://lmb.informatik.uni-freiburg.de/lmbsoft/unet.en.html) | NOASSERTION | `unknown` |
| 007 | 3D U-Net | [Author / institution](https://lmb.informatik.uni-freiburg.de/lmbsoft/unet.en.html) | NOASSERTION | `unknown` |
| 008 | V-Net | [Author / institution](https://github.com/faustomilletari/VNet) | [GPL-3.0](https://github.com/faustomilletari/VNet/blob/584c1502f369f2547e7ed384f9cc25beb09f65e1/LICENSE) | `restricted` |
| 009 | nnU-Net | [Author / institution](https://github.com/MIC-DKFZ/nnUNet) | [Apache-2.0](https://github.com/MIC-DKFZ/nnUNet/blob/bbf1222f24fee281470fbf71a4acb6a1c438fa7d/LICENSE) | `permitted_with_notice` |
| 010 | UNETR | [Author / institution](https://github.com/Project-MONAI/research-contributions) | [Apache-2.0](https://github.com/Project-MONAI/research-contributions/blob/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/LICENSE) | `permitted_with_notice` |
| 011 | Swin UNETR (SSL) | [Author / institution](https://github.com/Project-MONAI/research-contributions) | [Apache-2.0](https://github.com/Project-MONAI/research-contributions/blob/21ed8e57c7256834d4fbaf19579ca25ad3d135ee/LICENSE) | `permitted_with_notice` |
| 012 | MedNeXt | [Author / institution](https://github.com/MIC-DKFZ/MedNeXt) | [Apache-2.0](https://github.com/MIC-DKFZ/MedNeXt/blob/0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba/LICENSE) | `permitted_with_notice` |
| 013 | Models Genesis | [Author / institution](https://github.com/MrGiovanni/ModelsGenesis) | [LicenseRef-ASU-ModelsGenesis-NonCommercial](https://github.com/MrGiovanni/ModelsGenesis/blob/4c6c3bdee39e27c20622f1e9a2a8186833e97ceb/LICENSE) | `restricted` |
| 014 | VISTA3D | [Author / institution](https://github.com/Project-MONAI/VISTA) | [Apache-2.0](https://github.com/Project-MONAI/VISTA/blob/d4a8fe0dbf5cb4b76c531fccca8e29c1e6f6ee45/LICENSE.txt) | `permitted_with_notice` |
| 015 | S4 | [Author / institution](https://github.com/state-spaces/s4) | [Apache-2.0](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/LICENSE) | `permitted_with_notice` |
| 016 | Mamba | [Author / institution](https://github.com/state-spaces/mamba) | [Apache-2.0](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/LICENSE) | `permitted_with_notice` |
| 017 | VMamba | [Author / institution](https://github.com/MzeroMiko/VMamba) | [MIT](https://github.com/MzeroMiko/VMamba/blob/2ed52ead062a51a64521ed3871d52914bf532876/LICENSE) | `permitted_with_notice` |
| 018 | U-Mamba | [Author / institution](https://github.com/bowang-lab/U-Mamba) | [Apache-2.0](https://github.com/bowang-lab/U-Mamba/blob/28459e33ca03769800dd35e23c6e62491d1925b5/LICENSE) | `permitted_with_notice` |
| 019 | SegMamba | [Author / institution](https://github.com/ge-xing/SegMamba) | NOASSERTION | `unknown` |
| 020 | Transformer | [Author / institution](https://github.com/tensorflow/tensor2tensor) | [Apache-2.0](https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/LICENSE) | `permitted_with_notice` |
| 021 | Vision Transformer (ViT) | [Author / institution](https://github.com/google-research/vision_transformer) | [Apache-2.0](https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/LICENSE) | `permitted_with_notice` |
| 022 | Swin Transformer | [Author / institution](https://github.com/microsoft/Swin-Transformer) | [MIT](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/LICENSE) | `permitted_with_notice` |
| 023 | Mask2Former | [Author / institution](https://github.com/facebookresearch/Mask2Former) | [MIT](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/LICENSE) | `permitted_with_notice` |
| 024 | SimCLR | [Author / institution](https://github.com/google-research/simclr) | [Apache-2.0](https://github.com/google-research/simclr/blob/383d4143fd8cf7879ae10f1046a9baeb753ff438/LICENSE) | `permitted_with_notice` |
| 025 | MoCo | [Author / institution](https://github.com/facebookresearch/moco) | NOASSERTION | `unknown` |
| 026 | MAE | [Author / institution](https://github.com/facebookresearch/mae) | [CC-BY-NC-4.0](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/LICENSE) | `restricted` |
| 027 | DINO | [Author / institution](https://github.com/facebookresearch/dino) | [Apache-2.0](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/LICENSE) | `permitted_with_notice` |
| 028 | iBOT | [Author / institution](https://github.com/bytedance/ibot) | [Apache-2.0](https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/LICENSE) | `permitted_with_notice` |
| 029 | DINOv2 | [Author / institution](https://github.com/facebookresearch/dinov2) | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE) | `permitted_with_notice` |
| 030 | CLIP | [Author / institution](https://github.com/openai/CLIP) | [MIT](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE) | `permitted_with_notice` |
| 031 | BLIP-2 | [Author / institution](https://github.com/salesforce/LAVIS) | [BSD-3-Clause](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/LICENSE.txt) | `permitted_with_notice` |
| 032 | Segment Anything (SAM) | [Author / institution](https://github.com/facebookresearch/segment-anything) | [Apache-2.0](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/LICENSE) | `permitted_with_notice` |
| 033 | MedSAM | [Author / institution](https://github.com/bowang-lab/MedSAM) | [Apache-2.0](https://github.com/bowang-lab/MedSAM/blob/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/LICENSE) | `permitted_with_notice` |
| 034 | SAM 2 | [Author / institution](https://github.com/facebookresearch/sam2) | [Apache-2.0](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE) | `permitted_with_notice` |
| 035 | ConVIRT | [Paper](https://github.com/yuhaozhang/convirt) | NOASSERTION | `unknown` |
| 036 | GLoRIA | [Author / institution](https://github.com/marshuang80/gloria) | [Apache-2.0](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/LICENSE) | `permitted_with_notice` |
| 037 | BioViL | [Author / institution](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal) | [MIT](https://github.com/microsoft/hi-ml/blob/b67c1d27c6b17d8e8ff01f8c507f3cabdb307388/LICENSE) | `permitted_with_notice` |
| 038 | BiomedCLIP | [Author / institution](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/tree/9f341de24bfb00180f1b847274256e9b65a3a32e) | [MIT](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/9f341de24bfb00180f1b847274256e9b65a3a32e/LICENSE.md) | `permitted_with_notice` |
| 039 | UNI | [Author / institution](https://github.com/mahmoodlab/UNI) | [CC-BY-NC-ND-4.0](https://github.com/mahmoodlab/UNI/blob/42715efc11722a496e0a67f3369505a8f277206c/LICENSE) | `restricted` |
| 040 | CONCH | [Author / institution](https://github.com/mahmoodlab/CONCH) | [CC-BY-NC-ND-4.0](https://github.com/mahmoodlab/CONCH/blob/141cc09c7d4ff33d8eda562bd75169b457f71a62/LICENSE) | `restricted` |
| 041 | Prov-GigaPath | [Author / institution](https://github.com/prov-gigapath/prov-gigapath) | [Apache-2.0](https://github.com/prov-gigapath/prov-gigapath/blob/55431e04ff853a08f752a0ba42e6e9c48e60c776/LICENSE) | `permitted_with_notice` |
| 042 | Merlin | [Author / institution](https://github.com/StanfordMIMI/Merlin) | [MIT](https://github.com/StanfordMIMI/Merlin/blob/e1552df771367aa2589ec7dcfd3151dc91f8ce74/LICENSE) | `permitted_with_notice` |
| 043 | Deep Autoencoder | [Author / institution](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) | [LicenseRef-Hinton-Science-Code](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) | `permitted_with_notice` |
| 044 | GAN | [Author / institution](https://github.com/goodfeli/adversarial) | [BSD-3-Clause](https://github.com/goodfeli/adversarial/blob/1720041f51efcb92df61d1638f1a74aea2ae74c2/LICENSE) | `permitted_with_notice` |
| 045 | pix2pix | [Author / institution](https://github.com/phillipi/pix2pix) | [BSD-2-Clause AND BSD-3-Clause](https://github.com/phillipi/pix2pix/blob/89ff2a81ce441fbe1f1b13eca463b87f1e539df8/LICENSE) | `permitted_with_notice` |
| 046 | CycleGAN | [Author / institution](https://github.com/junyanz/CycleGAN) | [BSD-2-Clause AND BSD-3-Clause](https://github.com/junyanz/CycleGAN/blob/40b4498526de6b566f94a000d98fe18b9b8921db/LICENSE) | `permitted_with_notice` |
| 047 | FBPConvNet | [Author / institution](https://github.com/panakino/FBPConvNet) | NOASSERTION | `unknown` |
| 048 | Learned Primal-Dual | [Author / institution](https://github.com/adler-j/learned_primal_dual) | NOASSERTION | `unknown` |
| 049 | Variational Network | [Author / institution](https://github.com/VLOGroup/mri-variationalnetwork) | [MIT](https://github.com/VLOGroup/mri-variationalnetwork/blob/653630b9dcd0687387e6cfa494fa3d049abb83d8/LICENSE) | `permitted_with_notice` |
| 050 | MoDL | [Author / institution](https://github.com/hkaggarwal/modl) | NOASSERTION | `unknown` |
| 051 | VarNet | [Author / institution](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/varnet) | [MIT](https://github.com/facebookresearch/fastMRI/blob/91f2df4711adbb6d643df1810f234e4abcf5881b/LICENSE.md) | `permitted_with_notice` |
| 052 | DnCNN | [Author / institution](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/models/network_dncnn.py) | [MIT](https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/LICENSE) | `permitted_with_notice` |
| 053 | Noise2Noise | [Author / institution](https://github.com/NVlabs/noise2noise) | [CC-BY-NC-4.0](https://github.com/NVlabs/noise2noise/blob/7355519e7bfc49e0606cca8867748e736431244b/LICENSE.txt) | `restricted` |
| 054 | Noise2Self | [Author / institution](https://github.com/czbiohub-sf/noise2self) | [MIT](https://github.com/czbiohub-sf/noise2self/blob/b306ca7ee1a8e1d4c4ad8f3a081b6e81c2c8f37d/LICENSE) | `permitted_with_notice` |
| 055 | DDPM | [Author / institution](https://github.com/hojonathanho/diffusion) | NOASSERTION | `unknown` |
| 056 | DDIM | [Author / institution](https://github.com/ermongroup/ddim) | [MIT](https://github.com/ermongroup/ddim/blob/e8df66cf02e457719930748b5c26e37e3126dfdd/LICENSE) | `permitted_with_notice` |
| 057 | Score-SDE | [Author / institution](https://github.com/yang-song/score_sde) | [Apache-2.0](https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/LICENSE) | `permitted_with_notice` |
| 058 | Latent Diffusion | [Author / institution](https://github.com/CompVis/latent-diffusion) | [MIT](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/LICENSE) | `permitted_with_notice` |
| 059 | DDRM | [Author / institution](https://github.com/bahjat-kawar/ddrm) | [MIT](https://github.com/bahjat-kawar/ddrm/blob/32b6b3ccfda532ba01c9cc5b6e7456c3a06a6ca2/LICENSE) | `permitted_with_notice` |
| 060 | DPS | [Author / institution](https://github.com/DPS2022/diffusion-posterior-sampling) | NOASSERTION | `unknown` |
| 061 | DiffusionMBIR | [Author / institution](https://github.com/hyungjin-chung/DiffusionMBIR) | [Apache-2.0](https://github.com/hyungjin-chung/DiffusionMBIR/blob/bdfe460582baf226689c6d15f02a08fbc7c969e2/LICENSE) | `permitted_with_notice` |
| 062 | Flow Matching | [Canonical library](https://github.com/facebookresearch/flow_matching) | [CC-BY-NC-4.0](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/LICENSE) | `restricted` |
| 063 | Rectified Flow | [Author / institution](https://github.com/gnobitab/RectifiedFlow) | NOASSERTION | `unknown` |
| 064 | Consistency Models | [Author / institution](https://github.com/openai/consistency_models) | [MIT](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/LICENSE) | `permitted_with_notice` |
| 065 | DiT | [Author / institution](https://github.com/facebookresearch/DiT) | [CC-BY-NC-4.0](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/LICENSE.txt) | `restricted` |
| 066 | Spatial Transformer Network | [Canonical library](https://github.com/pytorch/tutorials/blob/main/intermediate_source/spatial_transformer_tutorial.py) | [BSD-3-Clause](https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/LICENSE) | `permitted_with_notice` |
| 067 | VoxelMorph | [Author / institution](https://github.com/voxelmorph/voxelmorph) | [Apache-2.0](https://github.com/voxelmorph/voxelmorph/blob/c4155e1baf04bc8f1774775b23a6027b58b37e00/LICENSE.md) | `permitted_with_notice` |
| 068 | GradICON | [Author / institution](https://github.com/uncbiag/ICON) | [Apache-2.0](https://github.com/uncbiag/ICON/blob/5987ff367d850cca7965c5b52e36524454c070fd/LICENSE) | `permitted_with_notice` |
| 069 | uniGradICON | [Author / institution](https://github.com/uncbiag/uniGradICON) | [Apache-2.0](https://github.com/uncbiag/uniGradICON/blob/7678cebe3967ce506bfd2c92489b324c349cb1cc/LICENSE) | `permitted_with_notice` |
| 070 | DiffPose | [Author / institution](https://github.com/eigenvivek/DiffPose) | [MIT](https://github.com/eigenvivek/DiffPose/blob/afc0159b3c1db4ea1e1bd55cd4fc85b0408c3974/LICENSE) | `permitted_with_notice` |
| 071 | X2CT-GAN | [Author / institution](https://github.com/kylekma/X2CT) | [LGPL-3.0](https://github.com/kylekma/X2CT/blob/a9f0473ee2b8acfc2abcc9acd7d7cf50435435e3/LICENSE) | `restricted` |
| 072 | NeRF | [Author / institution](https://github.com/bmild/nerf) | [MIT](https://github.com/bmild/nerf/blob/14c55567a6d0fbd75d3fd12b0411f98160ba3237/LICENSE) | `permitted_with_notice` |
| 073 | NAF | [Author / institution](https://github.com/ruyi-zha/naf_cbct) | [MIT](https://github.com/ruyi-zha/naf_cbct/blob/cff7da31a6f80e788ce0831b0fae6b4b4dae4cec/LICENSE) | `permitted_with_notice` |
| 074 | 3D Gaussian Splatting | [Author / institution](https://github.com/graphdeco-inria/gaussian-splatting) | [LicenseRef-Gaussian-Splatting-Research](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md) | `restricted` |
| 075 | GCN | [Author / institution](https://github.com/tkipf/gcn) | [MIT](https://github.com/tkipf/gcn/blob/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81/LICENCE) | `permitted_with_notice` |
| 076 | GAT | [Author / institution](https://github.com/PetarV-/GAT) | [MIT](https://github.com/PetarV-/GAT/blob/5af87e7fce2b90ae1cbd621cd58059036a3c7436/LICENSE) | `permitted_with_notice` |
| 077 | LSTM | [Canonical library](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py) | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/4c36a17f2d31b8c882ea38078a4ce9c5fe5e8856/LICENSE) | `permitted_with_notice` |
| 078 | Temporal Convolutional Network (TCN) | [Author / institution](https://github.com/locuslab/TCN) | [MIT](https://github.com/locuslab/TCN/blob/2f8c2b817050206397458dfd1f5a25ce8a32fe65/LICENSE) | `permitted_with_notice` |
| 079 | HeartBEiT | [Author / institution](https://github.com/akhilvaid/HeartBEiT) | NOASSERTION | `unknown` |
| 080 | LaBraM | [Author / institution](https://github.com/935963004/LaBraM) | [MIT](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/LICENSE) | `permitted_with_notice` |
| 081 | VGG-16/19 | [Canonical library](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py) | [BSD-3-Clause](https://github.com/pytorch/vision/blob/268ddfce8e7938226c4ec8ed23fc7b9c46b97316/LICENSE) | `permitted_with_notice` |
| 082 | GoogLeNet / Inception-v1 | [Author / institution](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) | [BSD-2-Clause](https://github.com/BVLC/caffe/blob/9b891540183ddc834a02b2bd81b31afae71b2153/LICENSE) | `permitted_with_notice` |
| 083 | ResNeXt | [Author / institution](https://github.com/facebookresearch/ResNeXt) | [BSD-3-Clause](https://github.com/facebookresearch/ResNeXt/blob/1f59a2c744edba655bb703e8e060d6f9ce3b289c/LICENSE) | `permitted_with_notice` |
| 084 | SENet | [Author / institution](https://github.com/hujie-frank/SENet) | [Apache-2.0](https://github.com/hujie-frank/SENet/blob/0262d43d44c561fd53c3dba210cc8bacfc60500d/LICENSE) | `permitted_with_notice` |
| 085 | EfficientNet | [Author / institution](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) | [Apache-2.0](https://github.com/tensorflow/tpu/blob/7dc921f846c9e4071f3884b768a7ddf521272a49/LICENSE) | `permitted_with_notice` |
| 086 | U-Net++ | [Author / institution](https://github.com/MrGiovanni/UNetPlusPlus) | [LicenseRef-ASU-UNetPlusPlus-NonCommercial](https://github.com/MrGiovanni/UNetPlusPlus/blob/e9153d4243e7bb67880c380ce803b176c81476f1/LICENSE) | `restricted` |
| 087 | Attention U-Net | [Author / institution](https://github.com/ozan-oktay/Attention-Gated-Networks) | [MIT](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/LICENSE) | `permitted_with_notice` |
| 088 | H-DenseUNet | [Author / institution](https://github.com/xmengli/H-DenseUNet) | NOASSERTION | `unknown` |
| 089 | DeiT | [Author / institution](https://github.com/facebookresearch/deit) | [Apache-2.0](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/LICENSE) | `permitted_with_notice` |
| 090 | PVT | [Author / institution](https://github.com/whai362/PVT) | [Apache-2.0](https://github.com/whai362/PVT/blob/57e2dfaa5a46f9050d76f306a4fcd9a7c061f520/LICENSE) | `permitted_with_notice` |
| 091 | SegFormer | [Author / institution](https://github.com/NVlabs/SegFormer) | [LicenseRef-NVIDIA-SegFormer-NonCommercial](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/LICENSE) | `restricted` |
| 092 | BEiT | [Author / institution](https://github.com/microsoft/unilm/tree/master/beit) | [MIT](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/LICENSE) | `permitted_with_notice` |
| 093 | BYOL | [Author / institution](https://github.com/google-deepmind/deepmind-research/tree/master/byol) | [Apache-2.0](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/LICENSE) | `permitted_with_notice` |
| 094 | SwAV | [Author / institution](https://github.com/facebookresearch/swav) | [CC-BY-NC-4.0](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE) | `restricted` |
| 095 | ALIGN | [Canonical library](https://github.com/huggingface/transformers/tree/main/src/transformers/models/align) | [Apache-2.0](https://github.com/huggingface/transformers/blob/8498ed425fa01bbee7d086efdf1a6634c2962aed/LICENSE) | `permitted_with_notice` |
| 096 | BLIP | [Author / institution](https://github.com/salesforce/BLIP) | [BSD-3-Clause](https://github.com/salesforce/BLIP/blob/056a169437371659074aa2732649d5de3bffb4a8/LICENSE.txt) | `permitted_with_notice` |
| 097 | Florence-2 | [Author / institution](https://huggingface.co/microsoft/Florence-2-base/tree/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac) | [MIT](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/LICENSE) | `permitted_with_notice` |
| 098 | MedCLIP | [Author / institution](https://github.com/RyanWangZf/MedCLIP) | NOASSERTION | `unknown` |
| 099 | Med-Flamingo | [Author / institution](https://github.com/snap-stanford/med-flamingo) | NOASSERTION | `unknown` |
| 100 | LLaVA-Med | [Author / institution](https://github.com/microsoft/LLaVA-Med) | [LicenseRef-Microsoft-Research](https://github.com/microsoft/LLaVA-Med/blob/30697ca50b5c29a8e955c99330b259776aef27b9/LICENSE) | `restricted` |
| 101 | RadFM | [Author / institution](https://github.com/chaoyi-wu/RadFM) | [MIT](https://github.com/chaoyi-wu/RadFM/blob/8d798c554ca0f40cd4a409f0335cc8512cc9d7cf/LICENSE) | `permitted_with_notice` |
| 102 | MAIRA-1 | [Paper](https://arxiv.org/abs/2311.13668) | NOASSERTION | `unknown` |
| 103 | MAIRA-2 | [Author / institution](https://huggingface.co/microsoft/maira-2/tree/795a2b1cd4a310624b4e3d14b5a23e41fd273deb) | [LicenseRef-Microsoft-Research](https://huggingface.co/microsoft/maira-2/blob/795a2b1cd4a310624b4e3d14b5a23e41fd273deb/LICENSE) | `restricted` |
| 104 | BiomedGPT | [Author / institution](https://github.com/taokz/BiomedGPT) | [Apache-2.0](https://github.com/taokz/BiomedGPT/blob/2c739bd0aa3e010da1a417d9e8591fafcfd1e4b5/LICENSE) | `permitted_with_notice` |
| 105 | Virchow | [Author / institution](https://huggingface.co/paige-ai/Virchow/tree/19eebc84ae33e79f1b2d866e6ff90ae50e522f9a) | [Apache-2.0](https://huggingface.co/paige-ai/Virchow/blob/19eebc84ae33e79f1b2d866e6ff90ae50e522f9a/README.md) | `permitted_with_notice` |
| 106 | CHIEF | [Author / institution](https://github.com/hms-dbmi/CHIEF) | [AGPL-3.0](https://github.com/hms-dbmi/CHIEF/blob/1abed334e674ad2b6ee0430d1ce7d0b566420bb4/LICENSE) | `restricted` |
| 107 | TITAN | [Author / institution](https://github.com/mahmoodlab/TITAN) | NOASSERTION | `unknown` |
| 108 | PRISM2 | [Author / institution](https://huggingface.co/paige-ai/Prism2/tree/450352d0ddc6b42b21ce20794ce0fbefe6b5a47a) | [CC-BY-NC-ND-4.0](https://huggingface.co/paige-ai/Prism2/blob/450352d0ddc6b42b21ce20794ce0fbefe6b5a47a/README.md) | `restricted` |
| 109 | BME-X | [Author / institution](https://github.com/DBC-Lab/Brain_MRI_Enhancement) | [MIT](https://github.com/DBC-Lab/Brain_MRI_Enhancement/blob/5a9c3e79667a940e21153b6e31fb917a99e7cfb9/LICENSE) | `permitted_with_notice` |
| 110 | Prima | [Author / institution](https://github.com/MLNeurosurg/Prima) | [MIT](https://github.com/MLNeurosurg/Prima/blob/604f1fc4dd7059c278d4d34bbbcaab3e404fbcd4/LICENSE) | `permitted_with_notice` |
| 111 | MARS | [Author / institution](https://github.com/zqiuak/MARS) | [Apache-2.0](https://github.com/zqiuak/MARS/blob/7075cf97ba7eb5715cbbf522701e5e2e46f5f3ac/LICENSE) | `permitted_with_notice` |
| 112 | Decipher-MR | [Paper](https://doi.org/10.1038/s41746-026-02596-4) | NOASSERTION | `unknown` |
| 113 | Noise2Void | [Author / institution](https://github.com/juglab/n2v) | [BSD-3-Clause](https://github.com/juglab/n2v/blob/c61b3b47c5cacc533f7e0e3db7cefab139d7b62a/LICENSE.txt) | `permitted_with_notice` |
| 114 | RED-Net | [Paper](https://papers.nips.cc/paper_files/paper/2016/hash/0ed9422357395a0d4879191c66f4faa2-Abstract.html) | NOASSERTION | `unknown` |
| 115 | SwinIR | [Author / institution](https://github.com/JingyunLiang/SwinIR) | [Apache-2.0](https://github.com/JingyunLiang/SwinIR/blob/6545850fbf8df298df73d81f3e8cba638787c8bd/LICENSE) | `permitted_with_notice` |
| 116 | Restormer | [Author / institution](https://github.com/swz30/Restormer) | [MIT](https://github.com/swz30/Restormer/blob/68dc6ac472db26f16361150cb7a96a1bc87da93f/LICENSE.md) | `permitted_with_notice` |
| 117 | AUTOMAP | [Author / institution](https://github.com/MattRosenLab/AUTOMAP) | [GPL-3.0](https://github.com/MattRosenLab/AUTOMAP/blob/172e0b081c0510c0ef3860ff295cb0f8c7355a0d/LICENSE) | `restricted` |
| 118 | LEARN | [Author / institution](https://github.com/maybe198376/LEARN) | [Apache-2.0](https://github.com/maybe198376/LEARN/blob/2d3f06b8a13112468ba8e345d4c829e313d907d7/LICENSE) | `permitted_with_notice` |
| 119 | DDNM | [Author / institution](https://github.com/wyhuai/DDNM) | [MIT](https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/LICENSE.txt) | `permitted_with_notice` |
| 120 | RealNVP | [Author / institution](https://github.com/tensorflow/models/tree/5eb294f84bd3f415b548980e69fee63db1f6f1df/research/real_nvp) | [Apache-2.0](https://github.com/tensorflow/models/blob/5eb294f84bd3f415b548980e69fee63db1f6f1df/LICENSE) | `permitted_with_notice` |
| 121 | Glow | [Author / institution](https://github.com/openai/glow) | [MIT](https://github.com/openai/glow/blob/91b2c577a5c110b2b38761fc56d81f7d87f077c1/LICENSE) | `permitted_with_notice` |
| 122 | PFGM++ | [Author / institution](https://github.com/Newbeeer/pfgmpp) | [CC-BY-NC-SA-4.0](https://github.com/Newbeeer/pfgmpp/blob/d57c1ee4488d8e4064a5b9c8548792f00395aa8b/LICENSE.txt) | `restricted` |
| 123 | SiT | [Author / institution](https://github.com/willisma/SiT) | [MIT](https://github.com/willisma/SiT/blob/cbde832a40b153ccc79603412409da9c9b0c568c/LICENSE.txt) | `permitted_with_notice` |
| 124 | LapIRN | [Author / institution](https://github.com/cwmok/LapIRN) | [MIT](https://github.com/cwmok/LapIRN/blob/c17fb7564ca752d37e97baf12cafff44e01da668/LICENSE.md) | `permitted_with_notice` |
| 125 | TransMorph | [Author / institution](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) | [MIT](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/LICENSE) | `permitted_with_notice` |
| 126 | IDIR | [Author / institution](https://github.com/MIAGroupUT/IDIR) | [MIT](https://github.com/MIAGroupUT/IDIR/blob/9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3/LICENSE) | `permitted_with_notice` |
| 127 | DeepDRR | [Author / institution](https://github.com/arcadelab/deepdrr) | [GPL-3.0](https://github.com/arcadelab/deepdrr/blob/cb2cb3ad1a1f68c09f79a11f4ce583bff9ff67bb/LICENSE) | `restricted` |
| 128 | pixelNeRF | [Author / institution](https://github.com/sxyu/pixel-nerf) | [BSD-2-Clause](https://github.com/sxyu/pixel-nerf/blob/91a044bdd62aebe0ed3a5685ca37cb8a9dc8e8ee/LICENSE) | `permitted_with_notice` |
| 129 | Instant-NGP | [Author / institution](https://github.com/NVlabs/instant-ngp) | [LicenseRef-NVIDIA-InstantNGP-NonCommercial](https://github.com/NVlabs/instant-ngp/blob/abe236ee00cf90cfca6e36e65c00435d5b21f50a/LICENSE.txt) | `restricted` |
| 130 | MedNeRF | [Author / institution](https://github.com/abrilcf/mednerf) | [MIT](https://github.com/abrilcf/mednerf/blob/b34c0228def12f4a506dfdb17234ac7e18744c6c/LICENSE) | `permitted_with_notice` |
| 131 | EndoNeRF | [Author / institution](https://github.com/med-air/EndoNeRF) | NOASSERTION | `unknown` |
| 132 | GraphSAGE | [Author / institution](https://github.com/williamleif/GraphSAGE) | [MIT](https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/LICENSE.txt) | `permitted_with_notice` |
| 133 | Graphormer | [Author / institution](https://github.com/microsoft/Graphormer) | [MIT](https://github.com/microsoft/Graphormer/blob/59c0decffcade9df81d29dcc178a489b31958bab/LICENSE) | `permitted_with_notice` |
| 134 | Voxel2Mesh | [Author / institution](https://github.com/cvlab-epfl/voxel2mesh) | [GPL-3.0](https://github.com/cvlab-epfl/voxel2mesh/blob/ed8805be68151fbbe6c3bcc6827373c690290f68/LICENSE) | `restricted` |
| 135 | Vox2Cortex | [Author / institution](https://github.com/ai-med/Vox2Cortex) | [GPL-3.0](https://github.com/ai-med/Vox2Cortex/blob/ef10ca2cf9999e1698a98a4bf9dce19b0000bd10/LICENSE) | `restricted` |
| 136 | GRU | [Canonical library](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py) | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/4c36a17f2d31b8c882ea38078a4ce9c5fe5e8856/LICENSE) | `permitted_with_notice` |
| 137 | ECGFounder | [Author / institution](https://github.com/PKUDigitalHealth/ECGFounder) | [MIT](https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/LICENSE) | `permitted_with_notice` |

| 138 | PhyDiCT | [Author source, not executed](https://github.com/batmanlab/PhyDiCT/tree/a5366fd19cc4a44e52445ce6385259f00cf770bb) | [Apache-2.0](https://github.com/batmanlab/PhyDiCT/blob/a5366fd19cc4a44e52445ce6385259f00cf770bb/LICENSE.txt) | `permitted_with_notice` |

## Validation performed

The offline validator matched all 137 IDs, numbers and titles against the actual Atlas catalog and checked mandatory fields, immutable SHA format and permission evidence. The optional network refresh path was exercised against the public BiomedCLIP metadata endpoint without writing changes or downloading weights. GitHub sources were audited through the connected repository interface because the execution environment's unauthenticated public API was rate-limited.


The 2026-10-17 maintenance admitted #138 PhyDiCT after reading its official conference page, pinned source metadata, selected inference files and full root license. Its original model was not run and its full PDF was unavailable. The offline identity/database checks now cover 138 entries. Earlier 137-entry source checks and 102 CPU execution records remain historical evidence; they were not repeated by this update. See [maintenance records](../maintenance/history.json).
