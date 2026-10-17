# Third-party notices / 第三方来源声明

Original teaching code is MIT; synthetic data is CC0. Third-party code and libraries retain their own licenses. A source catalog pin can differ from the library actually executed. No pretrained weights are bundled.

## Vendored source files

Each directory preserves its license text and a PROVENANCE.json with source URLs, revisions and modifications. Noncommercial and other restrictions continue to apply.

### extended_ecgfounder

[Provenance](vendor/extended_ecgfounder/PROVENANCE.json) · [LICENSE](vendor/extended_ecgfounder/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "license": "MIT",
  "commit": "04edac702b61c91face519774ddcc0cd712fef23",
  "scope": "Model architecture file only; pretrained resources and end-to-end training pipelines not included.",
  "sources": [
    {
      "source_url": "https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/net1d.py",
      "git_blob_sha": "1e48d2800698c89fb1e404debd80c5a7e51545f4",
      "original_path": "net1d.py"
    },
    {
      "source_url": "https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/LICENSE",
      "git_blob_sha": "689c66875909c6eb6eea1cc8fe82ee8d49d4a8c4",
      "original_path": "LICENSE"
    }
  ],
  "modifications": [],
  "notes": []
}
```

### extended_heartbeit

[Provenance](vendor/extended_heartbeit/PROVENANCE.json) · [LICENSE](vendor/extended_heartbeit/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "license": "MIT",
  "commit": "e065cca3c41d6bc5e852a44c6f7fc1fb51c4d45e",
  "scope": "Model architecture file only; pretrained resources and end-to-end training pipelines not included.",
  "sources": [
    {
      "source_url": "https://github.com/akhilvaid/HeartBEiT/blob/e065cca3c41d6bc5e852a44c6f7fc1fb51c4d45e/modeling_finetune.py",
      "git_blob_sha": "b3f672c9721b814fd696d1de7f94483165d659e2",
      "original_path": "modeling_finetune.py"
    },
    {
      "source_url": "https://github.com/microsoft/unilm/blob/master/LICENSE",
      "git_blob_sha": "ae241a567f3de21656c00c7b341a6f8701e405b3",
      "original_path": "LICENSE"
    }
  ],
  "modifications": [
    "Global timm registration decorators removed to prevent conflicts. Architecture constructors, forward functions and initialization unchanged."
  ],
  "notes": [
    "The author repository has no root license. This individual model file explicitly carries Microsoft's MIT license header and links microsoft/unilm/beit. Its inherited MIT license text is preserved here from microsoft/unilm/LICENSE. This permission is not generalized to other HeartBEiT repository files."
  ]
}
```

### extended_labram

[Provenance](vendor/extended_labram/PROVENANCE.json) · [LICENSE](vendor/extended_labram/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "license": "MIT",
  "commit": "c431221e6cfd23dbfa9950e0180682fb322b0548",
  "scope": "Model architecture file only; pretrained resources and end-to-end training pipelines not included.",
  "sources": [
    {
      "source_url": "https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/modeling_finetune.py",
      "git_blob_sha": "611bf7a8a36a25f7035cc45fcff0ca47fe971f79",
      "original_path": "modeling_finetune.py"
    },
    {
      "source_url": "https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/LICENSE",
      "git_blob_sha": "2208a148ac666cdf09a18dd6e355df8ed051f672",
      "original_path": "LICENSE"
    }
  ],
  "modifications": [
    "Global timm registration decorators removed to prevent conflicts. Architecture constructors, forward functions and initialization unchanged."
  ],
  "notes": []
}
```

### foundation_clip

[Provenance](vendor/foundation_clip/PROVENANCE.json) · [LICENSE](vendor/foundation_clip/LICENSE)

```json
{
  "upstream_repo": "openai/CLIP",
  "commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
  "checked_on": "2026-10-17",
  "modifications": "No byte changes. clip/clip.py stored as clip_reference.py for reference only; it is not imported. The teaching adapter imports original model.py directly.",
  "files": [
    {
      "path": "model.py",
      "upstream_path": "clip/model.py",
      "url": "https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py",
      "sha256": "9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c"
    },
    {
      "path": "LICENSE",
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE",
      "sha256": "987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db"
    },
    {
      "path": "clip_reference.py",
      "upstream_path": "clip/clip.py",
      "url": "https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py",
      "sha256": "9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6"
    }
  ]
}
```

### foundation_medsam

[Provenance](vendor/foundation_medsam/PROVENANCE.json) · [LICENSE](vendor/foundation_medsam/LICENSE)

```json
{
  "upstream_repo": "bowang-lab/MedSAM",
  "commit": "d71e8a1a99ad751840a22a7fa3ecfb4166fb1488",
  "checked_on": "2026-10-17",
  "modifications": "None. train_one_gpu.py retained for reference only; it is not imported or executed. CPU wrapper is outside this directory.",
  "files": [
    {
      "path": "__init__.py",
      "upstream_path": "segment_anything/modeling/__init__.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/__init__.py",
      "sha256": "7bd468bf97fea2758bcdf18977dc8e60d6f90d1d02f5413a2870740005ecfbe8"
    },
    {
      "path": "common.py",
      "upstream_path": "segment_anything/modeling/common.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/common.py",
      "sha256": "77581b8217e25c9ace51f7356fccd1a07ed2962d0f74aae2491f1001750565f1"
    },
    {
      "path": "image_encoder.py",
      "upstream_path": "segment_anything/modeling/image_encoder.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/image_encoder.py",
      "sha256": "afd56922b0c2903f0ddcc75ea2819b42c534c6a830082794f8b93028a2d0c19c"
    },
    {
      "path": "mask_decoder.py",
      "upstream_path": "segment_anything/modeling/mask_decoder.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/mask_decoder.py",
      "sha256": "270dea7cb1103e7ad04932b7c45e5b0d5ba0db0a8a13734a55bbce0e002fadf7"
    },
    {
      "path": "prompt_encoder.py",
      "upstream_path": "segment_anything/modeling/prompt_encoder.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/prompt_encoder.py",
      "sha256": "6961f2b6b336501ecaed497b5642e2626d60a81e6b861455822678df256f3664"
    },
    {
      "path": "sam.py",
      "upstream_path": "segment_anything/modeling/sam.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/sam.py",
      "sha256": "4a5eb2d47f10e2762c29eed6212c190a54709edc3ce01c1f136bce6d901473a6"
    },
    {
      "path": "transformer.py",
      "upstream_path": "segment_anything/modeling/transformer.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/segment_anything/modeling/transformer.py",
      "sha256": "9bd360481a8f693e0b4054a229b5a5bb60efc38bb92e004de138009eba98cd94"
    },
    {
      "path": "LICENSE",
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/LICENSE",
      "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    },
    {
      "path": "train_one_gpu.py",
      "upstream_path": "train_one_gpu.py",
      "url": "https://raw.githubusercontent.com/bowang-lab/MedSAM/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/train_one_gpu.py",
      "sha256": "c63c3a47ab60887bd6da1279b2fd8e61b4fe25b047e4709dc8f881eda5799b7d"
    }
  ]
}
```

### foundation_sam

[Provenance](vendor/foundation_sam/PROVENANCE.json) · [LICENSE](vendor/foundation_sam/LICENSE)

```json
{
  "upstream_repo": "facebookresearch/segment-anything",
  "commit": "dca509fe793f601edb92606367a655c15ac00fdf",
  "checked_on": "2026-10-17",
  "modifications": "None to upstream files. Import adapter and teaching loss are outside this directory.",
  "files": [
    {
      "path": "__init__.py",
      "upstream_path": "segment_anything/modeling/__init__.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/__init__.py",
      "sha256": "47f6363b86c0bfaa7a6abf723975136d17fae4a5f002d2814a5cce8834d18980"
    },
    {
      "path": "common.py",
      "upstream_path": "segment_anything/modeling/common.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/common.py",
      "sha256": "59a53bfa5d0a4d446df7f2bb57d2e95808b130601592b934600fb663d256ce17"
    },
    {
      "path": "image_encoder.py",
      "upstream_path": "segment_anything/modeling/image_encoder.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/image_encoder.py",
      "sha256": "1665ada2902722c9a8cd566c840e324d623ef2b12f7d9ae9f38b24aed0e39886"
    },
    {
      "path": "mask_decoder.py",
      "upstream_path": "segment_anything/modeling/mask_decoder.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/mask_decoder.py",
      "sha256": "6a2455e990515c67a061187c92c236b879b189f21ef6ebfce1627cae9a6b2f93"
    },
    {
      "path": "prompt_encoder.py",
      "upstream_path": "segment_anything/modeling/prompt_encoder.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/prompt_encoder.py",
      "sha256": "bd8f2f5903f0647f55c21c608bcbb334eec2901ebe76aa41061b1c513df4a06f"
    },
    {
      "path": "sam.py",
      "upstream_path": "segment_anything/modeling/sam.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py",
      "sha256": "2d501173b7345f6829f745a2ade240c98026f6b3237bf6cf8e9473405e86c54e"
    },
    {
      "path": "transformer.py",
      "upstream_path": "segment_anything/modeling/transformer.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/transformer.py",
      "sha256": "566a8f6db8b608398de939548bd0f485b21752fda2bdda36900e6444ef938d11"
    },
    {
      "path": "LICENSE",
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/LICENSE",
      "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    }
  ]
}
```

### foundation_transformers

[Provenance](vendor/foundation_transformers/PROVENANCE.json) · [LICENSE](vendor/foundation_transformers/LICENSE)

```json
{
  "source": "https://github.com/huggingface/transformers",
  "source_kind": "canonical_library",
  "commit": "052e652d6d53c2b26ffde87e039b723949a53493",
  "package": "transformers",
  "version": "4.46.3",
  "license_spdx": "Apache-2.0",
  "evidence_checked": "2026-10-17",
  "files": [],
  "installed_files": [
    {
      "module": "transformers.models.align.modeling_align",
      "installed_relative_path": "transformers/models/align/modeling_align.py",
      "upstream_path": "src/transformers/models/align/modeling_align.py",
      "url": "https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/align/modeling_align.py",
      "sha256": "9c1b8a33cf54c57def7c1b06932e72d8e9bf4e2906209f3284a8945be01a582e"
    },
    {
      "module": "transformers.models.align.configuration_align",
      "installed_relative_path": "transformers/models/align/configuration_align.py",
      "upstream_path": "src/transformers/models/align/configuration_align.py",
      "url": "https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/align/configuration_align.py",
      "sha256": "43a22f7b2d148e5de671076677abb10a755cd0c052289d2a15933b756c93e542"
    },
    {
      "module": "transformers.models.blip_2.modeling_blip_2",
      "installed_relative_path": "transformers/models/blip_2/modeling_blip_2.py",
      "upstream_path": "src/transformers/models/blip_2/modeling_blip_2.py",
      "url": "https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/blip_2/modeling_blip_2.py",
      "sha256": "015ab850e14977f4e0241849068a836d90fbc2911988366028d8842f986abdde"
    },
    {
      "module": "transformers.models.blip_2.configuration_blip_2",
      "installed_relative_path": "transformers/models/blip_2/configuration_blip_2.py",
      "upstream_path": "src/transformers/models/blip_2/configuration_blip_2.py",
      "url": "https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/blip_2/configuration_blip_2.py",
      "sha256": "0ae5e81604cb8c48c981ab93f0d2bf447ccdcd7931d7b9941a2cc6d66e4b2899"
    },
    {
      "module": "transformers.models.opt.modeling_opt",
      "installed_relative_path": "transformers/models/opt/modeling_opt.py",
      "upstream_path": "src/transformers/models/opt/modeling_opt.py",
      "url": "https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/opt/modeling_opt.py",
      "sha256": "cb1a8a514e9723fd9b376ec8308abe1c6da8524f2e94199bd611df3abeb42cfd"
    },
    {
      "module": "transformers.models.opt.configuration_opt",
      "installed_relative_path": "transformers/models/opt/configuration_opt.py",
      "upstream_path": "src/transformers/models/opt/configuration_opt.py",
      "url": "https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/models/opt/configuration_opt.py",
      "sha256": "78fc0912437d8cdcb8e50de214c891f53333353357fb73d8a2b4614db7f528a9"
    }
  ],
  "notes": [
    "Architecture is imported from the pinned installed package, not vendored. Installed file hashes are independently checked by tests.",
    "The local LICENSE copy retains the upstream grant. All class implementations remain unmodified."
  ],
  "upstream_bytes_verified": true
}
```

### image_mamba

[Provenance](vendor/image_mamba/PROVENANCE.json) · [LICENSE](vendor/image_mamba/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "sources": [
    {
      "key": "mamba_simple",
      "git_blob_sha": "4c8a388217c4cd4d21d5ea4704a5d571be294781",
      "commit": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
      "original_path": "mamba_ssm/modules/mamba_simple.py",
      "source_url": "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py",
      "license": "Apache-2.0"
    },
    {
      "key": "mamba_scan",
      "git_blob_sha": "271b7b15b3e0046cf2dbc93c0df28198c99bd528",
      "commit": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
      "original_path": "mamba_ssm/ops/selective_scan_interface.py",
      "source_url": "https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py",
      "license": "Apache-2.0"
    },
    {
      "key": "mamba_license",
      "git_blob_sha": "f4abe24eb520fbb077753ae4f34bfaa43cb3b83f",
      "commit": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
      "original_path": "LICENSE",
      "source_url": "https://github.com/state-spaces/mamba/blob/main/LICENSE",
      "license": "Apache-2.0"
    }
  ],
  "modifications": [
    "Backend imports replaced with local upstream selective_scan_ref; CUDA/Triton functions explicitly disabled.",
    "The exact selective_scan_ref function is extracted into a CPU-importable file with unchanged function body."
  ]
}
```

### image_mednext

[Provenance](vendor/image_mednext/PROVENANCE.json) · [LICENSE](vendor/image_mednext/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "sources": [
    {
      "key": "mednext_net",
      "git_blob_sha": "350f2b5dbf704dbc8331441a3e0a7b18f1958f67",
      "commit": "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba",
      "original_path": "nnunet_mednext/network_architecture/mednextv1/MedNextV1.py",
      "source_url": "https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/MedNextV1.py",
      "license": "Apache-2.0"
    },
    {
      "key": "mednext_blocks",
      "git_blob_sha": "f8fd4d725fd604c98da1bfab822e7ef41d96713c",
      "commit": "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba",
      "original_path": "nnunet_mednext/network_architecture/mednextv1/blocks.py",
      "source_url": "https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/blocks.py",
      "license": "Apache-2.0"
    },
    {
      "key": "mednext_license",
      "git_blob_sha": "8bbe09cd335d3da0509c52ee207647cb6cb79368",
      "commit": "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba",
      "original_path": "LICENSE",
      "source_url": "https://github.com/MIC-DKFZ/MedNeXt/blob/main/LICENSE",
      "license": "Apache-2.0"
    }
  ],
  "modifications": [
    "One absolute Python import changed to relative .blocks; all architecture code unchanged."
  ]
}
```

### image_pvt

[Provenance](vendor/image_pvt/PROVENANCE.json) · [LICENSE](vendor/image_pvt/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "sources": [
    {
      "source_url": "https://github.com/whai362/PVT/blob/v2/classification/pvt.py",
      "git_blob_sha": "0695dd4db6475186b05236fde1279796ec9005e3",
      "original_path": "pvt.py"
    },
    {
      "source_url": "https://github.com/whai362/PVT/blob/v2/LICENSE",
      "git_blob_sha": "b1395e94b016dd1b95b4c7e3ed493e1d0b342917",
      "original_path": "LICENSE"
    }
  ],
  "license": "Apache-2.0",
  "modifications": [
    "Removed timm global model registration decorators to avoid name conflicts; model class unchanged."
  ]
}
```

### image_s4

[Provenance](vendor/image_s4/PROVENANCE.json) · [LICENSE](vendor/image_s4/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "sources": [
    {
      "key": "s4_full",
      "git_blob_sha": "deb534192dbf2b7795b29283e14d9430b9b00195",
      "commit": "e757cef57d89e448c413de7325ed5601aceaac13",
      "original_path": "models/s4/s4.py",
      "source_url": "https://github.com/state-spaces/s4/blob/main/models/s4/s4.py",
      "license": "Apache-2.0"
    },
    {
      "key": "s4_license",
      "git_blob_sha": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
      "commit": "e757cef57d89e448c413de7325ed5601aceaac13",
      "original_path": "LICENSE",
      "source_url": "https://github.com/state-spaces/s4/blob/main/LICENSE",
      "license": "Apache-2.0"
    }
  ],
  "modifications": [
    "Lightning rank_zero_only import replaced by identity decorator for single-process CPU; all S4/DPLR equations unchanged."
  ]
}
```

### image_segformer

[Provenance](vendor/image_segformer/PROVENANCE.json) · [LICENSE](vendor/image_segformer/LICENSE)

```json
{
  "fetched_date": "2026-10-17",
  "sources": [
    {
      "source_url": "https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py",
      "git_blob_sha": "e4b7ea21fba402f1c475d6c3d0fc4fdd1633cb3c",
      "original_path": "mix_transformer.py"
    },
    {
      "source_url": "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
      "git_blob_sha": "c14f578420e1f6ff6f60ca714839d0ed329a1051",
      "original_path": "LICENSE"
    },
    {
      "source_url": "https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py",
      "git_blob_sha": "8ada6d72de1c45270374dcdcfc18caf74509f2cb",
      "original_path": "segformer_head.py"
    }
  ],
  "license": "NVIDIA Source Code License for SegFormer — noncommercial research/evaluation only",
  "modifications": [
    "Removed MMCV/MMSeg registry integration; CPU encoder equations and weights initialization unchanged.",
    "Optional init_weights(pretrained=...) explicitly raises; no checkpoint is silently loaded.",
    "The original SegFormer head equations are preserved using minimal PyTorch-only BaseDecodeHead, resize and ConvModule adapters.",
    "SyncBatchNorm becomes BatchNorm2d for single-process CPU; imports of unused IPython/attr/MMCV utilities and registration decorators removed."
  ]
}
```

### inverse_cyclegan

[Provenance](vendor/inverse_cyclegan/PROVENANCE.json) · [LICENSE](vendor/inverse_cyclegan/LICENSE)

```json
{
  "repository": "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix",
  "commit": "2a7afba2895d52556dd5dfe07e8555ef657ced6f",
  "license": "BSD-2-Clause with bundled DCGAN BSD-3 notices",
  "retrieved_date": "2026-10-17",
  "adaptation": "No source edits. Both pix2pix and CycleGAN wrappers instantiate original network classes with smaller settings.",
  "files": [
    {
      "upstream_path": "models/networks.py",
      "url": "https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/2a7afba2895d52556dd5dfe07e8555ef657ced6f/models/networks.py",
      "sha256": "2bfcb7adb19f7b7ff6fd8dd76e313200acf000e96784b920658a62e1a11fc04f",
      "bytes": 26800
    },
    {
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/2a7afba2895d52556dd5dfe07e8555ef657ced6f/LICENSE",
      "sha256": "5474d910fc756107f149fb5ea08e557364d1607acda4e79a03b8e8f7a6279593",
      "bytes": 3565
    }
  ]
}
```

### inverse_dit

[Provenance](vendor/inverse_dit/PROVENANCE.json) · [LICENSE.txt](vendor/inverse_dit/LICENSE.txt)

```json
{
  "repository": "https://github.com/facebookresearch/DiT",
  "commit": "ed81ce2229091fd4ecc9a223645f95cf379d582b",
  "license": "CC-BY-NC-4.0",
  "retrieved_date": "2026-10-17",
  "adaptation": "No source edits. Noncommercial license applies to this copied code. timm 1.0.11 supplies architecture utilities.",
  "files": [
    {
      "upstream_path": "models.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/DiT/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py",
      "sha256": "1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca",
      "bytes": 14995
    },
    {
      "upstream_path": "LICENSE.txt",
      "url": "https://raw.githubusercontent.com/facebookresearch/DiT/ed81ce2229091fd4ecc9a223645f95cf379d582b/LICENSE.txt",
      "sha256": "7c6858d0e71dc8b858255270ad4b5004e80b5b058bf5bbc408a28430d6ddbf4b",
      "bytes": 19334
    }
  ]
}
```

### inverse_dncnn

[Provenance](vendor/inverse_dncnn/PROVENANCE.json) · [LICENSE](vendor/inverse_dncnn/LICENSE)

```json
{
  "repository": "https://github.com/cszn/KAIR",
  "commit": "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e",
  "license": "MIT",
  "retrieved_date": "2026-10-17",
  "adaptation": "Files unchanged on disk. Runtime loader replaces only import models.basicblock as B with isolated module alias import biomed_lab_upstream_dncnn_blocks as B, to avoid package-name collisions.",
  "files": [
    {
      "upstream_path": "models/network_dncnn.py",
      "url": "https://raw.githubusercontent.com/cszn/KAIR/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/models/network_dncnn.py",
      "sha256": "8cee9e14df16da67f922ba96b9156b5bc5bcfc8c1cc2628906c871287d4b7c40",
      "bytes": 6298
    },
    {
      "upstream_path": "models/basicblock.py",
      "url": "https://raw.githubusercontent.com/cszn/KAIR/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/models/basicblock.py",
      "sha256": "48406db8867394ac5ae233ebeec7711ac10acfc3a6bbf0072c33aa77d659b6fd",
      "bytes": 24138
    },
    {
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/cszn/KAIR/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/LICENSE",
      "sha256": "448e69b705d64f21bf8cb86562301e0edd99ac79026064ddd75af8242b067be5",
      "bytes": 1066
    }
  ]
}
```

### inverse_fastmri

[Provenance](vendor/inverse_fastmri/PROVENANCE.json) · [LICENSE.md](vendor/inverse_fastmri/LICENSE.md)

```json
{
  "repository": "https://github.com/facebookresearch/fastMRI",
  "commit": "91f2df4711adbb6d643df1810f234e4abcf5881b",
  "license": "MIT",
  "retrieved_date": "2026-10-17",
  "adaptation": "Files unchanged. VarNet and Unet run from these files; installed fastmri 0.3.0 supplies Fourier and data transform utilities.",
  "files": [
    {
      "upstream_path": "fastmri/models/varnet.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/fastMRI/91f2df4711adbb6d643df1810f234e4abcf5881b/fastmri/models/varnet.py",
      "sha256": "dd2ccddfc899b346558b35e2a1fb23e2a1e2e2c0c81b7f62a71974bd98c63942",
      "bytes": 10666
    },
    {
      "upstream_path": "fastmri/models/unet.py",
      "url": "https://raw.githubusercontent.com/facebookresearch/fastMRI/91f2df4711adbb6d643df1810f234e4abcf5881b/fastmri/models/unet.py",
      "sha256": "b0c097adaa5c086ea93cdc34774bd508bd93dbf6146d74aad1024304ccec47a8",
      "bytes": 5981
    },
    {
      "upstream_path": "LICENSE.md",
      "url": "https://raw.githubusercontent.com/facebookresearch/fastMRI/91f2df4711adbb6d643df1810f234e4abcf5881b/LICENSE.md",
      "sha256": "52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85",
      "bytes": 1086
    }
  ]
}
```

### inverse_restormer

[Provenance](vendor/inverse_restormer/PROVENANCE.json) · [LICENSE.md](vendor/inverse_restormer/LICENSE.md)

```json
{
  "repository": "https://github.com/swz30/Restormer",
  "commit": "68dc6ac472db26f16361150cb7a96a1bc87da93f",
  "license": "MIT",
  "retrieved_date": "2026-10-17",
  "adaptation": "No source edits. Runtime dependency: einops 0.8.0.",
  "files": [
    {
      "upstream_path": "basicsr/models/archs/restormer_arch.py",
      "url": "https://raw.githubusercontent.com/swz30/Restormer/68dc6ac472db26f16361150cb7a96a1bc87da93f/basicsr/models/archs/restormer_arch.py",
      "sha256": "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a",
      "bytes": 11428
    },
    {
      "upstream_path": "LICENSE.md",
      "url": "https://raw.githubusercontent.com/swz30/Restormer/68dc6ac472db26f16361150cb7a96a1bc87da93f/LICENSE.md",
      "sha256": "2b93776512924bc095ec5d97a79d76cf1ab401ae641d1ae1f46e94f1c53a2e59",
      "bytes": 1090
    }
  ]
}
```

### inverse_sit

[Provenance](vendor/inverse_sit/PROVENANCE.json) · [LICENSE.txt](vendor/inverse_sit/LICENSE.txt)

```json
{
  "repository": "https://github.com/willisma/SiT",
  "commit": "cbde832a40b153ccc79603412409da9c9b0c568c",
  "license": "MIT",
  "retrieved_date": "2026-10-17",
  "adaptation": "No source edits. Original SiT class, linear interpolant velocity objective, reduced image-space settings.",
  "files": [
    {
      "upstream_path": "models.py",
      "url": "https://raw.githubusercontent.com/willisma/SiT/cbde832a40b153ccc79603412409da9c9b0c568c/models.py",
      "sha256": "677cfa2cf5a7db122abd014d6d92d7ac5f39745f571773a6adcc26c7d2f33d89",
      "bytes": 14984
    },
    {
      "upstream_path": "LICENSE.txt",
      "url": "https://raw.githubusercontent.com/willisma/SiT/cbde832a40b153ccc79603412409da9c9b0c568c/LICENSE.txt",
      "sha256": "cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173",
      "bytes": 1087
    }
  ]
}
```

### inverse_swinir

[Provenance](vendor/inverse_swinir/PROVENANCE.json) · [LICENSE](vendor/inverse_swinir/LICENSE)

```json
{
  "repository": "https://github.com/JingyunLiang/SwinIR",
  "commit": "6545850fbf8df298df73d81f3e8cba638787c8bd",
  "license": "Apache-2.0",
  "retrieved_date": "2026-10-17",
  "adaptation": "No source edits. Runtime dependency: timm 1.0.11; upstream deprecation warnings are preserved.",
  "files": [
    {
      "upstream_path": "models/network_swinir.py",
      "url": "https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py",
      "sha256": "9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913",
      "bytes": 36832
    },
    {
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/LICENSE",
      "sha256": "d66ccc11682c7458b19b376e1335e8665cbe412ff71f554bfec25f1791cddfa0",
      "bytes": 11348
    }
  ]
}
```

### inverse_transmorph

[Provenance](vendor/inverse_transmorph/PROVENANCE.json) · [LICENSE](vendor/inverse_transmorph/LICENSE)

```json
{
  "repository": "https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration",
  "commit": "6357a1d7fc44c36db9b1d1ccaa372409253142cf",
  "license": "MIT",
  "retrieved_date": "2026-10-17",
  "adaptation": "Files unchanged on disk. Runtime loader replaces only import models.configs_TransMorph as configs with isolated module alias import biomed_lab_upstream_transmorph_configs as configs. ml-collections 0.1.1 provides original ConfigDict. Original CUDA-only optional sinusoidal embeddings remain disabled as in baseline config. Original Grad3d loss is used directly. Sizes/widths/depth/window/drop-path are configuration changes, not architecture edits.",
  "training_reference": "https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/6357a1d7fc44c36db9b1d1ccaa372409253142cf/TransMorph/train_TransMorph.py",
  "files": [
    {
      "upstream_path": "TransMorph/models/TransMorph.py",
      "url": "https://raw.githubusercontent.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/6357a1d7fc44c36db9b1d1ccaa372409253142cf/TransMorph/models/TransMorph.py",
      "sha256": "988ef4181ae173c7d77fb8409fb87dae6d13ff1a236330e506ee8915ecf64076",
      "bytes": 38672
    },
    {
      "upstream_path": "TransMorph/models/configs_TransMorph.py",
      "url": "https://raw.githubusercontent.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/6357a1d7fc44c36db9b1d1ccaa372409253142cf/TransMorph/models/configs_TransMorph.py",
      "sha256": "7cc3f82026b95d5887195ef27147cca4c244a8030f593ea38ec18b7a64d32a31",
      "bytes": 10318
    },
    {
      "upstream_path": "TransMorph/losses.py",
      "url": "https://raw.githubusercontent.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/6357a1d7fc44c36db9b1d1ccaa372409253142cf/TransMorph/losses.py",
      "sha256": "e25b9ef9b9fa8623ee35aea201433e7b5892a38dd1dfaa2cd82b63f61aa5e00f",
      "bytes": 23692
    },
    {
      "upstream_path": "LICENSE",
      "url": "https://raw.githubusercontent.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/6357a1d7fc44c36db9b1d1ccaa372409253142cf/LICENSE",
      "sha256": "47b39efd556809b2d5ac1f7468ae5da0286c0dd28fb828e79523a41d98f96b1b",
      "bytes": 1067
    }
  ]
}
```

### research_mae

[Provenance](vendor/research_mae/PROVENANCE.json) · [LICENSE](vendor/research_mae/LICENSE)

```json
{
  "repository": "https://github.com/facebookresearch/mae",
  "commit": "efb2a8062c206524e35e47d04501ed4f544c0ae8",
  "license": "CC-BY-NC-4.0",
  "changes": [
    "models_mae.py: use relative pos_embed import; remove qk_scale=None argument removed by timm (default scale remains 1/sqrt(head_dim)).",
    "pos_embed.py: replace removed np.float alias with identical np.float64 dtype.",
    "Runtime uses 32x32 RGB replication, patch4, reduced widths/depth; original training/pretraining/results not reproduced."
  ],
  "files": {
    "models_mae_original.py": "102a53122866b6f7714afb04e2d083ac8019023c6d464d5df4aed64ca80e8444",
    "pos_embed_original.py": "ef67de3ea7038e2165ff479d4fe11d2d76455028bad81267c85e9eeb35c99f82",
    "models_mae.py": "f6218f7de157f3bba74db743c82215de9d360a712b6c2774f69a25ef8cdcfc45",
    "pos_embed.py": "951105688664030a9297b2e6ed2261284377013da866deebca36f6489fd9f978"
  }
}
```

### research_tcn

[Provenance](vendor/research_tcn/PROVENANCE.json) · [LICENSE](vendor/research_tcn/LICENSE)

```json
{
  "repository": "https://github.com/locuslab/TCN",
  "commit": "2f8c2b817050206397458dfd1f5a25ce8a32fe65",
  "license": "MIT",
  "changes": [],
  "files": {
    "tcn.py": "f9d8e3ec037ac60ab906053a90cb04fa62803fd644ab81d3cb08ef65eddc4dd7"
  }
}
```

## Installed libraries

The exact direct dependencies are in [requirements-cpu.txt](requirements-cpu.txt), [requirements-macos.txt](requirements-macos.txt) and [pyproject.toml](pyproject.toml); each execution record captures installed versions. Library code is installed from its upstream distribution, not relicensed here.

| Library | Actual execution version/source |
| --- | --- |
| PyTorch / torchvision | [2.5.1](https://github.com/pytorch/pytorch/tree/v2.5.1) / [0.20.1](https://github.com/pytorch/vision/tree/v0.20.1) |
| MONAI | [1.4.0](https://github.com/Project-MONAI/MONAI/tree/1.4.0) |
| timm | [1.0.11](https://github.com/huggingface/pytorch-image-models/tree/v1.0.11) |
| PyTorch Geometric | [2.6.1](https://github.com/pyg-team/pytorch_geometric/tree/2.6.1) |
| Transformers | [4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3) |
| fastMRI | [0.3.0](https://github.com/facebookresearch/fastMRI/tree/v0.3.0) |

The complete 137-item source/license audit, including repositories not copied here, is [sources.json](sources.json). NOASSERTION means that copying permission has not been established, not that a work has no owner. GPL is an open-source copyleft license; its obligations are different from noncommercial restrictions. HeartBEiT includes only the individual architecture file carrying an explicit inherited Microsoft MIT grant, not the whole unlicensed repository.
