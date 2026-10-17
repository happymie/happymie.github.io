"""Licensed foundation-family architectures, explicitly without pretraining.

These exercises expose the original architecture code used for downstream
classification. They do not reproduce the published foundation training,
checkpoints, disease ontology or cross-cohort results.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


SPECS = {
    79: {
        "name":"HeartBEiT (downstream image architecture)",
        "task":"sequence",
        "implementation":"upstream",
        "source_url":"https://github.com/akhilvaid/HeartBEiT/blob/e065cca3c41d6bc5e852a44c6f7fc1fb51c4d45e/modeling_finetune.py",
        "mechanism":"The author's MIT-attributed BEiT classification encoder receives rasterized synthetic traces; relative position bias, layer scale and mean token pooling are retained.",
        "limitations":[
            "Architecture-only exercise: HeartBEiT's DALL-E tokenization, ECG masked pretraining, pretrained weights and cardiac downstream studies are not reproduced.",
            "The default uses four dimensionless synthetic waveform traces rendered on a 64x64 grid; these are not physiological ECGs or calibrated paper ECG images.",
            "Width/depth/patch size are reduced; random weights carry no foundation-model capability.",
            "Only modeling_finetune.py has been copied under its explicit file-level Microsoft MIT notice; the repository's remaining files have no inferred redistribution permission."
        ],
        "config":{"batch_size":2,"trace_image_size":64,"patch_size":8,"hidden_size":64,"depth":2,"num_heads":4},
    },
    80: {
        "name":"LaBraM (downstream EEG architecture)",
        "task":"sequence",
        "implementation":"upstream",
        "source_url":"https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/modeling_finetune.py",
        "mechanism":"The authors' TemporalConv and NeuralTransformer operate on channel-by-time patches with separate learned electrode and time embeddings and Q/K normalization.",
        "limitations":[
            "Architecture-only exercise: VQ neural tokenizer, masked code prediction, full pretraining corpus and pretrained LaBraM capability are not reproduced.",
            "Synthetic waveforms are interpolated to two 200-sample patches per channel. This is a shape adapter, not an assertion of EEG sampling rate or a source of extra information.",
            "Synthetic channel IDs 1..C have no anatomical electrode identity. Real EEG requires explicit montage, channel-order, unit and sampling checks.",
            "Width 100 and depth two replace the original base preset; all weights are random."
        ],
        "config":{"batch_size":2,"patch_size":200,"patches":2,"temporal_channels":4,"depth":2,"num_heads":4},
    },
    137: {
        "name":"ECGFounder (Net1D architecture)",
        "task":"sequence",
        "implementation":"upstream",
        "source_url":"https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/net1d.py",
        "mechanism":"The authors' Net1D: SAME-padded grouped 1D bottlenecks, squeeze-excitation, residual downsampling, global time pooling and a task head.",
        "limitations":[
            "Architecture-only exercise: no ten-million-recording pretraining, 150-label diagnostic ontology, lead-dropout pretraining or published checkpoint is included.",
            "The default reads one synthetic waveform channel; it is not a physiological ECG or a validated single-lead disease predictor.",
            "Original seven-stage block counts and kernel size 16 are retained, but channels are reduced and the downstream head has three synthetic classes.",
            "No external weights are loaded. Low compute does not recreate the learned knowledge of ECGFounder."
        ],
        "config":{"batch_size":2,"in_channels":1,"base_filters":8,"filter_list":[8,16,16,24,24,32,32],"m_blocks_list":[2,2,2,3,3,4,4],"kernel_size":16,"groups_width":4},
    },
}


def render_synthetic_traces(sequence: torch.Tensor,size: int=64) -> torch.Tensor:
    """Rasterize the observed input traces; no label/target is accepted.

    Input is B,L,C with dimensionless amplitude. Fixed amplitude range and row
    allocation preserve relative amplitudes; no specimen-dependent rescaling.
    The Gaussian stroke is a sampling convention, not medical acquisition.
    """
    if sequence.ndim!=3 or sequence.shape[-1]<1:
        raise ValueError("Expected waveform shape (batch, time, channels)")
    values=F.interpolate(sequence.transpose(1,2),size=size,mode="linear",align_corners=False)
    channels=values.shape[1]
    rows=(torch.arange(channels,device=sequence.device,dtype=sequence.dtype)+.5)/channels
    centers=(rows[None,:,None]-values.clamp(-2,2)*(.2/channels))*(size-1)
    pixel_rows=torch.arange(size,device=sequence.device,dtype=sequence.dtype)[None,None,:,None]
    stroke=torch.exp(-.5*((pixel_rows-centers[:,:,None,:])/.6).square()).amax(1)
    return (1-stroke)[:,None]


class HeartBEiTArchitecture(nn.Module):
    def __init__(self,config):
        super().__init__()
        from vendor.extended_heartbeit.modeling_finetune import VisionTransformer
        self.size=config["trace_image_size"]
        self.network=VisionTransformer(img_size=self.size,patch_size=config["patch_size"],in_chans=3,num_classes=3,
                                       embed_dim=config["hidden_size"],depth=config["depth"],num_heads=config["num_heads"],
                                       mlp_ratio=2.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6),
                                       use_abs_pos_emb=False,use_rel_pos_bias=True,use_mean_pooling=True,init_values=.1)

    def forward(self,sequence):
        image=render_synthetic_traces(sequence,self.size).repeat(1,3,1,1)
        return self.network(image)

    def training_loss(self,batch):
        return F.cross_entropy(self(batch["sequence"]),batch["sequence_label"])


class LaBraMArchitecture(nn.Module):
    def __init__(self,config):
        super().__init__()
        from vendor.extended_labram.modeling_finetune import NeuralTransformer
        self.patch_size,self.patches=config["patch_size"],config["patches"]
        out=config["temporal_channels"]
        if out%4 or not (1<=self.patches<=16):
            raise ValueError("LaBraM TemporalConv requires temporal_channels divisible by 4 and 1..16 time patches")
        hidden=((self.patch_size+7)//8)*out
        if hidden%config["num_heads"]:
            raise ValueError("TemporalConv output width must divide evenly into num_heads")
        self.hidden_size=hidden
        self.network=NeuralTransformer(EEG_size=self.patch_size*self.patches,patch_size=self.patch_size,in_chans=1,
                                      out_chans=out,num_classes=3,embed_dim=hidden,depth=config["depth"],num_heads=config["num_heads"],
                                      mlp_ratio=2.,qkv_bias=True,qk_norm=partial(nn.LayerNorm,eps=1e-6),
                                      norm_layer=partial(nn.LayerNorm,eps=1e-6),use_abs_pos_emb=True,use_mean_pooling=True,init_values=.1)

    def prepare(self,sequence):
        if sequence.ndim!=3 or not 1<=sequence.shape[-1]<=128:
            raise ValueError("Expected B,L,C with 1..128 declared synthetic channels")
        b,_,channels=sequence.shape
        values=F.interpolate(sequence.transpose(1,2),size=self.patch_size*self.patches,mode="linear",align_corners=False)
        return values.reshape(b,channels,self.patches,self.patch_size),list(range(channels+1))

    def forward(self,sequence):
        patches,channel_ids=self.prepare(sequence)
        return self.network(patches,input_chans=channel_ids)

    def training_loss(self,batch):
        return F.cross_entropy(self(batch["sequence"]),batch["sequence_label"])


class ECGFounderArchitecture(nn.Module):
    def __init__(self,config):
        super().__init__()
        from vendor.extended_ecgfounder.net1d import Net1D
        self.in_channels=config["in_channels"]
        if len(config["filter_list"])!=len(config["m_blocks_list"]):
            raise ValueError("Each Net1D stage needs one width and one block count")
        if any(w%config["groups_width"] for w in config["filter_list"]):
            raise ValueError("All Net1D stage widths must be divisible by groups_width")
        self.network=Net1D(in_channels=self.in_channels,base_filters=config["base_filters"],ratio=1,
                           filter_list=config["filter_list"],m_blocks_list=config["m_blocks_list"],kernel_size=config["kernel_size"],
                           stride=2,groups_width=config["groups_width"],n_classes=3,use_bn=False,use_do=False,verbose=False)

    def forward(self,sequence):
        if sequence.ndim!=3 or sequence.shape[-1]<self.in_channels:
            raise ValueError("Input must provide the configured observed channels; missing channels are never fabricated")
        return self.network(sequence[...,:self.in_channels].transpose(1,2).contiguous())

    def training_loss(self,batch):
        return F.cross_entropy(self(batch["sequence"]),batch["sequence_label"])


def build_model(num:int,config:dict|None=None)->nn.Module:
    if num not in SPECS:
        raise KeyError(f"Extended model {num} is unavailable")
    c={**SPECS[num]["config"],**(config or {})}
    return {79:HeartBEiTArchitecture,80:LaBraMArchitecture,137:ECGFounderArchitecture}[num](c)
