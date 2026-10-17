"""Image and state-space lessons with explicit implementation boundaries.

Upstream weights are never downloaded at import or construction. CPU presets
reduce capacity, not the declared mechanism. ``SPECS`` says which lessons are
upstream architectures, paper ports, or bounded mechanism demonstrations.
See docs/IMAGE_MODELS.md and the individual vendor LICENSE/PROVENANCE files.
"""
from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def _spec(name, task, implementation, source_url, mechanism, limitations, **config):
    return dict(name=name, task=task, implementation=implementation,
                source_url=source_url, mechanism=mechanism,
                limitations=list(limitations), config=dict(batch_size=2, **config))


_NO_RESULTS = "Random initialization and synthetic data; no original benchmark result or pretrained capability is reproduced."
_SMALL = "CPU dimensions differ from the paper's training preset; see the explicit configuration."
_TV = "https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models"
_MONAI = "https://github.com/Project-MONAI/MONAI/blob/1.4.0/monai/networks/nets"

SPECS = {
    1: _spec("AlexNet", "classification", "upstream", _TV + "/alexnet.py",
             "Five convolution layers and three fully connected layers from torchvision; reduced classifier.",
             [_NO_RESULTS, "Torchvision uses its documented AlexNet variant; adaptive pooling 1x1 and classifier width 128 replace the 6x6/4096 head."], image_size=64, fc_width=128),
    2: _spec("ResNet", "classification", "upstream", _TV + "/resnet.py",
             "ResNet-18 BasicBlock residual stages, directly instantiated from torchvision.",
             [_NO_RESULTS, "Input is a synthetic 32-pixel grayscale image repeated to RGB; classifier has three classes."], image_size=32),
    3: _spec("DenseNet", "classification", "upstream", _TV + "/densenet.py",
             "Dense concatenation and compressed transition layers in the upstream DenseNet class.",
             [_NO_RESULTS, _SMALL], image_size=32, growth_rate=8, block_config=[2,2,2,2], initial_features=16),
    4: _spec("ConvNeXt", "classification", "upstream", _TV + "/convnext.py",
             "Depthwise 7x7 convolution, channels-last LayerNorm, inverted bottleneck and layer scale.",
             [_NO_RESULTS, _SMALL], image_size=32, widths=[16,32,64,128], depths=[1,1,3,1]),
    5: _spec("FCN", "segmentation", "paper_port", "https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/train.prototxt",
             "FCN-8s: VGG pool3/pool4 score fusion plus learned 2x/2x/8x transposed upsampling.",
             [_NO_RESULTS, "Reduced VGG channels/FC widths; same-padding replaces the original padding/cropping boundary convention."], width=8, fc_width=64),
    6: _spec("U-Net", "segmentation", "paper_port", "https://arxiv.org/abs/1505.04597",
             "Two convolutions per scale, contracting path, transposed upsampling and concatenated encoder skips.",
             [_NO_RESULTS, "Three pooling levels and same-padding instead of the original four-level valid-convolution/crop preset."], width=8, depth=3),
    7: _spec("3D U-Net", "segmentation3d", "paper_port", "https://arxiv.org/abs/1606.06650",
             "Volumetric convolution/pooling/upsampling and same-scale 3D encoder skips with BatchNorm.",
             [_NO_RESULTS, "Reduced channels; same-padding; densely labelled phantoms do not reproduce the sparse-annotation study."], width=4, depth=3),
    8: _spec("V-Net", "segmentation3d", "upstream", _MONAI + "/vnet.py",
             "MONAI VNet residual volumetric stages with 5x5x5 kernels; soft Dice training objective.",
             [_NO_RESULTS, "MONAI implementation with ELU, dropout disabled and two output classes; 32-cube teaching input."], volume_size=32),
    10: _spec("UNETR", "segmentation3d", "upstream", _MONAI + "/unetr.py",
              "Twelve-layer ViT encoder with intermediate token features decoded by a volumetric U-shaped network.",
              [_NO_RESULTS, _SMALL], volume_size=32, hidden_size=48, mlp_dim=96, feature_size=4, num_heads=4),
    11: _spec("Swin UNETR (architecture)", "segmentation3d", "upstream", _MONAI + "/swin_unetr.py",
              "Hierarchical shifted-window transformer encoder and volumetric skip decoder.",
              [_NO_RESULTS, "This lesson runs the segmentation architecture. The original SSL rotation/contrast/restoration pretraining and its checkpoint are NOT reproduced.", "GroupNorm replaces InstanceNorm to support tiny spatial grids."], volume_size=32, feature_size=12, depths=[2,2,2,2]),
    12: _spec("MedNeXt", "segmentation3d", "upstream", "https://github.com/MIC-DKFZ/MedNeXt",
              "Authors' MedNeXt encoder/decoder and depthwise inverted bottleneck blocks, with only a local import adapter.",
              [_NO_RESULTS, "Width 4, one block per scale, expansion 2 and kernel 3; original scaling/training/UpKern experiment is not reproduced."], volume_size=32, width=4, expansion=2, kernel_size=3),
    13: _spec("Models Genesis (pretext mechanisms)", "reconstruction3d", "mechanism_demo", "https://arxiv.org/abs/1908.06912",
              "Restore 3D inputs after local shuffling, Bezier intensity transformation, inpainting or outpainting.",
              [_NO_RESULTS, "Independent PyTorch teaching implementation; reduced 3D U-Net and corruption budget; no published pretrained model or transfer result."], width=4, shuffle_blocks=16),
    15: _spec("S4", "sequence", "upstream", "https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py",
              "Authors' full diagonal-plus-low-rank S4 kernel, HiPPO initialization and FFT convolution.",
              [_NO_RESULTS, "Single-process logging adapter; naive CPU Cauchy kernel instead of accelerated kernels; small classifier added."], d_model=16, d_state=16),
    16: _spec("Mamba", "sequence", "upstream", "https://github.com/state-spaces/mamba",
              "Authors' Mamba block, input-dependent delta/B/C, causal depthwise convolution and exact reference selective scan.",
              [_NO_RESULTS, "Python reference scan is slower than CUDA fused kernels; small sequence classification head added."], d_model=16, d_state=8),
    17: _spec("VMamba (SS2D mechanism)", "classification", "mechanism_demo", "https://github.com/MzeroMiko/VMamba",
              "Four independent selective scans along row/column orders and their reversals; restored 2D alignment and gated fusion.",
              [_NO_RESULTS, "Independent compact SS2D port with a small image stem/head; not the complete VSSM hierarchy, initialization or ImageNet recipe."], image_size=32, width=16, d_state=4),
    18: _spec("U-Mamba (bottleneck mechanism)", "segmentation3d", "mechanism_demo", "https://github.com/bowang-lab/U-Mamba",
              "Residual convolution U-shaped encoder/decoder with authors' Mamba block on flattened bottleneck features.",
              [_NO_RESULTS, "Compact bottleneck variant only; nnU-Net planning and the full U-Mamba_Bot/Enc training pipelines are not reproduced."], width=4, d_state=4),
    19: _spec("SegMamba (spatial scan mechanism)", "segmentation3d", "mechanism_demo", "https://github.com/ge-xing/SegMamba",
              "Gated spatial convolution branches and three sequence scan orders inside a volumetric skip network.",
              [_NO_RESULTS, "The three-order scan is a pedagogical analogue, not the authors' BiMamba-v3/fast-slice kernel or complete SegMamba architecture."], width=4, d_state=4),
    20: _spec("Transformer", "language", "upstream", "https://docs.pytorch.org/docs/2.5/generated/torch.nn.Transformer.html",
              "Encoder-decoder multihead attention, sinusoidal positions, causal target mask and teacher-forced token loss.",
              [_NO_RESULTS, "Small vocabulary and token reversal exercise instead of machine translation; no WMT training recipe."], d_model=32, depth=2, num_heads=4, vocab_size=32),
    21: _spec("Vision Transformer / ViT", "classification", "upstream", _TV + "/vision_transformer.py",
              "Patch projection, class token, learned positional embeddings and pre-norm transformer encoder.",
              [_NO_RESULTS, _SMALL], image_size=32, patch_size=4, hidden_size=32, depth=2, num_heads=4),
    22: _spec("Swin Transformer", "classification", "upstream", _TV + "/swin_transformer.py",
              "Alternating regular/shifted window attention, relative position bias and patch merging.",
              [_NO_RESULTS, _SMALL], image_size=32, width=16, depths=[2,2,2,2], window_size=4),
    23: _spec("Mask2Former (masked-attention mechanism)", "segmentation", "mechanism_demo", "https://github.com/facebookresearch/Mask2Former",
              "Query masks restrict cross-attention; bipartite matching supervises class/mask/dice set predictions.",
              [_NO_RESULTS, "Small convolution pixel decoder replaces multi-scale deformable attention; semantic foreground/background targets, not panoptic instance annotations.", "Matching, mask attention and set losses are real; the full Detectron2 pipeline is not reproduced."], width=32, num_queries=4, depth=3),
    81: _spec("VGG-16/19", "classification", "upstream", _TV + "/vgg.py",
              "Upstream make_layers with the VGG-16 D configuration, reduced channels and three-layer classifier.",
              [_NO_RESULTS, "Default runs VGG-16; set variant=19 for VGG-19. Adaptive pooling 1x1 and narrow classifier replace the original head."], image_size=32, width=8, fc_width=64, variant=16),
    82: _spec("GoogLeNet / Inception-v1", "classification", "upstream", _TV + "/googlenet.py",
              "Upstream GoogLeNet multi-branch Inception modules and global pooling.",
              [_NO_RESULTS, "Auxiliary classifiers and dropout disabled; torchvision's documented implementation details differ from the original Caffe model."], image_size=32),
    83: _spec("ResNeXt", "classification", "upstream", _TV + "/resnet.py",
              "Upstream ResNet Bottleneck class configured with grouped 3x3 transforms (cardinality 8).",
              [_NO_RESULTS, "One bottleneck per stage, cardinality 8 and base width 4; not the original 32x4d preset."], image_size=32, groups=8, width_per_group=4),
    84: _spec("SENet", "classification", "upstream", "https://github.com/huggingface/pytorch-image-models/blob/v1.0.11/timm/models/resnet.py",
              "timm SE-ResNet-18 with global squeeze and sigmoid channel excitation inside residual blocks.",
              [_NO_RESULTS, "SE-ResNet-18 family example, not SENet-154; one-channel input and three classes."], image_size=32),
    85: _spec("EfficientNet", "classification", "upstream", _TV + "/efficientnet.py",
              "Upstream MBConv, squeeze-excitation and compound width/depth configuration.",
              [_NO_RESULTS, "Quarter width/half depth B0-family teaching configuration is not a published EfficientNet checkpoint."], image_size=32, width_mult=0.25, depth_mult=0.5),
    86: _spec("U-Net++", "segmentation", "paper_port", "https://arxiv.org/abs/1807.10165",
              "Nested decoder grid concatenates all earlier same-scale nodes, with supervised heads at each decoder depth.",
              [_NO_RESULTS, "Three-level, narrow independent port; original pruning/clinical experiments and nnU-Net trainer not reproduced."], width=8, depth=3),
    87: _spec("Attention U-Net", "segmentation", "upstream", _MONAI + "/attentionunet.py",
              "MONAI additive spatial attention gates filter encoder skip tensors with decoder context.",
              [_NO_RESULTS, "Three resolution levels, width 8; MONAI implementation and synthetic foreground classes."], width=8),
    88: _spec("H-DenseUNet (hybrid fusion mechanism)", "segmentation3d", "mechanism_demo", "https://arxiv.org/abs/1709.07330",
              "2D dense encoder/decoder features are stacked in slice order and fused with 3D dense features before a volumetric head.",
              [_NO_RESULTS, "Compact independent hybrid port; staged ImageNet/LiTS training, original DenseNet depths and liver/tumor experiment are not reproduced."], width=4),
    89: _spec("DeiT (distillation mechanism)", "classification", "mechanism_demo", "https://github.com/facebookresearch/deit",
              "timm distilled ViT has separate class/distillation tokens and heads; hard teacher targets supervise the distillation head.",
              [_NO_RESULTS, "Default teacher is an explicit analytic classifier of synthetic shape fill ratio, not the paper's pretrained RegNetY. Supply teacher_logits for a real teacher.", _SMALL], image_size=32, patch_size=4, hidden_size=32, depth=2, num_heads=4),
    90: _spec("Pyramid Vision Transformer / PVT", "classification", "upstream", "https://github.com/whai362/PVT/blob/v2/classification/pvt.py",
              "Authors' PVT v1 class: four pyramid stages, spatial-reduction attention and last-stage class token.",
              [_NO_RESULTS, _SMALL], image_size=32, widths=[8,16,32,64], depths=[1,1,1,1]),
    91: _spec("SegFormer", "segmentation", "upstream", "https://github.com/NVlabs/SegFormer",
              "Authors' overlapping Mix Transformer encoder, depthwise Mix-FFN and all-MLP multi-scale decode head.",
              [_NO_RESULTS, "NVIDIA vendor code permits noncommercial research/evaluation only; see vendor/image_segformer/LICENSE.", "MMCV registry/checkpoint interfaces removed; SyncBatchNorm becomes ordinary BatchNorm on one CPU; dimensions reduced."], image_size=32, widths=[8,16,32,64], depths=[1,1,1,1], decoder_width=16),
}

UNAVAILABLE = {
    9: "nnU-Net is a self-configuring pipeline. No planner, preprocessing, cross-validation or model selection is replaced by a generic U-Net here. Use the official nnUNet repository.",
    14: "VISTA3D requires its actual prompt/automatic segmentation system and pretrained resources. A random small U-Net is not a reproduction; use the official MONAI VISTA resources.",
}


def _resize(x, size):
    if size is None or tuple(x.shape[2:]) == (size,) * (x.ndim-2):
        return x
    return F.interpolate(x, size=(size,)*(x.ndim-2), mode="bilinear" if x.ndim == 4 else "trilinear", align_corners=False)


class TaskAdapter(nn.Module):
    """Explicit modality, channel and output-grid adapter around a real model."""
    def __init__(self, network, task="classification", rgb=False, size=None, dice=False):
        super().__init__()
        self.network, self.task, self.rgb, self.size, self.dice = network, task, rgb, size, dice

    def forward(self, x):
        original_shape = x.shape[2:]
        x = _resize(x, self.size)
        if self.rgb and x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        y = self.network(x)
        if isinstance(y, dict):
            y = y["out"]
        if hasattr(y, "logits"):
            y = y.logits
        if isinstance(y, (tuple,list)):
            y = y[0]
        if self.task.startswith("segmentation") and tuple(y.shape[2:]) != tuple(original_shape):
            y = F.interpolate(y, size=original_shape, mode="bilinear" if y.ndim==4 else "trilinear", align_corners=False)
        return y

    def training_loss(self, batch):
        if self.task == "classification":
            return F.cross_entropy(self(batch["image"]), batch["label"])
        key, target = ("volume", "volume_mask") if self.task == "segmentation3d" else ("image", "mask")
        prediction = self(batch[key])
        if self.dice:
            p = prediction.softmax(1)
            y = F.one_hot(batch[target], p.shape[1]).movedim(-1,1).float()
            axes = tuple(range(2,p.ndim))
            return (1-(2*(p*y).sum(axes)+1e-5)/(p.square().sum(axes)+y.square().sum(axes)+1e-5)).mean()
        return F.cross_entropy(prediction, batch[target])


class ConvPair(nn.Module):
    def __init__(self, dim, in_c, out_c, norm=False, residual=False):
        super().__init__()
        conv = nn.Conv2d if dim==2 else nn.Conv3d
        bn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
        self.layers = nn.Sequential(conv(in_c,out_c,3,padding=1), bn(out_c) if norm else nn.Identity(), nn.ReLU(),
                                    conv(out_c,out_c,3,padding=1), bn(out_c) if norm else nn.Identity())
        self.skip = conv(in_c,out_c,1) if residual and in_c != out_c else nn.Identity()
        self.residual = residual

    def forward(self,x):
        return F.relu(self.layers(x) + self.skip(x) if self.residual else self.layers(x))


class UNetND(nn.Module):
    def __init__(self, dim=2, width=8, depth=3, in_channels=1, classes=2, norm=False, residual=False):
        super().__init__()
        conv = nn.Conv2d if dim==2 else nn.Conv3d
        trans = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
        pool = nn.MaxPool2d if dim==2 else nn.MaxPool3d
        self.dim, self.depth = dim, depth
        self.widths = [width*2**i for i in range(depth+1)]
        self.enc = nn.ModuleList([ConvPair(dim,in_channels if i==0 else self.widths[i-1],c,norm,residual) for i,c in enumerate(self.widths)])
        self.pool = pool(2)
        self.up = nn.ModuleList([trans(self.widths[i+1],self.widths[i],2,2) for i in range(depth)])
        self.dec = nn.ModuleList([ConvPair(dim,2*self.widths[i],self.widths[i],norm,residual) for i in range(depth)])
        self.head = conv(width,classes,1)

    def encode(self,x):
        features=[]
        for i,block in enumerate(self.enc):
            x=block(self.pool(x) if i else x)
            features.append(x)
        return features

    def decode(self,features):
        x=features[-1]
        for i in reversed(range(self.depth)):
            x=self.up[i](x)
            if x.shape[2:] != features[i].shape[2:]:
                x=F.interpolate(x,size=features[i].shape[2:],mode="nearest")
            x=self.dec[i](torch.cat([features[i],x],1))
        return self.head(x)

    def forward(self,x):
        return self.decode(self.encode(x))


class FCN8s(nn.Module):
    def __init__(self,width=8,fc_width=64):
        super().__init__()
        from torchvision.models.vgg import make_layers
        self.stages=nn.ModuleList()
        # Upstream layer builder assumes three input channels per call; build
        # the entire VGG feature chain once, then split at its pooling layers.
        cfg=[width,width,"M",2*width,2*width,"M",4*width,4*width,4*width,"M",
             8*width,8*width,8*width,"M",8*width,8*width,8*width,"M"]
        current=[]
        for layer in make_layers(cfg):
            current.append(layer)
            if isinstance(layer,nn.MaxPool2d):
                self.stages.append(nn.Sequential(*current)); current=[]
        self.fc=nn.Sequential(nn.Conv2d(8*width,fc_width,7,padding=3),nn.ReLU(),nn.Dropout2d(.1),
                              nn.Conv2d(fc_width,fc_width,1),nn.ReLU(),nn.Dropout2d(.1),nn.Conv2d(fc_width,2,1))
        self.score3, self.score4 = nn.Conv2d(4*width,2,1), nn.Conv2d(8*width,2,1)
        self.up2a=nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.up2b=nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.up8=nn.ConvTranspose2d(2,2,16,8,4,bias=False)
        # FCN initializes learned upsampling to channel-wise bilinear kernels.
        for layer in (self.up2a,self.up2b,self.up8):
            kernel=layer.kernel_size[0]
            factor=(kernel+1)//2
            center=factor-1 if kernel%2 else factor-.5
            line=1-(torch.arange(kernel)-center).abs()/factor
            with torch.no_grad():
                layer.weight.zero_()
                for channel in range(2):layer.weight[channel,channel].copy_(line[:,None]*line[None,:])

    def forward(self,x):
        x=x.repeat(1,3,1,1) if x.shape[1]==1 else x
        features=[]
        for stage in self.stages:
            x=stage(x);features.append(x)
        y=self.up2a(self.fc(features[4])) + self.score4(features[3])
        y=self.up2b(y)+self.score3(features[2])
        return self.up8(y)


class NestedUNet(nn.Module):
    def __init__(self,width=8,depth=3):
        super().__init__()
        self.depth=depth
        self.widths=[width*2**i for i in range(depth+1)]
        self.nodes=nn.ModuleDict()
        for i in range(depth+1):
            self.nodes[f"{i}_0"]=ConvPair(2,1 if i==0 else self.widths[i-1],self.widths[i],True)
        for j in range(1,depth+1):
            for i in range(depth+1-j):
                self.nodes[f"{i}_{j}"]=ConvPair(2,j*self.widths[i]+self.widths[i+1],self.widths[i],True)
        self.heads=nn.ModuleList([nn.Conv2d(width,2,1) for _ in range(depth)])

    def all_outputs(self,x):
        nodes={(0,0):self.nodes["0_0"](x)}
        for i in range(1,self.depth+1):
            nodes[i,0]=self.nodes[f"{i}_0"](F.max_pool2d(nodes[i-1,0],2))
        for j in range(1,self.depth+1):
            for i in range(self.depth+1-j):
                inputs=[nodes[i,k] for k in range(j)]
                inputs.append(F.interpolate(nodes[i+1,j-1],size=nodes[i,0].shape[2:],mode="bilinear",align_corners=True))
                nodes[i,j]=self.nodes[f"{i}_{j}"](torch.cat(inputs,1))
        return [head(nodes[0,j+1]) for j,head in enumerate(self.heads)]

    def forward(self,x):
        return self.all_outputs(x)[-1]

    def training_loss(self,batch):
        outputs=self.all_outputs(batch["image"])
        return sum(F.cross_entropy(y,batch["mask"]) for y in outputs)/len(outputs)


class Genesis(nn.Module):
    def __init__(self,width=4,shuffle_blocks=16):
        super().__init__()
        self.network=UNetND(3,width,2,classes=1,norm=True)
        self.shuffle_blocks=shuffle_blocks

    @staticmethod
    def nonlinear(x):
        # A cubic Bezier map; sorting both axes gives a monotone branch,
        # sorting only the input axis gives the intensity-reversal branch.
        t=torch.linspace(0,1,256,device=x.device,dtype=x.dtype)
        ctrl=torch.cat([x.new_zeros(1,2),torch.rand(2,2,device=x.device),x.new_ones(1,2)])
        basis=torch.stack([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3])
        curve=ctrl.T@basis
        u=curve[0].sort().values
        v=curve[1].sort().values if torch.rand(())<.5 else curve[1]
        j=torch.searchsorted(u.contiguous(),x.contiguous()).clamp(1,len(t)-1)
        alpha=(x-u[j-1])/(u[j]-u[j-1]+1e-6)
        return v[j-1]+alpha*(v[j]-v[j-1])

    def corrupt(self,x,mode=None):
        y=x.clone()
        for n in range(len(y)):
            z=y[n]
            # The source uses 10,000 tiny blocks. The CPU preset uses 16,
            # and at least two voxels/axis so 16-cube shuffling is nontrivial.
            for _ in range(self.shuffle_blocks):
                lengths=[max(2,d//8) for d in z.shape[1:]]
                starts=[int(torch.randint(0,d-k+1,())) for d,k in zip(z.shape[1:],lengths)]
                sl=(slice(None),)+tuple(slice(s,s+k) for s,k in zip(starts,lengths))
                p=z[sl].flatten();z[sl]=p[torch.randperm(p.numel(),device=z.device)].reshape(z[sl].shape)
            z=self.nonlinear(z)
            chosen=mode or ("inpaint" if torch.rand(())<.5 else "outpaint")
            if chosen not in {"inpaint","outpaint"}:
                raise ValueError("mode must be inpaint or outpaint")
            lengths=[max(2,d//2) for d in z.shape[1:]]
            starts=[int(torch.randint(0,d-k+1,())) for d,k in zip(z.shape[1:],lengths)]
            sl=(slice(None),)+tuple(slice(s,s+k) for s,k in zip(starts,lengths))
            if chosen=="inpaint":
                z[sl]=torch.rand_like(z[sl])
            else:
                canvas=torch.rand_like(z);canvas[sl]=z[sl];z=canvas
            y[n]=z
        return y

    def forward(self,x):
        return self.network(x).sigmoid()

    def training_loss(self,batch):
        clean=batch["volume"]
        return F.mse_loss(self(self.corrupt(clean)),clean)


class SequenceSSM(nn.Module):
    def __init__(self,kind,d_model=16,d_state=8):
        super().__init__()
        self.kind=kind
        self.input=nn.Linear(4,d_model)
        self.norm=nn.LayerNorm(d_model)
        if kind=="s4":
            from vendor.image_s4.s4 import S4Block
            self.mixer=S4Block(d_model=d_model,d_state=d_state,mode="dplr",init="legs",backend="naive",transposed=False,dropout=0.)
        else:
            from vendor.image_mamba.mamba_simple import Mamba
            self.mixer=Mamba(d_model=d_model,d_state=d_state,use_fast_path=False)
        self.head=nn.Linear(d_model,3)

    def forward(self,x):
        x=self.input(x)
        y=self.mixer(self.norm(x))
        if isinstance(y,tuple): y=y[0]
        return self.head((x+y).mean(1))

    def training_loss(self,batch):
        return F.cross_entropy(self(batch["sequence"]),batch["sequence_label"])


class SS2D(nn.Module):
    """A compact, explicit four-direction VMamba-v0-style selective scan."""
    def __init__(self,width=16,d_state=4):
        super().__init__()
        self.width,self.d_state=width,d_state
        self.in_proj=nn.Conv2d(width,width*2,1)
        self.conv=nn.Conv2d(width,width,3,padding=1,groups=width)
        self.select=nn.ModuleList([nn.Linear(width,width+2*d_state) for _ in range(4)])
        self.A_log=nn.Parameter(torch.log(torch.arange(1,d_state+1.).repeat(4,width,1)))
        self.D=nn.Parameter(torch.ones(4,width))
        self.norm=nn.LayerNorm(width)
        self.out=nn.Conv2d(width,width,1)

    @staticmethod
    def scan_orders(x):
        a=x.flatten(2)
        b=x.transpose(2,3).flatten(2)
        return [a,b,a.flip(-1),b.flip(-1)]

    @staticmethod
    def restore_orders(ys,height,width):
        return [ys[0].reshape(*ys[0].shape[:2],height,width),
                ys[1].reshape(*ys[1].shape[:2],width,height).transpose(2,3),
                ys[2].flip(-1).reshape(*ys[2].shape[:2],height,width),
                ys[3].flip(-1).reshape(*ys[3].shape[:2],width,height).transpose(2,3)]

    def forward(self,x):
        from vendor.image_mamba.selective_scan import selective_scan_ref
        u,z=self.in_proj(x).chunk(2,1)
        u=F.silu(self.conv(u));ys=[]
        for k,s in enumerate(self.scan_orders(u)):
            select=self.select[k](s.transpose(1,2))
            dt,b,c=torch.split(select,[self.width,self.d_state,self.d_state],-1)
            ys.append(selective_scan_ref(s,dt.transpose(1,2),-self.A_log[k].exp(),b.transpose(1,2),c.transpose(1,2),self.D[k],delta_softplus=True))
        y=sum(self.restore_orders(ys,*x.shape[2:]))
        y=self.norm(y.permute(0,2,3,1)).permute(0,3,1,2)
        return self.out(y*F.silu(z))


class VMambaMechanism(nn.Module):
    def __init__(self,width=16,d_state=4):
        super().__init__()
        self.stem=nn.Conv2d(1,width,4,4)
        self.scan=SS2D(width,d_state)
        self.head=nn.Linear(width,3)

    def forward(self,x):
        x=self.stem(x)
        return self.head((x+self.scan(x)).mean((2,3)))


class SpatialMamba(nn.Module):
    def __init__(self,width,d_state=4,three_orders=False):
        super().__init__()
        from vendor.image_mamba.mamba_simple import Mamba
        self.norm=nn.LayerNorm(width)
        self.three_orders=three_orders
        self.mixers=nn.ModuleList([Mamba(width,d_state=d_state,use_fast_path=False) for _ in range(3 if three_orders else 1)])

    def forward(self,x):
        orders=[(0,1,2,3,4),(0,1,3,4,2),(0,1,4,2,3)] if self.three_orders else [(0,1,2,3,4)]
        outputs=[]
        for order,mixer in zip(orders,self.mixers):
            u=x.permute(order).contiguous()
            y=mixer(self.norm(u.flatten(2).transpose(1,2))).transpose(1,2).reshape(u.shape)
            inverse=tuple(order.index(i) for i in range(5))
            outputs.append(y.permute(inverse))
        return x+sum(outputs)/len(outputs)


class SpatialGSC(nn.Module):
    def __init__(self,width):
        super().__init__()
        def unit(kernel):
            return nn.Sequential(nn.Conv3d(width,width,kernel,padding=kernel//2),nn.GroupNorm(1,width),nn.ReLU())
        self.spatial=nn.Sequential(unit(3),unit(3))
        self.point=unit(1)
        self.fuse=unit(1)

    def forward(self,x):
        return x+self.fuse(self.spatial(x)+self.point(x))


class UNetMambaMechanism(nn.Module):
    def __init__(self,width=4,d_state=4,segmamba=False):
        super().__init__()
        self.unet=UNetND(3,width,2,norm=True,residual=True)
        self.spatial=SpatialGSC(width*4) if segmamba else nn.Identity()
        self.mamba=SpatialMamba(width*4,d_state,three_orders=segmamba)

    def forward(self,x):
        features=self.unet.encode(x)
        features[-1]=self.mamba(self.spatial(features[-1]))
        return self.unet.decode(features)


class TokenTransformer(nn.Module):
    def __init__(self,d_model=32,depth=2,num_heads=4,vocab_size=32):
        super().__init__()
        self.d_model,self.vocab_size=d_model,vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.transformer=nn.Transformer(d_model=d_model,nhead=num_heads,num_encoder_layers=depth,num_decoder_layers=depth,
                                       dim_feedforward=2*d_model,dropout=0.,batch_first=True)
        self.head=nn.Linear(d_model,vocab_size)

    def encode_tokens(self,x):
        p=torch.arange(x.shape[1],device=x.device).float()[:,None]
        frequency=torch.exp(torch.arange(0,self.d_model,2,device=x.device)*(-math.log(10000.)/self.d_model))
        pos=torch.zeros(x.shape[1],self.d_model,device=x.device)
        pos[:,0::2]=torch.sin(p*frequency);pos[:,1::2]=torch.cos(p*frequency)
        return self.embedding(x)*math.sqrt(self.d_model)+pos[None]

    def forward(self,source,decoder_input=None):
        if decoder_input is None:
            # An explicitly autoregressive inference path (no target tokens).
            generated=torch.ones(source.shape[0],1,dtype=torch.long,device=source.device)
            logits=[]
            for _ in range(source.shape[1]):
                y=self.forward(source,generated)[:,-1]
                logits.append(y);generated=torch.cat([generated,y.argmax(-1,keepdim=True)],1)
            return torch.stack(logits,1)
        mask=torch.triu(torch.ones(decoder_input.shape[1],decoder_input.shape[1],device=source.device,dtype=torch.bool),diagonal=1)
        return self.head(self.transformer(self.encode_tokens(source),self.encode_tokens(decoder_input),tgt_mask=mask))

    def training_loss(self,batch):
        target=batch["tokens"].flip(1)
        decoder=torch.cat([torch.ones_like(target[:,:1]),target[:,:-1]],1)
        logits=self(batch["tokens"],decoder)
        return F.cross_entropy(logits.reshape(-1,self.vocab_size),target.reshape(-1))


class MaskedSetSegmentation(nn.Module):
    def __init__(self,width=32,num_queries=4,depth=3):
        super().__init__()
        self.width,self.num_queries,self.depth=width,num_queries,depth
        self.pixels=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1))
        self.queries=nn.Parameter(torch.randn(num_queries,width)*.02)
        self.cross=nn.ModuleList([nn.MultiheadAttention(width,4,batch_first=True) for _ in range(depth)])
        self.self_attn=nn.ModuleList([nn.MultiheadAttention(width,4,batch_first=True) for _ in range(depth)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3*depth)])
        self.ff=nn.ModuleList([nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width)) for _ in range(depth)])
        self.mask_embed=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width))
        self.class_head=nn.Linear(width,3)  # background, foreground, no-object

    @staticmethod
    def attention_mask(mask_logits,size,heads=4):
        mask=F.interpolate(mask_logits,size=size,mode="bilinear",align_corners=False).flatten(2).sigmoid()<.5
        mask=mask.detach()
        # A fully blocked row produces NaNs in softmax. The original decoder
        # releases such rows before attention; preserve that behavior here.
        mask=mask & ~mask.all(-1,keepdim=True)
        return mask[:,None].expand(-1,heads,-1,-1).reshape(-1,mask.shape[1],mask.shape[2])

    def predictions(self,x):
        pixels=self.pixels(x)
        q=self.queries[None].expand(x.shape[0],-1,-1)
        masks=torch.einsum("bqc,bchw->bqhw",self.mask_embed(q),pixels)
        for i in range(self.depth):
            scale=2**(1+i%3)
            memory=F.avg_pool2d(pixels,scale)
            block=self.attention_mask(masks,memory.shape[2:])
            memory=memory.flatten(2).transpose(1,2)
            update=self.cross[i](q,memory,memory,attn_mask=block,need_weights=False)[0]
            q=self.norms[3*i](q+update)
            q=self.norms[3*i+1](q+self.self_attn[i](q,q,q,need_weights=False)[0])
            q=self.norms[3*i+2](q+self.ff[i](q))
            masks=torch.einsum("bqc,bchw->bqhw",self.mask_embed(q),pixels)
        return self.class_head(q),masks

    def forward(self,x):
        classes,masks=self.predictions(x)
        scores=torch.einsum("bqc,bqhw->bchw",classes.softmax(-1)[...,:2],masks.sigmoid())
        return scores.clamp_min(1e-6).log()

    def set_loss(self,class_logits,mask_logits,targets):
        from scipy.optimize import linear_sum_assignment
        loss=class_logits.sum()*0
        for classes,masks,target in zip(class_logits,mask_logits,targets):
            labels=target.unique()
            gt=torch.stack([(target==label).float() for label in labels])
            flat=masks.flatten(1);g=gt.flatten(1)
            p=flat.sigmoid()
            # Pairwise BCE and soft Dice, without allocating Q*T*H*W tensors.
            bce=F.softplus(flat).mean(-1)[:,None]-(flat@g.T)/flat.shape[-1]
            dice=1-(2*(p@g.T)+1)/(p.sum(-1)[:,None]+g.sum(-1)[None]+1)
            cost=-classes.softmax(-1)[:,labels]+bce+dice
            a,b=linear_sum_assignment(cost.detach().cpu().numpy())
            a=torch.as_tensor(a,device=classes.device);b=torch.as_tensor(b,device=classes.device)
            cls_target=torch.full((len(classes),),2,device=classes.device,dtype=torch.long)
            cls_target[a]=labels[b]
            weight=classes.new_tensor([1.,1.,.1])
            loss=loss+F.cross_entropy(classes,cls_target,weight=weight)+F.binary_cross_entropy_with_logits(masks[a],gt[b])+dice[a,b].mean()
        return loss/len(targets)

    def training_loss(self,batch):
        return self.set_loss(*self.predictions(batch["image"]),batch["mask"])


class DenseFeatureBlock(nn.Module):
    def __init__(self,dim,in_channels,width,layers=2):
        super().__init__()
        conv=nn.Conv2d if dim==2 else nn.Conv3d
        self.layers=nn.ModuleList([nn.Sequential(nn.GroupNorm(1,in_channels+i*width),nn.ReLU(),conv(in_channels+i*width,width,3,padding=1)) for i in range(layers)])
        self.compress=conv(in_channels+layers*width,width,1)

    def forward(self,x):
        for layer in self.layers:x=torch.cat([x,layer(x)],1)
        return self.compress(x)


class TinyDenseUNet(nn.Module):
    def __init__(self,dim,in_channels,width):
        super().__init__()
        self.dim=dim
        self.enc=DenseFeatureBlock(dim,in_channels,width)
        self.low=DenseFeatureBlock(dim,width,2*width)
        self.dec=DenseFeatureBlock(dim,3*width,width)

    def forward(self,x):
        skip=self.enc(x)
        pool=F.avg_pool2d if self.dim==2 else F.avg_pool3d
        y=self.low(pool(skip,2))
        y=F.interpolate(y,size=skip.shape[2:],mode="nearest")
        return self.dec(torch.cat([skip,y],1))


class HybridDenseUNet(nn.Module):
    def __init__(self,width=4):
        super().__init__()
        self.in_slice=TinyDenseUNet(2,1,width)
        self.inter_slice=TinyDenseUNet(3,width+1,width)
        self.fusion=nn.Sequential(nn.Conv3d(2*width,width,3,padding=1),nn.ReLU(),nn.Conv3d(width,2,1))

    def forward(self,x):
        b,c,d,h,w=x.shape
        slices=x.permute(0,2,1,3,4).reshape(b*d,c,h,w)
        f2=self.in_slice(slices).reshape(b,d,-1,h,w).permute(0,2,1,3,4)
        f3=self.inter_slice(torch.cat([x,f2],1))
        return self.fusion(torch.cat([f2,f3],1))


class DistilledViT(nn.Module):
    def __init__(self,config):
        super().__init__()
        from timm.models.deit import VisionTransformerDistilled
        self.network=VisionTransformerDistilled(img_size=config["image_size"],patch_size=config["patch_size"],in_chans=1,num_classes=3,
                                               embed_dim=config["hidden_size"],depth=config["depth"],num_heads=config["num_heads"],mlp_ratio=2.)
        self.network.set_distilled_training(True)

    def forward(self,x):
        y=self.network(x)
        return (y[0]+y[1])/2 if isinstance(y,tuple) else y

    @staticmethod
    def synthetic_teacher(image):
        # Input-only analytic teacher for these declared geometric phantoms.
        # It is deliberately visible, and is not a pretrained vision model.
        mask=image[:,0]>.45
        height=mask.any(-1).sum(-1).clamp_min(1)
        width=mask.any(-2).sum(-1).clamp_min(1)
        ratio=mask.sum((-1,-2))/(height*width)
        prototypes=ratio.new_tensor([math.pi/4,1.,.5])
        return -80*(ratio[:,None]-prototypes[None]).square()

    def training_loss(self,batch):
        if not self.training:
            return F.cross_entropy(self(batch["image"]),batch["label"])
        cls,dist=self.network(batch["image"])
        with torch.no_grad():
            teacher=batch.get("teacher_logits")
            if teacher is None: teacher=self.synthetic_teacher(batch["image"])
        return .5*F.cross_entropy(cls,batch["label"])+.5*F.cross_entropy(dist,teacher.argmax(-1))


class SegFormerAdapter(nn.Module):
    def __init__(self,config):
        super().__init__()
        from vendor.image_segformer.mix_transformer import MixVisionTransformer
        from vendor.image_segformer.segformer_head import SegFormerHead
        widths=config["widths"]
        self.encoder=MixVisionTransformer(img_size=config["image_size"],in_chans=1,embed_dims=widths,num_heads=[1,2,4,8],
                                          mlp_ratios=[2]*4,depths=config["depths"],sr_ratios=[8,4,2,1],qkv_bias=True)
        self.decoder=SegFormerHead(feature_strides=[4,8,16,32],in_channels=widths,num_classes=2,decoder_params={"embed_dim":config["decoder_width"]})

    def forward(self,x):
        return F.interpolate(self.decoder(self.encoder(x)),size=x.shape[2:],mode="bilinear",align_corners=False)


def build_model(num: int, config: dict | None = None) -> nn.Module:
    """Build the declared lesson. Unknown/unavailable IDs fail explicitly."""
    if num in UNAVAILABLE:
        raise NotImplementedError(UNAVAILABLE[num])
    if num not in SPECS:
        raise KeyError(f"Image lesson {num} is not registered")
    c={**SPECS[num]["config"], **(config or {})}
    task=SPECS[num]["task"]
    if num in {1,2,3,4,21,22,81,82,83,85}:
        import torchvision.models as tv
        if num==1:
            net=tv.alexnet(weights=None,num_classes=3)
            net.avgpool=nn.AdaptiveAvgPool2d((1,1))
            fc=c["fc_width"]
            net.classifier=nn.Sequential(nn.Dropout(.1),nn.Linear(256,fc),nn.ReLU(),nn.Dropout(.1),nn.Linear(fc,fc),nn.ReLU(),nn.Linear(fc,3))
        elif num==2:net=tv.resnet18(weights=None,num_classes=3)
        elif num==3:
            net=tv.DenseNet(growth_rate=c["growth_rate"],block_config=tuple(c["block_config"]),num_init_features=c["initial_features"],bn_size=2,num_classes=3)
        elif num==4:
            from torchvision.models.convnext import ConvNeXt,CNBlockConfig
            widths=c["widths"]
            setting=[CNBlockConfig(w,widths[i+1] if i<3 else None,c["depths"][i]) for i,w in enumerate(widths)]
            net=ConvNeXt(setting,num_classes=3,stochastic_depth_prob=0.)
        elif num==21:
            net=tv.VisionTransformer(image_size=c["image_size"],patch_size=c["patch_size"],num_layers=c["depth"],num_heads=c["num_heads"],hidden_dim=c["hidden_size"],mlp_dim=2*c["hidden_size"],num_classes=3)
        elif num==22:
            net=tv.SwinTransformer(patch_size=[2,2],embed_dim=c["width"],depths=c["depths"],num_heads=[1,2,4,8],window_size=[c["window_size"]]*2,num_classes=3,stochastic_depth_prob=0.)
        elif num==81:
            from torchvision.models.vgg import make_layers,cfgs
            cfg=cfgs["E" if c["variant"]==19 else "D"]
            narrow=[max(c["width"],int(v*c["width"]/64)) if isinstance(v,int) else v for v in cfg]
            features=make_layers(narrow)
            fc=c["fc_width"]
            net=nn.Sequential(features,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(8*c["width"],fc),nn.ReLU(),nn.Dropout(.1),nn.Linear(fc,fc),nn.ReLU(),nn.Dropout(.1),nn.Linear(fc,3))
        elif num==82:net=tv.googlenet(weights=None,num_classes=3,aux_logits=False,init_weights=False,dropout=0.)
        elif num==83:
            from torchvision.models.resnet import ResNet,Bottleneck
            net=ResNet(Bottleneck,[1,1,1,1],groups=c["groups"],width_per_group=c["width_per_group"],num_classes=3)
        else:
            from torchvision.models.efficientnet import EfficientNet,_efficientnet_conf
            setting,last=_efficientnet_conf("efficientnet_b0",width_mult=c["width_mult"],depth_mult=c["depth_mult"])
            net=EfficientNet(setting,dropout=0.,stochastic_depth_prob=0.,num_classes=3,last_channel=last)
        return TaskAdapter(net,rgb=True,size=c.get("image_size"))
    if num==84:
        import timm
        return TaskAdapter(timm.create_model("seresnet18",pretrained=False,in_chans=1,num_classes=3))
    if num==5:return TaskAdapter(FCN8s(c["width"],c["fc_width"]),task)
    if num in {6,7}:return TaskAdapter(UNetND(2 if num==6 else 3,c["width"],c["depth"],norm=num==7),task)
    if num in {8,10,11,87}:
        from monai.networks import nets
        if num==8:net=nets.VNet(spatial_dims=3,in_channels=1,out_channels=2,dropout_prob_down=0.,dropout_prob_up=(None,0.))
        elif num==10:net=nets.UNETR(in_channels=1,out_channels=2,img_size=(c["volume_size"],)*3,feature_size=c["feature_size"],hidden_size=c["hidden_size"],mlp_dim=c["mlp_dim"],num_heads=c["num_heads"],norm_name=("group",{"num_groups":1}))
        elif num==11:net=nets.SwinUNETR(img_size=(c["volume_size"],)*3,in_channels=1,out_channels=2,feature_size=c["feature_size"],depths=c["depths"],norm_name=("group",{"num_groups":1}),use_checkpoint=False)
        else:net=nets.AttentionUnet(spatial_dims=2,in_channels=1,out_channels=2,channels=(c["width"],2*c["width"],4*c["width"]),strides=(2,2))
        return TaskAdapter(net,task,size=c.get("volume_size"),dice=num==8)
    if num==12:
        from vendor.image_mednext.MedNextV1 import MedNeXt
        net=MedNeXt(in_channels=1,n_channels=c["width"],n_classes=2,exp_r=c["expansion"],kernel_size=c["kernel_size"],deep_supervision=False,do_res=True,do_res_up_down=True,block_counts=[1]*9,dim="3d")
        return TaskAdapter(net,task,size=c["volume_size"])
    if num==13:return Genesis(c["width"],c["shuffle_blocks"])
    if num in {15,16}:return SequenceSSM("s4" if num==15 else "mamba",c["d_model"],c["d_state"])
    if num==17:return TaskAdapter(VMambaMechanism(c["width"],c["d_state"]))
    if num in {18,19}:return TaskAdapter(UNetMambaMechanism(c["width"],c["d_state"],num==19),task)
    if num==20:return TokenTransformer(c["d_model"],c["depth"],c["num_heads"],c["vocab_size"])
    if num==23:return MaskedSetSegmentation(c["width"],c["num_queries"],c["depth"])
    if num==86:return NestedUNet(c["width"],c["depth"])
    if num==88:return TaskAdapter(HybridDenseUNet(c["width"]),task)
    if num==89:return DistilledViT(c)
    if num==90:
        from vendor.image_pvt.pvt import PyramidVisionTransformer
        net=PyramidVisionTransformer(img_size=c["image_size"],patch_size=4,in_chans=1,num_classes=3,embed_dims=c["widths"],depths=c["depths"],num_heads=[1,2,4,8],mlp_ratios=[2]*4,sr_ratios=[8,4,2,1])
        return TaskAdapter(net)
    if num==91:return TaskAdapter(SegFormerAdapter(c),task)
    raise AssertionError(f"Missing constructor for declared image lesson {num}")
