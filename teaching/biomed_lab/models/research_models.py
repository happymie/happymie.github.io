"""Small self-supervised, graph, and waveform experiments.

Open-source components are reused for MAE, graph convolutions, recurrent cells,
and TCN. Other losses are independent, readable implementations of paper
equations with explicitly declared reductions; none supplies foundation weights.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


def _spec(name,task,mechanism,source,limitations,implementation='mechanism_demo',**config):
    return dict(name=name,task=task,implementation=implementation,source_url=source,mechanism=mechanism,
                limitations=limitations,config=config or {'width':32})


SPECS={
24:_spec('SimCLR','self_supervised','Two independent views, nonlinear projection and symmetric NT-Xent; self-similarities excluded.',
 'https://github.com/google-research/simclr',['Small patch Transformer replaces the original ResNet; grayscale synthetic augmentations; no ImageNet-scale batch/training.'],width=32,temperature=0.2),
25:_spec('MoCo','self_supervised','Gradient-trained query encoder, stopped EMA key encoder, and a FIFO negative queue.',
 'https://github.com/facebookresearch/moco',['Paper-equation CPU implementation; no copying of upstream code with unresolved license. Small encoder/queue, no distributed shuffle BN.'],width=32,queue_size=64,momentum=0.99,temperature=0.2),
26:_spec('MAE','masked_reconstruction','Author encoder sees only visible patches; a lightweight decoder reconstructs masked pixels.',
 'https://github.com/facebookresearch/mae',['Author source, CC BY-NC-4.0. Shrunk image/patch/width/depth; no pretrained weights. RGB obtained by repeating the grayscale teaching image.'],implementation='upstream',width=64,depth=2,decoder_width=32,mask_ratio=0.75),
27:_spec('DINO','self_supervised','Cross-view student–teacher distillation with stopped EMA teacher, centering and sharpened teacher probabilities.',
 'https://github.com/facebookresearch/dino',['Independent two-view mechanism implementation, not copied author DINO code. No multi-crop schedule, weight-normalized final layer or original training recipe.'],width=32,momentum=0.99,prototypes=16),
28:_spec('iBOT','self_supervised','Global distillation plus patch-token distillation from unmasked teacher to masked student.',
 'https://github.com/bytedance/ibot',['Shared-head two-view mechanism exercise; simplified mask sampler/encoder and no original training schedule or weights.'],width=32,momentum=0.99,prototypes=16),
29:_spec('DINOv2','self_supervised','DINO-style global and masked patch distillation with a nearest-neighbor entropy regularizer.',
 'https://github.com/facebookresearch/dinov2',['Mechanism subset only: local mini objectives do not reproduce DINOv2 data curation, Sinkhorn teacher balancing, untied heads, large training or learned representations.'],width=32,momentum=0.99,prototypes=16),
92:_spec('BEiT','masked_reconstruction','Masked patch prediction uses discrete target tokens instead of pixels.',
 'https://github.com/microsoft/unilm/tree/master/beit',['Teaching tokenizer is fixed patch-intensity quantization, NOT the pretrained dVAE tokenizer. Demonstrates the discrete-target objective only; no claim of BEiT architecture or result equivalence.'],width=32,vocabulary=8),
93:_spec('BYOL','self_supervised','Online encoder/projector/predictor regresses to an EMA target, with target stop-gradient and symmetric views.',
 'https://github.com/google-deepmind/deepmind-research/tree/master/byol',['Readable CPU mechanism port replaces author JAX training; tiny patch encoder, no BN or original augmentations/schedules.'],width=32,momentum=0.99),
94:_spec('SwAV','self_supervised','Entropy-regularized balanced Sinkhorn assignments from one view supervise prototype scores in the other.',
 'https://github.com/facebookresearch/swav',['Small two-view experiment; no multi-crop queue/distributed training. Tiny batches are an objective exercise, not evidence of learned transfer.'],width=32,prototypes=8),
75:_spec('GCN','graph','Symmetrically degree-normalized neighbor aggregation with added self-loops.',
 'https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GCNConv.html',['Direct PyG GCNConv; independently sampled small graphs replace citation/medical datasets.'],implementation='upstream',width=16),
76:_spec('GAT','graph','Masked additive attention over each node neighborhood, with multiple attention heads.',
 'https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.GATConv.html',['Direct PyG GATConv; two heads/two layers on small graphs. No paper benchmark claim.'],implementation='upstream',width=8,heads=2),
132:_spec('GraphSAGE','graph','A learned root transform and mean-neighbor aggregation support inductive node prediction.',
 'https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.SAGEConv.html',['Direct PyG SAGEConv; uses all neighbors on tiny graphs instead of stochastic large-graph sampling.'],implementation='upstream',width=16),
133:_spec('Graphormer','graph','Graph attention receives shortest-path spatial bias and degree/centrality embeddings.',
 'https://github.com/microsoft/Graphormer',['Independent mechanism subset: no multi-hop edge-feature encoding, graph-token objective, original training or medical mesh decoder.'],width=32,heads=4),
77:_spec('LSTM','sequence','Input/forget/output gates and an additive memory cell carry information across time.',
 'https://pytorch.org/docs/2.5/generated/torch.nn.LSTM.html',['Direct modern PyTorch LSTM includes forget gates introduced after the 1997 paper; this historical difference is explicit. Synthetic waves are not ECG/EEG.'],implementation='upstream',width=16,layers=1),
78:_spec('Temporal Convolutional Network (TCN)','sequence','Author residual causal convolutions use exponentially increasing dilation and weight normalization.',
 'https://github.com/locuslab/TCN',['Unmodified author TemporalConvNet with 3 small channel stages and a new task head. Synthetic waves; no paper benchmark.'],implementation='upstream',width=16,levels=3,kernel_size=3),
136:_spec('GRU','sequence','Update and reset gates control recurrent state without a separate cell state.',
 'https://pytorch.org/docs/2.5/generated/torch.nn.GRU.html',['Direct PyTorch GRU applies reset after the recurrent linear map; the Cho et al. original formulation places it before. Not claimed bitwise equivalent to that original equation.'],implementation='upstream',width=16,layers=1),
}


class Encoder(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        if width%4:raise ValueError('width must be divisible by 4')
        self.patch=nn.Conv2d(1,width,4,4)
        self.cls=nn.Parameter(torch.zeros(1,1,width));self.pos=nn.Parameter(torch.randn(1,65,width)*0.02)
        self.mask=nn.Parameter(torch.zeros(1,1,width))
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True,norm_first=True,activation='gelu')
        self.transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
    def forward(self,x,mask=None,tokens=False):
        x=F.interpolate(x,size=(32,32),mode='bilinear',align_corners=False)
        patches=self.patch(x).flatten(2).transpose(1,2)
        if mask is not None: patches=torch.where(mask[:,:,None],self.mask,patches)
        seq=torch.cat([self.cls.expand(x.shape[0],-1,-1),patches],dim=1)+self.pos
        out=self.norm(self.transformer(seq))
        return out if tokens else out[:,0]


def _views(image):
    def one(x):
        if torch.rand(())>0.5: x=x.flip(-1)
        return (x+torch.randn_like(x)*0.025)*torch.empty(x.shape[0],1,1,1,device=x.device).uniform_(0.85,1.15)
    return one(image),one(image)


def nt_xent(a,b,temperature=0.2):
    n=a.shape[0]
    if n<2:raise ValueError('Contrastive experiments require at least two specimens')
    z=F.normalize(torch.cat([a,b]),dim=-1)
    logits=z@z.T/temperature
    logits=logits.masked_fill(torch.eye(2*n,device=z.device,dtype=torch.bool),float('-inf'))
    labels=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(logits,labels)


class Contrastive(nn.Module):
    def __init__(self,width,temp):
        super().__init__();self.encoder=Encoder(width)
        self.projector=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width))
        self.temperature=temp
    def forward(self,x):return self.projector(self.encoder(x))
    def training_loss(self,b):
        a,c=_views(b['image']);return nt_xent(self(a),self(c),self.temperature)


@torch.no_grad()
def ema(online,target,momentum):
    for p,q in zip(online.parameters(),target.parameters()):q.mul_(momentum).add_(p,alpha=1-momentum)


class MoCo(nn.Module):
    def __init__(self,width,k,momentum,temp):
        super().__init__();self.online=Contrastive(width,temp);self.target=copy.deepcopy(self.online).requires_grad_(False)
        self.register_buffer('queue',F.normalize(torch.randn(k,width),dim=-1));self.register_buffer('pointer',torch.zeros((),dtype=torch.long))
        self.momentum=momentum;self.temperature=temp;self.pending=None
    def training_loss(self,b):
        a,c=_views(b['image']);q=F.normalize(self.online(a),dim=-1)
        with torch.no_grad():k=F.normalize(self.target(c),dim=-1)
        positive=(q*k).sum(-1,keepdim=True)
        logits=torch.cat([positive,q@self.queue.detach().clone().T],dim=1)/self.temperature
        if self.training:self.pending=k.detach()
        return F.cross_entropy(logits,torch.zeros(q.shape[0],device=q.device,dtype=torch.long))
    @torch.no_grad()
    def after_optimizer_step(self):
        ema(self.online,self.target,self.momentum)
        if self.pending is not None:
            for value in self.pending:
                self.queue[int(self.pointer)]=value;self.pointer.add_(1).remainder_(len(self.queue))
        self.pending=None


class MAE(nn.Module):
    def __init__(self,c):
        super().__init__()
        from vendor.research_mae.models_mae import MaskedAutoencoderViT
        self.model=MaskedAutoencoderViT(img_size=32,patch_size=4,in_chans=3,embed_dim=c['width'],depth=c['depth'],num_heads=4,
                    decoder_embed_dim=c['decoder_width'],decoder_depth=1,decoder_num_heads=4,mlp_ratio=2.)
        self.ratio=c['mask_ratio']
    def training_loss(self,b):
        x=F.interpolate(b['image'],size=(32,32),mode='bilinear',align_corners=False).repeat(1,3,1,1)
        return self.model(x,mask_ratio=self.ratio)[0]


class Bootstrap(nn.Module):
    def __init__(self,width,momentum):
        super().__init__();self.online=Contrastive(width,0.2);self.target=copy.deepcopy(self.online).requires_grad_(False)
        self.predictor=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,width));self.momentum=momentum
    def training_loss(self,b):
        a,c=_views(b['image'])
        pa=F.normalize(self.predictor(self.online(a)),dim=-1);pc=F.normalize(self.predictor(self.online(c)),dim=-1)
        with torch.no_grad():ta=F.normalize(self.target(a),dim=-1);tc=F.normalize(self.target(c),dim=-1)
        return ((2-2*(pa*tc).sum(-1))+(2-2*(pc*ta).sum(-1))).mean()/2
    @torch.no_grad()
    def after_optimizer_step(self):ema(self.online,self.target,self.momentum)


def koleo(z):
    z=F.normalize(z,dim=-1);distance=torch.cdist(z,z)
    distance=distance+torch.eye(len(z),device=z.device)*10
    return -torch.log(distance.min(dim=-1).values.clamp_min(1e-6)).mean()


class Distillation(nn.Module):
    def __init__(self,width,momentum,prototypes,patch=False,entropy=False):
        super().__init__();self.student=Encoder(width);self.teacher=copy.deepcopy(self.student).requires_grad_(False)
        self.head=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,prototypes))
        self.teacher_head=copy.deepcopy(self.head).requires_grad_(False)
        self.register_buffer('center',torch.zeros(1,prototypes));self.momentum=momentum
        self.patch=patch;self.entropy=entropy;self.pending_center=None
    def training_loss(self,b):
        a,c=_views(b['image']);mask=torch.rand(len(a),64,device=a.device)<0.4 if self.patch else None
        sa=self.student(a,mask,tokens=True);sc=self.student(c,mask,tokens=True)
        with torch.no_grad():
            ta=self.teacher_head(self.teacher(a,tokens=True));tc=self.teacher_head(self.teacher(c,tokens=True))
            qa=F.softmax((ta-self.center)/0.04,dim=-1);qc=F.softmax((tc-self.center)/0.04,dim=-1)
        la=F.log_softmax(self.head(sa)/0.1,dim=-1);lc=F.log_softmax(self.head(sc)/0.1,dim=-1)
        loss=(-(qa[:,0]*lc[:,0]).sum(-1).mean()-(qc[:,0]*la[:,0]).sum(-1).mean())/2
        if mask is not None:
            pl=(-(qa[:,1:]*la[:,1:]).sum(-1)-(qc[:,1:]*lc[:,1:]).sum(-1))/2
            loss=loss+(pl*mask).sum()/mask.sum().clamp_min(1)
        if self.entropy:loss=loss+0.1*(koleo(sa[:,0])+koleo(sc[:,0]))/2
        if self.training:self.pending_center=torch.cat([ta[:,0],tc[:,0]]).mean(0,keepdim=True)
        return loss
    @torch.no_grad()
    def after_optimizer_step(self):
        ema(self.student,self.teacher,self.momentum);ema(self.head,self.teacher_head,self.momentum)
        if self.pending_center is not None:self.center.mul_(0.9).add_(self.pending_center,alpha=0.1)
        self.pending_center=None


@torch.no_grad()
def sinkhorn(logits,epsilon=0.05,iterations=20):
    q=torch.exp((logits-logits.max())/epsilon).T
    q=q/q.sum().clamp_min(1e-12); k,n=q.shape
    for _ in range(iterations):
        q=q/q.sum(1,keepdim=True).clamp_min(1e-12)/k
        q=q/q.sum(0,keepdim=True).clamp_min(1e-12)/n
    return (q*n).T


class SwAV(nn.Module):
    def __init__(self,width,prototypes):
        super().__init__();self.encoder=Contrastive(width,.2);self.prototypes=nn.Linear(width,prototypes,bias=False)
    def training_loss(self,b):
        a,c=_views(b['image'])
        def scores(x):return F.normalize(self.encoder(x),dim=-1)@F.normalize(self.prototypes.weight,dim=-1).T
        sa,sc=scores(a),scores(c)
        qa,qc=sinkhorn(sa),sinkhorn(sc)
        return (-(qa*F.log_softmax(sc/0.1,dim=-1)).sum(-1).mean()-(qc*F.log_softmax(sa/0.1,dim=-1)).sum(-1).mean())/2


class DiscreteMasked(nn.Module):
    def __init__(self,width,vocab):
        super().__init__();self.encoder=Encoder(width);self.head=nn.Linear(width,vocab);self.vocab=vocab
    def training_loss(self,b):
        x=F.interpolate(b['image'],size=(32,32),mode='bilinear',align_corners=False)
        with torch.no_grad():targets=(F.avg_pool2d(x,4).flatten(1)*self.vocab).long().clamp(0,self.vocab-1)
        mask=torch.rand(x.shape[0],64,device=x.device)<.4;mask[:,0]=True
        logits=self.head(self.encoder(x,mask,tokens=True)[:,1:])
        return F.cross_entropy(logits[mask],targets[mask])


class GraphNetwork(nn.Module):
    def __init__(self,kind,width,heads=2):
        super().__init__()
        from torch_geometric.nn import GCNConv,GATConv,SAGEConv
        if kind==75:self.a=GCNConv(8,width);self.b=GCNConv(width,3)
        elif kind==76:self.a=GATConv(8,width,heads=heads,dropout=0.0);self.b=GATConv(width*heads,3,heads=1,concat=False,dropout=0.0)
        else:self.a=SAGEConv(8,width,aggr='mean');self.b=SAGEConv(width,3,aggr='mean')
    def forward(self,features,adjacency):
        edges=adjacency.nonzero().T.contiguous()
        return self.b(F.relu(self.a(features,edges)),edges)
    def training_loss(self,b):
        return torch.stack([F.cross_entropy(self(x,a),y) for x,a,y in zip(b['node_features'],b['adjacency'],b['node_labels'])]).mean()


def shortest_paths(adjacency):
    n=len(adjacency);d=torch.where(adjacency.bool(),torch.ones_like(adjacency),torch.full_like(adjacency,n+1))
    d.fill_diagonal_(0)
    for k in range(n): d=torch.minimum(d,d[:,k,None]+d[None,k,:])
    return d.long()


class GraphTransformer(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.project=nn.Linear(8,width);self.degree=nn.Embedding(33,width)
        self.spatial=nn.Embedding(9,heads);self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norm=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
        self.head=nn.Linear(width,3)
    def forward(self,x,a):
        h=self.project(x)+self.degree(a.sum(-1).long().clamp_max(32))
        bias=self.spatial(shortest_paths(a).clamp_max(8)).permute(2,0,1)
        z=self.attention(h[None],h[None],h[None],attn_mask=bias,need_weights=False)[0][0]
        h=self.norm(h+z);return self.head(h+self.ff(h))
    def training_loss(self,b):
        return torch.stack([F.cross_entropy(self(x,a),y) for x,a,y in zip(b['node_features'],b['adjacency'],b['node_labels'])]).mean()


class Recurrent(nn.Module):
    def __init__(self,kind,width,layers):
        super().__init__();cls=nn.LSTM if kind==77 else nn.GRU
        self.rnn=cls(4,width,num_layers=layers,batch_first=True);self.head=nn.Linear(width,3)
    def forward(self,x):return self.head(self.rnn(x)[0][:,-1])
    def training_loss(self,b):return F.cross_entropy(self(b['sequence']),b['sequence_label'])


class TCN(nn.Module):
    def __init__(self,width,levels,kernel_size):
        super().__init__()
        if kernel_size<2:raise ValueError('Author TCN Chomp1d requires kernel_size >= 2')
        from vendor.research_tcn.tcn import TemporalConvNet
        self.network=TemporalConvNet(4,[width]*levels,kernel_size=kernel_size,dropout=0)
        self.head=nn.Linear(width,3)
    def forward(self,x):return self.head(self.network(x.transpose(1,2))[:,:,-1])
    def training_loss(self,b):return F.cross_entropy(self(b['sequence']),b['sequence_label'])


def build_model(num,config):
    c=dict(SPECS[num]['config']);c.update(config);w=c.get('width',32)
    if not isinstance(w,int) or w<4 or w>512:raise ValueError('width must be an integer in 4..512')
    if num==24:return Contrastive(w,c['temperature'])
    if num==25:return MoCo(w,c['queue_size'],c['momentum'],c['temperature'])
    if num==26:return MAE(c)
    if num in {27,28,29}:return Distillation(w,c['momentum'],c['prototypes'],patch=num!=27,entropy=num==29)
    if num==92:return DiscreteMasked(w,c['vocabulary'])
    if num==93:return Bootstrap(w,c['momentum'])
    if num==94:return SwAV(w,c['prototypes'])
    if num in {75,76,132}:return GraphNetwork(num,w,c.get('heads',2))
    if num==133:return GraphTransformer(w,c['heads'])
    if num in {77,136}:return Recurrent(num,w,c['layers'])
    if num==78:return TCN(w,c['levels'],c['kernel_size'])
    raise KeyError(num)
