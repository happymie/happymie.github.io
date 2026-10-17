"""Forward/backward and modality contracts for three upstream architectures."""
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from biomed_lab.data import demo_batch
from biomed_lab.models.extended_models import SPECS,build_model,render_synthetic_traces

torch.set_num_threads(2)


@pytest.mark.parametrize("num",sorted(SPECS))
def test_foundation_architecture_training_and_evaluation_paths(num):
    torch.manual_seed(412)
    model=build_model(num,{}).train()
    batch=demo_batch(kind="waves")
    loss=model.training_loss(batch)
    assert torch.isfinite(loss) and loss.ndim==0
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum()>0 for g in gradients)
    torch.optim.AdamW(model.parameters(),lr=1e-3).step()
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model.training_loss(batch))
        assert model(batch["sequence"]).shape==(2,3)


def test_waveform_renderer_has_fixed_scale_and_preserves_observed_input():
    batch=demo_batch(kind="waves")
    sequence=batch["sequence"].clone().requires_grad_()
    original=sequence.detach().clone()
    picture=render_synthetic_traces(sequence,64)
    assert picture.shape==(2,1,64,64)
    assert picture.min()>=0 and picture.max()<=1
    picture.square().mean().backward()
    torch.testing.assert_close(sequence.detach(),original)
    assert torch.isfinite(sequence.grad).all() and sequence.grad.abs().sum()>0


def test_labram_channel_identity_follows_declared_montage_permutation():
    torch.manual_seed(19)
    model=build_model(80,{}).eval()
    patches,ids=model.prepare(demo_batch(kind="waves")["sequence"])
    assert patches.shape==(2,4,2,200)
    assert model.network.patch_embed(patches).shape==(2,8,100)
    order=torch.tensor([2,0,3,1])
    moved_ids=[0]+[ids[i+1] for i in order.tolist()]
    with torch.no_grad():
        a=model.network.forward_features(patches,input_chans=ids)
        b=model.network.forward_features(patches[:,order],input_chans=moved_ids)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("length",[31,32,65])
def test_ecgfounder_same_padding_supports_odd_lengths(length):
    from vendor.extended_ecgfounder.net1d import MyConv1dPadSame
    conv=MyConv1dPadSame(in_channels=1,out_channels=4,kernel_size=16,stride=2)
    out=conv(torch.randn(2,1,length))
    assert out.shape==(2,4,math.ceil(length/2))


def test_ecgfounder_missing_observed_channels_are_not_repeated():
    model=build_model(137,{"in_channels":12})
    with pytest.raises(ValueError,match="missing channels"):
        model(torch.randn(2,64,4))


def test_extended_sources_match_preserved_blob_bytes():
    root=Path(__file__).resolve().parents[1]/"vendor"
    for folder in sorted(root.glob("extended_*")):
        info=json.loads((folder/"PROVENANCE.json").read_text())
        for source in info["sources"]:
            name=Path(source["original_path"]).name
            path=folder/"original"/name
            if not path.exists():path=folder/name
            b=path.read_bytes()
            actual=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
            assert actual==source["git_blob_sha"],(path,actual,source["git_blob_sha"])
    json.dumps(SPECS)
