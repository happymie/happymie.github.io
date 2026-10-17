"""Exercise actual training paths and architecture-specific failure modes."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from biomed_lab.data import demo_batch
from biomed_lab.models.image_models import (SPECS, UNAVAILABLE, build_model,
    MaskedSetSegmentation, SS2D, TokenTransformer, NestedUNet, Genesis)

torch.set_num_threads(2)


@pytest.mark.parametrize("num", sorted(SPECS))
def test_each_declared_lesson_runs_real_backward(num):
    torch.manual_seed(27)
    config=SPECS[num]["config"]
    model=build_model(num,config).train()
    batch=demo_batch(image_size=config.get("image_size",32),kind="phantoms3d")
    loss=model.training_loss(batch)
    assert loss.ndim==0 and torch.isfinite(loss), (num,loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), num
    assert any(g.abs().sum()>0 for g in grads), num
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optimizer.step()


def test_unavailable_frameworks_fail_explicitly():
    for num in UNAVAILABLE:
        with pytest.raises(NotImplementedError):
            build_model(num,{})
    json.dumps(SPECS)


def test_mamba_batch_scan_matches_independent_streaming_step():
    from vendor.image_mamba.mamba_simple import Mamba
    torch.manual_seed(10)
    model=Mamba(8,d_state=4,use_fast_path=False).eval()
    x=torch.randn(2,13,8)
    with torch.no_grad():
        batch=model(x)
        conv,state=model.allocate_inference_cache(2,13)
        outputs=[]
        for step in x.split(1,dim=1):
            y,conv,state=model.step(step,conv,state)
            outputs.append(y)
        torch.testing.assert_close(batch,torch.cat(outputs,1),atol=1e-6,rtol=2e-5)


def test_s4_uses_full_dplr_and_is_causal():
    from vendor.image_s4.s4 import SSMKernelDPLR
    model=build_model(15,{}).eval()
    assert isinstance(model.mixer.layer.kernel,SSMKernelDPLR)
    x=torch.randn(2,24,16)
    changed=x.clone();changed[:,12:]+=8
    with torch.no_grad():
        y=model.mixer(x)[0]
        yy=model.mixer(changed)[0]
    torch.testing.assert_close(y[:,:12],yy[:,:12],atol=2e-5,rtol=2e-5)


def test_vmamba_orders_restore_non_square_grid_without_axis_swap():
    x=torch.arange(2*3*5*7).reshape(2,3,5,7)
    restored=SS2D.restore_orders(SS2D.scan_orders(x),5,7)
    for y in restored:torch.testing.assert_close(x,y)


def test_transformer_target_mask_prevents_future_token_leakage():
    torch.manual_seed(12)
    model=TokenTransformer(depth=1).eval()
    source=torch.randint(0,32,(2,10))
    decoder=torch.randint(0,32,(2,10))
    changed=decoder.clone();changed[:,6:]=(changed[:,6:]+5)%32
    with torch.no_grad():
        a=model(source,decoder);b=model(source,changed)
    torch.testing.assert_close(a[:,:6],b[:,:6],atol=1e-6,rtol=1e-5)


def test_mask2former_set_loss_is_query_permutation_invariant():
    torch.manual_seed(31)
    model=MaskedSetSegmentation()
    cls=torch.randn(2,4,3)
    masks=torch.randn(2,4,16,16)
    target=torch.randint(0,2,(2,16,16))
    order=torch.tensor([2,0,3,1])
    torch.testing.assert_close(model.set_loss(cls,masks,target),model.set_loss(cls[:,order],masks[:,order],target))
    all_blocked=torch.full((2,4,8,8),-100.)
    assert not model.attention_mask(all_blocked,(4,4)).any()


def test_unetplusplus_deep_heads_all_receive_supervision():
    model=NestedUNet()
    model.training_loss(demo_batch()).backward()
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum()>0 for head in model.heads)


def test_genesis_corruptions_preserve_clean_target():
    model=Genesis(shuffle_blocks=2)
    x=torch.rand(2,1,16,16,16)
    original=x.clone()
    for mode in ("inpaint","outpaint"):
        y=model.corrupt(x,mode)
        torch.testing.assert_close(x,original)
        assert y.shape==x.shape and not torch.equal(x,y)
        assert torch.isfinite(y).all()


def test_distillation_updates_both_tokens_and_heads():
    model=build_model(89,{}).train()
    model.training_loss(demo_batch()).backward()
    for p in [model.network.cls_token,model.network.dist_token,model.network.head.weight,model.network.head_dist.weight]:
        assert p.grad is not None and p.grad.abs().sum()>0


def test_original_vendor_bytes_match_github_blob_identifiers():
    root=Path(__file__).resolve().parents[1]/"vendor"
    for folder in sorted(root.glob("image_*")):
        record=json.loads((folder/"PROVENANCE.json").read_text())
        for entry in record["sources"]:
            name=Path(entry["original_path"]).name
            path=folder/"original"/name
            if not path.exists():path=folder/name
            assert path.exists(), path
            data=path.read_bytes()
            digest=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            assert digest==entry["git_blob_sha"], (path,digest,entry["git_blob_sha"])
