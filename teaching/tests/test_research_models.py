import torch
import pytest
from biomed_lab.data import demo_batch
from biomed_lab.models import research_models as m

torch.set_num_threads(2)


@pytest.mark.parametrize('num',sorted(m.SPECS))
def test_finite_backward_and_parameter_update(num):
    torch.manual_seed(19)
    model=m.build_model(num,m.SPECS[num]['config'])
    kind='graphs' if m.SPECS[num]['task']=='graph' else 'waves' if m.SPECS[num]['task']=='sequence' else 'shapes2d'
    b=demo_batch(kind=kind)
    before=[p.detach().clone() for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss=model.training_loss(b);assert loss.ndim==0 and torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    if hasattr(model,'after_optimizer_step'):model.after_optimizer_step()
    after=[p.detach() for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(a,c) for a,c in zip(before,after))


def test_infonce_positives_and_symmetry():
    x=torch.eye(6)
    good=m.nt_xent(x,x)
    bad=m.nt_xent(x,x.roll(1,0))
    assert good<bad
    assert torch.allclose(m.nt_xent(x,x*.5),m.nt_xent(x*.5,x))


def test_teacher_no_grad_and_queue_moves():
    model=m.build_model(25,m.SPECS[25]['config']);q=model.queue.clone()
    model.training_loss(demo_batch()).backward()
    assert all(p.grad is None for p in model.target.parameters())
    assert torch.equal(q,model.queue)
    model.after_optimizer_step();assert int(model.pointer)==2
    assert not torch.equal(q,model.queue)


def test_sinkhorn_balances_assignments():
    torch.manual_seed(9)
    q=m.sinkhorn(torch.randn(32,4)*.05,iterations=100)
    assert torch.allclose(q.sum(1),torch.ones(32),atol=1e-5)
    assert torch.allclose(q.sum(0),torch.full((4,),8.),atol=1e-4)


def test_tcn_is_causal():
    model=m.build_model(78,m.SPECS[78]['config']).eval()
    x=torch.randn(2,4,64);altered=x.clone();altered[:,:,40:]+=10
    with torch.no_grad():a=model.network(x);b=model.network(altered)
    assert torch.allclose(a[:,:,:40],b[:,:,:40],atol=1e-6)


def test_graph_permutation_equivariance():
    batch=demo_batch(kind='graphs');x=batch['node_features'][0];a=batch['adjacency'][0]
    p=torch.randperm(32)
    for num in [75,76,132,133]:
        model=m.build_model(num,m.SPECS[num]['config']).eval()
        with torch.no_grad():base=model(x,a);permuted=model(x[p],a[p][:,p])
        assert torch.allclose(base[p],permuted,atol=2e-5),num


def test_mae_encoder_removes_masked_tokens():
    model=m.build_model(26,m.SPECS[26]['config']).eval()
    x=demo_batch()['image'].repeat(1,3,1,1)
    with torch.no_grad():latent,mask,restore=model.model.forward_encoder(x,mask_ratio=.75)
    assert latent.shape[1]==17  # 16 visible patches + CLS, not 65 filled tokens
    assert mask.sum().item()==2*48
