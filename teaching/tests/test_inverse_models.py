"""Numerical and information-boundary checks for the inverse-model lessons."""
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from biomed_lab.data import demo_batch
from biomed_lab.models import inverse_models as m


@pytest.fixture(autouse=True, scope="module")
def small_cpu_budget():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("number", sorted(m.SPECS))
def test_each_declared_implementation_trains_and_evaluates(number):
    torch.manual_seed(173 + number)
    model = m.build_model(number, {})
    batch = demo_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss = model.training_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer.step()
    if hasattr(model, "after_optimizer_step"):
        model.after_optimizer_step()
    assert any(not torch.equal(before[k], p) for k, p in model.named_parameters())
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model.training_loss(batch))


def test_ct_forward_adjoint_inner_product():
    torch.manual_seed(61)
    op = m.ParallelBeam(16, 11).double()
    x = torch.randn(2, 1, 16, 16, dtype=torch.float64)
    y = torch.randn(2, 1, 11, 16, dtype=torch.float64)
    lhs = (op(x) * y).sum()
    rhs = (x * op.adjoint(y)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_fbp_disk_has_the_correct_physical_amplitude():
    # An independent continuum sanity check: unit attenuation stays close to
    # one in the interior after ramp filtering and angular integration.
    op = m.ParallelBeam(16, 32)
    axis = (torch.arange(16) + .5) / 16 - .5
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    phantom = (x.square() + y.square() < .2 ** 2).float()[None, None]
    reconstruction = op.fbp(op(phantom))[0, 0]
    mean = reconstruction[x.square() + y.square() < .1 ** 2].mean()
    assert .85 < mean < 1.15  # Expected finite-grid quadrature error, not a benchmark.


def test_complex_mri_adjoint_and_complete_acquisition_identity():
    torch.manual_seed(18)
    op = m.FourierOperator(16)
    x = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    y = torch.randn(2, 2, 16, 16, dtype=torch.complex64)
    lhs = torch.vdot(op(x).flatten(), y.flatten())
    rhs = torch.vdot(x.flatten(), op.adjoint(y).flatten())
    torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)
    op.mask.fill_(1)
    torch.testing.assert_close(op.normal(x), x, atol=5e-6, rtol=5e-6)


def test_cg_matches_independent_dense_linear_solve_and_backpropagates():
    torch.manual_seed(9)
    a = torch.randn(6, 6, dtype=torch.float64)
    matrix = a.T @ a + torch.eye(6, dtype=torch.float64)
    rhs = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    solution = m.conjugate_gradient(lambda x: x @ matrix.T, rhs, steps=8)
    expected = torch.linalg.solve(matrix, rhs.T).T
    torch.testing.assert_close(solution, expected, atol=1e-9, rtol=1e-9)
    solution.square().sum().backward()
    assert torch.isfinite(rhs.grad).all()


@pytest.mark.parametrize("void", [False, True])
def test_blind_spot_prediction_cannot_read_any_held_out_pixel(void):
    torch.manual_seed(17)
    model = m.BlindSpot(void=void).eval()
    x = torch.randn(1, 1, 17, 19)
    mask = model.partition(x, 5)
    corrupted = x.clone()
    corrupted[mask] += torch.randn_like(corrupted[mask]) * 100
    original = model.masked_prediction(x, mask)
    changed = model.masked_prediction(corrupted, mask)
    torch.testing.assert_close(original[mask], changed[mask], atol=0, rtol=0)


def test_noise2noise_does_not_read_clean_targets():
    torch.manual_seed(3)
    model = m.Restoration(number=53).eval()
    batch = demo_batch()
    actual = model.training_loss(batch)
    expected = F.mse_loss(model(batch["noisy"]), batch["noisy2"])
    batch["clean"] = torch.full_like(batch["clean"], float("nan"))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.training_loss(batch), actual)
    batch["noisy2"] = batch["noisy"]
    with pytest.raises(ValueError, match="independent"):
        model.training_loss(batch)


def test_gan_generator_objective_does_not_update_discriminator():
    torch.manual_seed(10)
    model = m.GAN()
    batch = demo_batch()
    real = m._image(batch)
    torch.manual_seed(27)
    fake = model(torch.randn(len(real), model.latent))
    discriminator_only = F.softplus(-model.discriminator(real)).mean() + F.softplus(model.discriminator(fake.detach())).mean()
    expected = torch.autograd.grad(discriminator_only, list(model.discriminator.parameters()))
    torch.manual_seed(27)
    model.training_loss(batch).backward()
    for p, grad in zip(model.discriminator.parameters(), expected):
        torch.testing.assert_close(p.grad, grad)


def test_ddim_eta_zero_does_not_depend_on_reverse_noise():
    torch.manual_seed(72)
    model = m.Diffusion(number=56).eval()
    x = torch.randn(2, 1, 16, 16)
    t = torch.full((2,), 400, dtype=torch.long)
    previous = torch.full((2,), 300, dtype=torch.long)
    a = model.ddim_step(x, t, previous, eta=0., noise=torch.randn_like(x))
    b = model.ddim_step(x, t, previous, eta=0., noise=torch.randn_like(x) * 99)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    initial = torch.randn_like(x)
    torch.testing.assert_close(model.sample(initial=initial, steps=4), model.sample(initial=initial, steps=4), atol=0, rtol=0)


def test_one_step_diffusion_starts_at_the_terminal_trained_time():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []
        def forward(self, x, t):
            self.times.extend(t.tolist())
            return torch.zeros_like(x)
    model = m.Diffusion(number=56)
    model.net = Recorder()
    model.sample(initial=torch.ones(1, 1, 16, 16), steps=1)
    assert model.net.times == [999]
    for number in [59, 119]:
        posterior = m.DiffusionInverse(number=number)
        result = posterior.reconstruct(torch.zeros(1, 1, 16, 16), steps=1)
        assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="Sampling steps"):
        model.sample(steps=0)


def test_shortened_beta_tables_cannot_silently_assume_gaussian_terminal_data():
    model = m.Diffusion(number=56, timesteps=100)
    assert model.alpha_bar[-1] > .3
    with pytest.raises(ValueError, match="noise horizon"):
        model.sample(steps=10)


def test_score_ode_time_grid_converges_to_an_independent_gaussian_solution():
    class ExactPointMassScore(m.ScoreSDE):
        def forward(self, x, t):
            _, std = self.marginal(t)
            return -x / m._scalar(std.square(), x)
    model = ExactPointMassScore()
    initial = torch.ones(1, 1, 16, 16)
    coarse = model.sample(steps=20, probability_flow=True, initial=initial)
    finer = model.sample(steps=100, probability_flow=True, initial=initial)
    _, end_std = model.marginal(torch.tensor([.001]))
    _, start_std = model.marginal(torch.tensor([1.]))
    exact = end_std / start_std
    coarse_error = (coarse - exact).abs().mean()
    fine_error = (finer - exact).abs().mean()
    assert (coarse > 0).all() and (finer > 0).all()
    assert fine_error < coarse_error
    assert fine_error < .005


def test_ddpm_and_ddim_share_the_same_training_objective():
    ddpm, ddim = m.Diffusion(number=55), m.Diffusion(number=56)
    ddim.load_state_dict(ddpm.state_dict())
    batch = demo_batch()
    torch.manual_seed(41)
    loss_a = ddpm.training_loss(batch)
    torch.manual_seed(41)
    loss_b = ddim.training_loss(batch)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="DDIM"):
        ddpm.sample(steps=4)


def test_sit_sampling_integrates_velocity_not_epsilon():
    class ConstantVelocity(nn.Module):
        def forward(self, x, t, label):
            return torch.ones_like(x) * .2
    model = m.TransformerGenerator(number=123)
    model.net = ConstantVelocity()
    initial = torch.zeros(1, 1, 16, 16)
    torch.testing.assert_close(model.sample(initial=initial, steps=5), torch.full_like(initial, .2), atol=1e-7, rtol=1e-7)


def test_ddnm_preserves_range_and_leaves_the_null_space_unchanged():
    model = m.DiffusionInverse(number=119)
    x = torch.randn(2, 1, 16, 16)
    y = model.measure(torch.randn_like(x))
    corrected = model.ddnm_project(x, y)
    torch.testing.assert_close(model.measure(corrected), y, atol=1e-6, rtol=1e-6)
    null = 1 - model.measurement_mask
    torch.testing.assert_close(null * corrected, null * x, rtol=0, atol=0)
    sampled = model.reconstruct(y, steps=3)
    torch.testing.assert_close(model.measure(sampled), y, atol=2e-5, rtol=1e-5)


def test_ddrm_implements_distinct_spectral_noise_regimes():
    model = m.DiffusionInverse(number=59)
    x = torch.zeros(1, 1, 16, 16)
    y = model.measure(torch.ones_like(x))
    mask = model.measurement_mask.bool()
    # eta=0 suppresses stochasticity in the below-measurement-noise branch.
    below = model.ddrm_spectral_step(x, x, y, sigma_next=.1, sigma_y=.2, eta=0)
    torch.testing.assert_close(below[mask], torch.full_like(below[mask], .5))
    noiseless_end = model.ddrm_spectral_step(x, x, y, sigma_next=0., sigma_y=0., eta=0.)
    torch.testing.assert_close(noiseless_end, y)


def test_dps_guidance_differentiates_the_measurement_path():
    class Zero(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)
    model = m.DiffusionInverse(number=60)
    model.net = Zero()
    x = torch.ones(1, 1, 16, 16)
    y = torch.zeros_like(x)
    t = torch.tensor([100])
    torch.manual_seed(80)
    base = model.dps_step(x, t, y, strength=0.)
    torch.manual_seed(80)
    guided = model.dps_step(x, t, y, strength=.1)
    mask = model.measurement_mask.bool()
    assert (guided[mask] < base[mask]).all()
    torch.testing.assert_close(guided[~mask], base[~mask], rtol=0, atol=0)


@pytest.mark.parametrize("number", [62, 63])
def test_flow_velocity_is_the_time_derivative_of_its_path(number):
    torch.manual_seed(8)
    flow = m.FlowMatching(number=number)
    a, b = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    t = torch.tensor([.2, .7])
    _, velocity = flow.path_and_velocity(a, b, t)
    plus, _ = flow.path_and_velocity(a, b, t + 1e-3)
    minus, _ = flow.path_and_velocity(a, b, t - 1e-3)
    torch.testing.assert_close((plus - minus) / .002, velocity, atol=2e-4, rtol=2e-4)


def test_consistency_boundary_is_exact_and_ema_follows_optimizer():
    model = m.Consistency()
    x = torch.randn(2, 1, 16, 16)
    sigma = torch.full((2,), model.sigma_min)
    torch.testing.assert_close(model(x, sigma), x, rtol=0, atol=0)
    target_before = [p.clone() for p in model.target.parameters()]
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(1.)
    model.after_optimizer_step()
    for old, target in zip(target_before, model.target.parameters()):
        torch.testing.assert_close(target, old + .05, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cls", [m.RealNVP, m.Glow])
def test_flow_inverse_and_logdet_match_a_dense_autograd_jacobian(cls):
    torch.manual_seed(19)
    model = cls(size=2, width=8, layers=2).double()
    # Initialize data-dependent ActNorm before differentiating the fixed map.
    model(torch.randn(5, 1, 2, 2, dtype=torch.float64))
    x = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    z, logdet = model(x)
    torch.testing.assert_close(model.inverse(z), x, atol=1e-9, rtol=1e-9)
    jac = torch.autograd.functional.jacobian(lambda v: model(v.view(1, 1, 2, 2))[0].flatten(), x.flatten())
    exact_logdet = torch.linalg.slogdet(jac)[1]
    torch.testing.assert_close(logdet[0], exact_logdet, atol=1e-8, rtol=1e-8)


def test_pfgm_radial_second_moment_has_the_correct_finite_d_factor():
    # For D>2, E[R^2]=N sigma^2 D/(D-2). This distinguishes finite-D
    # Poisson perturbations from an accidentally substituted Gaussian law.
    torch.manual_seed(808)
    model = m.PFGMPP(size=4, augmented_dim=8)
    x = torch.zeros(8192, 1, 4, 4)
    noise = model.perturb(x, torch.ones(len(x)))
    actual = noise.flatten(1).square().sum(1).mean() / 16
    assert abs(float(actual) - 8 / 6) < .06


def test_identity_warp_and_stn_preserve_coordinates():
    volume = torch.randn(1, 1, 8, 9, 10)
    flow = torch.zeros(1, 3, 8, 9, 10)
    torch.testing.assert_close(m.warp(volume, flow), volume, atol=2e-6, rtol=2e-6)
    image = torch.randn(1, 1, 16, 16)
    warped, affine = m.SpatialTransformer()(image, image)
    torch.testing.assert_close(affine[0], torch.eye(2, 3), atol=0, rtol=0)
    torch.testing.assert_close(warped, image, atol=5e-6, rtol=5e-6)


def test_gradicon_penalizes_derivative_not_translation_offset():
    u = torch.ones(1, 3, 8, 8, 8) * .1
    v = torch.ones_like(u) * .2
    assert m.inverse_composition(u, v).square().mean() > .08
    assert float(m.gradicon_penalty(u, v)) < 1e-10
    u[:, 0] = m.coordinate_grid((8, 8, 8))[..., 0] * .1
    assert m.gradicon_penalty(u, v) > 1e-4


def test_rigid_rotation_and_parallel_ray_physical_units():
    pose = torch.tensor([[.15, -.07, .23, .2, .1, -.1]], requires_grad=True)
    transform = m.rigid_matrix(pose)
    rotation = transform[:, :, :3]
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation, torch.eye(3)[None], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(1), atol=1e-6, rtol=1e-6)
    uniform = torch.ones(1, 1, 16, 16, 16)
    integral = m.parallel_drr(uniform, torch.zeros(1, 6))
    torch.testing.assert_close(integral, torch.full_like(integral, 2.), atol=1e-6, rtol=1e-6)
    m.parallel_drr(torch.randn_like(uniform), pose).square().mean().backward()
    assert torch.isfinite(pose.grad).all()


def test_volume_render_matches_constant_medium_beer_lambert_solution():
    sigma = torch.full((1, 3, 20), 2.)
    color = torch.ones(1, 3, 20, 3)
    delta = torch.full((20,), .15)
    rgb, _, weights = m.volume_render(sigma, color, delta)
    expected = 1 - math.exp(-2 * 3)
    torch.testing.assert_close(rgb, torch.full_like(rgb, expected), atol=1e-6, rtol=1e-6)
    assert (weights >= 0).all()
    assert (weights.sum(-1) <= 1 + 1e-6).all()


def test_naf_density_has_no_view_direction_dependency():
    torch.manual_seed(91)
    lesson = m.FieldLesson(number=73)
    points = torch.rand(1, 8, 4, 3) * 1.5 - .75
    density_a, _ = lesson.query(points, F.normalize(torch.randn_like(points), dim=-1))
    density_b, _ = lesson.query(points, F.normalize(torch.randn_like(points), dim=-1))
    torch.testing.assert_close(density_a, density_b, atol=0, rtol=0)


def test_endonerf_parameterization_can_represent_its_known_synthetic_motion():
    class ExactTranslationLogits(nn.Module):
        def forward(self, input):
            time = input[..., 3:]
            ratio = -.15 * torch.sin(2 * math.pi * time) / time.clamp_min(1e-8)
            return torch.cat((torch.atanh(ratio), torch.zeros_like(time), torch.zeros_like(time)), -1)
    model = m.FieldLesson(number=131)
    model.deformation = ExactTranslationLogits()
    points = torch.zeros(1, 2, 4, 3)
    for time in [.001, .01, .25, .5, .75, .99]:
        actual = model.deform_points(points, time)
        expected = points.clone()
        expected[..., 0] = -.15 * math.sin(2 * math.pi * time)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_hash_interpolation_preserves_constants_and_coordinate_gradients():
    encoding = m.HashGrid(levels=2, features=2)
    x = (torch.rand(10, 3) * 1.8 - .9).requires_grad_(True)
    output = encoding(x)
    output.sum().backward()
    assert x.grad.abs().sum() > 0 and torch.isfinite(x.grad).all()
    with torch.no_grad():
        for table in encoding.tables:
            table.fill_(.37)
    torch.testing.assert_close(encoding(x), torch.full_like(output, .37), atol=1e-6, rtol=1e-6)


def test_projected_gaussian_covariances_remain_positive_definite():
    splats = m.GaussianSplattingDemo()
    _, covariance, _ = splats.projected()
    assert (torch.linalg.eigvalsh(covariance) > 0).all()
    splats().sum().backward()
    assert torch.isfinite(splats.position.grad).all()


def test_transmorph_voxel_dhw_flow_matches_explicit_normalized_xyz_sampling():
    model = m.TransMorph(width=4)
    source = torch.randn(1, 1, 32, 32, 32)
    flow = torch.zeros(1, 3, 32, 32, 32)
    flow[:, 0] = .2
    flow[:, 2] = -.4
    native = model.net.spatial_trans(source, flow)
    grid = m.coordinate_grid((32, 32, 32)) + model.normalized_xyz_flow(flow).movedim(1, -1)
    independent = F.grid_sample(source, grid, align_corners=True, padding_mode="zeros")
    torch.testing.assert_close(native, independent, atol=2e-5, rtol=2e-5)


def test_transmorph_keeps_four_hierarchy_levels_and_shifted_attention():
    model = m.TransMorph(width=4)
    layers = model.net.transformer.layers
    assert len(layers) == 4
    for layer in layers:
        assert len(layer.blocks) == 2
        assert tuple(layer.blocks[0].shift_size) == (0, 0, 0)
        assert tuple(layer.blocks[1].shift_size) == (1, 1, 1)


@pytest.mark.parametrize("number", [72, 73, 74, 126, 129, 131])
def test_single_scene_checkpoints_restore_the_actual_scene(number):
    model = m.build_model(number, {})
    model.training_loss(demo_batch())
    state = model.state_dict()
    restored = m.build_model(number, {})
    restored.load_state_dict(state)
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert m.SPECS[number]["evaluation_scope"] == "same_scene"


def test_unmodified_vendor_files_match_recorded_hashes():
    manifests = sorted(m._VENDOR.glob("inverse_*/PROVENANCE.json"))
    assert len(manifests) == 8
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["commit"]) == 40
        for file in manifest["files"]:
            local = manifest_path.parent / file["upstream_path"]
            assert hashlib.sha256(local.read_bytes()).hexdigest() == file["sha256"]


@pytest.mark.parametrize("number", [45, 46, 51, 52, 65, 115, 116, 123, 125])
def test_upstream_width_setting_really_reduces_parameters(number):
    small = m.build_model(number, {"width": 4})
    larger = m.build_model(number, {"width": 8})
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(small) < count(larger)
    loss = small.training_loss(demo_batch())
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("number", [58, 61, 69, 127, 130])
def test_source_only_workflows_are_not_renamed_substitutes(number):
    with pytest.raises(KeyError, match="original-source route"):
        m.build_model(number, {})
