"""Small, inspectable inverse-problem and generative-model lessons.

The catalogue describes *what is actually executed*, not a benchmark claim.
Unmodified upstream architectures live in teaching/vendor/inverse_*.  Formula
ports and deliberately incomplete mechanism demonstrations are labelled in
SPECS and docs/INVERSE_MODELS.md. All default observations are synthetic.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from contextlib import contextmanager
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

_VENDOR = Path(__file__).resolve().parents[2] / "vendor"


def _upstream(name, path, substitutions=None):
    """Load an isolated module; never shadow another package's ``models``."""
    key = "biomed_lab_upstream_" + name
    if key not in sys.modules:
        source = _VENDOR / path
        spec = importlib.util.spec_from_file_location(key, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        if substitutions:
            text = source.read_text()
            for old, replacement in substitutions.items():
                if old not in text:
                    raise RuntimeError("Upstream import bridge no longer matches source")
                text = text.replace(old, replacement)
            exec(compile(text, str(source), "exec"), module.__dict__)
        else:
            spec.loader.exec_module(module)
    return sys.modules[key]


def _size(x, size=16):
    return F.interpolate(x, (size, size), mode="area") if x.shape[-2:] != (size, size) else x


def _image(batch, key="image", size=16):
    return _size(batch[key].float(), size)


def _scalar(t, x):
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def _diffusion_schedule(timesteps, steps, device):
    steps = 20 if steps is None else steps
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= timesteps:
        raise ValueError("Sampling steps must be an integer between 1 and the trained timestep count")
    if steps == 1:
        return torch.tensor([timesteps - 1], device=device, dtype=torch.long)
    return torch.linspace(0, timesteps - 1, steps, device=device).long()


@contextmanager
def _frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


def _conv(cin, cout, width=8, dim=2, depth=3):
    C = nn.Conv2d if dim == 2 else nn.Conv3d
    layers = [C(cin, width, 3, padding=1), nn.LeakyReLU(.1)]
    for _ in range(depth - 2):
        layers += [C(width, width, 3, padding=1), nn.LeakyReLU(.1)]
    return nn.Sequential(*layers, C(width, cout, 3, padding=1))


class SmallUNet(nn.Module):
    def __init__(self, cin=1, cout=1, width=8, dim=2):
        super().__init__()
        self.dim = dim
        self.enc = _conv(cin, width, width, dim)
        self.mid = _conv(width, 2 * width, 2 * width, dim)
        self.dec = _conv(3 * width, cout, width, dim)

    def forward(self, x):
        h = self.enc(x)
        pool = F.avg_pool2d if self.dim == 2 else F.avg_pool3d
        z = self.mid(pool(h, 2))
        z = F.interpolate(z, size=h.shape[2:], mode="nearest")
        return self.dec(torch.cat((h, z), 1))


class TimeUNet(nn.Module):
    """A small noise/velocity network; not an original paper's full backbone."""
    def __init__(self, width=8):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.inp = nn.Conv2d(1, width, 3, padding=1)
        self.body = SmallUNet(width, 1, width)
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(8) / 8))

    def forward(self, x, t):
        phase = t.float()[:, None] * self.frequencies[None]
        emb = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        return self.body(F.silu(self.inp(x) + emb[:, :, None, None]))


class RBM(nn.Module):
    """Bernoulli-visible RBM, with Bernoulli or Gaussian hidden units."""
    def __init__(self, visible, hidden, gaussian_hidden=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(visible, hidden) * .01)
        self.vbias = nn.Parameter(torch.zeros(visible))
        self.hbias = nn.Parameter(torch.zeros(hidden))
        self.gaussian_hidden = gaussian_hidden

    def encode(self, v):
        h = v @ self.weight + self.hbias
        return h if self.gaussian_hidden else h.sigmoid()

    def free_energy(self, v):
        h = v @ self.weight + self.hbias
        term = .5 * h.square() if self.gaussian_hidden else F.softplus(h)
        return -(v * self.vbias).sum(-1) - term.sum(-1)

    def cd1_loss(self, v):
        with torch.no_grad():
            hp = self.encode(v)
            h = hp + torch.randn_like(hp) if self.gaussian_hidden else torch.bernoulli(hp)
            vn = torch.bernoulli((h @ self.weight.T + self.vbias).sigmoid())
        return (self.free_energy(v) - self.free_energy(vn)).mean()


class DeepAutoencoder(nn.Module):
    def __init__(self, size=16, width=32, latent=8, pretrain_steps=1, **_):
        super().__init__()
        self.size, self.pretrain_steps = size, int(pretrain_steps)
        self.rbms = nn.ModuleList([RBM(size * size, width), RBM(width, latent, True)])
        self.enc = nn.ModuleList([nn.Linear(size * size, width), nn.Linear(width, latent)])
        self.dec = nn.ModuleList([nn.Linear(latent, width), nn.Linear(width, size * size)])
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        self._unfold()

    @torch.no_grad()
    def _unfold(self):
        for i, rbm in enumerate(self.rbms):
            self.enc[i].weight.copy_(rbm.weight.T)
            self.enc[i].bias.copy_(rbm.hbias)
            self.dec[1 - i].weight.copy_(rbm.weight)
            self.dec[1 - i].bias.copy_(rbm.vbias)

    def forward(self, x):
        v = _size(x, self.size).flatten(1)
        z = self.enc[1](self.enc[0](v).sigmoid())
        return self.dec[1](self.dec[0](z).sigmoid()).sigmoid().view(-1, 1, self.size, self.size)

    def training_loss(self, batch):
        x = _image(batch, size=self.size)
        step = int(self.calls)
        self.calls.add_(1)
        if step < 2 * self.pretrain_steps:
            layer = step // self.pretrain_steps
            v = x.flatten(1)
            for rbm in self.rbms[:layer]:
                v = rbm.encode(v).detach()
            return self.rbms[layer].cd1_loss(v)
        if step == 2 * self.pretrain_steps:
            self._unfold()
        return F.mse_loss(self(x), x)


class GAN(nn.Module):
    def __init__(self, size=16, width=32, latent=16, **_):
        super().__init__()
        self.size, self.latent = size, latent
        self.generator = nn.Sequential(nn.Linear(latent, width), nn.ReLU(), nn.Linear(width, size * size), nn.Sigmoid())
        self.discriminator = nn.Sequential(nn.Flatten(), nn.Linear(size * size, width), nn.LeakyReLU(.2), nn.Linear(width, 1))

    def forward(self, z):
        return self.generator(z).view(-1, 1, self.size, self.size)

    def training_loss(self, batch):
        real = _image(batch, size=self.size)
        fake = self(torch.randn(real.shape[0], self.latent, device=real.device))
        d = F.softplus(-self.discriminator(real)).mean() + F.softplus(self.discriminator(fake.detach())).mean()
        with _frozen(self.discriminator):
            g = F.softplus(-self.discriminator(fake)).mean()
        # D gets only discriminator gradients; G gets non-saturating gradients.
        return d + g


class PairedGAN(nn.Module):
    def __init__(self, cycle=False, width=8, **_):
        super().__init__()
        src = _upstream("cyclegan", "inverse_cyclegan/models/networks.py")
        self.cycle = cycle
        norm = src.get_norm_layer("instance")
        if cycle:
            self.generator = src.ResnetGenerator(1, 1, ngf=width, norm_layer=norm, n_blocks=3)
            self.reverse = src.ResnetGenerator(1, 1, ngf=width, norm_layer=norm, n_blocks=3)
            self.discriminator = src.NLayerDiscriminator(1, ndf=width, n_layers=3, norm_layer=norm)
            self.discriminator_x = src.NLayerDiscriminator(1, ndf=width, n_layers=3, norm_layer=norm)
        else:
            self.generator = src.UnetGenerator(1, 1, num_downs=5, ngf=width, norm_layer=norm)
            self.discriminator = src.NLayerDiscriminator(2, ndf=width, n_layers=3, norm_layer=norm)
        src.init_weights(self, init_type="normal", init_gain=.02)

    def forward(self, x):
        # 64px ensures the original 70px PatchGAN has valid intermediate sizes.
        return self.generator(_size(x, 64) * 2 - 1) / 2 + .5

    def training_loss(self, batch):
        x = _image(batch, "noisy", 64) * 2 - 1
        y = _image(batch, "clean", 64) * 2 - 1
        if not self.cycle:
            fake = self.generator(x)
            real_logit = self.discriminator(torch.cat((x, y), 1))
            fake_logit = self.discriminator(torch.cat((x, fake.detach()), 1))
            d = .5 * (F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean())
            with _frozen(self.discriminator):
                g = F.softplus(-self.discriminator(torch.cat((x, fake), 1))).mean()
            return d + g + 100 * F.l1_loss(fake, y)
        # Shuffle the target domain: no paired target occurs in the objective.
        y = y[torch.randperm(y.shape[0], device=y.device)]
        fy, fx = self.generator(x), self.reverse(y)
        d = .5 * ((self.discriminator(y) - 1).square().mean() + self.discriminator(fy.detach()).square().mean())
        d = d + .5 * ((self.discriminator_x(x) - 1).square().mean() + self.discriminator_x(fx.detach()).square().mean())
        with _frozen(self.discriminator), _frozen(self.discriminator_x):
            g = (self.discriminator(fy) - 1).square().mean() + (self.discriminator_x(fx) - 1).square().mean()
        cycle = F.l1_loss(self.reverse(fy), x) + F.l1_loss(self.generator(fx), y)
        identity = F.l1_loss(self.generator(y), y) + F.l1_loss(self.reverse(x), x)
        return d + g + 10 * cycle + 5 * identity


class Restoration(nn.Module):
    def __init__(self, number=52, width=8, **_):
        super().__init__()
        self.number = number
        if number == 52:
            _upstream("dncnn_blocks", "inverse_dncnn/models/basicblock.py")
            src = _upstream("dncnn", "inverse_dncnn/models/network_dncnn.py", {
                "import models.basicblock as B": "import biomed_lab_upstream_dncnn_blocks as B"})
            self.net = src.DnCNN(in_nc=1, out_nc=1, nc=width, nb=17, act_mode="BR")
        elif number == 115:
            src = _upstream("swinir", "inverse_swinir/models/network_swinir.py")
            self.net = src.SwinIR(img_size=32, in_chans=1, embed_dim=3 * width,
                                 depths=[2, 2], num_heads=[3, 3], window_size=4,
                                 mlp_ratio=2, upscale=1, upsampler="", img_range=1., drop_path_rate=0.)
        elif number == 116:
            src = _upstream("restormer", "inverse_restormer/basicsr/models/archs/restormer_arch.py")
            if width % 2:
                raise ValueError("Restormer width must be even for pixel unshuffle/shuffle")
            self.net = src.Restormer(inp_channels=1, out_channels=1, dim=width,
                                    num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
                                    heads=[1, 2, 4, 8], ffn_expansion_factor=2.)
        elif number == 114:
            self.encoder = nn.ModuleList([nn.Conv2d(1 if i == 0 else width, width, 3, padding=1) for i in range(5)])
            self.decoder = nn.ModuleList([nn.ConvTranspose2d(width, width if i < 4 else 1, 3, padding=1) for i in range(5)])
        else:
            self.net = SmallUNet(width=width)

    def forward(self, x):
        if self.number == 114:
            original, features = x, []
            for conv in self.encoder:
                x = F.relu(conv(x))
                features.append(x)
            for i, conv in enumerate(self.decoder):
                x = conv(x)
                if i < 4:
                    # Symmetric feature skips at every second encoder depth.
                    if i % 2 == 0:
                        x = x + features[-2 - i]
                    x = F.relu(x)
            return original + x
        return self.net(x)

    def training_loss(self, batch):
        if self.number == 53:
            # No clean tensor is read here. Zero-mean independent noise targets
            # must be prepared without clipping, which would bias their mean.
            x, target = batch["noisy"], batch["noisy2"]
            if x.data_ptr() == target.data_ptr():
                raise ValueError("Noise2Noise requires two independent acquisitions, not the same tensor")
            return F.mse_loss(self(x), target)
        return F.mse_loss(self(batch["noisy"]), batch["clean"])


class BlindSpot(nn.Module):
    def __init__(self, void=False, width=8, **_):
        super().__init__()
        self.net, self.void = SmallUNet(width=width), void
        self.register_buffer("iteration", torch.zeros((), dtype=torch.long))

    @staticmethod
    def partition(x, index, period=3):
        h, w = x.shape[-2:]
        row = torch.arange(h, device=x.device)[:, None]
        col = torch.arange(w, device=x.device)[None, :]
        return ((row % period) * period + col % period == index)[None, None].expand_as(x)

    def masked_prediction(self, x, mask):
        if self.void:
            # Neighbour samples never come from held-out J. The lattice spacing
            # also prevents a masked pixel from leaking into another replacement.
            source = torch.roll(x, shifts=1, dims=-1)
            safe = ~torch.roll(mask, shifts=1, dims=-1)
            replacement = torch.where(safe, source, torch.zeros_like(source))
        else:
            replacement = torch.zeros_like(x)
        return self.net(torch.where(mask, replacement, x))

    def forward(self, x):
        result = torch.zeros_like(x)
        for j in range(9):
            mask = self.partition(x, j)
            result = result + self.masked_prediction(x, mask) * mask
        return result

    def training_loss(self, batch):
        x = batch["noisy"]
        j = int(torch.randint(9, ()).item()) if self.void else int(self.iteration % 9)
        self.iteration.add_(1)
        mask = self.partition(x, j)
        return (self.masked_prediction(x, mask)[mask] - x[mask]).square().mean()


class Diffusion(nn.Module):
    def __init__(self, number=55, size=16, width=8, timesteps=1000, **_):
        super().__init__()
        self.number, self.size, self.timesteps = number, size, timesteps
        self.net = TimeUNet(width)
        beta = torch.linspace(.0001, .02, timesteps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))

    def forward(self, x, t):
        return self.net(x, t)

    def q_sample(self, x0, t, noise):
        a = _scalar(self.alpha_bar[t], x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def training_loss(self, batch):
        x = _image(batch, size=self.size) * 2 - 1
        t = torch.randint(self.timesteps, (len(x),), device=x.device)
        noise = torch.randn_like(x)
        return F.mse_loss(self(self.q_sample(x, t, noise), t), noise)

    def x0_estimate(self, x, t):
        a = _scalar(self.alpha_bar[t], x)
        eps = self(x, t)
        return (x - (1 - a).sqrt() * eps) / a.sqrt(), eps

    def _check_gaussian_start(self, initial):
        if initial is None and self.timesteps < 1000:
            raise ValueError("Gaussian-start generation requires the default 1000-step noise horizon; use fewer DDIM sampling steps, not fewer training timesteps, or supply an explicit terminal state")

    def ddim_step(self, x, t, previous, eta=0., noise=None):
        if not 0 <= eta <= 1:
            raise ValueError("eta must lie in [0,1]")
        at = _scalar(self.alpha_bar[t], x)
        ap = _scalar(torch.where(previous >= 0, self.alpha_bar[previous.clamp_min(0)], torch.ones_like(at.flatten())), x)
        x0, eps = self.x0_estimate(x, t)
        sigma = eta * ((1 - ap) / (1 - at) * (1 - at / ap)).clamp_min(0).sqrt()
        noise = torch.randn_like(x) if noise is None else noise
        return ap.sqrt() * x0 + (1 - ap - sigma.square()).clamp_min(0).sqrt() * eps + sigma * noise

    @torch.no_grad()
    def sample(self, count=1, steps=None, initial=None, eta=0.):
        device = self.beta.device
        self._check_gaussian_start(initial)
        x = torch.randn(count, 1, self.size, self.size, device=device) if initial is None else initial.clone()
        if self.number == 55:
            if steps is not None and steps != self.timesteps:
                raise ValueError("DDPM ancestral sampling uses every trained step; select DDIM for step skipping")
            for i in reversed(range(self.timesteps)):
                t = torch.full((len(x),), i, device=device, dtype=torch.long)
                b, a = self.beta[i], 1 - self.beta[i]
                mean = (x - b / (1 - self.alpha_bar[i]).sqrt() * self(x, t)) / a.sqrt()
                prev = self.alpha_bar[i - 1] if i else torch.ones((), device=device)
                variance = b * (1 - prev) / (1 - self.alpha_bar[i])
                x = mean + variance.sqrt() * torch.randn_like(x) if i else mean
        else:
            seq = _diffusion_schedule(self.timesteps, steps, device)
            for k in reversed(range(len(seq))):
                t = seq[k].expand(len(x))
                p = (seq[k - 1] if k else torch.tensor(-1, device=device)).expand(len(x))
                x = self.ddim_step(x, t, p, eta)
        return x


class ScoreSDE(nn.Module):
    """Variance-preserving SDE and denoising score matching."""
    def __init__(self, size=16, width=8, **_):
        super().__init__()
        self.size, self.net = size, TimeUNet(width)

    @staticmethod
    def marginal(t):
        mean = torch.exp(-.25 * (20. - .1) * t.square() - .5 * .1 * t)
        return mean, (1 - mean.square()).sqrt()

    def forward(self, x, t):
        _, std = self.marginal(t)
        return self.net(x, t * 999) / _scalar(std.clamp_min(1e-4), x)

    def training_loss(self, batch):
        x = _image(batch, size=self.size) * 2 - 1
        t = torch.rand(len(x), device=x.device) * .999 + .001
        mean, std = self.marginal(t)
        noise = torch.randn_like(x)
        xt = _scalar(mean, x) * x + _scalar(std, x) * noise
        return (_scalar(std, x) * self(xt, t) + noise).square().mean()

    @torch.no_grad()
    def sample(self, count=1, steps=20, probability_flow=False, initial=None):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("SDE/ODE integration steps must be a positive integer")
        device = next(self.parameters()).device
        x = torch.randn(count, 1, self.size, self.size, device=device) if initial is None else initial.clone()
        edges = torch.linspace(1, .001, steps + 1, device=device)
        for value, following in zip(edges[:-1], edges[1:]):
            dt = following - value
            t = value.expand(len(x))
            beta = .1 + 19.9 * value
            drift = -.5 * beta * x - (.5 if probability_flow else 1.) * beta * self(x, t)
            x = x + drift * dt
            if not probability_flow:
                x = x + (-beta * dt).sqrt() * torch.randn_like(x)
        return x


class FlowMatching(nn.Module):
    def __init__(self, number=62, size=16, width=8, sigma_min=.001, **_):
        super().__init__()
        self.number, self.size, self.sigma_min = number, size, sigma_min
        self.net = TimeUNet(width)

    def forward(self, x, t):
        return self.net(x, t * 999)

    def path_and_velocity(self, noise, data, t):
        t = _scalar(t, data)
        s = self.sigma_min if self.number == 62 else 0.
        return (1 - (1 - s) * t) * noise + t * data, data - (1 - s) * noise

    def pair_loss(self, noise, data, t=None):
        if t is None:
            t = torch.rand(len(data), device=data.device)
        x, v = self.path_and_velocity(noise, data, t)
        return F.mse_loss(self(x, t), v)

    def training_loss(self, batch):
        x = _image(batch, size=self.size) * 2 - 1
        return self.pair_loss(torch.randn_like(x), x)

    @torch.no_grad()
    def sample(self, initial, steps=20):
        x = initial.clone()
        for i in range(steps):
            t = torch.full((len(x),), i / steps, device=x.device)
            x = x + self(x, t) / steps
        return x

    @torch.no_grad()
    def reflow_pairs(self, initial, steps=20):
        if self.number != 63:
            raise ValueError("Reflow endpoint recoupling belongs to Rectified Flow")
        return initial.detach().clone(), self.sample(initial, steps).detach()


class Consistency(nn.Module):
    def __init__(self, size=16, width=8, sigma_min=.002, sigma_max=5., **_):
        super().__init__()
        self.size, self.sigma_min, self.sigma_max = size, sigma_min, sigma_max
        self.net = TimeUNet(width)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def denoise(self, x, sigma, target=False):
        s = _scalar(sigma, x)
        c_skip = .25 / ((s - self.sigma_min).square() + .25)
        c_out = (s - self.sigma_min) * .5 / (s.square() + .25).sqrt()
        net = self.target if target else self.net
        return c_skip * x + c_out * net(x / (s.square() + .25).sqrt(), .25 * sigma.log())

    def forward(self, x, sigma):
        return self.denoise(x, sigma)

    @torch.no_grad()
    def update_ema(self, decay=.95):
        for p, q in zip(self.net.parameters(), self.target.parameters()):
            q.lerp_(p, 1 - decay)

    def training_loss(self, batch):
        self.calls.add_(1)
        x = _image(batch, size=self.size) * 2 - 1
        # Fixed K=16 for the bounded lesson. Original training grows K.
        grid = torch.linspace(self.sigma_min ** (1 / 7), self.sigma_max ** (1 / 7), 16, device=x.device).pow(7)
        i = torch.randint(15, (len(x),), device=x.device)
        noise = torch.randn_like(x)
        low, high = grid[i], grid[i + 1]
        with torch.no_grad():
            target = self.denoise(x + _scalar(low, x) * noise, low, target=True)
        return F.mse_loss(self.denoise(x + _scalar(high, x) * noise, high), target)

    def after_optimizer_step(self):
        self.update_ema()


class TransformerGenerator(Diffusion):
    def __init__(self, number=65, size=16, **cfg):
        super().__init__(number=number, size=size, **cfg)
        hidden_size = 8 * int(cfg.get("width", 8))
        if number == 65:
            src = _upstream("dit", "inverse_dit/models.py")
            self.net = src.DiT(input_size=size, patch_size=4, in_channels=1,
                               hidden_size=hidden_size, depth=2, num_heads=4, num_classes=3, learn_sigma=False)
        else:
            src = _upstream("sit", "inverse_sit/models.py")
            self.net = src.SiT(input_size=size, patch_size=4, in_channels=1,
                               hidden_size=hidden_size, depth=2, num_heads=4, num_classes=3, learn_sigma=False)

    def forward(self, x, t, label=None):
        if label is None:
            label = torch.zeros(len(x), device=x.device, dtype=torch.long)
        return self.net(x, t, label)

    def training_loss(self, batch):
        x = _image(batch, size=self.size) * 2 - 1
        label = batch["label"].long()
        noise = torch.randn_like(x)
        if self.number == 123:
            t = torch.rand(len(x), device=x.device)
            xt = (1 - _scalar(t, x)) * noise + _scalar(t, x) * x
            return F.mse_loss(self(xt, t, label), x - noise)
        t = torch.randint(self.timesteps, (len(x),), device=x.device)
        return F.mse_loss(self(self.q_sample(x, t, noise), t, label), noise)

    @torch.no_grad()
    def sample(self, count=1, steps=None, initial=None, eta=0.):
        if self.number == 65:
            return super().sample(count, steps, initial, eta)
        # SiT predicts velocity here; a DDIM epsilon update would be incorrect.
        x = torch.randn(count, 1, self.size, self.size, device=self.beta.device) if initial is None else initial.clone()
        steps = steps or 20
        for i in range(steps):
            x = x + self(x, x.new_full((len(x),), i / steps)) / steps
        return x


class PFGMPP(nn.Module):
    def __init__(self, size=16, width=8, augmented_dim=64, **_):
        super().__init__()
        self.size, self.augmented_dim = size, augmented_dim
        self.net = TimeUNet(width)

    def perturb(self, x, sigma):
        n, d = x[0].numel(), self.augmented_dim
        distribution = torch.distributions.Beta(torch.tensor(n / 2, device=x.device), torch.tensor(d / 2, device=x.device))
        b = distribution.sample((len(x),)).clamp(1e-6, 1 - 1e-6)
        radius = sigma * math.sqrt(d) * (b / (1 - b)).sqrt()
        direction = F.normalize(torch.randn_like(x).flatten(1), dim=1)
        return (direction * radius[:, None]).view_as(x)

    def forward(self, x, sigma):
        s = _scalar(sigma, x)
        return .25 / (s.square() + .25) * x + s * .5 / (s.square() + .25).sqrt() * self.net(x / (s.square() + .25).sqrt(), .25 * sigma.log())

    def training_loss(self, batch):
        x = _image(batch, size=self.size) * 2 - 1
        sigma = (torch.randn(len(x), device=x.device) * 1.2 - 1.2).exp()
        noise = self.perturb(x, sigma)
        weight = (sigma.square() + .25) / (sigma * .5).square()
        return (_scalar(weight, x) * (self(x + noise, sigma) - x).square()).mean()


class ParallelBeam(nn.Module):
    """Exact discrete bilinear ray-sum matrix and its algebraic transpose.

    Coordinates, detector spacing, and line integration are in a unit square;
    this is a 2D parallel-beam teaching geometry, not a clinical CT scanner.
    """
    def __init__(self, size=16, angles=12):
        super().__init__()
        self.size, self.angles = size, angles
        coords = (torch.arange(size) + .5) / size - .5
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        # Pixel-driven linear binning supplies a precisely defined A. Every
        # pixel contributes its line-length scale 1/size to two detector bins.
        rows = []
        for angle in torch.arange(angles) * math.pi / angles:
            detector = (xx * angle.cos() + yy * angle.sin()) * size + (size - 1) / 2
            base = detector.floor().long().flatten()
            weight = detector.flatten() - base
            mat = torch.zeros(size, size * size)
            col = torch.arange(size * size)
            for shift, w in [(0, 1 - weight), (1, weight)]:
                index = base + shift
                valid = (index >= 0) & (index < size)
                mat[index[valid], col[valid]] += w[valid] / size
            rows.append(mat)
        matrix = torch.cat(rows, 0)
        self.register_buffer("matrix", matrix)
        self.register_buffer("norm2", torch.linalg.matrix_norm(matrix, 2).square())

    def forward(self, image):
        return (image.flatten(1) @ self.matrix.T).view(-1, 1, self.angles, self.size)

    def adjoint(self, measurements):
        return (measurements.flatten(1) @ self.matrix).view(-1, 1, self.size, self.size)

    def fbp(self, sinogram):
        padded = 2 ** math.ceil(math.log2(2 * self.size))
        # Ramp filter in detector cycles per unit length, followed by A*.
        ramp = 2 * torch.fft.rfftfreq(padded, d=1 / self.size, device=sinogram.device)
        filtered = torch.fft.irfft(torch.fft.rfft(sinogram, n=padded) * ramp, n=padded)[..., :self.size]
        return self.adjoint(filtered) * math.pi * self.size / (2 * self.angles)


class FourierOperator(nn.Module):
    """Synthetic two-coil Cartesian acquisition with a unitary FFT."""
    def __init__(self, size=16, coils=2):
        super().__init__()
        self.size = size
        y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        maps = []
        for i in range(coils):
            angle = 2 * math.pi * i / coils
            magnitude = 1 + .25 * (x * math.cos(angle) + y * math.sin(angle))
            maps.append(magnitude * torch.exp(1j * (x * math.sin(angle) + y * math.cos(angle))))
        sensitivity = torch.stack(maps)
        sensitivity = sensitivity / sensitivity.abs().square().sum(0).sqrt()
        mask = torch.zeros(1, 1, size, size)
        mask[..., ::4, :] = 1
        # Unshifted FFT: central calibration frequencies wrap around the ends.
        mask[..., :2, :] = 1
        mask[..., -2:, :] = 1
        self.register_buffer("sensitivity", sensitivity[None])
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self.mask * torch.fft.fft2(x * self.sensitivity, norm="ortho")

    def adjoint(self, k):
        return (self.sensitivity.conj() * torch.fft.ifft2(k * self.mask, norm="ortho")).sum(1, keepdim=True)

    def normal(self, x):
        return self.adjoint(self(x))


def _channels(z):
    return torch.cat((z.real, z.imag), 1)


def _complex(x):
    return torch.complex(x[:, :1], x[:, 1:2])


def conjugate_gradient(operator, rhs, steps=8, initial=None):
    """Batched Hermitian positive-definite CG, differentiable through iterations."""
    x = torch.zeros_like(rhs) if initial is None else initial
    r = rhs - operator(x)
    p = r
    inner = lambda a, b: (a.conj() * b).real.flatten(1).sum(1)
    rr = inner(r, r)
    for _ in range(steps):
        ap = operator(p)
        alpha = rr / inner(p, ap).clamp_min(1e-12)
        x = x + _scalar(alpha, x) * p
        r = r - _scalar(alpha, r) * ap
        new = inner(r, r)
        p = r + _scalar(new / rr.clamp_min(1e-12), r) * p
        rr = new
    return x


class CTReconstructor(nn.Module):
    def __init__(self, number=47, size=16, width=8, stages=3, **_):
        super().__init__()
        self.number, self.size, self.stages = number, size, stages
        self.operator = ParallelBeam(size)
        if number == 47:
            self.net = SmallUNet(width=width)
        elif number == 48:
            self.primal = nn.ModuleList([_conv(6, 5, width) for _ in range(stages)])
            self.dual = nn.ModuleList([_conv(7, 5, width) for _ in range(stages)])
        else:
            self.regularizers = nn.ModuleList([_conv(1, 1, width) for _ in range(stages)])
            self.step = nn.Parameter(torch.full((stages,), -1.))

    def forward(self, sinogram):
        if self.number == 47:
            fbp = self.operator.fbp(sinogram)
            return fbp + self.net(fbp)
        if self.number == 48:
            primal = sinogram.new_zeros(len(sinogram), 5, self.size, self.size)
            dual = sinogram.new_zeros(len(sinogram), 5, self.operator.angles, self.size)
            for p, d in zip(self.primal, self.dual):
                dual = dual + d(torch.cat((dual, self.operator(primal[:, 1:2]), sinogram), 1))
                primal = primal + p(torch.cat((primal, self.operator.adjoint(dual[:, :1])), 1))
            return primal[:, :1]
        x = self.operator.adjoint(sinogram) / self.operator.norm2
        for step, cnn in zip(self.step, self.regularizers):
            residual = self.operator.adjoint(self.operator(x) - sinogram)
            x = x - F.softplus(step) / self.operator.norm2 * residual - cnn(x)
        return x

    def training_loss(self, batch):
        x = _image(batch, size=self.size)
        return F.mse_loss(self(self.operator(x)), x)


class RBFRegularizer(nn.Module):
    """K^T phi(Kx), with trainable Gaussian-RBF influence functions."""
    def __init__(self, width=8):
        super().__init__()
        self.filters = nn.Parameter(torch.randn(width, 2, 3, 3) * .05)
        self.coefficients = nn.Parameter(torch.randn(width, 11) * .01)
        self.register_buffer("centers", torch.linspace(-1, 1, 11))

    def forward(self, x):
        # Filter constraints of the variational-network parameterization.
        kernels = self.filters - self.filters.mean((1, 2, 3), keepdim=True)
        kernels = kernels / kernels.flatten(1).norm(dim=1).clamp_min(1.)[:, None, None, None]
        response = F.conv2d(x, kernels, padding=1)
        phi = torch.exp(-.5 * ((response[..., None] - self.centers) / .2).square())
        phi = (phi * self.coefficients[None, :, None, None]).sum(-1)
        return F.conv_transpose2d(phi, kernels, padding=1)


class MRIReconstructor(nn.Module):
    def __init__(self, number=50, size=16, width=8, stages=3, **_):
        super().__init__()
        self.number, self.size, self.stages = number, size, stages
        self.operator = FourierOperator(size)
        self.weight = nn.Parameter(torch.full((stages if number == 49 else 1,), -1.))
        if number == 49:
            self.regularizers = nn.ModuleList([RBFRegularizer(width) for _ in range(stages)])
        else:
            self.denoiser = _conv(2, 2, width, depth=5)

    def forward(self, kspace):
        rhs = self.operator.adjoint(kspace)
        x = rhs
        for i in range(self.stages):
            if self.number == 49:
                x = x - _complex(self.regularizers[i](_channels(x))) - F.softplus(self.weight[i]) * (self.operator.normal(x) - rhs)
            else:
                # One shared denoiser and lambda across all MoDL stages.
                lam = F.softplus(self.weight[0])
                denoised = x - _complex(self.denoiser(_channels(x)))
                x = conjugate_gradient(lambda v: self.operator.normal(v) + lam * v,
                                       rhs + lam * denoised, steps=8, initial=x)
        return x

    def training_loss(self, batch):
        x = _image(batch, size=self.size).to(torch.complex64)
        return (self(self.operator(x)) - x).abs().square().mean()


class VarNet(nn.Module):
    def __init__(self, size=32, width=8, stages=2, **_):
        super().__init__()
        import fastmri  # Pinned runtime supplies exact Fourier/data utilities.
        parent = "biomed_lab_upstream_fastmri_models"
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = [str(_VENDOR / "inverse_fastmri/fastmri/models")]
            sys.modules[parent] = pkg
        import importlib
        src = importlib.import_module(parent + ".varnet")
        self.size, self.fm = size, fastmri
        self.operator = FourierOperator(size)
        self.net = src.VarNet(num_cascades=stages, sens_chans=4, sens_pools=2, chans=width, pools=2)

    def acquire(self, image):
        coil_images = image * self.operator.sensitivity
        kspace = self.fm.fft2c(torch.view_as_real(coil_images))
        # fastMRI uses centred FFT and a [B,1,1,W,1] Boolean line mask.
        mask = torch.zeros(1, 1, 1, self.size, 1, device=image.device, dtype=torch.bool)
        mask[..., ::4, :] = True
        middle = self.size // 2
        mask[..., middle - 2:middle + 2, :] = True
        return kspace * mask, mask

    def forward(self, kspace, mask):
        return self.net(kspace, mask, num_low_frequencies=4)

    def training_loss(self, batch):
        x = _image(batch, size=self.size)
        kspace, mask = self.acquire(x)
        return F.mse_loss(self(kspace, mask), x[:, 0])


class AUTOMAP(nn.Module):
    def __init__(self, size=16, width=8, **_):
        super().__init__()
        self.size = size
        self.transform = nn.Sequential(nn.Linear(2 * size * size, size * size), nn.Tanh(),
                                       nn.Linear(size * size, size * size), nn.Tanh())
        self.refine = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                    nn.ConvTranspose2d(width, 1, 3, padding=1))

    def forward(self, kspace):
        image = self.transform(torch.view_as_real(kspace).flatten(1)).view(-1, 1, self.size, self.size)
        return self.refine(image)

    def training_loss(self, batch):
        x = _image(batch, size=self.size)
        return F.mse_loss(self(torch.fft.fft2(x, norm="ortho")), x)


class DiffusionInverse(Diffusion):
    """Prior training and three *different* measurement-conditioned samplers.

    The bounded training run trains a prior. Call reconstruct explicitly for
    inverse sampling. A few smoke steps cannot produce a useful learned prior.
    """
    def __init__(self, number=119, **cfg):
        super().__init__(number=number, **cfg)
        mask = torch.ones(1, 1, self.size, self.size)
        mask[..., self.size // 4:3 * self.size // 4, self.size // 4:3 * self.size // 4] = 0
        self.register_buffer("measurement_mask", mask)

    def measure(self, x):
        return x * self.measurement_mask

    def ddnm_project(self, x0, measurement):
        # For a binary inpainting operator, A†=A and A†A=A.
        return x0 + self.measurement_mask * (measurement - x0)

    def dps_step(self, x, t, measurement, strength=.1):
        # Enable gradients even during an enclosing inference/validation block.
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            x0, eps = self.x0_estimate(x, t)
            residual = (self.measure(x0) - measurement).flatten(1).norm(dim=1).sum()
            guidance = torch.autograd.grad(residual, x)[0]
            b = _scalar(self.beta[t], x)
            ab = _scalar(self.alpha_bar[t], x)
            ap = _scalar(torch.where(t > 0, self.alpha_bar[(t - 1).clamp_min(0)], torch.ones_like(t, dtype=x.dtype)), x)
            mean = (x - b / (1 - ab).sqrt() * eps) / (1 - b).sqrt()
            variance = b * (1 - ap) / (1 - ab)
            sample = mean + variance.sqrt() * torch.randn_like(x)
            return (sample - strength * guidance).detach()

    def ddrm_spectral_step(self, x0, eps, measurement, sigma_next, sigma_y=.05, eta=.85):
        """DDRM SVD cases for a binary diagonal H (nonzero singulars=1).

        All values are in x/sqrt(alpha_bar) coordinates. Missing components,
        observed components noisier than y, and those cleaner than y have
        separate conditional transitions. This is not a generic projection.
        """
        sigma = torch.as_tensor(sigma_next, device=x0.device, dtype=x0.dtype)
        std = sigma * eta
        deterministic = sigma * math.sqrt(1 - eta * eta)
        missing = x0 + deterministic * eps + std * torch.randn_like(x0)
        if sigma >= sigma_y:
            observed = measurement + (sigma.square() - sigma_y ** 2).clamp_min(0).sqrt() * torch.randn_like(x0)
        else:
            observed = x0 + deterministic * (measurement - x0) / max(sigma_y, 1e-12) + std * torch.randn_like(x0)
        return torch.where(self.measurement_mask.bool(), observed, missing)

    @torch.no_grad()
    def reconstruct(self, measurement, steps=None, sigma_y=.05, strength=.1, initial=None):
        device = measurement.device
        self._check_gaussian_start(initial)
        x = torch.randn_like(measurement) if initial is None else initial.clone()
        if self.number == 60:
            if steps is not None and steps != self.timesteps:
                raise ValueError("The DPS lesson uses consecutive DDPM steps; step skipping changes this algorithm")
            for i in reversed(range(self.timesteps)):
                x = self.dps_step(x, torch.full((len(x),), i, device=device, dtype=torch.long), measurement, strength)
            return x
        seq = _diffusion_schedule(self.timesteps, steps, device)
        if self.number == 59:
            a = self.alpha_bar[seq[-1]]
            sigma = ((1 - a) / a).sqrt()
            if sigma < sigma_y:
                raise ValueError("Largest diffusion scale must exceed measurement-noise scale")
            observed = measurement + (sigma.square() - sigma_y ** 2).sqrt() * x
            x = a.sqrt() * torch.where(self.measurement_mask.bool(), observed, sigma * x)
        for k in reversed(range(len(seq))):
            t = seq[k].expand(len(x))
            previous = seq[k - 1] if k else -1
            ap = self.alpha_bar[previous] if k else x.new_tensor(1.)
            x0, eps = self.x0_estimate(x, t)
            if self.number == 119:
                x0 = self.ddnm_project(x0, measurement)
                # DDNM's noiseless range correction, with deterministic DDIM.
                x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps
            else:
                spectral = self.ddrm_spectral_step(x0, eps, measurement, ((1 - ap) / ap).sqrt(), sigma_y)
                x = ap.sqrt() * spectral
        return x


class AffineCoupling(nn.Module):
    def __init__(self, features, width=32, flip=False):
        super().__init__()
        mask = (torch.arange(features) % 2 == int(flip)).float()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(features, width), nn.ReLU(), nn.Linear(width, 2 * features))

    def _params(self, x):
        s, t = self.net(x * self.mask).chunk(2, -1)
        return s.tanh() * (1 - self.mask), t * (1 - self.mask)

    def forward(self, x):
        s, t = self._params(x)
        return x * s.exp() + t, s.sum(-1)

    def inverse(self, y):
        s, t = self._params(y)
        return (y - t) * (-s).exp()


class RealNVP(nn.Module):
    def __init__(self, size=8, width=32, layers=4, **_):
        super().__init__()
        self.size = size
        self.layers = nn.ModuleList([AffineCoupling(size * size, width, bool(i % 2)) for i in range(layers)])

    def forward(self, x):
        x = x.flatten(1)
        logdet = x.new_zeros(len(x))
        for layer in self.layers:
            x, change = layer(x)
            logdet = logdet + change
        return x, logdet

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z.view(-1, 1, self.size, self.size)

    def training_loss(self, batch):
        x = _image(batch, size=self.size)
        # Uniform dequantization of synthetic 8-bit observations; report a
        # discrete-density bound, not the likelihood of an undequantized grid.
        x = (torch.floor(x.clamp(0, 1) * 255) + torch.rand_like(x)) / 256 - .5
        z, ld = self(x)
        return (.5 * (z.square() + math.log(2 * math.pi)).sum(1) - ld + z.shape[1] * math.log(256)).mean() / z.shape[1]


class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x):
        if not bool(self.initialized):
            with torch.no_grad():
                self.bias.copy_(-x.mean((0, 2, 3), keepdim=True))
                self.log_scale.copy_(-x.std((0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-3).log())
                self.initialized.fill_(True)
        y = (x + self.bias) * self.log_scale.exp()
        return y, self.log_scale.sum() * x.shape[2] * x.shape[3]

    def inverse(self, y):
        if not bool(self.initialized):
            raise ValueError("Initialize ActNorm with a data batch before inverse sampling")
        return y * (-self.log_scale).exp() - self.bias


class GlowStep(nn.Module):
    def __init__(self, channels=4, width=8):
        super().__init__()
        self.norm = ActNorm(channels)
        orthogonal, _ = torch.linalg.qr(torch.randn(channels, channels))
        self.mix = nn.Parameter(orthogonal)
        self.net = _conv(channels // 2, channels, width)

    def forward(self, x):
        x, ld = self.norm(x)
        x = F.conv2d(x, self.mix[:, :, None, None])
        ld = ld + torch.linalg.slogdet(self.mix)[1] * x.shape[2] * x.shape[3]
        a, b = x.chunk(2, 1)
        s, t = self.net(a).chunk(2, 1)
        s = s.tanh()
        return torch.cat((a, b * s.exp() + t), 1), ld + s.flatten(1).sum(1)

    def inverse(self, y):
        a, b = y.chunk(2, 1)
        s, t = self.net(a).chunk(2, 1)
        x = torch.cat((a, (b - t) * (-s.tanh()).exp()), 1)
        x = F.conv2d(x, torch.linalg.inv(self.mix)[:, :, None, None])
        return self.norm.inverse(x)


class Glow(RealNVP):
    def __init__(self, size=8, width=8, layers=3, **_):
        nn.Module.__init__(self)
        if size % 2:
            raise ValueError("Glow squeeze requires an even image size")
        self.size = size
        self.layers = nn.ModuleList([GlowStep(4, width) for _ in range(layers)])

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        ld = x.new_zeros(len(x))
        for layer in self.layers:
            x, delta = layer(x)
            ld = ld + delta
        return x.flatten(1), ld

    def inverse(self, z):
        x = z.view(-1, 4, self.size // 2, self.size // 2)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return F.pixel_shuffle(x, 2)


def coordinate_grid(shape, device=None, dtype=torch.float32):
    axes = [torch.linspace(-1, 1, n, device=device, dtype=dtype) for n in shape]
    grid = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(tuple(reversed(grid)), -1)[None]


def warp(moving, flow):
    """Pull-back sampling: flow=(x,y,z) in dimensionless [-1,1] coordinates."""
    identity = coordinate_grid(moving.shape[2:], moving.device, moving.dtype)
    grid = identity + flow.movedim(1, -1)
    return F.grid_sample(moving, grid, align_corners=True, padding_mode="border")


def gradient_energy(flow):
    return sum((torch.diff(flow, dim=axis) * (flow.shape[axis] - 1) / 2).square().mean()
               for axis in range(2, flow.ndim))


def inverse_composition(u, v):
    return u + warp(v, u)


def gradicon_penalty(u, v):
    # Differentiate composition error, NOT merely penalize its displacement.
    return gradient_energy(inverse_composition(u, v))


def _registration_pair(batch):
    fixed = batch["volume"].float()
    if fixed.shape[-3:] != (16, 16, 16):
        fixed = F.interpolate(fixed, (16, 16, 16), mode="trilinear", align_corners=True)
    flow = fixed.new_zeros(len(fixed), 3, *fixed.shape[2:])
    flow[:, 0] = .12
    flow[:, 1] = -.06
    moving = warp(fixed, flow)
    return moving, fixed


class SpatialTransformer(nn.Module):
    def __init__(self, width=8, **_):
        super().__init__()
        self.localizer = nn.Sequential(_conv(2, width, width), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 6))
        nn.init.zeros_(self.localizer[-1].weight)
        with torch.no_grad():
            self.localizer[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, moving, fixed):
        theta = self.localizer(torch.cat((moving, fixed), 1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, moving.shape, align_corners=True)
        return F.grid_sample(moving, grid, align_corners=True, padding_mode="border"), theta

    def training_loss(self, batch):
        fixed = _image(batch, size=32)
        shift = fixed.new_tensor([[1., 0., .15], [0., 1., -.1]])[None].expand(len(fixed), -1, -1)
        moving = F.grid_sample(fixed, F.affine_grid(shift, fixed.shape, align_corners=True), align_corners=True)
        return F.mse_loss(self(moving, fixed)[0], fixed)


class Registration(nn.Module):
    def __init__(self, number=67, width=4, **_):
        super().__init__()
        self.number = number
        self.net = SmallUNet(2, 3, width, dim=3)

    def flow(self, moving, fixed):
        return .15 * self.net(torch.cat((moving, fixed), 1)).tanh()

    def forward(self, moving, fixed):
        displacement = self.flow(moving, fixed)
        return warp(moving, displacement), displacement

    def training_loss(self, batch):
        moving, fixed = _registration_pair(batch)
        predicted, u = self(moving, fixed)
        if self.number == 67:
            return F.mse_loss(predicted, fixed) + .01 * gradient_energy(u)
        reverse, v = self(fixed, moving)
        return F.mse_loss(predicted, fixed) + F.mse_loss(reverse, moving) + .1 * (gradicon_penalty(u, v) + gradicon_penalty(v, u))


class LapIRN(nn.Module):
    def __init__(self, width=4, **_):
        super().__init__()
        self.levels = nn.ModuleList([SmallUNet(2 if i == 0 else 5, 3, width, dim=3) for i in range(3)])

    def forward(self, moving, fixed):
        flow = None
        for level, size in zip(self.levels, [4, 8, 16]):
            m = F.interpolate(moving, (size,) * 3, mode="trilinear", align_corners=True)
            f = F.interpolate(fixed, (size,) * 3, mode="trilinear", align_corners=True)
            if flow is None:
                flow = .15 * level(torch.cat((m, f), 1)).tanh()
            else:
                # Normalized coordinates keep the same displacement units at
                # all scales; no erroneous factor-two multiplication is needed.
                flow = F.interpolate(flow, (size,) * 3, mode="trilinear", align_corners=True)
                increment = .1 * level(torch.cat((warp(m, flow), f, flow), 1)).tanh()
                flow = increment + warp(flow, increment)
        return warp(moving, flow), flow

    def training_loss(self, batch):
        moving, fixed = _registration_pair(batch)
        predicted, flow = self(moving, fixed)
        return F.mse_loss(predicted, fixed) + .01 * gradient_energy(flow)


class TransMorph(nn.Module):
    """Original four-level TransMorph hierarchy, decoder and voxel-unit warp."""
    def __init__(self, width=8, size=32, **_):
        super().__init__()
        if size < 32 or size % 32:
            raise ValueError("Original TransMorph patch/merge hierarchy requires size divisible by 32")
        if width < 2 or width % 2:
            raise ValueError("TransMorph embedding width must be an even integer >= 2")
        config_src = _upstream("transmorph_configs", "inverse_transmorph/TransMorph/models/configs_TransMorph.py")
        src = _upstream("transmorph", "inverse_transmorph/TransMorph/models/TransMorph.py", {
            "import models.configs_TransMorph as configs": "import biomed_lab_upstream_transmorph_configs as configs"})
        losses = _upstream("transmorph_losses", "inverse_transmorph/TransMorph/losses.py")
        cfg = config_src.get_3DTransMorph_config()
        cfg.img_size = (size, size, size)
        cfg.embed_dim = width
        cfg.depths = (2, 2, 2, 2)
        cfg.num_heads = (1, 2, 4, 8)
        cfg.window_size = (2, 2, 2)
        cfg.reg_head_chan = max(2, width // 2)
        cfg.drop_path_rate = 0.
        # Original baseline already uses relative position embeddings; its
        # CUDA-only optional sinusoidal-position branch remains disabled.
        self.size, self.net = size, src.TransMorph(cfg)
        self.regularity = losses.Grad3d(penalty="l2")

    def forward(self, moving, fixed):
        return self.net(torch.cat((moving, fixed), dim=1))

    @staticmethod
    def normalized_xyz_flow(voxel_dhw_flow):
        d, h, w = voxel_dhw_flow.shape[-3:]
        scale = voxel_dhw_flow.new_tensor([2 / (w - 1), 2 / (h - 1), 2 / (d - 1)]).view(1, 3, 1, 1, 1)
        return voxel_dhw_flow[:, [2, 1, 0]] * scale

    def training_loss(self, batch):
        moving, fixed = _registration_pair(batch)
        moving = F.interpolate(moving, (self.size,) * 3, mode="trilinear", align_corners=True)
        fixed = F.interpolate(fixed, (self.size,) * 3, mode="trilinear", align_corners=True)
        forward, forward_flow = self(moving, fixed)
        reverse, reverse_flow = self(fixed, moving)
        # The official baseline training script uses MSE + 0.02 Grad3d(L2),
        # with both directions. The lesson accumulates both before one update.
        return .5 * (F.mse_loss(forward, fixed) + F.mse_loss(reverse, moving)
                     + .02 * (self.regularity(forward_flow, fixed) + self.regularity(reverse_flow, moving)))


class SineLayer(nn.Module):
    def __init__(self, cin, cout, first=False, omega=30.):
        super().__init__()
        self.linear, self.omega = nn.Linear(cin, cout), omega
        bound = 1 / cin if first else math.sqrt(6 / cin) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class VariableBufferModule(nn.Module):
    """Restore cached scenes/pairs from a checkpoint with their true shape."""
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("fixed", "moving", "scene", "observation"):
            key = prefix + name
            if name in self._buffers and key in state_dict:
                old = self._buffers[name]
                if old.shape != state_dict[key].shape:
                    self._buffers[name] = torch.empty_like(state_dict[key], device=old.device)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class IDIR(VariableBufferModule):
    def __init__(self, width=32, **_):
        super().__init__()
        self.field = nn.Sequential(SineLayer(3, width, True), SineLayer(width, width), nn.Linear(width, 3))
        nn.init.zeros_(self.field[-1].bias)
        nn.init.normal_(self.field[-1].weight, std=1e-3)
        self.register_buffer("fixed", torch.empty(0))
        self.register_buffer("moving", torch.empty(0))

    def forward(self, coordinates):
        return self.field(coordinates) * .1

    def training_loss(self, batch):
        if not self.fixed.numel():
            moving, fixed = _registration_pair(batch)
            # IDIR optimizes one image pair at test time. Never silently fit one
            # unconditional deformation field to several different patients.
            self.fixed, self.moving = fixed[:1].detach().clone(), moving[:1].detach().clone()
        with torch.enable_grad():
            coords = coordinate_grid(self.fixed.shape[2:], self.fixed.device).requires_grad_(True)
            disp = self(coords)
            prediction = F.grid_sample(self.moving, coords + disp, align_corners=True, padding_mode="border")
            jac = [torch.autograd.grad(disp[..., i].sum(), coords, create_graph=True)[0] for i in range(3)]
            regularity = sum(j.square().mean() for j in jac)
            return F.mse_loss(prediction, self.fixed) + .01 * regularity


def rigid_matrix(pose):
    rotation, translation = pose[:, :3], pose[:, 3:]
    zero = torch.zeros_like(rotation[:, 0])
    x, y, z = rotation.unbind(-1)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).view(-1, 3, 3)
    return torch.cat((torch.matrix_exp(skew), translation[..., None]), -1)


def parallel_drr(volume, pose):
    # A normalized object-frame parallel-beam DRR, not a calibrated C-arm.
    theta = rigid_matrix(pose)
    grid = F.affine_grid(theta, volume.shape, align_corners=True)
    sample = F.grid_sample(volume, grid, align_corners=True, padding_mode="zeros")
    attenuation = torch.trapezoid(sample, dx=2 / (volume.shape[2] - 1), dim=2)
    return attenuation


class DiffPoseDemo(nn.Module):
    def __init__(self, width=8, **_):
        super().__init__()
        self.initializer = nn.Sequential(_conv(1, width, width), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 6))
        nn.init.zeros_(self.initializer[-1].weight)
        nn.init.zeros_(self.initializer[-1].bias)

    def forward(self, projection):
        return .25 * self.initializer(projection).tanh()

    def training_loss(self, batch):
        volume = batch["volume"].float()
        true_pose = torch.randn(len(volume), 6, device=volume.device) * .1
        target = parallel_drr(volume, true_pose)
        pose = self(target)
        return F.mse_loss(pose, true_pose) + F.mse_loss(parallel_drr(volume, pose), target)

    def refine(self, volume, target, initial, steps=10, learning_rate=.2):
        pose = initial.detach().clone()
        for _ in range(steps):
            with torch.enable_grad():
                pose.requires_grad_(True)
                loss = F.mse_loss(parallel_drr(volume, pose), target)
                grad = torch.autograd.grad(loss, pose)[0]
                pose = (pose - learning_rate * grad).detach()
        return pose


class X2CTDemo(nn.Module):
    def __init__(self, width=4, **_):
        super().__init__()
        self.ap = _conv(1, width, width)
        self.lat = _conv(1, width, width)
        self.decoder = SmallUNet(2 * width, 1, width, dim=3)
        self.discriminator = nn.Sequential(_conv(1, width, width, dim=3), nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(width, 1))

    def forward(self, ap, lateral):
        # volume axes D,H,W: AP integrates D; lateral integrates W.
        size = ap.shape[-1]
        ap_features = self.ap(ap).unsqueeze(2).expand(-1, -1, size, -1, -1)
        lateral_features = self.lat(lateral).unsqueeze(-1).expand(-1, -1, -1, -1, size)
        return self.decoder(torch.cat((ap_features, lateral_features), 1)).sigmoid()

    def training_loss(self, batch):
        volume = batch["volume"].float()
        ap, lateral = volume.mean(2), volume.mean(4)
        fake = self(ap, lateral)
        d = F.softplus(-self.discriminator(volume)).mean() + F.softplus(self.discriminator(fake.detach())).mean()
        with _frozen(self.discriminator):
            g = F.softplus(-self.discriminator(fake)).mean()
        projections = F.l1_loss(fake.mean(2), ap) + F.l1_loss(fake.mean(4), lateral)
        return d + .01 * g + F.l1_loss(fake, volume) + projections


class FourierFeatures(nn.Module):
    def __init__(self, dimensions=3, levels=4):
        super().__init__()
        self.output_dim = dimensions * (1 + 2 * levels)
        self.register_buffer("frequencies", math.pi * 2 ** torch.arange(levels).float())

    def forward(self, x):
        phase = x[..., None] * self.frequencies
        return torch.cat((x, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


class HashGrid(nn.Module):
    """Differentiable multiresolution hash encoding with trilinear lookup."""
    def __init__(self, levels=4, features=2, table_size=256):
        super().__init__()
        self.tables = nn.ParameterList([nn.Parameter(torch.empty(table_size, features).uniform_(-1e-4, 1e-4)) for _ in range(levels)])
        self.resolutions = [4 * 2 ** level for level in range(levels)]
        self.output_dim, self.table_size = levels * features, table_size
        self.register_buffer("corners", torch.tensor([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=torch.long))

    def forward(self, coordinates):
        unit = (coordinates + 1) / 2
        outputs = []
        for resolution, table in zip(self.resolutions, self.tables):
            scaled = unit * resolution
            base = scaled.floor().long()
            frac = scaled - base
            vertex = base[..., None, :] + self.corners
            index = (vertex[..., 0] ^ (vertex[..., 1] * 2654435761) ^ (vertex[..., 2] * 805459861)) % self.table_size
            weights = torch.where(self.corners.bool(), frac[..., None, :], 1 - frac[..., None, :]).prod(-1)
            outputs.append((table[index] * weights[..., None]).sum(-2))
        return torch.cat(outputs, -1)


class RadianceField(nn.Module):
    def __init__(self, width=32, hashed=False, condition_dim=0):
        super().__init__()
        self.encoding = HashGrid() if hashed else FourierFeatures()
        self.direction_encoding = FourierFeatures(levels=2)
        self.body = nn.Sequential(nn.Linear(self.encoding.output_dim + condition_dim, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.density = nn.Linear(width, 1)
        self.color = nn.Sequential(nn.Linear(width + self.direction_encoding.output_dim, width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, points, directions, condition=None):
        encoding = self.encoding(points)
        if condition is not None:
            encoding = torch.cat((encoding, condition), -1)
        h = self.body(encoding)
        sigma = F.softplus(self.density(h))[..., 0]
        color = self.color(torch.cat((h, self.direction_encoding(directions)), -1)).sigmoid()
        inside = (points.abs() <= 1).all(-1)
        return sigma * inside, color


def camera_rays(size=8, samples=16, yaw=0., device=None):
    yy, xx = torch.meshgrid(torch.linspace(-.45, .45, size, device=device), torch.linspace(-.45, .45, size, device=device), indexing="ij")
    direction = F.normalize(torch.stack((xx, yy, torch.ones_like(xx)), -1).view(-1, 3), dim=-1)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], device=device)
    direction = direction @ rotation.T
    origin = torch.tensor([0., 0., -2.], device=device) @ rotation.T
    # Midpoint quadrature on t∈[1,3.5], with true Euclidean ray step length.
    edges = torch.linspace(1., 3.5, samples + 1, device=device)
    t = (edges[:-1] + edges[1:]) / 2
    points = origin[None, None] + direction[:, None] * t[None, :, None]
    return points, direction[:, None].expand_as(points), edges[1:] - edges[:-1], t


def volume_render(sigma, color, delta, distances=None):
    optical = sigma * delta
    alpha = -torch.expm1(-optical)
    # Exclusive accumulated optical thickness; avoids inclusive-transmittance
    # and arbitrary final 1e10-depth errors on this finite-domain phantom.
    transmittance = torch.exp(-(torch.cumsum(optical, -1) - optical))
    weights = alpha * transmittance
    rgb = (weights[..., None] * color).sum(-2)
    depth = None if distances is None else (weights * distances).sum(-1)
    return rgb, depth, weights


def _sample_volume(volume, points):
    if points.ndim == 3:
        points = points[None].expand(len(volume), -1, -1, -1)
    grid = points[:, :, :, None, :]
    return F.grid_sample(volume, grid, align_corners=True, padding_mode="zeros")[:, 0, :, :, 0]


class FieldLesson(VariableBufferModule):
    def __init__(self, number=72, width=32, ray_size=8, ray_samples=16, **_):
        super().__init__()
        self.number, self.ray_size, self.ray_samples = number, ray_size, ray_samples
        self.field = RadianceField(width, hashed=number in {73, 129}, condition_dim=8 if number == 128 else 0)
        if number == 128:
            self.encoder = _conv(3, 8, 8)
        if number == 131:
            self.deformation = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, 3))
        self.register_buffer("scene", torch.empty(0))

    def truth(self, volume, points, delta, distances, time=0.):
        canonical = points.clone()
        if self.number == 131:
            canonical[..., 0] -= .15 * math.sin(2 * math.pi * time)
        density = _sample_volume(volume, canonical) * 3
        color = ((canonical + 1) / 2).clamp(0, 1).expand(len(volume), -1, -1, -1)
        return volume_render(density, color, delta, distances)

    def deform_points(self, points, time):
        if self.number == 131:
            t = torch.full_like(points[..., :1], time)
            # The target translation is .15*sin(2*pi*t): its derivative at 0
            # has magnitude .3*pi < 1. Anchoring by t with unit amplitude can
            # represent the entire target path; a .1*t bound cannot.
            return points + time * self.deformation(torch.cat((points, t), -1)).tanh()
        return points

    def query(self, points, directions, reference_features=None, time=0.):
        if points.ndim == 3:
            points = points[None]
            directions = directions[None]
        canonical = self.deform_points(points, time)
        features = None
        if reference_features is not None:
            # Known pinhole reference camera at (0,0,-2), pixel domain ±.45.
            uv = points[..., :2] / (points[..., 2:] + 2).clamp_min(.1) / .45
            features = F.grid_sample(reference_features, uv, align_corners=True, padding_mode="zeros").movedim(1, -1)
        return self.field(canonical, directions, features)

    def forward(self, yaw=0., reference_features=None, time=0.):
        device = next(self.parameters()).device
        p, d, delta, distance = camera_rays(self.ray_size, self.ray_samples, yaw, device)
        if reference_features is not None:
            p, d = p[None].expand(len(reference_features), -1, -1, -1), d[None].expand(len(reference_features), -1, -1, -1)
        sigma, color = self.query(p, d, reference_features, time)
        if self.number == 73:
            # X-rays obey attenuation line integrals, not emitted RGB compositing.
            return (sigma * delta).sum(-1)
        return volume_render(sigma, color, delta, distance)

    def training_loss(self, batch):
        volume = batch["volume"].float()
        if self.number != 128:
            if not self.scene.numel():
                self.scene = volume[:1].detach().clone()
            volume = self.scene
        device = volume.device
        yaw = float(torch.rand(()) * 1.2 - .6)
        time = float(torch.rand(())) if self.number == 131 else 0.
        points, _, delta, distances = camera_rays(self.ray_size, self.ray_samples, yaw, device)
        if self.number == 73:
            target = (_sample_volume(volume, points) * 3 * delta).sum(-1)
            return F.mse_loss(self(yaw), target)
        target_rgb, target_depth, _ = self.truth(volume, points, delta, distances, time)
        features = None
        if self.number == 128:
            p, _, delta0, t0 = camera_rays(self.ray_size, self.ray_samples, 0., device)
            reference, _, _ = self.truth(volume, p, delta0, t0)
            reference = reference.view(len(volume), self.ray_size, self.ray_size, 3).movedim(-1, 1)
            features = self.encoder(reference)
        rgb, depth, _ = self(yaw, features, time)
        if self.number == 131:
            # Synthetic instrument occlusion: exclude a moving stripe. Ground
            # truth depth is simulated, not claimed to be clinical stereo data.
            col = torch.arange(self.ray_size, device=device).repeat(self.ray_size)
            visible = (col != int(time * (self.ray_size - 1)))[None].expand(len(volume), -1)
            return (rgb[visible] - target_rgb[visible]).square().mean() + .1 * (depth[visible] - target_depth[visible]).square().mean()
        return F.mse_loss(rgb, target_rgb)


class GaussianSplattingDemo(VariableBufferModule):
    """CPU dense rasterizer: anisotropic 3D covariances and alpha compositing."""
    def __init__(self, gaussians=16, size=16, **_):
        super().__init__()
        self.size = size
        self.position = nn.Parameter(torch.randn(gaussians, 3) * .3 + torch.tensor([0., 0., 2.]))
        self.log_scale = nn.Parameter(torch.full((gaussians, 3), -2.))
        self.rotation = nn.Parameter(torch.zeros(gaussians, 3))
        self.opacity = nn.Parameter(torch.full((gaussians,), -1.))
        self.color = nn.Parameter(torch.randn(gaussians, 3) * .1)
        self.register_buffer("observation", torch.empty(0))

    def projected(self):
        p = self.position
        z = p[:, 2].clamp_min(.1)
        mean = p[:, :2] / z[:, None]
        pose = torch.cat((self.rotation, torch.zeros_like(self.rotation)), -1)
        rotation = rigid_matrix(pose)[:, :, :3]
        covariance = rotation @ torch.diag_embed((2 * self.log_scale).exp()) @ rotation.transpose(-1, -2)
        zero = torch.zeros_like(z)
        jacobian = torch.stack((1 / z, zero, -p[:, 0] / z.square(), zero, 1 / z, -p[:, 1] / z.square()), -1).view(-1, 2, 3)
        covariance_2d = jacobian @ covariance @ jacobian.transpose(-1, -2)
        covariance_2d = covariance_2d + torch.eye(2, device=p.device) * 1e-4
        return mean, covariance_2d, z

    def forward(self):
        mean, covariance, z = self.projected()
        grid = coordinate_grid((self.size, self.size), mean.device)[0].reshape(-1, 2) * .5
        offset = grid[None] - mean[:, None]
        exponent = torch.einsum("gpi,gij,gpj->gp", offset, torch.linalg.inv(covariance), offset)
        alpha = (self.opacity.sigmoid()[:, None] * torch.exp(-.5 * exponent)).clamp_max(.99)
        order = torch.argsort(z)
        alpha = alpha[order]
        transmission = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1 - alpha[:-1]), 0), 0)
        color = self.color.sigmoid()[order]
        rgb = (transmission[..., None] * alpha[..., None] * color[:, None]).sum(0)
        return rgb.T.view(1, 3, self.size, self.size)

    def training_loss(self, batch):
        # This per-scene explicit representation fits one synthetic observation.
        if not self.observation.numel():
            self.observation = _image(batch, size=self.size)[:1].expand(-1, 3, -1, -1).detach().clone()
        return F.mse_loss(self(), self.observation)


_ROWS = {
    43: ("Deep Autoencoder", "generative", "paper_port", "https://www.cs.toronto.edu/~hinton/absps/science.pdf", "Layerwise contrastive-divergence RBM pretraining, Gaussian bottleneck, then unfolded untied autoencoder fine-tuning.", "Two compact RBMs replace the paper's large four-layer example; default first two training calls pretrain and later calls fine-tune."),
    44: ("GAN", "generative", "paper_port", "https://arxiv.org/abs/1406.2661", "Separate discriminator and non-saturating generator gradients implement the two-player game.", "Small MLP generator/discriminator; one simultaneous update instead of a tuned adversarial training schedule."),
    45: ("pix2pix", "translation", "upstream", "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix", "Original recursive U-Net and 70x70 conditional PatchGAN with logistic adversarial and paired L1 objectives.", "Width 8 and five U-Net downsamplings at 64px; default examples are synthetic noisy/clean pairs."),
    46: ("CycleGAN", "translation", "upstream", "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix", "Original ResNet generators and PatchGANs; dual LSGAN, cycle and identity losses on shuffled domains.", "Three residual blocks; no replay buffer or long-horizon learning-rate schedule; synthetic domain translation only."),
    47: ("FBPConvNet", "reconstruction", "mechanism_demo", "https://doi.org/10.1109/TMI.2017.2713099", "A discrete parallel-beam ramp-filtered backprojection precedes residual image-domain U-Net refinement.", "Reduced U-Net and pixel-driven teaching projector; not the original MatConvNet/CT geometry implementation."),
    48: ("Learned Primal-Dual", "reconstruction", "paper_port", "https://arxiv.org/abs/1707.06474", "Five-channel primal and dual memories update through distinct CNNs and explicit A and its exact transpose at every stage.", "Three stages and a 16px parallel-beam phantom replace the full ODL experiment."),
    49: ("Variational Network", "reconstruction", "paper_port", "https://doi.org/10.1002/mrm.26977", "MRI energy-gradient updates use constrained convolution filters, learned RBF influence functions and coil-aware Fourier data gradients.", "Small filter bank and three stages; synthetic smooth coil maps are supplied, not learned."),
    50: ("MoDL", "reconstruction", "paper_port", "https://arxiv.org/abs/1712.02862", "One shared residual CNN alternates with differentiable CG solves of (A* A + lambda I)x=A*y+lambda D(x).", "Eight CG iterations approximate each solve; three unrolls and synthetic Cartesian two-coil data."),
    51: ("VarNet", "reconstruction", "upstream", "https://github.com/facebookresearch/fastMRI", "Original fastMRI sensitivity estimation and normalized-U-Net k-space cascades.", "Two cascades, smaller widths and four calibration lines; synthetic phantom MSE replaces the original benchmark training loss/data."),
    52: ("DnCNN", "restoration", "upstream", "https://github.com/cszn/KAIR", "Author-recommended DnCNN code predicts the additive noise residual and subtracts it from the observation.", "Seventeen convolution layers retained, width reduced from 64 to 8; no pretrained weights."),
    53: ("Noise2Noise", "restoration", "paper_port", "https://arxiv.org/abs/1803.04189", "Regress a second independent unclipped zero-mean noisy acquisition from the first; clean data is not read by the loss.", "Compact U-Net instead of the original TensorFlow architecture; independence is a dataset assumption, not inferred from image appearance."),
    54: ("Noise2Self", "restoration", "paper_port", "https://proceedings.mlr.press/v97/batson19a.html", "Nine disjoint pixel subsets are masked; predictions on J do not depend on any measured input in J.", "Deterministic zero replacement and a compact U-Net; full J-invariant inference needs nine forward passes."),
    55: ("DDPM", "generative", "paper_port", "https://arxiv.org/abs/2006.11239", "Original linear beta schedule, epsilon-MSE objective and consecutive ancestral reverse transitions.", "Compact time-conditioned U-Net; default prior training uses 1000 time indices, and ancestral generation requires all 1000 steps."),
    56: ("DDIM", "generative", "paper_port", "https://arxiv.org/abs/2010.02502", "The same DDPM epsilon objective feeds a separate non-Markovian sampler; eta=0 is deterministic conditional on initial noise.", "No distinct DDIM training objective; twenty-step generation is a speed/accuracy setting, not a reproduced benchmark."),
    57: ("Score-SDE", "generative", "paper_port", "https://arxiv.org/abs/2011.13456", "Continuous VP marginal noising and denoising score matching; reverse SDE and probability-flow ODE have different score coefficients.", "VP branch and Euler integration only; original predictor-corrector networks and published settings are not recreated."),
    59: ("DDRM", "reconstruction", "paper_port", "https://arxiv.org/abs/2201.11793", "Three spectral reverse-update cases condition a diffusion prior on noisy measurements and zero/nonzero singular values.", "Exactly diagonal binary inpainting operator with unit nonzero singular values; a useful pretrained prior is not supplied."),
    60: ("DPS", "reconstruction", "paper_port", "https://arxiv.org/abs/2209.14687", "Differentiate measurement discrepancy through the Tweedie x0 estimate, then guide consecutive DDPM sampling updates.", "Binary inpainting and Gaussian noise example; training_loss trains the prior, reconstruct() runs posterior sampling separately."),
    62: ("Flow Matching", "generative", "paper_port", "https://arxiv.org/abs/2210.02747", "Conditional optimal-transport Gaussian path and analytical velocity regression require no training-time ODE solve.", "Small positive sigma_min leaves endpoint smoothing; this is the conditional OT path, not every flow-matching variant."),
    63: ("Rectified Flow", "generative", "paper_port", "https://arxiv.org/abs/2209.03003", "Regress straight noise/data path velocities; reflow_pairs explicitly recouples a trained ODE's initial and terminal samples.", "Default smoke training is first-round rectified flow; iterative reflow requires user-run endpoint generation and retraining."),
    64: ("Consistency Models", "generative", "paper_port", "https://proceedings.mlr.press/v202/song23a.html", "Shared-noise adjacent sigma pairs, a stop-gradient EMA target, and exact sigma_min boundary parameterization.", "Fixed sixteen-level consistency-training grid and MSE; no pretrained diffusion distillation or original progressive schedule."),
    65: ("DiT", "generative", "upstream", "https://github.com/facebookresearch/DiT", "Original patch Transformer, time/class embeddings and adaLN-Zero backbone, trained with epsilon prediction.", "Two Transformer blocks on small image pixels; no pretrained latent VAE. Original code is CC BY-NC 4.0, not MIT."),
    66: ("Spatial Transformer Network", "registration", "paper_port", "https://arxiv.org/abs/1506.02025", "Identity-initialized localization network predicts affine parameters for differentiable grid generation and resampling.", "Synthetic 2D alignment rather than classification; affine grid coordinates are normalized, not patient millimetres."),
    67: ("VoxelMorph", "registration", "mechanism_demo", "https://arxiv.org/abs/1809.05231", "A 3D encoder-decoder predicts a dense displacement; image similarity plus spatial smoothness trains through warping.", "Reduced U-Net, known synthetic translations and normalized coordinates; no pretrained VoxelMorph or diffeomorphic integration."),
    68: ("GradICON", "registration", "mechanism_demo", "https://arxiv.org/abs/2206.05897", "Bidirectional warps are composed and the spatial derivative of composition error is penalized.", "Finite-difference Jacobians and a small shared network; no original multistep registration architecture or topology guarantee."),
    70: ("DiffPose", "registration", "mechanism_demo", "https://arxiv.org/abs/2312.06358", "A pose initializer and test-time gradient refinement share a differentiable rigid volume projector.", "Normalized parallel-beam renderer; no calibrated cone-beam DiffDRR, patient-specific published pretraining or geodesic pose loss."),
    71: ("X2CT-GAN", "reconstruction3d", "mechanism_demo", "https://openaccess.thecvf.com/content_CVPR_2019/html/Ying_X2CT-GAN_Reconstructing_CT_From_Biplanar_X-Rays_With_Generative_Adversarial_Networks_CVPR_2019_paper.html", "Two orthogonal image encoders lift features along their missing axes; 3D fusion uses adversarial, voxel and projection losses.", "Small custom encoders/decoder, synthetic orthogonal mean projections; not the original chest-CT model, and two views are required."),
    72: ("NeRF", "neural_field", "mechanism_demo", "https://arxiv.org/abs/2003.08934", "Coordinate/direction-conditioned density and radiance integrate using exclusive transmittance and emission-absorption compositing.", "One synthetic scene; compact MLP and uniform finite-interval rays; no hierarchical coarse/fine sampling or original benchmark training."),
    73: ("NAF", "neural_field", "mechanism_demo", "https://github.com/ruyi-zha/naf_cbct", "A position-only hash-encoded attenuation field fits cone-ray line-integral measurements, with Beer-Lambert transmission exp(-integral).", "Normalized synthetic attenuation values; no scanner calibration, CUDA marching or original CBCT architecture/configuration."),
    74: ("3D Gaussian Splatting", "neural_field", "mechanism_demo", "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/", "Perspective Jacobians project anisotropic 3D Gaussian covariances; depth-sorted opacities alpha-compose colors.", "Dense CPU rasterizer with sixteen Gaussians; no CUDA tile renderer, densification, pruning or nonconstant spherical harmonics."),
    113: ("Noise2Void", "restoration", "paper_port", "https://arxiv.org/abs/1811.10980", "Neighbourhood replacement removes selected central observations from the input; only those held-out targets contribute to loss.", "Randomized regular partitions ensure no selected pixel leaks through another replacement; smaller U-Net and modified mask sampling."),
    114: ("RED-Net", "restoration", "mechanism_demo", "https://arxiv.org/abs/1606.08921", "Symmetric convolution/deconvolution stacks carry mirrored additive feature skips and a residual image path.", "Ten layers, small channels and simplified every-second-depth skips; not the full RED30 architecture/training."),
    115: ("SwinIR", "restoration", "upstream", "https://github.com/JingyunLiang/SwinIR", "Original residual Swin Transformer blocks, alternating shifted windows, relative position bias and reconstruction head.", "Two two-block stages, 24 embedding channels and 4px windows; denoising task only, no pretrained super-resolution model."),
    116: ("Restormer", "restoration", "upstream", "https://github.com/swz30/Restormer", "Original multiscale Restormer with channel-transposed MDTA attention and gated depthwise-convolution feed-forward blocks.", "Width 8 and one block per level; no pretrained restoration weights."),
    117: ("AUTOMAP", "reconstruction", "mechanism_demo", "https://doi.org/10.1038/nature25988", "Dense sensor-to-image layers learn a global complex Fourier-to-image transform, followed by convolutional refinement.", "A 16px complete synthetic Fourier acquisition; reduced layer/kernel sizes and no original manifold-pretraining/data augmentation."),
    118: ("LEARN", "reconstruction", "paper_port", "https://pmc.ncbi.nlm.nih.gov/articles/PMC6019143/", "Each stage combines its own three-layer CNN regularizer with an explicit A-transpose projection residual.", "Three unrolls and a pixel-driven parallel-beam operator; positive step parameters are bounded through softplus scaling."),
    119: ("DDNM", "reconstruction", "paper_port", "https://arxiv.org/abs/2212.00490", "Replace estimated x0's measurement-range component with A-dagger y while retaining its null-space component.", "Noiseless inpainting where A-dagger=A; DDNM+ noisy-data and time-travel variants are not included."),
    120: ("RealNVP", "generative", "paper_port", "https://arxiv.org/abs/1605.08803", "Alternating affine coupling layers give exact inverse transforms and exact triangular-Jacobian log determinants.", "Vectorized single-scale coupling, uniform dequantization, small MLPs; no original multiscale image architecture."),
    121: ("Glow", "generative", "paper_port", "https://arxiv.org/abs/1807.03039", "Squeeze, data-initialized ActNorm, invertible learned 1x1 mixing and affine coupling provide exact likelihood and inversion.", "One scale, full matrix determinant/inverse rather than LU acceleration, no hierarchical latent splits."),
    122: ("PFGM++", "generative", "paper_port", "https://proceedings.mlr.press/v202/xu23m.html", "A Beta(N/2,D/2) radial perturbation law and EDM-preconditioned denoising learn the finite-D augmented Poisson field.", "Finite augmented dimension D=64 and compact network; original PFGM corresponds to D=1, not a D-to-zero limit; no full ODE sampler."),
    123: ("SiT", "generative", "upstream", "https://github.com/willisma/SiT", "Original SiT Transformer with velocity regression on its linear stochastic-interpolant path.", "Two blocks on pixels; no pretrained VAE or alternative score/noise parameterizations and SDE samplers."),
    124: ("LapIRN", "registration", "mechanism_demo", "https://arxiv.org/abs/2006.16148", "Three coarse-to-fine registration stages warp previous estimates and compose residual displacement corrections.", "Small custom networks and displacement composition; not the original Laplacian residual-feature decoder or diffeomorphic variant."),
    125: ("TransMorph", "registration", "upstream", "https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration", "Original four-level 3D Swin hierarchy, convolutional decoder, transformer/convolution skips and voxel-unit SpatialTransformer, trained with original MSE plus Grad3d loss.", "32-cubed resampled synthetic volumes; embedding width8, two blocks per level, 2-cubed windows and no stochastic depth; both directions accumulated into one optimizer step. Raw flow components are D/H/W voxel displacements; normalized_xyz_flow converts them explicitly."),
    126: ("IDIR", "registration", "paper_port", "https://proceedings.mlr.press/v172/wolterink22a.html", "A SIREN coordinate network fits one image pair's displacement at test time with autograd spatial derivatives.", "Only the first synthetic pair is fitted; smoothness regularization rather than every original Jacobian/bending variant; normalized coordinates."),
    128: ("pixelNeRF", "neural_field", "mechanism_demo", "https://arxiv.org/abs/2012.02190", "Project world points into a known reference camera, sample aligned CNN features and condition a radiance field for a new view.", "Single reference view, small encoder/MLP and synthetic scenes; no original ResNet encoder or multiview feature aggregation."),
    129: ("Instant-NGP", "neural_field", "mechanism_demo", "https://nvlabs.github.io/instant-ngp/", "Learnable multiresolution hash tables interpolate features at eight cell corners before a small radiance decoder.", "CPU PyTorch implementation; no fused CUDA kernels, occupancy skipping or claimed real-time performance."),
    131: ("EndoNeRF", "neural_field", "mechanism_demo", "https://github.com/med-air/EndoNeRF", "A time-dependent deformation maps observations to a canonical radiance field; synthetic occlusions mask RGB/depth supervision.", "One synthetic moving phantom; no clinical stereo preprocessing, tool segmentation or original depth-guided importance sampler."),
}


def _default_config(number):
    if number == 43:
        return {"size": 16, "width": 32, "latent": 8, "pretrain_steps": 1}
    if number == 44:
        return {"size": 16, "width": 32, "latent": 16}
    if number in {45, 46, 52, 53, 54, 66, 70, 113, 114, 115, 116}:
        return {"width": 8}
    if number in {47, 48, 49, 50, 118}:
        return {"size": 16, "width": 8, "stages": 3}
    if number == 51:
        return {"size": 32, "width": 8, "stages": 2}
    if number in {55, 56, 59, 60, 65, 119, 123}:
        return {"size": 16, "width": 8, "timesteps": 1000}
    if number in {57, 62, 63, 64, 117}:
        return {"size": 16, "width": 8}
    if number in {67, 68, 71, 124}:
        return {"width": 4}
    if number == 125:
        return {"width": 8, "size": 32}
    if number in {72, 73, 128, 129, 131}:
        return {"width": 32, "ray_size": 8, "ray_samples": 16}
    if number == 74:
        return {"gaussians": 16, "size": 16}
    if number in {120, 121}:
        return {"size": 8, "width": 32 if number == 120 else 8, "layers": 4 if number == 120 else 3}
    if number == 122:
        return {"size": 16, "width": 8, "augmented_dim": 64}
    return {"width": 32}


SPECS = {num: dict(name=row[0], task=row[1], implementation=row[2], source_url=row[3],
                  mechanism=row[4], limitations=[row[5], "Synthetic teaching data only; no clinical validity or published benchmark is claimed."],
                  config=_default_config(num), evaluation_scope="same_scene" if num in {72, 73, 74, 126, 129, 131} else "held_out_synthetic")
         for num, row in _ROWS.items()}


def build_model(num: int, config: dict) -> nn.Module:
    if num not in SPECS:
        raise KeyError(f"Model {num} has no executable implementation in inverse_models.py; use its original-source route")
    cfg = {**SPECS[num]["config"], **config}
    if num == 43:
        return DeepAutoencoder(**cfg)
    if num == 44:
        return GAN(**cfg)
    if num in {45, 46}:
        return PairedGAN(cycle=num == 46, **cfg)
    if num in {47, 48, 118}:
        return CTReconstructor(number=num, **cfg)
    if num in {49, 50}:
        return MRIReconstructor(number=num, **cfg)
    if num == 51:
        return VarNet(**cfg)
    if num in {52, 53, 114, 115, 116}:
        return Restoration(number=num, **cfg)
    if num in {54, 113}:
        return BlindSpot(void=num == 113, **cfg)
    if num in {55, 56}:
        return Diffusion(number=num, **cfg)
    if num == 57:
        return ScoreSDE(**cfg)
    if num in {59, 60, 119}:
        return DiffusionInverse(number=num, **cfg)
    if num in {62, 63}:
        return FlowMatching(number=num, **cfg)
    if num == 64:
        return Consistency(**cfg)
    if num in {65, 123}:
        return TransformerGenerator(number=num, **cfg)
    if num == 66:
        return SpatialTransformer(**cfg)
    if num in {67, 68}:
        return Registration(number=num, **cfg)
    if num == 70:
        return DiffPoseDemo(**cfg)
    if num == 71:
        return X2CTDemo(**cfg)
    if num in {72, 73, 128, 129, 131}:
        return FieldLesson(number=num, **cfg)
    if num == 74:
        return GaussianSplattingDemo(**cfg)
    if num == 117:
        return AUTOMAP(**cfg)
    if num == 120:
        return RealNVP(**cfg)
    if num == 121:
        return Glow(**cfg)
    if num == 122:
        return PFGMPP(**cfg)
    if num == 124:
        return LapIRN(**cfg)
    if num == 125:
        return TransMorph(**cfg)
    if num == 126:
        return IDIR(**cfg)
    raise AssertionError(f"Unreachable factory entry: {num}")
