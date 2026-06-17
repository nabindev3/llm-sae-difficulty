"""SAE architecture variants for the design-space sweep (Limitation 2).

The committed result uses a single TopK-32 SAE. To show the null
(SAE features add no incremental difficulty signal over raw activations) is
not a TopK-32 artifact, we sweep alternative dictionary-learning objectives:

  * ``GatedSAE``      -- Rajamanoharan et al. 2024, "Improving Dictionary
                         Learning with Gated Sparse Autoencoders".
  * ``JumpReLUSAE``   -- Rajamanoharan et al. 2024, "Jumping Ahead: Improving
                         Reconstruction Fidelity with JumpReLU SAEs".
  * ``TopKTranscoder``-- TopK dictionary that maps one activation site to
                         another (MLP-in -> MLP-out) rather than autoencoding.

Uniform contract so one trainer / one probe-loader handle every variant:
  - ``encode(x) -> feature_acts``           (the codes a probe consumes)
  - ``forward(x) -> (acts, recon, aux)``    (matches sae_model.TopKSAE's tuple)
  - ``compute_loss(x, target=None) -> (loss, metrics)``  (training objective)
  - ``VARIANT`` class attribute + ``config()`` so checkpoints are self-describing.

The original ``sae_model.TopKSAE`` is left untouched (probe.py / causal_ablation.py
depend on its exact ``forward`` signature); ``train_variants.py`` adds a thin
``compute_loss`` adapter for it instead of editing it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gated SAE
# ---------------------------------------------------------------------------
class GatedSAE(nn.Module):
    """Gated SAE with encoder weight sharing (Rajamanoharan et al. 2024, §3).

    A single encoder direction ``W_gate`` feeds two paths:
      gate path:      pi_gate = (x - b_dec) W_gate + b_gate
      magnitude path: pi_mag  = (x - b_dec) (W_gate * exp(r_mag)) + b_mag
    Feature activations:  f = 1[pi_gate > 0] * ReLU(pi_mag).

    Loss = ||x - x_hat||^2
         + lam * ||ReLU(pi_gate)||_1                       (sparsity)
         + ||x - (ReLU(pi_gate) W_dec.detach() + b_dec.detach())||^2  (aux recon).
    """

    VARIANT = "gated"

    def __init__(self, d_model, d_hidden, l1_coeff=1e-3, **_ignored):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.W_gate = nn.Parameter(torch.empty(d_model, d_hidden))
        self.b_gate = nn.Parameter(torch.zeros(d_hidden))
        self.r_mag = nn.Parameter(torch.zeros(d_hidden))   # log magnitude scale
        self.b_mag = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_gate)
        # Tie decoder to encoder direction at init, then unit-normalize columns.
        self.W_dec.data = self.W_gate.data.T.clone()
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def _pre(self, x):
        xc = x - self.b_dec
        pi_gate = xc @ self.W_gate + self.b_gate
        W_mag = self.W_gate * torch.exp(self.r_mag)
        pi_mag = xc @ W_mag + self.b_mag
        return pi_gate, pi_mag

    def encode(self, x):
        pi_gate, pi_mag = self._pre(x)
        gate = (pi_gate > 0).to(pi_mag.dtype)
        return gate * F.relu(pi_mag)

    def forward(self, x, dead_mask=None):
        acts = self.encode(x)
        recon = acts @ self.W_dec + self.b_dec
        return acts, recon, 0.0

    def compute_loss(self, x, target=None):
        pi_gate, pi_mag = self._pre(x)
        gate = (pi_gate > 0).to(pi_mag.dtype)
        acts = gate * F.relu(pi_mag)
        recon = acts @ self.W_dec + self.b_dec

        l_recon = F.mse_loss(recon, x)
        gated_relu = F.relu(pi_gate)
        l_sparsity = self.l1_coeff * gated_relu.sum(dim=-1).mean()
        # Auxiliary: gating path must itself reconstruct, via a frozen decoder.
        aux_recon = gated_relu @ self.W_dec.detach() + self.b_dec.detach()
        l_aux = F.mse_loss(aux_recon, x)

        loss = l_recon + l_sparsity + l_aux
        with torch.no_grad():
            l0 = (acts > 0).float().sum(dim=-1).mean().item()
        return loss, {"l_recon": l_recon.item(), "l_aux": l_aux.item(), "l0": l0}

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def config(self):
        return {"variant": self.VARIANT, "d_model": self.d_model,
                "d_hidden": self.d_hidden, "l1_coeff": self.l1_coeff}


# ---------------------------------------------------------------------------
# JumpReLU SAE
# ---------------------------------------------------------------------------
class _RectSTE(torch.autograd.Function):
    """Straight-through estimators for JumpReLU with a rectangular kernel.

    Returns (jumprelu_out, l0_step) where
        jumprelu_out = pre * H(pre - theta)
        l0_step      = H(pre - theta)
    and the threshold ``theta`` receives pseudo-gradients (Rajamanoharan 2024):
        d jumprelu / d theta  ~  -(theta / eps) * K((pre - theta)/eps)
        d l0       / d theta  ~  -(1   / eps) * K((pre - theta)/eps)
    with K the rectangle kernel 1[|z| <= 1/2]. The pre-activation keeps its
    natural gate gradient d jumprelu / d pre = H(pre - theta).
    """

    @staticmethod
    def forward(ctx, pre, theta, eps):
        gate = (pre > theta).to(pre.dtype)
        ctx.save_for_backward(pre, theta, gate)
        ctx.eps = eps
        return pre * gate, gate

    @staticmethod
    def backward(ctx, grad_out, grad_l0):
        pre, theta, gate = ctx.saved_tensors
        eps = ctx.eps
        # Rectangle kernel evaluated at (pre - theta)/eps.
        within = (((pre - theta) / eps).abs() <= 0.5).to(pre.dtype)
        kernel = within / eps

        grad_pre = grad_out * gate                       # natural gate gradient
        # Threshold pseudo-grads, summed contributions from both outputs.
        grad_theta = grad_out * (-theta * kernel) + grad_l0 * (-kernel)
        return grad_pre, grad_theta, None


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE with learnable per-feature thresholds and L0 penalty."""

    VARIANT = "jumprelu"

    def __init__(self, d_model, d_hidden, l0_coeff=1e-2, bandwidth=1e-3,
                 theta_init=1e-3, **_ignored):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.l0_coeff = l0_coeff
        self.bandwidth = bandwidth

        self.W_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        # Threshold parametrized in log space so theta = exp(log_theta) > 0.
        self.log_theta = nn.Parameter(torch.full((d_hidden,), float(torch.log(torch.tensor(theta_init)))))

        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def _acts_and_l0(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        theta = torch.exp(self.log_theta)
        acts, gate = _RectSTE.apply(pre, theta, self.bandwidth)
        return acts, gate

    def encode(self, x):
        return self._acts_and_l0(x)[0]

    def forward(self, x, dead_mask=None):
        acts, _ = self._acts_and_l0(x)
        recon = acts @ self.W_dec + self.b_dec
        return acts, recon, 0.0

    def compute_loss(self, x, target=None):
        acts, gate = self._acts_and_l0(x)
        recon = acts @ self.W_dec + self.b_dec
        l_recon = F.mse_loss(recon, x)
        l0_per_example = gate.sum(dim=-1).mean()
        loss = l_recon + self.l0_coeff * l0_per_example
        with torch.no_grad():
            l0 = (acts > 0).float().sum(dim=-1).mean().item()
        return loss, {"l_recon": l_recon.item(), "l0": l0}

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def config(self):
        return {"variant": self.VARIANT, "d_model": self.d_model,
                "d_hidden": self.d_hidden, "l0_coeff": self.l0_coeff,
                "bandwidth": self.bandwidth}


# ---------------------------------------------------------------------------
# TopK transcoder (input site -> target site)
# ---------------------------------------------------------------------------
class TopKTranscoder(nn.Module):
    """TopK dictionary mapping one activation site to another.

    Unlike an autoencoder it reconstructs a *target* (e.g. MLP output) from the
    *input* (e.g. MLP input), so ``compute_loss`` requires ``target``. The
    feature codes (``encode``) are what a probe consumes, exactly like the SAE
    variants, so it slots into the same sweep.
    """

    VARIANT = "transcoder"

    def __init__(self, d_model, d_hidden, k=32, d_out=None, **_ignored):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.d_out = d_out or d_model

        self.W_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, self.d_out))
        self.b_dec = nn.Parameter(torch.zeros(self.d_out))
        self.b_in = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def encode(self, x):
        pre = (x - self.b_in) @ self.W_enc + self.b_enc
        top_acts, top_idx = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, top_idx, F.relu(top_acts))
        return acts

    def forward(self, x, dead_mask=None):
        acts = self.encode(x)
        out = acts @ self.W_dec + self.b_dec
        return acts, out, 0.0

    def compute_loss(self, x, target=None):
        if target is None:
            raise ValueError("TopKTranscoder.compute_loss requires a target site tensor")
        acts = self.encode(x)
        out = acts @ self.W_dec + self.b_dec
        l_recon = F.mse_loss(out, target)
        with torch.no_grad():
            l0 = (acts > 0).float().sum(dim=-1).mean().item()
        return l_recon, {"l_recon": l_recon.item(), "l0": l0}

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def config(self):
        return {"variant": self.VARIANT, "d_model": self.d_model,
                "d_hidden": self.d_hidden, "k": self.k, "d_out": self.d_out}


BUILDERS = {
    GatedSAE.VARIANT: GatedSAE,
    JumpReLUSAE.VARIANT: JumpReLUSAE,
    TopKTranscoder.VARIANT: TopKTranscoder,
}


def build_variant(variant, **kwargs):
    if variant not in BUILDERS:
        raise ValueError(f"unknown variant '{variant}'. choices: {list(BUILDERS)}")
    return BUILDERS[variant](**kwargs)
