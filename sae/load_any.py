"""Reconstruct any SAE checkpoint (TopK or a variant) for probing.

Two checkpoint formats coexist:
  * legacy TopK SAE -- a bare ``state_dict`` containing ``W_enc`` (saved by
    sae/train_sae.py via ``torch.save(model.state_dict())``).
  * variant SAE -- a dict ``{variant, config, state_dict}`` (saved by
    sae/train_variants.py).

``load_sae_any`` detects which and returns ``(model, d_hidden)``. Every model
exposes the same ``forward(x) -> (acts, recon, aux)`` contract, so downstream
probing code (probe.encode_sae_codes) is variant-agnostic.
"""
import os
import sys

import torch

from sae.sae_model import TopKSAE
from sae.sae_variants import build_variant


def load_sae_any(sae_ckpt, k, device):
    if not os.path.exists(sae_ckpt):
        sys.exit(f"[probe] SAE checkpoint '{sae_ckpt}' not found. "
                 f"Train the SAE first; refusing to probe with random weights.")

    obj = torch.load(sae_ckpt, map_location=device, weights_only=False)

    # Variant checkpoint: self-describing.
    if isinstance(obj, dict) and "variant" in obj and "state_dict" in obj:
        cfg = dict(obj["config"])
        cfg.pop("variant", None)
        model = build_variant(obj["variant"], **cfg).to(device)
        model.load_state_dict(obj["state_dict"])
        model.eval()
        print(f"Loaded '{obj['variant']}' SAE: d_model={cfg['d_model']}, d_hidden={cfg['d_hidden']}")
        return model, cfg["d_hidden"]

    # Legacy TopK: bare state_dict with W_enc.
    if isinstance(obj, dict) and "W_enc" in obj:
        d_model_ckpt, d_hidden_ckpt = obj["W_enc"].shape
        print(f"Loaded TopK SAE: d_model={d_model_ckpt}, d_hidden={d_hidden_ckpt}")
        model = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=k).to(device)
        model.load_state_dict(obj)
        model.eval()
        return model, d_hidden_ckpt

    sys.exit(f"[probe] '{sae_ckpt}' is neither a TopK state_dict nor a variant checkpoint.")
