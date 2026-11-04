# ASRT vNext V1 — Illumination–Reference Prior Modulated Restoration

## Overview

ASRT vNext V1 replaces Stage-A IGAB channel attention with a new **IRP-MSA**
(Illumination–Reference Prior Modulated Self-Attention) inside the existing
CSFI-UNet multi-scale restore skeleton (`n_feats=64` at all scales).

\[
\text{Illumination-aware Attention}
+
\text{Reference-aware Attention}
\rightarrow
\text{Dual-Prior Modulated Restoration}
\]

Top-level API:

```python
from asrt.models.asrt_irp import ASRTIRP, build_asrt_irp
model = build_asrt_irp(cfg)          # prior_source: none | low | nano
y = model(low, ref=None)             # RGB [-1,1] → RGB [-1,1]
```

## Data flow

```text
Low X ──► IlluminationPriorEncoder ──► P_I {h4,h2,h1}
     └──► SFE / down / CSFI Encoder
Nano R ──► ReferencePriorEncoder ──► P_R {h2,h1}   (A2/A3 only)

H/4: coarse_irp  (illumination-only)
H/2: middle_irp  (dual-prior when P_R available)
H:   fine_irp    (dual-prior when P_R available)
     └──► MergeTail → residual + light-up base → Enhanced RGB
```

Legacy `IlluminationEstimator` light-up is retained for the Low-only base path.
IRP illumination prior is **independent** (`P_I = 1 - max_c(X)`).

## Priors

### Illumination (`IlluminationPriorEncoder`)

```text
X = (low + 1)/2 ∈ [0,1]
P_I = 1 - X.amax(dim=1)     # [B,1,H,W]
→ Resize + Conv projection → {h1,h2,h4} each [B,64,·,·]
```

Only from Low. Never from Nano Ref or GT.

### Reference (`ReferencePriorEncoder`)

```text
H_R = R - G_σ(R),  σ=7
G_R = sobel_mag(R),  G_X = sobel_mag(X)
Z_R = cat[H_R, G_R, G_X]    # [B,9,H,W]
→ Conv stem → Prior H + downsampled Prior H/2
```

A2 and A3 share this encoder; only the second image source differs
(`low` vs `nano ref`).

## IRP-MSA (`IRPTransformerBlock`)

Windowed spatial MHSA (`window=8`, `heads=4`, `head_dim=16`).

\[
\bar F=\mathrm{Norm}(F),\quad
M_I=\mathrm{SA}_I(\bar F),\quad
M_R=\mathrm{SA}_R(\bar F)
\]

\[
Z_I=M_I\odot\phi_I(P_I),\quad
Z_R=M_R\odot\tanh(\phi_R(P_R))
\]

\[
M_E=\alpha Z_I+\beta Z_R,\quad
F'=F+W_o(M_E),\quad
F_{\mathrm{out}}=F'+\mathrm{FFN}(\mathrm{Norm}(F'))
\]

- Q/K/V from restoration features only (no Reference Cross-Attention).
- Independent Q/K/V for SA_I and SA_R.
- Per-block learnable scalars \(\alpha\) (init 1.0), \(\beta\) (init 0.1).
- Padding tokens masked out of attention softmax.

## Modes

| Mode | `prior_source` | Ref encoder | Middle/Fine ref branch |
|------|----------------|-------------|------------------------|
| A1   | `none`         | not built   | `use_reference=False`  |
| A2   | `low`          | Low→Low     | dual; ignores external ref |
| A3   | `nano`         | Low+Nano    | dual; `ref=None` bypasses |

Dataset `ref_mode`: `correct|wrong|shuffle|ref_low|noref` only remaps the
Reference Prior input; Illumination Prior always uses original Low.

## Files

```text
asrt/models/irp_attention.py
asrt/models/illumination_prior_encoder.py
asrt/models/reference_prior_encoder.py
asrt/models/irp_csfi_unet.py
asrt/models/asrt_irp.py
asrt/data/reference_modes.py
configs/asrt_irp_v1.yaml
scripts/train_irp.py
scripts/smoke_irp.py
```

## Explicitly out of scope (V1)

MoE, LoRA, TTSR matching, Reference Cross-Attention, RGB blending,
reliability gate, semantic seg, GAN/diffusion/perceptual loss,
Stage-A checkpoint warm-start.
