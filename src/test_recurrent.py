"""CPU self-check of --arch recurrent (run from src/: python test_recurrent.py)."""
import torch

from model import Dynamics, PolicyHeadMTP
from train_dynamics import EmaRms, dynamics_pretrain_loss


def build(n_levels=3):
    torch.manual_seed(0)
    dyn = Dynamics(d_model=32, d_bottleneck=8, d_spatial=16, n_spatial=4, n_register=1, n_agent=1, n_heads=4,
                   depth=2, k_max=8, dropout=0.0, mlp_ratio=2.0, time_every=1, arch="recurrent",
                   n_levels=n_levels, unit_layers=1, head_layers=2, unit_mlp_ratio=1.0)
    for p in dyn.parameters():  # zero-init output heads would hide gradient paths
        torch.nn.init.normal_(p, std=0.2) if p.dim() > 1 else None
    return dyn


B, T, A = 2, 5, 16
z = torch.randn(B, T, 4, 16)
act, mask = torch.randn(B, T, A), torch.ones(B, T, A)
step, sig = torch.full((B, T), 3), torch.full((B, T), 6)
dyn = build()
hier = dyn.transformer

# Causal: frame t's output does not depend on frames after t.
z2 = z.clone()
z2[:, 3:] += 1.0
with torch.no_grad():
    y1, y2 = dyn(act, step, sig, z)[0], dyn(act, step, sig, z2)[0]
assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5) and not torch.allclose(y1[:, 3], y2[:, 3])

# Prefill + denoise step = the full scan at the last frame, and the step leaves the states alone.
with torch.no_grad():
    _, _, cache = dyn(act[:, :-1], step[:, :-1], sig[:, :-1], z[:, :-1], return_kv_cache=True)
    kept = [s.clone() for s in cache]
    y_step = dyn(act[:, -1:], step[:, -1:], sig[:, -1:], z[:, -1:], kv_cache=cache)[0]
    # A prefill continued from a cache = one scan over the whole window.
    _, _, c_half = dyn(act[:, :2], step[:, :2], sig[:, :2], z[:, :2], return_kv_cache=True)
    _, _, c_cont = dyn(act[:, 2:-1], step[:, 2:-1], sig[:, 2:-1], z[:, 2:-1], kv_cache=c_half, return_kv_cache=True)
assert torch.allclose(y_step[:, 0], y1[:, -1], atol=1e-5)
assert all(torch.equal(a, b) for a, b in zip(cache, kept))
assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(c_cont, cache))

# Both losses: flow + BC (not detached) reach every level; the local loss trains each unit only.
pol = PolicyHeadMTP(d_model=32, L=2, act_dim_max=A, dropout=0.0, mlp_ratio=2.0)
kw = dict(z1=z, actions=act, act_mask=mask, k_max=8, B_self=0, step=0, bootstrap_start=10**9, policy_head=pol,
          bc_weight=1.0, bc_detach=False, rms_flow=EmaRms(), rms_bc=EmaRms())
loss, aux = dynamics_pretrain_loss(dyn, pred_weight=0.0, **kw)
loss.backward()
for n, u in enumerate(hier.units):
    assert sum(p.grad.abs().sum() for p in u.parameters() if p.grad is not None) > 0, f"global losses miss level {n}"
assert aux["pred_levels"].shape == (3,) and aux["loss_pred"] > 0
dyn.zero_grad()
dynamics_pretrain_loss(dyn, pred_weight=1.0, rms_pred=EmaRms(), **{**kw, "bc_weight": 0.0})[0].backward()
assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in hier.aux_pred.parameters())

# The local loss alone: level 1's predictor error does not reach the head or the level below.
dyn.zero_grad()
dyn(act, step, sig, z)
hier.aux_loss.mean().backward()
assert all(p.grad is None or p.grad.abs().sum() == 0 for p in hier.head.parameters()), "local loss leaked into the head"
assert all(p.grad is not None for p in hier.units[2].parameters() if p.dim() > 1)
print("recurrent ok")
