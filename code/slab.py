#!/usr/bin/env python3
"""Minimum-mass axisymmetric slab, analytic Jacobian via FTC.

Find b(r') minimizing mass subject to:
  sqrt((g_z(r) - g0)^2 + g_r(r)^2) <= epsilon   for all r <= R

The Jacobian dg/db is obtained by evaluating kernels at the slab boundary
z'=-b(r'), requiring no z-integration or autograd.  Forward field uses a
lightweight midpoint sum.
"""

import numpy as np
from datetime import datetime
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

torch.cuda.empty_cache()

d = {'device': 'cuda', 'dtype': torch.float32}
print(d)

# --- Parameters ---
disk_r    = 0.5
g0        = 1.0
epsilon   = 0.005
n_src     = 50000
n_obs     = 300
n_z       = 8         # forward-model z-slices (midpoint rule)
smoothing = 1e3
R_ext     = 2 * 3.0

# --- Elliptic integral lookup tables ---
N_table = 100_000
m_np  = np.linspace(0, 1 - 1e-7, N_table)
K_tbl = torch.tensor(ellipk(m_np), **d)
E_tbl = torch.tensor(ellipe(m_np), **d)

def elliptic_KE(m):
    m_c = torch.clamp(m, 0.0, 1.0 - 1e-7)
    idx = m_c * (N_table - 1)
    i0  = idx.long().clamp(0, N_table - 2)
    t   = idx - i0.float()
    K   = K_tbl[i0] * (1 - t) + K_tbl[i0 + 1] * t
    E   = E_tbl[i0] * (1 - t) + E_tbl[i0 + 1] * t
    return K, E

# --- Grids ---
r_src_np = np.linspace(0, R_ext, n_src + 1)
r_src_np = 0.5 * (r_src_np[:-1] + r_src_np[1:])
dr = float(R_ext / n_src)
r_obs_np = np.linspace(disk_r / n_obs, disk_r * 0.99, n_obs)

r_src = torch.tensor(r_src_np, **d)
r_obs = torch.tensor(r_obs_np, **d)

# Pre-expand for (n_obs, n_src) broadcasting
ro      = r_obs[:, None]            # (n_obs, 1)
rs      = r_src[None, :]            # (1, n_src)
ro_safe = ro.clamp(min=1e-10)

def _kernels(z):
    """Kz, Kr at positive depth z.  z shape broadcasts to (n_obs, n_src)."""
    alpha2 = (ro - rs)**2 + z**2
    beta2  = (ro + rs)**2 + z**2
    beta   = torch.sqrt(beta2.clamp(min=1e-30))
    k2     = (4 * ro * rs / beta2.clamp(min=1e-30)).clamp(0, 1 - 1e-7)
    K, E   = elliptic_KE(k2)
    Kz = z * 4 * E / (beta * alpha2.clamp(min=1e-20))
    Kr = 2 / (ro_safe * beta) * (
        (ro**2 - rs**2 - z**2) / alpha2.clamp(min=1e-20) * E + K)
    return Kz, Kr

def compute_field(b):
    """g_z, g_r via midpoint sum over z (no autograd)."""
    bv = b[None, :]
    wt = bv * rs * dr / n_z
    gz = torch.zeros(n_obs, **d)
    gr = torch.zeros(n_obs, **d)
    for k in range(n_z):
        Kz, Kr = _kernels(bv * ((k + 0.5) / n_z))
        gz = gz + (Kz * wt).sum(1)
        gr = gr + (Kr * wt).sum(1)
    return gz, gr

def compute_jacobian(b):
    """dg/db via FTC: kernel at z=b(r') times r' dr'.  Returns (n_obs, n_src)."""
    Kz, Kr = _kernels(b[None, :])
    return Kz * (rs * dr), Kr * (rs * dr)

# --- Validation ---
with torch.no_grad():
    b_test = torch.full((n_src,), 0.1, **d)
    gz_t, gr_t = compute_field(b_test)
    print(f"Validation — uniform slab b=0.1:")
    print(f"  ε = {epsilon:.4f}")
    print(f"  g_z = {gz_t.mean().item():.4f} (infinite-slab limit: {2*np.pi*0.1:.4f})")
    print(f"  g_r max = {gr_t.abs().max().item():.4f}")
    b0_val = float(abs(g0) / gz_t.mean().item() * 0.1)
    print(f"  Estimated b0 for |g_z|=1: {b0_val:.4f}")

# --- Optimization (manual Adam, analytic gradient) ---
log_b = torch.full((n_src,), float(np.log(b0_val)), **d)

m_adam = torch.zeros(n_src, **d)
v_adam = torch.zeros(n_src, **d)
beta1, beta2a, eps_a = 0.9, 0.999, 1e-8
step_g = 0
lr_base = 3e-3
eta_min = 1e-4
T_max   = 3000

def run_opt(lam, n_steps=2000, log_every=500):
    global step_g, m_adam, v_adam, log_b

    for step in range(n_steps):
        step_g += 1
        b = torch.exp(log_b)

        gz, gr = compute_field(b)
        Jz, Jr = compute_jacobian(b)

        # --- loss components ---
        mass = 2 * np.pi * (b * r_src * dr).sum()
        err  = torch.sqrt((gz - g0)**2 + gr**2)

        # penalty gradient  d/db[ mean relu(err-ε)^2 ]
        scale  = 2 * torch.relu(err - epsilon) / (n_obs * err.clamp(min=1e-10))
        dp_dgz = scale * (gz - g0)                        # (n_obs,)
        dp_dgr = scale * gr

        grad_b = 2 * np.pi * r_src * dr + lam * (dp_dgz @ Jz + dp_dgr @ Jr)

        # smoothness on log_b
        diff = log_b[1:] - log_b[:-1]
        s_grad = torch.zeros_like(log_b)
        s_grad[:-1] -= 2 * smoothing * diff / (n_src - 1)
        s_grad[1:]  += 2 * smoothing * diff / (n_src - 1)

        grad = grad_b * b + s_grad          # chain rule to log_b

        # Adam update
        m_adam = beta1  * m_adam + (1 - beta1)  * grad
        v_adam = beta2a * v_adam + (1 - beta2a) * grad**2
        m_hat  = m_adam / (1 - beta1  ** step_g)
        v_hat  = v_adam / (1 - beta2a ** step_g)
        t_cos  = min(step_g, T_max) / T_max
        lr = eta_min + (lr_base - eta_min) * 0.5 * (1 + np.cos(np.pi * t_cos))
        log_b = (log_b - lr * m_hat / (torch.sqrt(v_hat) + eps_a)).clamp(-8, 2)

        if step % log_every == 0:
            smooth_val = smoothing * (diff**2).mean()
            print(f"  step={step:4d}: mass={mass.item():.4f}, "
                  f"max_err={err.max().item():.4f}, smooth={smooth_val.item():.1f}")

# Progressive lambda schedule
for lam in [10000, 100000]:
    print(f"\n--- Lambda = {lam} ---")
    run_opt(lam, n_steps=3000, log_every=1000)

# --- Final evaluation ---
with torch.no_grad():
    b_opt = torch.exp(log_b)
    gz_opt, gr_opt = compute_field(b_opt)
    err_opt = torch.sqrt((gz_opt - g0)**2 + gr_opt**2)

    b_np   = b_opt.cpu().numpy()
    gz_np  = gz_opt.cpu().numpy()
    gr_np  = gr_opt.cpu().numpy()
    err_np = err_opt.cpu().numpy()
    mass_final = float(2 * np.pi * (b_opt * r_src * dr).sum().item())

print(f"\n--- Final ---")
print(f"Mass:               {mass_final:.4f}")
print(f"Max field error:    {err_np.max():.4f}  (ε={epsilon})")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
r_plot = np.concatenate([-r_src_np[::-1], r_src_np])
b_plot = np.concatenate([b_np[::-1], b_np])
ax.fill_between(r_plot, 0, -b_plot, color='steelblue', alpha=0.8)
ax.plot(r_plot, -b_plot, color='black', linewidth=0.1)
ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=3, label='Disk')
ax.axvline( disk_r, color='r', ls='--', alpha=0.3)
ax.axvline(-disk_r, color='r', ls='--', alpha=0.3)
ax.set_xlim(-R_ext/2, R_ext/2)
ax.set_ylim(-b_np.max()*1.3, b_np.max()*0.3)
ax.set_aspect('equal')
ax.set_title(f'Min-mass slab (ε={epsilon}, actual ε_max={err_np.max():.3f})')
ax.set_xlabel('r'); ax.set_ylabel('z'); ax.legend()

ax = axes[1]
ax.plot(r_obs_np, gz_np, 'b-', label='$g_z$')
ax.plot(r_obs_np, gr_np, 'r-', label='$g_r$')
ax.axhline(g0, color='b', ls='--', alpha=0.5)
ax.axhline(0,  color='r', ls='--', alpha=0.5)
ax.fill_between(r_obs_np, g0 - epsilon, g0 + epsilon, alpha=0.1, color='blue')
ax.set_title('Field on disk'); ax.set_xlabel('r'); ax.legend()

ax = axes[2]
ax.plot(r_obs_np, err_np, 'k-')
ax.axhline(epsilon, color='r', ls='--', label=f'ε={epsilon}')
ax.set_title('|g - target|'); ax.set_xlabel('r'); ax.legend()

plt.tight_layout()
fig.text(0.01, 0.01, text:=f'n_src={n_src}  n_obs={n_obs}  n_z={n_z} smooth={smoothing}',
         fontsize=8, color='gray', va='bottom', ha='left')
print(text)
plt.savefig('/www/flatearth/minmass.png', dpi=150)
plt.savefig(f'/www/flatearth/archive/minmass_{datetime.utcnow().isoformat()}.png', dpi=150)
print("Saved to /www/flatearth/minmass.png")

np.savez('/www/flatearth_result.npz',
         b_opt=b_np, r_src=r_src_np,
         gz=gz_np, gr=gr_np, err=err_np, r_obs=r_obs_np,
         meta=np.array([epsilon, g0, disk_r, float(n_src), float(n_obs)]))
print(f"Saved results to /www/flatearth_result.npz")
