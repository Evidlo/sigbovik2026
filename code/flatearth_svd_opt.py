#!/usr/bin/env python3
"""SVD analysis linearized around the optimized b(r).

Loads b_opt saved by flatearth_minmass.py and computes the Jacobian
d(g_z, g_r)/db analytically at b_opt.

Analytical Jacobian (no autograd):
  d g_z(r_i) / d b_j = gz_kernel(r_i, r_j, z=b_j) * r_j * dr
  d g_r(r_i) / d b_j = gr_kernel(r_i, r_j, z=b_j) * r_j * dr

This is just the integrand evaluated at the slab bottom — the
fundamental theorem of calculus applied to the z integral.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from scipy.special import ellipk, ellipe

# --- Load results ---
data     = np.load('/www/flatearth_result.npz')
b_opt    = data['b_opt']
r_src_np = data['r_src']
gz_opt   = data['gz']
gr_opt   = data['gr']
err      = data['err']
r_obs_np = data['r_obs']
epsilon, g0, disk_r, n_src_f, n_obs_f = data['meta']
epsilon = float(epsilon); g0 = float(g0); disk_r = float(disk_r)

n_src = len(b_opt)
n_obs = len(r_obs_np)
R_ext = r_src_np[-1] + (r_src_np[1] - r_src_np[0]) / 2
dr    = float(R_ext / n_src)

print(f"Loaded: n_src={n_src}, R_ext={R_ext:.3f}, ε={epsilon}")
print(f"b_opt: min={b_opt.min():.4f}, max={b_opt.max():.4f}")
print(f"Field error: mean={err.mean():.4f}, max={err.max():.4f}")

# =====================================================================
# Analytical Jacobian at b_opt
#
# g_z(r_i) = sum_j integral_0^{b_j} gz_kern(r_i, r_j, z) dz * r_j * dr
# => d g_z(r_i) / d b_j = gz_kern(r_i, r_j, b_j) * r_j * dr
#
# Same for g_r.
# =====================================================================
print("Computing analytical Jacobian...")

ro = r_obs_np[:, None]   # (n_obs, 1)
rs = r_src_np[None, :]   # (1, n_src)
z  = b_opt[None, :]      # (1, n_src)  — evaluate at slab bottom

beta2  = (ro + rs)**2 + z**2
alpha2 = np.maximum((ro - rs)**2 + z**2, 1e-20)
beta   = np.sqrt(np.maximum(beta2, 1e-30))
k2     = np.clip(4 * ro * rs / np.maximum(beta2, 1e-30), 0.0, 1.0 - 1e-9)

K = ellipk(k2)
E = ellipe(k2)

gz_kern = z * 4 * E / (beta * alpha2)
ro_safe = np.maximum(ro, 1e-10)
gr_kern = 2 / (ro_safe * beta) * ((ro**2 - rs**2 - z**2) / alpha2 * E + K)

weight = rs * dr   # (1, n_src) broadcast

J_gz = gz_kern * weight   # (n_obs, n_src)
J_gr = gr_kern * weight   # (n_obs, n_src)
J    = np.vstack([J_gz, J_gr])   # (2*n_obs, n_src)

print(f"Jacobian shape: {J.shape}")

# =====================================================================
# SVD — plain and active-set-weighted
#
# At the optimum the Hessian of L is J^T W J where W = diag(w_i) with
# w_i proportional to the active Lagrange multipliers.  Observations
# deep inside the feasible set (err_i << ε) have zero multiplier.
# We approximate w_i = relu(err_i - ε_slack) / ε, where ε_slack is a
# fraction of ε (soft active-set indicator).
# =====================================================================
print("Computing SVD...")

# Active-set weights: ramp from 0 at err=0.9ε to 1 at err=ε
ε_slack  = 0.9 * epsilon
w_obs    = np.clip((err - ε_slack) / (epsilon - ε_slack), 0, None)  # (n_obs,)
sqrtw    = np.sqrt(w_obs)
n_active = int(np.sum(w_obs > 0))
print(f"Active observations (err > 0.9ε): {n_active}/{n_obs}")

# Weighted Jacobian: stack gz and gr rows, each scaled by sqrt(w)
Jw = np.vstack([J_gz * sqrtw[:, None],
                J_gr * sqrtw[:, None]])   # (2*n_obs, n_src)

# Unweighted (all obs equal)
U,    s,    Vt    = np.linalg.svd(J,    full_matrices=False)
# Weighted (active constraints only)
Uw,   sw,   Vtw   = np.linalg.svd(Jw,   full_matrices=False)
# gz-only unweighted (for reference)
U_z,  s_z,  Vt_z  = np.linalg.svd(J_gz, full_matrices=False)

V    = Vt.T
Vw   = Vtw.T
V_z  = Vt_z.T

def n_good_fn(sv): return int(np.sum(sv > 0.01 * sv[0]))
n_good    = n_good_fn(s)
n_good_w  = n_good_fn(sw)
n_good_z  = n_good_fn(s_z)

print(f"Unweighted  (gz+gr): condition={s[0]/s[-1]:.2e},  well-determined={n_good}")
print(f"Weighted    (gz+gr): condition={sw[0]/sw[-1]:.2e}, well-determined={n_good_w}")
print(f"gz only (unweighted): well-determined={n_good_z}")
print(f"Singular values unweighted: {s[:10]}")
print(f"Singular values   weighted: {sw[:10]}")

# =====================================================================
# Plots
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Singular value spectra: unweighted vs weighted
ax = axes[0, 0]
ax.semilogy(s,  'b.-', markersize=3, label=f'Unweighted gz+gr ({n_good} modes)')
ax.semilogy(sw, 'g.-', markersize=3, label=f'Active-weighted gz+gr ({n_good_w} modes)')
ax.semilogy(s_z,'r.--',markersize=3, alpha=0.6, label=f'gz only ({n_good_z} modes)')
ax.axhline(0.01 * s[0],  color='b', ls=':', alpha=0.4)
ax.axhline(0.01 * sw[0], color='g', ls=':', alpha=0.4)
ax.set_xlabel('Mode index'); ax.set_ylabel('Singular value')
ax.set_title('SVD spectra at b_opt'); ax.legend(fontsize=8)

# Active-set weight profile
ax = axes[0, 2]
ax.plot(r_obs_np, err,   'k-',  label='field error')
ax.axhline(epsilon,      color='r',  ls='--', label=f'ε={epsilon}')
ax.axhline(ε_slack,      color='r',  ls=':',  label=f'0.9ε (slack threshold)', alpha=0.5)
ax.fill_between(r_obs_np, 0, w_obs * epsilon, alpha=0.25, color='green', label='active weight')
ax.set_xlabel('r_obs'); ax.set_title('Active-set weights')
ax.legend(fontsize=8)

# Well-determined modes (weighted)
ax = axes[0, 1]
for i in range(min(8, n_good_w)):
    ax.plot(r_src_np, Vw[:, i], label=f'Mode {i} (σ={sw[i]:.3f})')
ax.axvline(disk_r, color='r', ls='--', alpha=0.4, label='Disk edge')
ax.set_xlabel('r'); ax.set_ylabel('Vw_i(r)')
ax.set_title(f'Well-determined modes (active-weighted, {n_good_w} total)')
ax.set_xlim(0, R_ext / 3); ax.legend(fontsize=7)

# b_opt cross-section
ax = axes[1, 0]
r_plot = np.concatenate([-r_src_np[::-1], r_src_np])
b_plot = np.concatenate([b_opt[::-1], b_opt])
ax.fill_between(r_plot, 0, -b_plot, color='steelblue', alpha=0.7)
ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=3, label='Disk')
ax.axvline( disk_r, color='r', ls='--', alpha=0.3)
ax.axvline(-disk_r, color='r', ls='--', alpha=0.3)
ax.set_xlim(-R_ext/3, R_ext/3)
ax.set_title('b_opt cross-section')
ax.set_xlabel('r'); ax.set_ylabel('z'); ax.legend()

# Jacobian row norms (sensitivity per observation point)
ax = axes[1, 1]
jz_norm = np.linalg.norm(J_gz, axis=1)
jr_norm = np.linalg.norm(J_gr, axis=1)
ax.plot(r_obs_np, jz_norm, 'b-', label='||J_gz row|| (gz sensitivity)')
ax.plot(r_obs_np, jr_norm, 'r-', label='||J_gr row|| (gr sensitivity)')
ax.set_xlabel('r_obs'); ax.set_ylabel('Row norm')
ax.set_title('Jacobian sensitivity per observation')
ax.legend(fontsize=8)

# Jacobian column norms (how much each source point matters)
ax = axes[1, 2]
jz_col = np.linalg.norm(J_gz, axis=0)
jr_col = np.linalg.norm(J_gr, axis=0)
ax.plot(r_src_np, jz_col, 'b-', label='||J_gz col|| (gz)')
ax.plot(r_src_np, jr_col, 'r-', label='||J_gr col|| (gr)')
ax.axvline(disk_r, color='k', ls='--', alpha=0.4, label='Disk edge')
ax.set_xlabel('r_src'); ax.set_ylabel('Column norm')
ax.set_title('Jacobian influence per source point')
ax.set_xlim(0, R_ext / 3); ax.legend(fontsize=8)

plt.tight_layout()
fig.text(0.01, 0.01, f'n_src={n_src}  n_obs={n_obs}  analytical Jacobian at b_opt',
         fontsize=8, color='gray')
plt.savefig('/www/flatearth_svd_opt.png', dpi=150)
print("Saved to /www/flatearth_svd_opt.png")

# =====================================================================
# Figure 2: successive projections of b_opt onto first n modes
# Layout: original + n=1..11 => 13 panels, 2 columns x 7 rows
# =====================================================================
n_proj_max = n_good_w   # use active-weighted well-determined modes
panels = ['original'] + list(range(1, n_proj_max + 1))
n_panels = len(panels)
n_rows = (n_panels + 1) // 2

fig2, axes2 = plt.subplots(n_rows, 2, figsize=(12, n_rows * 2.2))
axes2_flat = axes2.flatten()

# b(r) zlim shared across all panels
b_max = b_opt.max() * 1.3

r_plot = np.concatenate([-r_src_np[::-1], r_src_np])

for idx, panel in enumerate(panels):
    ax = axes2_flat[idx]
    if panel == 'original':
        b_show = b_opt
        title  = 'original b(r)'
        color  = 'steelblue'
    else:
        n = panel
        # project b_opt onto first n active-weighted right singular vectors
        coeffs  = Vw[:, n:n+1].T @ b_opt        # (n,)
        b_show  = Vw[:, n:n+1] @ coeffs          # (n_src,)
        title   = f'n={n} modes  (σ_n={sw[n-1]:.3f})'
        color   = 'darkorange'

    b_p = np.concatenate([b_show[::-1], b_show])
    ax.fill_between(r_plot, 0, -b_p, color=color, alpha=0.7)
    ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=2)
    ax.axvline( disk_r, color='r', ls='--', alpha=0.3, lw=0.8)
    ax.axvline(-disk_r, color='r', ls='--', alpha=0.3, lw=0.8)
    ax.set_xlim(-R_ext / 3, R_ext / 3)
    ax.set_ylim(-b_max, b_max * 0.15)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('r', fontsize=8)
    ax.set_ylabel('z', fontsize=8)
    ax.tick_params(labelsize=7)

# hide any unused panels
for idx in range(n_panels, len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

plt.suptitle('b(r) projected onto first n SVD modes', fontsize=12)
plt.tight_layout()
plt.savefig('/www/flatearth/svd_projections.png', dpi=150)
print("Saved to /www/flatearth/svd_projections.png")
