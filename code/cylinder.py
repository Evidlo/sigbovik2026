#!/usr/bin/env python3
"""Cylinder parameter study: ε contours in (r₀, b₀) space.

For a uniform-density cylinder (radius r₀, depth b₀), compute the
gravitational field on the observation disk and plot ε contours.
Uses PyTorch + GPU with the same elliptic-integral lookup as slab.py.
"""

import numpy as np
import torch
from scipy.special import ellipk, ellipe
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

torch.cuda.empty_cache()
d = {'device': 'cuda', 'dtype': torch.float32}

# --- Parameters ---
disk_r = 0.5
g0     = 1.0
n_obs  = 300
n_z    = 200

r_obs = torch.linspace(disk_r / n_obs, disk_r * 0.99, n_obs, **d)

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


def cylinder_field(b0, r0, n_r=500):
    """Compute (gz, gr) on observation disk for cylinder of radius r0, depth b0."""
    r_min = disk_r / n_r
    edges = np.geomspace(r_min, r0, n_r + 1)
    r_src_np = 0.5 * (edges[:-1] + edges[1:])
    dr_np    = np.diff(edges)
    r_src = torch.tensor(r_src_np, **d)
    dr    = torch.tensor(dr_np,    **d)
    dz = b0 / n_z

    ro = r_obs[:, None]          # (n_obs, 1)
    rs = r_src[None, :]          # (1, n_r)
    ro_safe = ro.clamp(min=1e-30)

    gz = torch.zeros(n_obs, **d)
    gr = torch.zeros(n_obs, **d)

    for k in range(n_z):
        z = (k + 0.5) * dz
        alpha2 = (ro - rs)**2 + z**2
        beta2  = (ro + rs)**2 + z**2
        beta   = torch.sqrt(beta2)
        k2     = (4 * ro * rs / beta2).clamp(0, 1 - 1e-7)
        K, E   = elliptic_KE(k2)

        Kz = z * 4 * E / (beta * alpha2.clamp(min=1e-20))
        Kr = 2 / (ro_safe * beta) * (
            (ro**2 - rs**2 - z**2) / alpha2.clamp(min=1e-20) * E + K)

        gz += (Kz * rs * dr * dz).sum(1)
        gr += (Kr * rs * dr * dz).sum(1)

    return gz, gr


# --- Scan grid ---
n_r0, n_b0 = 80, 80
r0_vals = np.geomspace(0.3, 5e3, n_r0)
b0_vals = np.geomspace(0.02, 5e3, n_b0)

eps_grid  = np.zeros((n_b0, n_r0))
mass_grid = np.zeros((n_b0, n_r0))

print(f"Scanning {n_r0}x{n_b0} = {n_r0*n_b0} cylinder configurations...")
with torch.no_grad():
    for i, r0 in enumerate(r0_vals):
        for j, b0 in enumerate(b0_vals):
            gz, gr = cylinder_field(b0, r0)
            scale = g0 / gz.mean()
            gz_s = gz * scale
            gr_s = gr * scale
            err = torch.sqrt((gz_s - g0)**2 + gr_s**2)
            eps_grid[j, i]  = err.max().item()
            mass_grid[j, i] = np.pi * r0**2 * b0 * scale.item()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_r0} r₀ values done")

print("Scan complete.")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
levels = [0.001, 0.005, 0.01, 0.05]
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

logR0, logB0 = np.meshgrid(np.log10(r0_vals / disk_r), np.log10(b0_vals / disk_r))
log_fmt = FuncFormatter(lambda x, _: f'$10^{{{int(x)}}}$')
int_loc = MultipleLocator(1)

# ε contours
ax = axes[0]
cf = ax.contourf(logR0, logB0, np.log10(eps_grid), levels=50, cmap='viridis')
plt.colorbar(cf, ax=ax, label='log₁₀(ε)')
cs = ax.contour(logR0, logB0, eps_grid, levels=levels, colors=colors, linewidths=2)
ax.clabel(cs, fmt='%.3f', fontsize=9)
ax.xaxis.set_major_locator(int_loc); ax.xaxis.set_major_formatter(log_fmt)
ax.yaxis.set_major_locator(int_loc); ax.yaxis.set_major_formatter(log_fmt)
ax.set_xlabel('r₀ / R')
ax.set_ylabel('b₀ / R')
ax.set_title('Max field error ε')

# mass contours with ε contours overlaid
ax = axes[1]
cf2 = ax.contourf(logR0, logB0, np.log10(mass_grid), levels=50, cmap='magma')
plt.colorbar(cf2, ax=ax, label='log₁₀(mass)')
cs2 = ax.contour(logR0, logB0, eps_grid, levels=levels, colors=colors, linewidths=2)
ax.clabel(cs2, fmt='%.3f', fontsize=9)
ax.xaxis.set_major_locator(int_loc); ax.xaxis.set_major_formatter(log_fmt)
ax.yaxis.set_major_locator(int_loc); ax.yaxis.set_major_formatter(log_fmt)
ax.set_xlabel('r₀ / R')
ax.set_ylabel('b₀ / R')
ax.set_title('Mass (with ε contours)')

plt.tight_layout()
plt.savefig('/www/flatearth/cylinder_scan.png', dpi=150)
print("Saved to /www/flatearth/cylinder_scan.png")
