#!/usr/bin/env python3
"""Minimum-mass axisymmetric slab, eps=0.

Find b(r) minimizing mass subject to:
  sqrt((g_z(r) - g0)^2 + g_r(r)^2) <= epsilon   for all r <= R

Slab occupies z in [-b(r), 0], unit density, axisymmetric.

Kernel formulas (derived from azimuthal integral of 1/r^3 force):
  g_z from ring at (r', -z') at observer (r, 0):
    = z' * 4*E(k) / (beta * alpha^2)

  g_r from ring at (r', -z') at observer (r, 0):
    = 2/(r*beta) * [(r^2 - r'^2 - z'^2)/alpha^2 * E(k) + K(k)]

  where beta^2 = (r+r')^2 + z'^2
        alpha^2 = (r-r')^2 + z'^2
        k^2 = 4*r*r' / beta^2

Depth integral over z' from 0 to b(r') done numerically.
"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')

# --- Parameters ---
disk_r = 0.5
g0     = 1.0      # target g_z magnitude (kernel gives positive downward values)
epsilon = 0.25    # field tolerance (start loose, tighten later)

n_src = 60        # b(r) nodes
n_obs = 40        # observation points on disk
n_z   = 25        # z quadrature points per source column
R_ext = disk_r * 5.0

# Grids
r_src = np.linspace(0, R_ext, n_src + 1)
r_src = 0.5 * (r_src[:-1] + r_src[1:])
dr    = R_ext / n_src

r_obs = np.linspace(disk_r / n_obs, disk_r * 0.99, n_obs)

z_frac = (np.arange(n_z) + 0.5) / n_z   # uniform quadrature on [0,1]


def compute_field(b_vals):
    """Vectorized field at r_obs for slab with bottom at b_vals.

    Returns g_z, g_r arrays of shape (n_obs,).
    """
    # Shapes: (n_obs, n_src, n_z)
    ro = r_obs[:, None, None]
    rs = r_src[None, :, None]
    z_ = b_vals[None, :, None] * z_frac[None, None, :]
    dz = b_vals[None, :, None] / n_z

    beta2  = (ro + rs)**2 + z_**2
    alpha2 = np.maximum((ro - rs)**2 + z_**2, 1e-20)
    beta   = np.sqrt(np.maximum(beta2, 1e-30))
    k2     = np.clip(4 * ro * rs / np.maximum(beta2, 1e-30), 0, 1 - 1e-12)

    K = ellipk(k2)
    E = ellipe(k2)

    gz_k = z_ * 4 * E / (beta * alpha2)

    ro_safe = np.maximum(ro, 1e-10)
    gr_k = 2 / (ro_safe * beta) * ((ro**2 - rs**2 - z_**2) / alpha2 * E + K)

    # Integrate: sum over z (axis=2) and r_src (axis=1)
    gz = np.sum(gz_k * dz * rs * dr, axis=(1, 2))
    gr = np.sum(gr_k * dz * rs * dr, axis=(1, 2))

    return gz, gr


# --- Validate ---
b_test = np.full(n_src, 0.1)
gz_t, gr_t = compute_field(b_test)
print(f"Validation — uniform slab b=0.1:")
print(f"  g_z = {gz_t.mean():.4f} (infinite-slab limit: {2*np.pi*0.1:.4f})")
print(f"  g_r max = {np.max(np.abs(gr_t)):.4f} (expect ~0)")

# Scale initial guess to hit g0
b0 = abs(g0) / gz_t.mean() * 0.1
print(f"  Initial b0 = {b0:.4f}")

# --- Optimization ---
def objective(log_b):
    return 2 * np.pi * np.sum(np.exp(log_b) * r_src * dr)

def constraint(log_b):
    gz, gr = compute_field(np.exp(log_b))
    return epsilon**2 - (gz - g0)**2 - gr**2  # >= 0

x0 = np.full(n_src, np.log(b0))

gz0, gr0 = compute_field(np.exp(x0))
print(f"\nInitial field: gz=[{gz0.min():.3f},{gz0.max():.3f}], gr_max={np.max(np.abs(gr0)):.4f}")
print(f"Initial mass: {objective(x0):.4f}")
print(f"Constraint violations: {np.sum(constraint(x0) < 0)}/{n_obs}")

print(f"\nOptimizing (n_src={n_src}, n_obs={n_obs}, ε={epsilon})...")
result = minimize(
    objective, x0,
    method='SLSQP',
    constraints={'type': 'ineq', 'fun': constraint},
    bounds=[(-8, 2)] * n_src,
    options={'maxiter': 1000, 'ftol': 1e-8, 'disp': True}
)

b_opt = np.exp(result.x)
gz_opt, gr_opt = compute_field(b_opt)
err = np.sqrt((gz_opt - g0)**2 + gr_opt**2)

print(f"\nMax field error: {err.max():.4f} (ε={epsilon})")
print(f"Constraint violations: {np.sum(err > epsilon)}/{n_obs}")
print(f"Optimal mass: {objective(result.x):.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
r_plot = np.concatenate([-r_src[::-1], r_src])
b_plot = np.concatenate([b_opt[::-1], b_opt])
ax.fill_between(r_plot, 0, -b_plot, color='steelblue', alpha=0.8)
ax.plot([-disk_r, disk_r], [0, 0], 'r-', lw=3, label='Disk')
ax.axvline( disk_r, color='r', ls='--', alpha=0.3)
ax.axvline(-disk_r, color='r', ls='--', alpha=0.3)
ax.set_xlim(-R_ext/2, R_ext/2)
ax.set_ylim(-np.max(b_opt)*1.3, np.max(b_opt)*0.3)
ax.set_aspect('equal')
ax.set_title(f'Min-mass slab (ε={epsilon})')
ax.set_xlabel('r'); ax.set_ylabel('z')
ax.legend()

ax = axes[1]
ax.plot(r_obs, gz_opt, 'b-', label='$g_z$')
ax.plot(r_obs, gr_opt, 'r-', label='$g_r$')
ax.axhline(g0, color='b', ls='--', alpha=0.5)
ax.axhline(0,  color='r', ls='--', alpha=0.5)
ax.fill_between(r_obs, g0 - epsilon, g0 + epsilon, alpha=0.1, color='blue', label='ε band')
ax.set_title('Field on disk'); ax.set_xlabel('r'); ax.legend()

ax = axes[2]
ax.plot(r_obs, err, 'k-')
ax.axhline(epsilon, color='r', ls='--', label=f'ε={epsilon}')
ax.set_title('|g - target|'); ax.set_xlabel('r'); ax.legend()

plt.tight_layout()
plt.savefig('/www/flatearth_minmass.png', dpi=150)
print("Saved to /www/flatearth_minmass.png")
