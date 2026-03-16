# Lower Bounds on M*(ε)

We seek a provable lower bound M*(ε) ≥ h(ε, R, g₀) where h→∞ as ε→0.

## Setup

An axisymmetric mass distribution ρ(r',z') ≥ 0 in z' ≤ 0 produces a
gravitational field (g_z, g_r) on the disk r ∈ [0,R] at z=0. The constraint
is |g_z(r) - g₀| ≤ ε and |g_r(r)| ≤ ε for all r ∈ [0,R]. Mass is
M = 2π ∫∫ ρ(r',z') r' dr' dz'.

We already have a "solid angle" lower bound M ≥ (g₀-ε)R²/(2π) which
approaches a finite constant as ε→0 — useless for showing divergence.

## Approach 1: Energy integral

The gravitational potential Φ is harmonic in z > 0 with ∂Φ/∂z = -g_z on the
surface. The field energy in the upper half-space is:

E = ∫_{z>0} |∇Φ|² dV

### Q1: Energy lower bound from the constraint

Given that g_z(r) = g₀ ± ε on [0,R] and Φ → 0 at infinity, is there a lower
bound E ≥ f(ε) that diverges as ε→0?

The intuition: Φ must approximate g₀z (which has infinite energy) near the
disk but decay at infinity. As ε→0 the approximation must extend further,
increasing E.

**The energy is finite even at ε=0, so this link fails.**

The minimum-energy potential satisfying ∂Φ/∂z = −g₀ on [0,R] with Φ→0 at
infinity solves a mixed BVP (Neumann on the disk, Dirichlet outside). The
solution is the classical "electrified disk" dual: Φ(r,0) = (2g₀/π)√(R²−r²),
giving finite E₀ = O(g₀²R³). The key point is that g_z outside the disk is
unconstrained, so the optimizer can smooth the Hankel spectrum and keep E
finite. There is no divergence as ε→0.

### Q2: Energy upper bound from mass

Is there an inequality E ≤ C · M^α (or similar) relating the field energy
to the total source mass? This would close the chain E ≥ f(ε) → M ≥ f(ε)^{1/α}/C.

Note: Φ is produced by sources in z ≤ 0, so E is the energy of the field
in the source-free region z > 0 only.

**No useful bound exists.**

Energy depends on both mass and geometry. A point mass m at depth h gives
E ∼ m²/h — for fixed m, E is unbounded (h→0) or zero (h→∞). Without
constraining source geometry, E and M are decoupled.

### Q3: Does the chain close?

Combining Q1 and Q2, do we get M ≥ h(ε) with h→∞? Or does one of the
links saturate?

**No.** Q1 gives finite energy even at ε=0, and Q2 provides no useful
energy-to-mass link. The chain does not close.

## Approach 2: Multipole expansion

Express Φ in terms of its multipole moments. The monopole is proportional
to M. Higher moments (quadrupole, etc.) must be tuned to produce near-uniform
g_z on [0,R].

### Q4: How many multipole moments must be controlled?

If g_z = g₀ ± ε on [0,R], how many Legendre/Zernike coefficients of the
field must be O(ε)? Is this number finite for each ε > 0, or does it grow
as ε→0?

**Near-field issue; finite number of effective modes.**

The multipole expansion doesn't converge on the disk (inside the source
region's bounding sphere). The Hankel-domain representation is more
appropriate: ĝ_z(k) must match g₀R J₁(kR)/k for k ≲ 1/R. The SVD analysis
(see AGENT.md) shows ~11 well-determined modes — this count is set by the
geometry (R, depth range) and doesn't grow as ε→0. Only the *precision* to
which these modes must be matched increases.

### Q5: Multipole moment bounds from mass

The ℓ-th multipole moment of a distribution with total mass M and support
radius R_max satisfies |Q_ℓ| ≤ M · R_max^ℓ (or similar). If many moments
must be small, does this force M or R_max to grow?

**Decoupled from mass.**

Mass is the k=0 moment: M = 2π lim_{k→0} ĝ_z(k)/k. Uniformity constrains
k>0 modes. The Hankel-domain kernel is
ĝ_z(k) = 2πk ∫∫ ρ e^{kz'} J₀(kr') r' dr' dz',
and the factor e^{kz'} ≤ 1 gives ĝ_z(k) ≤ kM, but this is tight only at
k→0 where it just recovers Gauss's law. No useful coupling between finite-k
constraints and M.

### Q6: Does the chain close?

Combining Q4 and Q5, do we get a bound M ≥ h(ε) or R_max ≥ h(ε) that
diverges?

**No.** The near-field convergence issue (Q4) and mass-spectrum decoupling
(Q5) prevent closure.

## Approach 3: Analytic continuation / harmonic measure

The field (g_z, g_r) on the disk is the boundary trace of a harmonic function.
A harmonic function that is ε-close to constant on a disk and decays at
infinity is constrained by the Poisson kernel / harmonic measure of the disk.

### Q7: Quantitative rigidity

For a harmonic function u in z > 0 with u = 0 at infinity: if u ≈ g₀z on
the disk (to within ε in gradient), what is the minimum L² norm (or other
natural norm) of u on the boundary z=0? Does this quantity diverge as ε→0?

**Norms are finite at ε=0.**

The harmonic function Φ in z>0 approximating −g₀z on the disk has finite
H^{1/2} boundary norm. The Dirichlet-to-Neumann map relates ∂Φ/∂z on [0,R]
to Φ on all of z=0, and the freedom to choose Φ (equivalently g_z) outside
the disk keeps all norms bounded. The three-lines / Hadamard arguments require
constraints on *all* of the boundary, not just a compact subset.

### Q8: Connection to mass

Can the boundary norm of Φ (or ∂Φ/∂z) on z=0 be bounded in terms of M?
Note that outside the disk, g_z(r) on the surface is unconstrained — it can
be anything the mass distribution produces.

**Same geometry-dependence issue as Q2.**

Φ(r,0) = ∫∫ ρ G(r,r',z') r' dr' dz' where G diverges as (r',z')→(r,0).
The sup of G over the source half-space is infinite, so ||Φ||_{L²} ≤ M · (∞).
No useful bound without constraining source geometry.

## Approach 4: Constructive ring upper bound

Place a single ring at (r'=a, z'=-h) with mass m. Compute the field error
as a function of (a, h, m). Optimize to get an explicit upper bound
M*(ε) ≤ M_ring(ε).

### Q9: Single ring field

What are g_z(r) and g_r(r) produced by a ring of mass m at radius a and
depth h? Express in terms of elliptic integrals. What is the field error
|g_z(r) - g₀| as a function of r, for optimal (a, h, m)?

Ring of mass m at (a, −h):

g_z(r) = (m/2π) · 4hE(k²)/(βα²)

g_r(r) = (m/2π) · (2/rβ)[(r²−a²−h²)E(k²)/α² + K(k²)]

where α² = (r−a)²+h², β² = (r+a)²+h², k² = 4ra/β², and K, E are complete
elliptic integrals of the first and second kind.

On-axis (a=0): g_z(r) = mh/(r²+h²)^{3/2}. Variation on [0,R]:
g_z(R) − g_z(0) ≈ −3g₀R²/(2h²) for R ≪ h.

### Q10: Scaling of M_ring(ε)

How does the optimal ring mass scale with ε? Is M_ring ~ 1/ε, 1/ε², log(1/ε),
or something else?

**M_ring ∼ g₀²R²/ε (i.e., 1/ε scaling).**

Axial ring (a=0): uniformity requires h ≳ R√(g₀/ε), giving m = g₀h² ∼
g₀²R²/ε. Off-axis ring (a≈R) can partially help at the disk edge but the
dominant scaling remains 1/ε. Multiple rings cannot beat 1/ε with finitely
many rings for a clean lower bound, though the constant improves.

## Summary

### Q11: Which approach is most likely to succeed?

Given the answers above, which approach gives the tightest or most tractable
bound on M*(ε)?

**Approach 4 for upper bounds; LP duality for lower bounds.**

Approach 4 gives a concrete, computable upper bound M*(ε) ≤ O(g₀²R²/ε).
Approaches 1–3 all fail because the field energy / boundary norms are finite
at ε=0, and the link from field quantities to mass is broken by geometry
dependence.

The most promising route for a *lower* bound (not among the listed approaches)
is **LP duality**: the dual of the min-mass LP gives

M ≥ max_w {∫w(r)g₀ r dr − ε∫|w(r)|r dr}

subject to 2πr'∫w(r)K_z(r,r',z')r dr ≤ 1 ∀(r',z'). Showing the dual
objective diverges as ε→0 would establish M*(ε)→∞.

## Bonus results

### LP duality framework for lower bounds

The minimum-mass problem is a linear program: minimize M = 2π∫∫ρ r'dr'dz'
subject to ρ≥0 and |∫∫ρ K_z r'dr'dz' − g₀| ≤ ε on [0,R]. Its dual provides
a certificate for any lower bound: find a test function w(r) on [0,R] such
that ∫w(r)K_z(r,r',z')r dr ≤ 1/(2πr') for all source points (r',z'), and
then M ≥ ∫w(r)g₀ r dr − ε∫|w(r)|r dr. The challenge is constructing w with
a large gap between these two integrals.

### On-axis analyticity argument

g_z(0,z) for z>0 satisfies g_z(0,0)=g₀, |g_z''(0,0)|=O(ε/R²), and
g_z(0,z) ≤ M/z² for large z. The transition from ~g₀ to ~M/z² forces
√(M/g₀) ≳ g₀R²/ε^{?} — suggestive of divergence but hard to make rigorous
without bounding all Taylor coefficients.

### Numerical Pareto frontier

Trace M*(ε) computationally (via the LP or the SVD-based optimizer) to
extract the empirical scaling exponent. This would guide which theoretical
bound to pursue and validate whether the 1/ε scaling from the single-ring
upper bound is tight.

## Are there other obvious approaches?

The file covers the main routes. A few gaps:

**Hankel-domain uncertainty principle** — implicit in Q10 but never stated as a
standalone approach. The source at depth h acts as a low-pass filter with
cutoff ~1/h. Achieving ε-uniformity requires suppressing modes up to k~1/R,
forcing h ≳ R·f(ε) and driving mass up. Most physically transparent argument
and could potentially be made rigorous for general sources. Most obviously
missing.

**Slab truncation argument** — the ε=0 solution is the infinite uniform slab.
Any finite-mass source is a truncated slab; truncation error at the disk edge
could be bounded below. Dual to Approach 4 from the other direction.

**Gravimetric inverse problem literature** — Backus-Gilbert theory and related
resolution/stability results in potential field inversion may give off-the-shelf
results for exactly this formulation.

**Capacity-based arguments** — Newtonian capacity of the disk (cap = 2R/π)
gives a natural scale, but likely dead-ends for the same reason as Approach 3:
freedom to choose the field outside [0,R] breaks the link to mass.

Of these, the Hankel-domain uncertainty principle is most promising and most
obviously absent. LP duality remains the strongest path to a rigorous lower
bound.
