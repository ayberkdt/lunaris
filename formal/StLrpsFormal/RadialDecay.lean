import Mathlib.Analysis.SpecificLimits.Basic

set_option autoImplicit false

/-!
# Radial decay of the residual field (`embedding_type = "physical_radial_decay"`)

`PhysicalRadialDecayEncoding` in `networks/models.py` feeds the network the
features `ρ, ρ², ρ³, ρ⁴` with `ρ = r_ref / r` (`physical_radial_decay_max_power = 4`,
`r_ref_m = 1738000`).  The degree-`n` term of the spherical-harmonic expansion of
the potential scales as `(R/r)^(n+1)`, so the residual field ST-LRPS learns
(degrees 26 … 200) is a sum of geometrically decaying terms in `ρ`.  The
theorems below are the exact facts that justify the feature choice — and that
predict the altitude-dependent loss of directional accuracy measured in the
evaluation (`angular_error_by_altitude.csv`).
-/

namespace StLrpsFormal

/-- Above the reference sphere the decay base `ρ = R / r` is strictly below 1. -/
theorem rho_lt_one {R r : ℝ} (hR : 0 < R) (h : R < r) : R / r < 1 := by
  rw [div_lt_one (hR.trans h)]
  exact h

theorem rho_nonneg {R r : ℝ} (hR : 0 < R) (h : R < r) : 0 ≤ R / r :=
  div_nonneg hR.le (hR.trans h).le

/-- Higher degrees decay at least as fast: `ρ^n ≤ ρ^m` for `m ≤ n`.  The
degree-200 tail is always dominated by the degree-26 head. -/
theorem rho_pow_antitone {R r : ℝ} (hR : 0 < R) (h : R < r) {m n : ℕ} (hmn : m ≤ n) :
    (R / r) ^ n ≤ (R / r) ^ m :=
  pow_le_pow_of_le_one (rho_nonneg hR h) (rho_lt_one hR h).le hmn

/-- Strict version: each additional degree strictly loses amplitude. -/
theorem rho_pow_strictAnti {R r : ℝ} (hR : 0 < R) (h : R < r) {m n : ℕ} (hmn : m < n) :
    (R / r) ^ n < (R / r) ^ m :=
  pow_lt_pow_right_of_lt_one₀ (div_pos hR (hR.trans h)) (rho_lt_one hR h) hmn

/-- Every feature `ρ^k` (k ≥ 1) is strictly inside `(0, 1)`: the encoding is
bounded on the whole training shell (100 – 1000 km), so no feature scaling is
needed and the sine arguments stay well-conditioned. -/
theorem rho_pow_mem_Ioo {R r : ℝ} (hR : 0 < R) (h : R < r) {k : ℕ} (hk : k ≠ 0) :
    0 < (R / r) ^ k ∧ (R / r) ^ k < 1 :=
  ⟨pow_pos (div_pos hR (hR.trans h)) k, pow_lt_one₀ (rho_nonneg hR h) (rho_lt_one hR h) hk⟩

/-- Geometric tail: the sum of all decay factors from degree `N` upward equals
`ρ^N / (1 - ρ)`.  With bounded coefficients this is the envelope of the entire
residual field beyond the analytic baseline, and it shrinks geometrically with
altitude — the residual becomes numerically negligible at high altitude, which
is exactly where the measured direction error grows. -/
theorem tail_geometric {q : ℝ} (h0 : 0 ≤ q) (h1 : q < 1) (N : ℕ) :
    ∑' n : ℕ, q ^ (n + N) = q ^ N * (1 - q)⁻¹ := by
  simp_rw [pow_add]
  rw [tsum_mul_right, tsum_geometric_of_lt_one h0 h1, mul_comm]

/-- The tail is summable, so truncating the spherical-harmonic series at
`degree_max = 200` leaves a finite, geometrically small remainder. -/
theorem tail_summable {q : ℝ} (h0 : 0 ≤ q) (h1 : q < 1) (N : ℕ) :
    Summable fun n : ℕ => q ^ (n + N) := by
  simp_rw [pow_add]
  exact (summable_geometric_of_lt_one h0 h1).mul_right _

end StLrpsFormal
