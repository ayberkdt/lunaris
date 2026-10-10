import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Tactic

set_option autoImplicit false

/-!
# Residual formulation, sign convention and residual blocks

* `target_mode = "residual"`, `baseline_kind = "spherical_harmonics"`,
  `degree_min = 25`: the network learns `ΔU = U_200 - U_25`; the analytic
  baseline carries the large-scale field.
* `resolved_a_sign = +1`: geodesy convention `a = +∇U` with `U = μ/r + …`.
* `use_residual_blocks = True`: `SirenResBlock` initialises its output
  projection to zero, so each block starts as the identity map.
-/

namespace StLrpsFormal

/-- Linearity of the derivative: differentiating `U_base + ΔU` gives
`a_base + Δa`.  This is the whole content of the residual design — the exact
baseline acceleration is added back untouched, and the network's error is only
the error in `Δa`. -/
theorem residual_deriv_split {U₀ ΔU : ℝ → ℝ} {a₀ Δa x : ℝ}
    (h₀ : HasDerivAt U₀ a₀ x) (hΔ : HasDerivAt ΔU Δa x) :
    HasDerivAt (fun r => U₀ r + ΔU r) (a₀ + Δa) x :=
  h₀.add hΔ

/-- The same statement for the Fréchet derivative on any normed space, in
particular `ℝ³`: the gradient of the total potential is the sum of gradients. -/
theorem residual_fderiv_split {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {U₀ ΔU : E → ℝ} {L₀ LΔ : E →L[ℝ] ℝ} {p : E}
    (h₀ : HasFDerivAt U₀ L₀ p) (hΔ : HasFDerivAt ΔU LΔ p) :
    HasFDerivAt (fun q => U₀ q + ΔU q) (L₀ + LΔ) p :=
  h₀.add hΔ

/-- Point-mass potential in the geodesy sign convention: `U(r) = μ / r` has
radial derivative `-μ / r²`. -/
theorem hasDerivAt_pointMass (μ r : ℝ) (hr : r ≠ 0) :
    HasDerivAt (fun s : ℝ => μ * s⁻¹) (μ * (-(r ^ 2)⁻¹)) r :=
  (hasDerivAt_inv hr).const_mul μ

/-- For `μ > 0`, `r > 0` that derivative is negative: `+∇U` points *inward*,
i.e. the geodesy convention `a = +∇U` (`a_sign = +1`) is attractive.  A dataset
stamped with the physics convention `a = -∇Φ` would need `a_sign = -1`; the
contract stores the sign so the two can never be silently mixed. -/
theorem pointMass_deriv_neg {μ r : ℝ} (hμ : 0 < μ) (hr : 0 < r) :
    μ * (-(r ^ 2)⁻¹) < 0 :=
  mul_neg_of_pos_of_neg hμ (neg_neg_of_pos (inv_pos.mpr (pow_pos hr 2)))

/-- A residual block whose output projection is the zero map is the identity:
`x + 0·h = x`.  `SirenResBlock` zero-initialises `lin2`, so at initialisation
the whole shared stack is the identity and depth cannot hurt the initial
signal. -/
theorem resblock_identity_at_init {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (x h : E) : x + (0 : E →L[ℝ] E) h = x := by
  simp

end StLrpsFormal
