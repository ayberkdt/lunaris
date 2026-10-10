import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.Calculus.ContDiff.Basic

set_option autoImplicit false

/-!
# The sine activation (`activation = "sine"`)

`Sine(w0)` in `src/lunaris/surrogate/st_lrps/networks/models.py` computes
`sin (w0 * x)`.  ST-LRPS trains the *gradient* of the network (Sobolev /
`potential_autograd`: `a = ∇ΔU`) and logs a Hutchinson Laplacian, so the
activation must have well-behaved first and second derivatives everywhere.
These theorems isolate exactly the properties that choice relies on.
-/

open Real
open scoped ContDiff

namespace StLrpsFormal

/-- The SIREN unit: `Sine(w0)(x) = sin (w0 * x)`. -/
noncomputable def sirenUnit (w0 : ℝ) : ℝ → ℝ := fun x => Real.sin (w0 * x)

/-- Sine is bounded, so pre-activation magnitudes are governed by the weights
alone; there is no activation blow-up at any depth. -/
theorem sirenUnit_abs_le_one (w0 x : ℝ) : |sirenUnit w0 x| ≤ 1 :=
  Real.abs_sin_le_one _

/-- The derivative of a sine unit is `w0 * cos (w0 x)`: the exact analytic
gradient that `torch.autograd` returns for the acceleration head. -/
theorem hasDerivAt_sirenUnit (w0 x : ℝ) :
    HasDerivAt (sirenUnit w0) (w0 * Real.cos (w0 * x)) x := by
  have h := (Real.hasDerivAt_sin (w0 * x)).comp x (hasDerivAt_const_mul w0 (x := x))
  unfold sirenUnit
  simpa [Function.comp_def, mul_comm] using h

/-- `cos (w0 x) = sin (w0 x + π/2)`: the derivative of a sine unit is a
*phase-shifted sine unit*.  Hence every derivative of the network is again a
SIREN and inherits the same boundedness and smoothness (Sitzmann et al. 2020,
§3.1).  This is why differentiating the potential network to obtain the
acceleration does not degrade the function class. -/
theorem deriv_sirenUnit_is_shifted_sine (w0 x : ℝ) :
    w0 * Real.cos (w0 * x) = w0 * Real.sin (w0 * x + Real.pi / 2) := by
  rw [Real.sin_add_pi_div_two]

/-- The derivative is bounded by `|w0|`, so the network gradient (the
predicted acceleration) cannot exceed the product of weight norms and band
frequencies: `w0` directly controls the Lipschitz budget of `∇ΔU`. -/
theorem abs_deriv_sirenUnit_le (w0 x : ℝ) :
    |w0 * Real.cos (w0 * x)| ≤ |w0| := by
  rw [abs_mul]
  calc |w0| * |Real.cos (w0 * x)| ≤ |w0| * 1 :=
        mul_le_mul_of_nonneg_left (Real.abs_cos_le_one _) (abs_nonneg _)
    _ = |w0| := mul_one _

/-- A sine unit is `C^n` for every `n` (in particular `C^∞`).  The Hutchinson
Laplacian diagnostic (`use_laplacian_regularization`, `n_hutchinson_samples`)
needs second derivatives that exist and are continuous everywhere; ReLU-type
activations have a zero second derivative almost everywhere and a kink at 0,
which is why they are unsuitable for a Sobolev-trained potential. -/
theorem contDiff_sirenUnit (w0 : ℝ) (n : WithTop ℕ∞) :
    ContDiff ℝ n (sirenUnit w0) := by
  have h : ContDiff ℝ n (fun x : ℝ => w0 * x) := contDiff_const.mul contDiff_id
  exact Real.contDiff_sin.comp h

/-- Period of a sine unit is `2π / w0`: the band frequency `w0` fixes the
spatial wavelength the unit can represent in network-scaled coordinates. -/
theorem sirenUnit_periodic (w0 : ℝ) (hw : w0 ≠ 0) (x : ℝ) :
    sirenUnit w0 (x + 2 * Real.pi / w0) = sirenUnit w0 x := by
  unfold sirenUnit
  have h : w0 * (2 * Real.pi / w0) = 2 * Real.pi := by field_simp
  rw [mul_add, h, Real.sin_add_two_pi]

end StLrpsFormal
