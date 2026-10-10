import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

set_option autoImplicit false

/-!
# SIREN initialisation (`siren_init_first_`, `siren_init_hidden_`, head init)

`networks/models.py` initialises

* first layer:   `weight ~ U(-1/n, 1/n)`                       (`siren_init_first_`)
* hidden layers: `weight ~ U(-b, b)`, `b = √(6/n) / w0`        (`siren_init_hidden_`)
* output head:   `weight ~ U(-0.1·b, 0.1·b)`, bias `0`
* residual blocks: `lin1` as hidden, `lin2 = 0`                (`SirenResBlock`)

with `w0_first = w0_hidden = 30`, `hidden = n = 512`.  The theorems below are
the analytic identities behind these constants (Sitzmann et al. 2020, Thm 1.8
and §3.2).  The probabilistic step — that `sin` of a wide pre-activation has
second moment `1/2` (arcsine law) — is an *input assumption* here, not proved.
-/

open Finset

namespace StLrpsFormal

/-- Second moment of the uniform law on `[-b, b]` (density `1 / (2b)`) is
`b² / 3`.  This is the variance every `nn.init.uniform_` call above injects. -/
theorem uniform_second_moment {b : ℝ} (hb : 0 < b) :
    (1 / (2 * b)) * ∫ x in (-b)..b, x ^ 2 = b ^ 2 / 3 := by
  have hb0 : b ≠ 0 := hb.ne'
  rw [integral_pow]
  push_cast
  field_simp
  ring

/-- Hidden-layer bound `b = √(6/n) / w0`: with weight variance `b²/3`, `n`
independent inputs of second moment `1/2`, the pre-activation `w0 · Σ wᵢ xᵢ`
has variance exactly `1`, *independently of `n` and `w0`*.  This is the reason
the bound divides by `w0`: the frequency multiplier is cancelled at
initialisation so that every layer of a depth-5, width-512 network starts in
the same regime. -/
theorem siren_hidden_bound_unit_variance {n : ℕ} (hn : 0 < n) {w0 : ℝ} (hw : w0 ≠ 0) :
    w0 ^ 2 * ((n : ℝ) * ((Real.sqrt (6 / n) / w0) ^ 2 / 3) * (1 / 2)) = 1 := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hn0 : (n : ℝ) ≠ 0 := hn'.ne'
  rw [div_pow, Real.sq_sqrt (by positivity)]
  field_simp
  ring

/-- First-layer bound `1/n`: for inputs in `[-1, 1]` (the scaled coordinates,
`x_scale_policy = max_norm_from_origin`) the pre-activation `Σ wᵢ xᵢ` is
confined to `[-1, 1]`. -/
theorem first_layer_preactivation_bound {n : ℕ} (hn : 0 < n) (w x : Fin n → ℝ)
    (hw : ∀ i, |w i| ≤ 1 / n) (hx : ∀ i, |x i| ≤ 1) :
    |∑ i, w i * x i| ≤ 1 := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  calc |∑ i, w i * x i| ≤ ∑ i, |w i * x i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin n, (1 / n : ℝ) * 1 := by
        apply Finset.sum_le_sum
        intro i _
        rw [abs_mul]
        exact mul_le_mul (hw i) (hx i) (abs_nonneg _) (by positivity)
    _ = 1 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        field_simp

/-- Hence the first sine sees arguments in `[-w0_first, w0_first] = [-30, 30]`,
i.e. up to `30/π ≈ 9.5` half-periods.  The purpose of `w0_first = 30` is
precisely to spread the initial activations over many periods of `sin`, so the
first layer is not stuck in the linear regime near 0 (Sitzmann et al. §3.2). -/
theorem first_layer_argument_bound {n : ℕ} (hn : 0 < n) (w x : Fin n → ℝ) (w0 : ℝ)
    (hw : ∀ i, |w i| ≤ 1 / n) (hx : ∀ i, |x i| ≤ 1) :
    |w0 * ∑ i, w i * x i| ≤ |w0| := by
  rw [abs_mul]
  calc |w0| * |∑ i, w i * x i| ≤ |w0| * 1 :=
        mul_le_mul_of_nonneg_left (first_layer_preactivation_bound hn w x hw hx) (abs_nonneg _)
    _ = |w0| := mul_one _

/-- The output head uses a damped bound `0.1 · b`: its initial variance is
`1/100` of a hidden layer's, so the initial residual prediction `ΔU ≈ 0` and
the exact baseline dominates at step 0 while gradients to the backbone stay
non-zero. -/
theorem head_init_is_damped {b : ℝ} (hb : 0 < b) : (1 / 10 : ℝ) * b < b := by
  linarith

end StLrpsFormal
