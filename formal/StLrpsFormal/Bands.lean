import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

set_option autoImplicit false

/-!
# Multi-scale bands (`n_bands = 2`, `w0_bands = [15.3, 42.4]`)

`compute_harmonic_w0_bands(n_bands, degree_min, degree_max)` in
`networks/models.py` places `n_bands` centre degrees geometrically between
`lo = degree_min + 1 = 26` and `hi = degree_max = 200` and sets the band
frequency to `w0 = 3 √deg` (then rounds to one decimal and clamps to
`[10, 100]`).  A degree-`n` spherical harmonic has characteristic spatial
frequency proportional to `n`, and the SIREN frequency needed to represent a
feature grows with its bandwidth; the square-root law is the implementation's
compromise between covering the residual spectrum and keeping `w0` inside the
range where SIREN initialisation is stable.  The theorems below pin down the
exact numbers used in the run.
-/

namespace StLrpsFormal

/-- The band-frequency law before rounding/clamping. -/
noncomputable def bandFreq (deg : ℝ) : ℝ := 3 * Real.sqrt deg

/-- Inverse law: the degree a band targets is recovered from its frequency. -/
theorem bandFreq_sq_div_nine {deg : ℝ} (h : 0 ≤ deg) : (bandFreq deg) ^ 2 / 9 = deg := by
  unfold bandFreq
  rw [mul_pow, Real.sq_sqrt h]
  ring

/-- Higher target degree ⇒ higher band frequency: the ordering of the bands
matches the ordering of the spectrum they cover. -/
theorem bandFreq_mono {a b : ℝ} (hab : a ≤ b) : bandFreq a ≤ bandFreq b := by
  unfold bandFreq
  exact mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hab) (by norm_num)

/-- With two bands, `t ∈ {0, 1}` and the geometric interpolation
`exp (log lo + t (log hi - log lo))` returns exactly the endpoints `26`, `200`. -/
theorem band_centres_two_bands :
    Real.exp (Real.log 26 + 0 * (Real.log 200 - Real.log 26)) = 26 ∧
    Real.exp (Real.log 26 + 1 * (Real.log 200 - Real.log 26)) = 200 := by
  constructor
  · rw [zero_mul, add_zero, Real.exp_log (by norm_num)]
  · have h : Real.log 26 + 1 * (Real.log 200 - Real.log 26) = Real.log 200 := by ring
    rw [h, Real.exp_log (by norm_num)]

theorem sqrt26_bounds : (5.084 : ℝ) < Real.sqrt 26 ∧ Real.sqrt 26 < 5.116 := by
  constructor
  · rw [Real.lt_sqrt (by norm_num)]
    norm_num
  · rw [Real.sqrt_lt' (by norm_num)]
    norm_num

theorem sqrt200_bounds : (14.117 : ℝ) < Real.sqrt 200 ∧ Real.sqrt 200 < 14.149 := by
  constructor
  · rw [Real.lt_sqrt (by norm_num)]
    norm_num
  · rw [Real.sqrt_lt' (by norm_num)]
    norm_num

/-- Low band: `bandFreq 26 ∈ (15.25, 15.35)`, hence rounds to `15.3` — the
value recorded in `config.json` (`w0_bands[0]`). -/
theorem band_low_bounds : (15.25 : ℝ) < bandFreq 26 ∧ bandFreq 26 < 15.35 := by
  unfold bandFreq
  obtain ⟨h1, h2⟩ := sqrt26_bounds
  constructor <;> linarith

/-- High band: `bandFreq 200 ∈ (42.35, 42.45)`, hence rounds to `42.4` — the
value recorded in `config.json` (`w0_bands[1]`), also used by every shared
residual block (`w0_deep = w0_bands[-1]`). -/
theorem band_high_bounds : (42.35 : ℝ) < bandFreq 200 ∧ bandFreq 200 < 42.45 := by
  unfold bandFreq
  obtain ⟨h1, h2⟩ := sqrt200_bounds
  constructor <;> linarith

/-- Both bands lie strictly inside the clamp window `[10, 100]`, so the clamp
in `compute_harmonic_w0_bands` is inactive for this run: the recorded
frequencies are the pure `3 √deg` law. -/
theorem bands_inside_clamp :
    (10 : ℝ) < bandFreq 26 ∧ bandFreq 200 < 100 := by
  obtain ⟨h1, _⟩ := band_low_bounds
  obtain ⟨_, h2⟩ := band_high_bounds
  constructor <;> linarith

end StLrpsFormal
