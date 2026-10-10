import Mathlib.Algebra.Ring.Parity
import Mathlib.Tactic

set_option autoImplicit false

/-!
# The Condon–Shortley phase (`spherical_harmonic_convention = 4pi_geodesy_no_condon_shortley_v1`)

GRAIL / JGGRX coefficients are fully normalised **without** the Condon–Shortley
factor `(-1)^m`.  The legacy training data applied that factor by accident
(`src/lunaris/physics/spherical_harmonics.py`, lines 166–177 explain the fix).
These theorems state precisely why that was a corruption and not a harmless
re-parametrisation, and why zonal-only tests could not detect it.
-/

namespace StLrpsFormal

/-- On odd orders the phase is exactly `-1`. -/
theorem cs_phase_odd {m : ℕ} (hm : Odd m) : ((-1 : ℝ) ^ m) = -1 :=
  hm.neg_one_pow

/-- On even orders the phase is `1`. -/
theorem cs_phase_even {m : ℕ} (hm : Even m) : ((-1 : ℝ) ^ m) = 1 :=
  hm.neg_one_pow

/-- Zonal terms (`m = 0`, e.g. `J₂`) are untouched: a test that only checks the
zonal field (`J₂` secular rates, polar orbits) is blind to the bug. -/
theorem cs_phase_zonal_invariant : ((-1 : ℝ) ^ (0 : ℕ)) = 1 :=
  pow_zero _

/-- Applying the phase to a non-zero odd-order term *changes* it.  Since the
tesseral/sectoral field has non-zero odd-order coefficients, the resulting
potential is a different function — the labels of the legacy dataset were a
different field, not a re-scaled one. -/
theorem cs_phase_changes_odd_term {m : ℕ} (hm : Odd m) {t : ℝ} (ht : t ≠ 0) :
    ((-1 : ℝ) ^ m) * t ≠ t := by
  rw [cs_phase_odd hm, neg_one_mul]
  intro h
  exact ht (by linarith)

/-- The phase is an involution: applying it twice restores the coefficients.
This is why a model trained on phase-flipped labels can still be internally
consistent (it learned *a* valid field) while disagreeing with the truth. -/
theorem cs_phase_involutive (m : ℕ) (t : ℝ) :
    ((-1 : ℝ) ^ m) * (((-1 : ℝ) ^ m) * t) = t := by
  rw [← mul_assoc, ← pow_add, ← two_mul, pow_mul]
  simp

end StLrpsFormal
