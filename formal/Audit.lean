import StLrpsFormal

/-!
Axiom audit: every theorem must depend only on the three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).  Any `sorryAx` here would mean
an unproved claim.  Run with `lake env lean Audit.lean`.
-/

open StLrpsFormal

-- Activation
#print axioms sirenUnit_abs_le_one
#print axioms hasDerivAt_sirenUnit
#print axioms deriv_sirenUnit_is_shifted_sine
#print axioms abs_deriv_sirenUnit_le
#print axioms contDiff_sirenUnit
#print axioms sirenUnit_periodic
-- Init
#print axioms uniform_second_moment
#print axioms siren_hidden_bound_unit_variance
#print axioms first_layer_preactivation_bound
#print axioms first_layer_argument_bound
#print axioms head_init_is_damped
-- Bands
#print axioms bandFreq_sq_div_nine
#print axioms bandFreq_mono
#print axioms band_centres_two_bands
#print axioms sqrt26_bounds
#print axioms sqrt200_bounds
#print axioms band_low_bounds
#print axioms band_high_bounds
#print axioms bands_inside_clamp
-- RadialDecay
#print axioms rho_lt_one
#print axioms rho_nonneg
#print axioms rho_pow_antitone
#print axioms rho_pow_strictAnti
#print axioms rho_pow_mem_Ioo
#print axioms tail_geometric
#print axioms tail_summable
-- Residual
#print axioms residual_deriv_split
#print axioms residual_fderiv_split
#print axioms hasDerivAt_pointMass
#print axioms pointMass_deriv_neg
#print axioms resblock_identity_at_init
-- CondonShortley
#print axioms cs_phase_odd
#print axioms cs_phase_even
#print axioms cs_phase_zonal_invariant
#print axioms cs_phase_changes_odd_term
#print axioms cs_phase_involutive
