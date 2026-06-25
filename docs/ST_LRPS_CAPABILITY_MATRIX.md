# ST-LRPS Capability Matrix

<!-- GENERATED FROM lunaris.surrogate.st_lrps.shared.capabilities.
     Do not edit by hand — run the regenerate helper and commit. A test
     asserts this file equals render_capability_matrix_markdown(). -->

Single source of truth for *which ST-LRPS system implements which feature, and to what extent*. Every unavailable capability raises `UnsupportedCapability` (a `NotImplementedError` subclass) routed through this registry, never an ad-hoc error.

Legend: [x] supported - [ ] not yet implemented - [-] not applicable by design.

| Component | Subject | Feature | Status | Notes |
|---|---|---|---|---|
| runtime | `potential_autograd` | scalar_residual_potential | [x] supported | Reference kind: the network outputs DeltaU(x); the scalar residual potential is read off directly. |
| runtime | `potential_autograd` | residual_acceleration | [x] supported | Delta_a = a_sign * grad(DeltaU_scaled) * (u_scale/x_scale) via autograd. |
| runtime | `potential_autograd` | total_acceleration | [x] supported | a_total = a_base + residual acceleration, with the analytical baseline. |
| runtime | `force_direct` | residual_acceleration | [x] supported | The network outputs the 3-vector residual acceleration directly. |
| runtime | `force_direct` | total_acceleration | [x] supported | a_total = a_base + predicted residual acceleration. |
| runtime | `force_direct` | scalar_residual_potential | [-] N/A by design | A direct-force model predicts acceleration only; there is no scalar potential DeltaU to return. Use the acceleration API instead. |
| baseline | `residual` | any_baseline_kind | [x] supported | Residual datasets already store baseline-subtracted labels, so the in-layer baseline is exactly zero. |
| baseline | `full` | none | [x] supported | Full-field labels with no analytical baseline; the network sees the full field. |
| baseline | `full` | point_mass | [x] supported | Monopole baseline U=mu/r, a=-mu r/|r|^3 subtracted analytically. |
| baseline | `full` | spherical_harmonics | [x] supported | SH baseline through base_degree, evaluated by lunaris.physics.spherical_harmonics; requires the source gravity model coefficients (threaded in via meta.gravity_model_path). |
| training | `potential_autograd` | sobolev_autograd | [x] supported | Main trainer (lunaris-train-st-lrps): scalar-potential SIREN with a Sobolev U/a objective and autograd acceleration. |
| training | `force_direct` | direct_residual_accel | [x] supported | Separate trainer (lunaris-train-force-direct): regress the residual acceleration directly with a 3-output head. |
