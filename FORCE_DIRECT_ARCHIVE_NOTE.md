# Why this branch exists

This branch (`experimental/force-direct-archive`, tag
`force-direct-archive-20260705`) preserves the last state of the repository
in which the **force_direct** ST-LRPS variant was fully wired: the 3-output
direct residual-acceleration model kind, its trainer
(`lunaris-train-force-direct` / `training/force_direct_cli.py`), evaluator
(`evaluation/force_direct_eval.py`), the `force_direct` runtime branches in
`runtime/force_model.py` / `surrogate/runtime/gravity_provider.py`, and the
`gpu_st_lrps_direct` batch backend.

## Why it was archived (2026-07-05)

ST-LRPS's defensible identity is a **conservative residual scalar potential
surrogate**: `a = a_SH_baseline + grad(residual potential)` via autograd.
force_direct learns acceleration directly, which:

- breaks the conservative-field guarantee and makes the `is_conservative`
  flag + symplectic non-conservative guard chain meaningless for the model;
- has no scalar potential, so curl/energy-based validation does not apply;
- splits the paper narrative into two model identities with different
  validation obligations.

Decision source: external system review 2026-07-05, item R01 in
`LUNARIS_ROADMAP_PLAN.md` on main. This supersedes the earlier
"ST-LRPS clarity remediation" Phase 2 plan that would have made
force_direct first-class.

## Status on main

Main's runtime rejects force_direct artifacts fail-closed with an error
pointing at this branch. No force_direct training/evaluation entry point
exists on main. If the direct-force idea is ever revived, it restarts from
here as an explicitly experimental line with its own validation contract
(curl + orbit-level), not by silently re-entering main.
