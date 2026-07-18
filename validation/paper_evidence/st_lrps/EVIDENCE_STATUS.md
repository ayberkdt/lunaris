# ST-LRPS Evidence Status

This file is the claim-scope gate for legacy and current ST-LRPS results. It
does not upgrade a result merely because the files exist.

## Legacy five-day float32 result

The previously circulated **1.106 km median five-day result was produced in
float32** with `paper_safe.enabled=false`. In this regime the observed error
floor is dominated by numerical precision/fixed-step effects and the compared
gravity models do not separate reliably. Therefore:

- it may be used only as throughput or exploratory engineering evidence;
- it must not be cited as trajectory-accuracy, model-equivalence, or
  paper-validation evidence;
- any slide, report, or table containing it must label the dtype and this scope;
- the repository contains no tracked presentation deck to patch, so this file
  is the durable repository-side restriction.

## Accuracy replacement

The canonical five-day replacement template is
`configs/st_lrps/paper/benchmark_5day_general.json`. It requires float64 and
paper-safe validation. The template is not itself evidence: an accuracy claim
requires a completed non-dry run whose benchmark manifest, validation report,
checkpoint hash, split/scaler provenance, and generated CSV tables all form one
consistent artifact chain.

Until that run exists, the five-day accuracy status is **not established**.

## Other historical values

The historical 0.626 km and 15.83 cm one-day values were documented as float64,
but their raw run manifests are not retained in this tree. They remain
documentation-level historical results, not substitutes for the canonical
paper-evidence pipeline.

