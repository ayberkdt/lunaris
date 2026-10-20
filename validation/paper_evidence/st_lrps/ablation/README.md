# ablation/

ST-LRPS ablation suite evidence (A0..A6 ladder from raw xyz to the shipped
model; optional A8/A9 encodings) written by `--stage ablation`:
`st_lrps_ablation_summary.csv` / `.md` (one row per arm, mean ± std over seeds)
plus per-run dirs `<arm>/seed<N>/`. Ablation is **secondary** and does not block
final-candidate evidence. Default is dry-run (commands + manifest); pass
`--execute-ablation` to launch training.
