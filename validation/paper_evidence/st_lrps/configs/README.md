# configs/

The **canonical** ST-LRPS paper training configs live in the repository configs
tree, not here:

    configs/st_lrps/paper/train_full_seed42.json
    configs/st_lrps/paper/train_full_seed123.json
    configs/st_lrps/paper/train_full_seed2026.json

Point the runner at those paths (or copies you place here) with
`--config`. Keeping the canonical configs under `configs/` keeps them versioned
alongside the rest of the project configuration.
