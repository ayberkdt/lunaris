"""ST-LRPS Paper Evidence Pipeline — reproducible training/evidence infrastructure.

Part 1 provides:
* canonical paper-safe training configs (``configs/st_lrps/paper/``),
* a validator that rejects unsafe paper configs before training,
* an evidence manifest (hashes + environment + provenance),
* a staged runner whose ``train`` stage wires into the existing ST-LRPS trainer.

Field validation, orbit benchmarks, worst-case analysis, ablation, and figures
are intentionally NOT part of this module — they belong to Parts 2/3.
"""

from __future__ import annotations

from lunaris.surrogate.st_lrps.paper_evidence.config_validation import (
    PaperConfigError,
    load_paper_training_config,
    validate_st_lrps_paper_training_config,
)
from lunaris.surrogate.st_lrps.paper_evidence.evidence_manifest import (
    artifact_record,
    collect_environment,
    compute_file_sha256,
    compute_json_hash,
    write_evidence_manifest,
)
from lunaris.surrogate.st_lrps.paper_evidence.multi_seed import (
    aggregate_multi_seed,
    collect_seed_entry,
    write_multi_seed_outputs,
)
from lunaris.surrogate.st_lrps.paper_evidence.paper_tables import (
    csv_to_markdown_table,
    generate_paper_figures,
    generate_paper_tables,
)
from lunaris.surrogate.st_lrps.paper_evidence.training_argv import build_training_argv
from lunaris.surrogate.st_lrps.paper_evidence.worst_case import (
    analyze_worst_cases,
    write_worst_case_outputs,
)

__all__ = [
    "PaperConfigError",
    "aggregate_multi_seed",
    "analyze_worst_cases",
    "artifact_record",
    "build_training_argv",
    "collect_environment",
    "collect_seed_entry",
    "compute_file_sha256",
    "compute_json_hash",
    "csv_to_markdown_table",
    "generate_paper_figures",
    "generate_paper_tables",
    "load_paper_training_config",
    "validate_st_lrps_paper_training_config",
    "write_evidence_manifest",
    "write_multi_seed_outputs",
    "write_worst_case_outputs",
]
