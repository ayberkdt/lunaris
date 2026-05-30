"""Selection-semantics and manifest-schema tests for the ``lunaris-data`` CLI.

Complements ``tests/test_data_downloader.py`` (download/verify mechanics). All
tests are network-free and use either the committed manifest or small synthetic
in-memory manifests; none require external data, CUDA, PyTorch, or SPICE kernels.
"""

from __future__ import annotations

from lunaris.cli import data as data_cli

REQUIRED_FIELDS = ("name", "group", "filename", "target_subdir", "required")


def _manifest(datasets):
    return {"schema_version": data_cli.SCHEMA_VERSION, "datasets": datasets}


def _sample():
    return _manifest([
        {"name": "req1", "group": "ephemeris", "filename": "a",
         "target_subdir": "ephemeris_models", "required": True},
        {"name": "opt1", "group": "ephemeris", "filename": "b",
         "target_subdir": "ephemeris_models", "required": False},
        {"name": "req2", "group": "gravity", "filename": "c",
         "target_subdir": "gravity_models", "required": True},
        {"name": "opt2", "group": "gravity", "filename": "d",
         "target_subdir": "gravity_models", "required": False},
    ])


# --------------------------------------------------------------------------- #
# Committed manifest schema
# --------------------------------------------------------------------------- #
def test_repo_manifest_schema_and_required_fields():
    manifest = data_cli.load_manifest(data_cli.default_manifest_path())
    assert manifest["schema_version"] == 1
    for entry in manifest["datasets"]:
        for field in REQUIRED_FIELDS:
            assert field in entry, f"{entry.get('name')} missing field {field!r}"
        assert entry["group"] in data_cli.GROUPS
        assert entry["target_subdir"] in data_cli.CANONICAL_SUBDIRS


def test_target_path_stays_under_root_and_never_in_src(tmp_path):
    manifest = data_cli.load_manifest(data_cli.default_manifest_path())
    root = tmp_path / "dataroot"
    for entry in manifest["datasets"]:
        target = data_cli.dataset_target_path(root, entry)
        assert root in target.parents
        assert "src" not in target.relative_to(root).parts


# --------------------------------------------------------------------------- #
# Download selection semantics
# --------------------------------------------------------------------------- #
def test_group_selection_defaults_to_required_only():
    names = [d["name"] for d in data_cli.select_for_download(_sample(), group="ephemeris")]
    assert names == ["req1"]


def test_group_selection_include_optional():
    names = {d["name"] for d in data_cli.select_for_download(
        _sample(), group="ephemeris", include_optional=True)}
    assert names == {"req1", "opt1"}


def test_all_selection_defaults_to_required_only():
    names = {d["name"] for d in data_cli.select_for_download(_sample(), all_groups=True)}
    assert names == {"req1", "req2"}


def test_all_selection_include_optional():
    names = {d["name"] for d in data_cli.select_for_download(
        _sample(), all_groups=True, include_optional=True)}
    assert names == {"req1", "opt1", "req2", "opt2"}


def test_name_selects_exact_optional_entry():
    names = [d["name"] for d in data_cli.select_for_download(_sample(), name="opt1")]
    assert names == ["opt1"]


# --------------------------------------------------------------------------- #
# Acceptance criteria via the CLI (dry-run on the committed manifest)
# --------------------------------------------------------------------------- #
def test_cli_group_dry_run_required_only(tmp_path, capsys):
    rc = data_cli.main(["--data-dir", str(tmp_path), "download", "--group", "ephemeris", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "naif_lsk_naif0012" in out        # required
    assert "naif_spk_de440" in out           # required
    assert "naif_spk_de440s" not in out      # optional -> excluded by default
    assert "naif_moon_fk_de440" not in out   # optional -> excluded


def test_cli_group_dry_run_include_optional(tmp_path, capsys):
    rc = data_cli.main(["--data-dir", str(tmp_path), "download", "--group", "ephemeris",
                        "--include-optional", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "naif_spk_de440s" in out
    assert "naif_moon_fk_de440" in out


def test_cli_name_dry_run_selects_optional(tmp_path, capsys):
    rc = data_cli.main(["--data-dir", str(tmp_path), "download",
                        "--name", "naif_spk_de440s", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "naif_spk_de440s" in out


# --------------------------------------------------------------------------- #
# verify_entry statuses
# --------------------------------------------------------------------------- #
def test_verify_entry_missing_vs_manual(tmp_path):
    root = tmp_path / "root"
    url_entry = {"name": "u", "filename": "x", "target_subdir": "ephemeris_models", "url": "https://x/y"}
    manual_entry = {"name": "m", "filename": "y", "target_subdir": "gravity_models", "url": None}
    assert data_cli.verify_entry(url_entry, root) == "missing"
    assert data_cli.verify_entry(manual_entry, root) == "manual_missing"


def test_verify_entry_present_without_hash(tmp_path):
    root = tmp_path / "root"
    (root / "ephemeris_models").mkdir(parents=True)
    (root / "ephemeris_models" / "p.bin").write_bytes(b"data")
    entry = {"name": "p", "filename": "p.bin", "target_subdir": "ephemeris_models",
             "url": "https://x/y", "sha256": None}
    assert data_cli.verify_entry(entry, root) == "present"
