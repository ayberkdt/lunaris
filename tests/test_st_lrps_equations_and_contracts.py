"""ST-LRPS network equations, artifact format, naming and critical defaults.

Every check runs on a production-architecture artifact (two-band SIREN with
residual blocks, physical radial-decay input encoding, rho**26 envelope) written
by the same functions as a training run, and loaded through the runtime the
propagators use. Each equation test carries a negative control: the same check
applied to a deliberately broken variant must fail, so a test that cannot detect
the error it is named after fails too.

Equations (paper Eq. 15 and the frame definition):
    dU(x)  = u_scale * f(x / x_scale) + u_mean,  f = backbone * rho**p
    rho    = R_ref / |x|
    da(x)  = a_sign * grad dU(x)                 (conservative, curl-free)
    a_i(r) = C^T a_f(C r),  C = q2m(q_i2f)        (scalar-first, active)
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from st_lrps_production_artifact import (  # noqa: E402
    ENVELOPE_POWER,
    R_REF_M,
    X_SCALE_M,
    make_production_like_run,
)

# Points inside the 100-1000 km shell, Moon-fixed, metres.
_PTS = np.array([
    [R_REF_M + 150e3, 0.0, 0.0],
    [0.0, -(R_REF_M + 420e3), 1.0e5],
    [(R_REF_M + 800e3) * 0.48, (R_REF_M + 800e3) * 0.6, (R_REF_M + 800e3) * 0.64],
    [3.0e4, 2.0e4, R_REF_M + 250e3],  # near the pole
])


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return make_production_like_run(tmp_path_factory.mktemp("prod") / "run")


@pytest.fixture(scope="module")
def runtime(run_dir):
    from lunaris.surrogate.runtime import SurrogateGravityModel

    with pytest.warns(RuntimeWarning):
        model = SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")
    model._force_runtime.to_device("cpu", torch.float64)
    return model


def _independent_residual(run_dir: Path, x: np.ndarray, *, u_scale=None, a_sign=None):
    """dU and da from config.json + scaler.json + checkpoint, without the runtime."""
    from lunaris.surrogate.st_lrps.networks.models import build_model_from_config

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    scaler = json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))
    state = torch.load(run_dir / "checkpoints" / "ckpt_best.pt", map_location="cpu", weights_only=False)
    net = build_model_from_config(cfg, dtype=torch.float64)
    net.load_state_dict(state["model_state_dict"])
    net.eval()
    xs = float(scaler["x"]["scale"])
    us = float(scaler["u"]["scale"]) if u_scale is None else u_scale
    um = float(np.asarray(scaler["u"]["mean"]).reshape(-1)[0])
    sign = float(cfg["resolved_a_sign"]) if a_sign is None else a_sign
    xt = torch.as_tensor(x, dtype=torch.float64).requires_grad_(True)
    du = us * net(xt / xs) + um
    (grad,) = torch.autograd.grad(du.sum(), xt)
    return du.detach().numpy().reshape(-1), sign * grad.numpy()


# --------------------------------------------------------------------------- equations


def test_runtime_matches_an_independent_evaluation_of_the_equations(runtime, run_dir) -> None:
    fr = runtime._force_runtime
    du_ref, da_ref = _independent_residual(run_dir, _PTS)
    np.testing.assert_allclose(fr.predict_residual_potential(_PTS), du_ref, rtol=1e-10, atol=0)
    np.testing.assert_allclose(fr.predict_residual_accel(_PTS), da_ref, rtol=1e-9, atol=0)
    # negative controls: a wrong sign or a wrong potential scale is detected
    _, da_bad = _independent_residual(run_dir, _PTS, a_sign=-1.0)
    assert not np.allclose(fr.predict_residual_accel(_PTS), da_bad, rtol=1e-3)
    du_bad, _ = _independent_residual(run_dir, _PTS, u_scale=8.8 * 1.001)
    assert not np.allclose(fr.predict_residual_potential(_PTS), du_bad, rtol=1e-6)


def _fd_gradient(fn, x: np.ndarray, h: float) -> np.ndarray:
    out = np.empty_like(x)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        out[:, k] = (np.asarray(fn(x + e)).reshape(-1) - np.asarray(fn(x - e)).reshape(-1)) / (2 * h)
    return out


def test_residual_acceleration_is_the_gradient_of_the_residual_potential(runtime) -> None:
    fr = runtime._force_runtime
    a = fr.predict_residual_accel(_PTS)
    a_fd = _fd_gradient(fr.predict_residual_potential, _PTS, h=20.0)
    rel = np.linalg.norm(a - a_fd, axis=1) / np.linalg.norm(a, axis=1)
    assert rel.max() < 1e-5, rel
    # negative control: the opposite sign is a factor-2 mismatch
    rel_bad = np.linalg.norm(-a - a_fd, axis=1) / np.linalg.norm(a, axis=1)
    assert rel_bad.min() > 1.0


def test_total_field_is_baseline_plus_residual_and_conservative(runtime) -> None:
    u, a = runtime.predict_potential_and_acceleration_fixed(_PTS)
    fr = runtime._force_runtime
    r = np.linalg.norm(_PTS, axis=1, keepdims=True)
    a_pm = -float(runtime.GM_m3s2) * _PTS / r**3
    np.testing.assert_allclose(a - a_pm, fr.predict_residual_accel(_PTS), rtol=0, atol=1e-12)
    a_fd = _fd_gradient(lambda x: runtime.predict_potential_and_acceleration_fixed(x)[0], _PTS, h=5.0)
    assert (np.linalg.norm(a - a_fd, axis=1) / np.linalg.norm(a, axis=1)).max() < 1e-7


def test_residual_field_is_curl_free(runtime) -> None:
    fr = runtime._force_runtime
    h = 20.0
    for p in _PTS:
        jac = np.empty((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            jac[:, j] = (fr.predict_residual_accel(p + e) - fr.predict_residual_accel(p - e)).reshape(-1) / (2 * h)
        scale = np.abs(jac).max()
        assert np.abs(jac - jac.T).max() < 1e-4 * scale, jac


def test_output_envelope_scales_as_rho_to_the_power_p() -> None:
    """With a constant backbone the network output is exactly c * (R_ref/r)**p."""
    from lunaris.surrogate.st_lrps.networks.models import PhysicsNet

    class _Const(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1), 3.0, dtype=x.dtype)

    def net(p: float) -> PhysicsNet:
        return PhysicsNet(_Const(), envelope_power=p, x_scale_m=X_SCALE_M, r_ref_m=R_REF_M)

    radii = np.array([R_REF_M + 100e3, R_REF_M + 500e3, R_REF_M + 1000e3])
    x = torch.as_tensor(np.c_[radii, np.zeros(3), np.zeros(3)] / X_SCALE_M, dtype=torch.float64)
    expected = 3.0 * (R_REF_M / radii) ** ENVELOPE_POWER
    np.testing.assert_allclose(net(ENVELOPE_POWER)(x).numpy().reshape(-1), expected, rtol=1e-12)
    assert not np.allclose(net(ENVELOPE_POWER - 1)(x).numpy().reshape(-1), expected, rtol=1e-3)
    np.testing.assert_allclose(net(0.0)(x).numpy().reshape(-1), 3.0)


def _q2m(q: np.ndarray) -> np.ndarray:
    """SPICE q2m: scalar-first unit quaternion -> rotation matrix (inertial -> fixed)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def test_inertial_evaluation_rotates_into_the_moon_fixed_frame_and_back(runtime) -> None:
    fr = runtime._force_runtime
    q = np.array([0.82, 0.31, -0.27, 0.40])
    q /= np.linalg.norm(q)
    c = _q2m(q)
    r_inertial = _PTS @ c  # rows: C^T r_fixed
    a_i = fr.predict_residual_accel_inertial(r_inertial, q)
    expected = (c.T @ fr.predict_residual_accel(_PTS).T).T
    np.testing.assert_allclose(a_i, expected, rtol=1e-9, atol=0)
    # negative control: the transposed rotation is a different field
    wrong = (c @ fr.predict_residual_accel((c @ c @ _PTS.T).T).T).T
    assert not np.allclose(a_i, wrong, rtol=1e-3)


def test_frozen_search_quaternion_convention_matches_spice_q2m() -> None:
    from lunaris.analysis.frozen.search import quaternion_i2f_to_matrix

    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        m = quaternion_i2f_to_matrix(q)
        np.testing.assert_allclose(m, _q2m(q), atol=1e-14)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-14)
        assert math.isclose(np.linalg.det(m), 1.0, abs_tol=1e-13)


def test_runtime_returns_float64_arrays_with_documented_shapes(runtime) -> None:
    fr = runtime._force_runtime
    one = fr.predict_residual_accel(_PTS[0])
    many = fr.predict_residual_accel(_PTS)
    assert one.dtype == np.float64 and many.dtype == np.float64
    assert one.shape in {(3,), (1, 3)} and many.shape == (len(_PTS), 3)
    assert isinstance(fr.predict_residual_potential(_PTS[0]), float)
    assert fr.predict_residual_potential(_PTS).shape == (len(_PTS),)
    a_total = runtime.acceleration_fixed(_PTS[0])
    assert a_total.shape == (3,) and a_total.dtype == np.float64


# --------------------------------------------------------------------------- artifact format


def test_artifact_files_carry_the_fields_the_runtime_reads(run_dir) -> None:
    from lunaris.surrogate.runtime.gravity_provider import _PHYSICS_CONFIG_KEYS
    from lunaris.surrogate.st_lrps.networks.models import ARCH_SIGNATURE_FIELDS

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    scaler = json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))
    ckpt = torch.load(run_dir / "checkpoints" / "ckpt_best.pt", map_location="cpu", weights_only=False)
    missing_arch = [k for k in ARCH_SIGNATURE_FIELDS if k not in cfg and k != "output_envelope_power"]
    assert missing_arch == [], missing_arch
    for key in ("resolved_a_sign", "resolved_mu_si", "resolved_r_ref_m", "x_scale_m", "w0_bands",
                "model_builder_version", "architecture_signature", "best_epoch"):
        assert key in cfg, key
    assert {"x", "u", "a"} <= set(scaler) and all({"mean", "scale"} <= set(scaler[k]) for k in "xua")
    assert {"model_state_dict", "config", "kind", "epoch_display"} <= set(ckpt)
    for key in _PHYSICS_CONFIG_KEYS & set(cfg):
        assert ckpt["config"].get(key) == cfg[key], key
    assert ckpt["config"]["best_epoch"] == cfg["best_epoch"]


def _copy_run(src: Path, dst: Path) -> Path:
    import shutil

    shutil.copytree(src, dst)
    return dst


@pytest.mark.parametrize(("key", "value"), [
    ("resolved_a_sign", -1.0),
    ("output_envelope_power", 25.0),
    ("hidden", 32),
    ("n_bands", 3),
    ("w0_bands", [10.0, 50.0]),
])
def test_an_edited_config_json_never_loads_silently(run_dir, tmp_path, key, value) -> None:
    """config.json and the checkpoint must agree on everything that changes the field."""
    from lunaris.surrogate.runtime import SurrogateGravityModel

    run = _copy_run(run_dir, tmp_path / "edited")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg[key] != value
    cfg[key] = value
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises((ValueError, RuntimeError)):
            SurrogateGravityModel.from_model_dir(run, device_preference="cpu")


def test_an_edited_scaler_never_loads_silently(run_dir, tmp_path) -> None:
    from lunaris.surrogate.runtime import SurrogateGravityModel

    run = _copy_run(run_dir, tmp_path / "scaler")
    scaler = json.loads((run / "scaler.json").read_text(encoding="utf-8"))
    scaler["u"]["scale"] = float(scaler["u"]["scale"]) * 1.01
    (run / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises((ValueError, RuntimeError)):
            SurrogateGravityModel.from_model_dir(run, device_preference="cpu")


# --------------------------------------------------------------------------- naming


def _trainer_parser_actions(monkeypatch, tmp_path) -> list:
    """The trainer's own argparse actions, captured while it parses a minimal argv."""
    import argparse
    import sys

    from lunaris.surrogate.st_lrps.training.config import parse_args

    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def spy(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    monkeypatch.setattr(sys, "argv", ["lunaris-train", "--train-data", str(tmp_path / "t.h5"),
                                      "--val-data", str(tmp_path / "v.h5"), "--out", str(tmp_path / "r")])
    parse_args()
    return [a for a in captured[0]._actions if a.dest not in {"help"}]


def test_every_trainer_flag_is_used(monkeypatch, tmp_path) -> None:
    """A flag whose destination is neither a TrainConfig field nor read while the
    config is built is accepted and silently dropped."""
    import dataclasses
    import re

    from lunaris.surrogate.st_lrps.training import config as tc

    fields = {f.name for f in dataclasses.fields(tc.TrainConfig)}
    src = Path(tc.__file__).read_text(encoding="utf-8")
    read = set(re.findall(r"\ba\.([a-z0-9_]+)", src)) | set(re.findall(r'getattr\(a, "([a-z0-9_]+)"', src))
    dests = {a.dest for a in _trainer_parser_actions(monkeypatch, tmp_path)}
    dropped = sorted(d for d in dests if d not in fields and d not in read)
    assert dropped == [], f"trainer flags that are parsed and never used: {dropped}"
    # negative control: a destination nobody reads is reported
    assert "zz_unused_flag" not in fields | read


def test_model_presets_agree_between_trainer_and_builder() -> None:
    from lunaris.surrogate.st_lrps.networks.models import _encoding_flags_from_preset
    from lunaris.surrogate.st_lrps.training.config import MODEL_PRESETS

    for preset in MODEL_PRESETS:
        _encoding_flags_from_preset({"model_preset": preset})  # must not raise
    with pytest.raises(ValueError):
        _encoding_flags_from_preset({"model_preset": "recommended_physical_radial_decay_v2"})


def _scenario_rows():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "hpc"))
    import run_training_scenario as launcher

    root = Path(__file__).resolve().parents[1] / "hpc" / "scenarios"
    return [(p.name, s) for p in sorted(root.glob("*.jsonl")) for s in launcher.load_scenarios(p)]


def test_hpc_scenario_names_are_unique_and_match_their_seed() -> None:
    """A run named ..._Seed43 that trains with --seed 42 mislabels a replicate."""
    import re

    rows = _scenario_rows()
    names = [s["name"] for _, s in rows]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert dupes == [], dupes
    for file, s in rows:
        m = re.search(r"Seed(\d+)$", s["name"])
        flags = [str(t) for t in s.get("flags", [])]
        if m and "--seed" in flags:
            assert flags[flags.index("--seed") + 1] == m.group(1), (file, s["name"])


# --------------------------------------------------------------------------- critical defaults


def test_critical_defaults_are_pinned() -> None:
    """Defaults that silently change a result. Changing one must be deliberate:
    update this test and say why in the commit."""
    import inspect

    from lunaris.analysis.frozen.search_backends import ClassicalSHValidationPropagator
    from lunaris.cli import frozen_search
    from lunaris.common.lunar_data import LUNAR_GM_REL_TOL
    from lunaris.surrogate.runtime.gravity_provider import SurrogateGravityModel
    from lunaris.surrogate.st_lrps.runtime.canonical_runtime import enforce_altitude_envelope_torch

    fz = frozen_search.build_parser().parse_args(["--out", "x"])
    assert fz.screening_torch_dtype == "float64"       # float32 also integrates the state in float32
    assert fz.st_lrps_allow_extrapolation is False     # strict envelope unless asked
    val = inspect.signature(ClassicalSHValidationPropagator.__init__).parameters
    assert val["rtol"].default <= 1e-10 and val["atol"].default <= 1e-9
    # Those tolerances converge over 60 days only with the Nyquist step cap
    # (pre-run audit 2026-10-28: 0.2-0.9 mm with it, 7-500 m without).
    from lunaris.common.type_defs import PropagatorConfig

    assert PropagatorConfig().use_nyquist_max_step is True
    assert inspect.signature(SurrogateGravityModel.from_model_dir).parameters["strict_domain"].default is False
    assert inspect.signature(enforce_altitude_envelope_torch).parameters["tol_km"].default == 1.0
    assert LUNAR_GM_REL_TOL == 5.0e-4
