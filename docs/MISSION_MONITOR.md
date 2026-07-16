# Mission Monitor

The Mission Monitor is Lunaris Mission Studio's observation console: a
dockable, multi-widget workspace that follows an active propagation **live**
and re-opens completed runs in **replay**. It is strictly an observation
layer — it never influences the integrator, the step size, or any scientific
result.

## Live vs Replay

| | Live | Replay |
|---|---|---|
| Source | `[TELEMETRY]` stdout lines from the propagation subprocess | `telemetry.ndjson` run artifact (or the just-finished live session in place) |
| Cursor | follows the latest sample | one shared timeline cursor for **all** widgets |
| Controls | — | play/pause, step ±1 sample, jump to start/end/event, speeds 0.25×/1×/5×/20× |
| Mode badge | `LIVE` (→ `LIVE · ENDED`) | `REPLAY` |

Both modes use the same widget infrastructure and the same bounded telemetry
store; there is no second charting stack. Playback speed semantics: **1×
replays the full run in about one minute of wall time**, independent of
mission duration; other speeds scale that rate (stated in the speed control's
tooltip — 1× is deliberately *not* "real time" for multi-day missions).

## Telemetry protocol (producer side)

Structured telemetry is versioned (`lunaris_telemetry_v1`) and typed via
`lunaris.common.telemetry_contract`:

- `[TELEMETRY_META] {json}` — once per run: requested backend facts,
  integrator, gravity model/degree, config hash, git commit, μ, cadence
  (`TelemetryProvenance`). Emitted by the CLI before propagation.
- `[TELEMETRY] {json}` — one `TelemetrySample` per telemetry cadence tick:
  SI state vector (m, m/s), radius/altitude/speed, osculating orbital
  elements, optional terrain channels, monotonic `sequence_id`.

Rules the implementation enforces:

- Emission is opt-in (`--enable-telemetry` / `--telem-cadence-s`) and
  best-effort: telemetry can never fail or slow a run beyond the cadence-gated
  single float comparison per RHS evaluation.
- Non-finite values are dropped at encode time (`allow_nan=False`); the
  consumer maps anything non-finite to "channel missing", never zero.
- **Singularity honesty**: for near-circular orbits `argp` is omitted, for
  near-equatorial orbits `raan` is omitted. The Orbital Elements widget shows
  "undefined (circular/equatorial orbit)" instead of a fake 0°.
- Unknown `schema_version` fails closed: the line is surfaced in the console
  and a warning banner appears on the Monitor page.
- Legacy bare-JSON telemetry (`{"t_s": ..., "alt_km": ...}`) is still parsed
  through a compatibility adapter, so old logs and third-party producers keep
  working (with reduced channel availability).

## Cadences (three, deliberately separate)

1. **Integrator cadence** — internal steps/RHS evaluations; never throttled or
   observed per-evaluation.
2. **Telemetry sampling cadence** — `--telem-cadence-s` (UI default derives
   from the output cadence); sample construction/serialization runs only at
   this cadence.
3. **UI paint cadence** — the monitor controller batches store updates through
   a ~60 ms QTimer (≤ ~16 Hz); widgets re-render from store snapshots, never
   per incoming line.

## Bounded store

`lunaris.ui.monitor.store.TelemetryStore` keeps per-channel ring buffers
(default 50 000 samples live; replay stores are sized to the artifact with a
200 000-sample hard cap, truncation is reported). Sequence gaps, duplicates
and out-of-order samples are counted and shown in Integrator Health.
Time-series widgets render display-resolution snapshots produced by a
**bucketed min/max envelope** downsampler (spikes survive; plain striding is
not used). Full-resolution science lives in the run artifacts, never in UI
memory.

## Widgets (first release)

| Widget | Shows | Honesty notes |
|---|---|---|
| 3D Orbit View | Moon, display-decimated trajectory trace, current-position marker (follows the replay cursor), impact/terminal markers | reuses the Orbit Setup GL scene pattern; offscreen/no-GL platforms get an explicit fallback note, never a blank scene |
| Altitude / Radius | mean-radius altitude, radius, terrain clearance; current/min/max | altitude definition (`r − R_ref`, R_ref value) visible on the widget; missing metrics are absent from the selector |
| Orbital Elements | osculating 2-body a, e, i, Ω, ω, ν | singular angles are "undefined", not 0; frame + convention footer |
| State Vector | x/y/z, vx/vy/vz, norms, sample time; inertial/body-fixed selector | body-fixed disabled with an explicit reason when the run has no `state_fixed` channel; SI source, km display |
| Integrator Health | sim/wall time, throughput, progress/ETA, sample & sequence counters, end-of-run `[DIAG]` fields (nfev, backend, stop reason) | only backend-reported fields appear |
| Event Timeline | periapsis/impact/terminal/fallback/run events, deduplicated, chronological | double-click jumps the replay timeline |
| Backend & Provenance | requested vs effective backend, gravity model/degree, ST-LRPS artifact, config hash, git commit, fallback reason | unknown facts render "—" (Unavailable); fallback is prominent |
| Batch Progress | `[BATCH_PROGRESS]` stage/samples/ETA; `[BATCH_METRICS]` impact counts with the runner's 95% CI, requested→actual backend, SH degree, device, ST-LRPS runtime kind | batch payloads are controller-scoped and never touch the live run's store/provenance |

Declared for later phases (open as honest placeholders, never fake content):
Invariant Monitor, Force Contribution, ST-LRPS Domain Status.

Every widget carries a unit/frame/source badge chip and a Live/Replay mode
badge, renders an explicit empty state ("Waiting for telemetry", "Channel
unavailable for this run/backend") instead of fake axes, and sits behind a
widget-level error boundary: one failing widget becomes a labelled
placeholder without affecting the rest of the workspace.

## Replay artifact

`--telemetry-artifact on` (UI-launched runs enable it automatically) mirrors
the emitted protocol lines into the canonical run directory as
`telemetry.ndjson` — one meta line followed by sample lines, exactly the
stdout payloads. See `docs/CONFIG_AND_ARTIFACT_CONTRACTS.md`. The Run History
indexer (`lunaris.ui.core.results_index.RunRecord.has_telemetry`) flags runs
that can be replayed. Loading happens on a worker thread in batches; the UI
thread never blocks on file IO.

A run without the artifact still opens: provenance/diagnostics widgets render
from `run_config.json`/`run_diagnostics.json` context where available, and
time-series widgets state that the run carries no telemetry samples.

## Workspace

Presets: Orbit Overview (default), Numerical Health, Force Model Monitor,
Batch/Ensemble, ST-LRPS. Widgets referenced by a preset but not implemented in
this build are skipped and listed in the toolbar notice. Widgets live in
QDockWidgets inside a nested QMainWindow: move, resize, float, tabify, close,
re-open (Add Widget menu), Reset Layout. Unknown widget ids restore as
graceful placeholders.

Multiple dashboard tabs (each an independent dock layout over the same run)
can be added with the "+" button; the last tab cannot be closed away.

## Layout persistence

The tab set, per-tab open widgets, dock geometry, active preset and the last
replay artifact persist through a versioned schema
(`lunaris_monitor_layout_v1`, `monitor_layout.json` in the app data dir,
written atomically on window close). A corrupt or foreign-schema layout file
is quarantined to `.bak` and the default Orbit Overview preset opens with a
console warning — a layout file can never block application startup.

## Performance envelope

- Telemetry disabled (default for library callers): zero overhead.
- Telemetry enabled: one float comparison per RHS evaluation; sample build +
  JSON encode only at telemetry cadence.
- UI memory is bounded by the store capacity regardless of run length; the
  contract is tested (100 000 appends into a fixed-capacity ring).
- Hidden widgets skip refresh work and catch up on show.

## Scientific interpretation caveats

- Orbital elements are **derived** (osculating, 2-body, Vallado convention)
  from the integrated state — badge says "derived".
- Altitude is relative to the mean reference radius unless the terrain
  channel is shown; the impact threshold at 0 km refers to the mean radius.
- Telemetry samples are taken at telemetry cadence, not at integrator steps;
  min/max stats reflect the sampled series, not the continuous trajectory.
- "Effective backend" is measured by the engine and arrives with the
  end-of-run diagnostics; before that, only requested facts are shown.
