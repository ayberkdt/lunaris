# Third-Party Notices

Lunaris is licensed under the MIT License (see [LICENSE](LICENSE)). The wheel
itself vendors **no third-party code**; the dependencies below are installed
separately by `pip`/`uv` from their own distributions under their own licenses.
This inventory exists so that organizations deploying Lunaris can review their
license obligations in one place.

Snapshot date: 2026-07-11 (versions as resolved by `uv.lock` for the `all`
extra on CPython 3.11). Regenerate with:

```bash
pip install pip-licenses
pip-licenses -f markdown --with-urls
```

## Direct runtime dependencies

| Package | License |
|---|---|
| numpy | BSD-3-Clause |
| numba | BSD-2-Clause |
| llvmlite (numba dependency) | BSD-2-Clause |
| scipy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| matplotlib | PSF-based (matplotlib license) |
| spiceypy | MIT |
| tqdm | MPL-2.0 AND MIT |
| torch (optional, `ml`/`hpc`/`all` extras) | BSD-3-Clause |
| h5py (optional) | BSD-3-Clause |
| psutil (optional) | BSD-3-Clause |
| PySide6 / PySide6_Essentials / PySide6_Addons / shiboken6 (optional, `ui`/`all` extras) | **LGPL-3.0-only** OR GPL-2.0-only OR GPL-3.0-only |
| pyqtgraph (optional) | MIT |
| PyOpenGL (optional) | BSD-3-Clause |
| QtAwesome / QtPy (optional) | MIT |
| reportlab (optional, `reports` extra) | BSD-3-Clause |

Transitive dependencies (Jinja2, MarkupSafe, pillow, sympy, networkx, fsspec,
filelock, python-dateutil, pytz, contourpy, cycler, fonttools, kiwisolver,
pyparsing, mpmath, typing_extensions, colorama, six, packaging, tzdata, …) are
BSD/MIT/Apache/PSF-family licenses; the regeneration command above lists the
complete set for the exact environment you deploy.

### LGPL notice — PySide6 (Qt for Python)

The desktop UI depends on PySide6, used under the **LGPL-3.0** option:

- Lunaris links to PySide6/Qt **dynamically** (standard Python imports of
  separately installed wheels); no Qt code is statically linked, vendored, or
  modified.
- Users can replace the PySide6/Qt libraries with their own builds simply by
  installing a different PySide6 wheel into the environment, which satisfies
  the LGPL re-linking requirement.
- PySide6/Qt source code is available from the Qt Project:
  <https://code.qt.io/cgit/pyside/pyside-setup.git/>.
- If you redistribute a bundled environment that includes PySide6 (e.g. a
  frozen installer or container image), ship this notice with it and keep the
  Qt licensing documentation reachable: <https://doc.qt.io/qt-6/licensing.html>.

Headless deployments can avoid the LGPL surface entirely by installing without
the `ui` extra (`pip install lunaris[hpc]`).

### MPL notice — tqdm

tqdm is MPL-2.0 (file-level copyleft) and is used unmodified as an installed
dependency; no obligations beyond preserving its notices arise from normal use.

## External data assets (not bundled)

Lunaris downloads mission data at the operator's request via `lunaris-data`;
nothing below ships inside the package. Recorded SHA-256 digests in
`data/data_sources.json` are verified on download.

| Asset | Source | Terms |
|---|---|---|
| SPICE kernels (`naif0012.tls`, `de440.bsp`, `de440s.bsp`, `pck00011.tpc`, `gm_de440.tpc`, `moon_pa_de440*.bpc/.tf`) | NASA/JPL NAIF | U.S. Government work — public domain; NAIF requests acknowledgment |
| GRAIL lunar gravity field (JGGRX) | NASA PDS Geosciences Node | U.S. Government work — public domain; cite the GRAIL science team |
| LOLA topography (LDEM) and albedo (LDAM) | NASA PDS (LRO/LOLA) | U.S. Government work — public domain; cite the LOLA team |
| Diviner thermal maps (DGDR) | NASA PDS (LRO/Diviner) | U.S. Government work — public domain; cite the Diviner team |
| LROC color texture (manual download) | NASA/GSFC/Arizona State University | Publicly released imagery; credit "NASA/GSFC/Arizona State University" |
| `st_lrps_cloud_suite` | Generated locally by Lunaris | Not an external asset (project output) |

NASA data carries no copyright inside the U.S. but is provided **without
warranty**; scientific attribution of the producing instrument teams is
expected in publications. If your deployment redistributes these files
internally (e.g. an offline mirror per [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)),
preserve the original file names and this attribution table.
