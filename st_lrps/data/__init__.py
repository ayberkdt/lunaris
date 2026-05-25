# st_lrps/data/__init__.py
"""ST-LRPS data subpackage: dataset contracts and spatial point-cloud generation.

Import-safe: importing this package must not load datasets, scan files, or pull
heavy dependencies. Import the specific submodule you need
(e.g. ``st_lrps.data.datasets``, ``st_lrps.data.spatial_cloud_generator``).
"""

from __future__ import annotations

__all__: list[str] = []
