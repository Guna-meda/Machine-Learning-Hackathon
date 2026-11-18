"""Read one trip folder (.track, .nodes, .arcs, .route) into a Trip object.

Every trip has its own road network, so node ids and arc ids are only valid
inside the trip they come from.  All geometry is converted to metres with a
local equirectangular projection (see project_lonlat).
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame            # columns lon, lat, t
    nodes: np.ndarray              # (n_nodes, 2) lon, lat
    arcs: np.ndarray               # (n_arcs, 2) from_node, to_node   (row index = arc_id)
    route: np.ndarray | None       # ordered arc ids, None if file missing
    lon0: float = 0.0
    lat0: float = 0.0
    nodes_xy: np.ndarray = field(default=None)   # metres
    track_xy: np.ndarray = field(default=None)   # metres


def project_lonlat(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres around (lon0, lat0).

    x = R * (lon - lon0) * cos(lat0),  y = R * (lat - lat0).  Trips are a few
    km wide, so the distortion is negligible compared with GPS noise.
    """
    x = np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def _read_table(path: Path, ncols: int, dtype):
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr


def load_trip(trip_dir, with_route: bool = True) -> Trip:
    """Load a trip.  with_route=False never opens the .route file (used by the API predictor)."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = pd.DataFrame(_read_table(trip_dir / f"{tid}.track", 3, float), columns=["lon", "lat", "t"])
    nodes = _read_table(trip_dir / f"{tid}.nodes", 2, float)
    arcs = _read_table(trip_dir / f"{tid}.arcs", 2, np.int64)
    route = None
    if with_route and (trip_dir / f"{tid}.route").exists():
        route = _read_table(trip_dir / f"{tid}.route", 1, np.int64).ravel()
    trip = Trip(tid, track, nodes, arcs, route)
    set_projection(trip)
    return trip


def set_projection(trip: Trip):
    valid = trip.track.dropna()
    trip.lon0 = float(valid["lon"].mean()) if len(valid) else 0.0
    trip.lat0 = float(valid["lat"].mean()) if len(valid) else 0.0
    trip.nodes_xy = project_lonlat(trip.nodes[:, 0], trip.nodes[:, 1], trip.lon0, trip.lat0)
    trip.track_xy = project_lonlat(trip.track["lon"].values, trip.track["lat"].values, trip.lon0, trip.lat0)


def list_trip_dirs(split_dir) -> list[Path]:
    """All sub-folders of a split folder that look like trips (never hardcode ids)."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if p.is_dir())
