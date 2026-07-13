"""loop.py — geometry of the nabta 1.0 km figure-eight loop.

Parametrizes the walking/jogging loop of the nabta wellness-park entry as a
closed 2D polyline inside a 150 m x 100 m site (a stylized stand-in for the
Al Safa Park 2 parcel). The base curve is a Gerono lemniscate (figure-eight);
a smooth meander is superimposed along the local normal and its amplitude is
solved numerically so the closed loop measures exactly 1.0 km. The polyline
is then resampled to uniform arc-length spacing so every sample represents
the same length of path.

Coordinate frame: x = east (m), y = north (m), origin at site centre.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Site + loop constants
# ---------------------------------------------------------------------------
SITE_W = 150.0          # site width  (m, east-west)
SITE_H = 100.0          # site height (m, north-south)
LOOP_LENGTH = 1000.0    # target closed-loop length (m)

_A = 64.0               # lemniscate half-width  (m)
_B = 37.0               # lemniscate half-height (m)
_MEANDER_LOBES = 20     # integer -> meander closes with the loop
_N_DENSE = 20000        # dense parameter samples used for arc-length solve


def _base_curve(t: np.ndarray) -> np.ndarray:
    """Gerono figure-eight: x = A sin t, y = B sin t cos t (scaled)."""
    x = _A * np.sin(t)
    y = 2.0 * _B * np.sin(t) * np.cos(t)  # = B sin(2t)
    return np.stack([x, y], axis=-1)


def _unit_normals(pts: np.ndarray) -> np.ndarray:
    """Unit normals of a closed polyline via central differences."""
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    return np.stack([-tang[:, 1], tang[:, 0]], axis=1)


def _polyline_length(pts: np.ndarray) -> float:
    seg = np.roll(pts, -1, axis=0) - pts
    return float(np.linalg.norm(seg, axis=1).sum())


def _meandered(amp: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, _N_DENSE, endpoint=False)
    base = _base_curve(t)
    normals = _unit_normals(base)
    offset = amp * np.sin(_MEANDER_LOBES * t)
    return base + offset[:, None] * normals


def _solve_meander_amplitude(target: float = LOOP_LENGTH) -> float:
    """Bisection on meander amplitude so the closed loop length == target."""
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _polyline_length(_meandered(mid)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_loop(n_points: int = 1000) -> dict:
    """Build the 1.0 km loop and resample it at uniform arc length.

    Returns a dict with:
      points   (n,2)  loop vertices, uniform arc-length spacing, closed (wraps)
      tangents (n,2)  unit tangents at each vertex
      normals  (n,2)  unit normals (tangent rotated +90 deg)
      s        (n,)   arc length of each vertex (m), s[0] = 0
      length   float  total loop length (m)
    """
    amp = _solve_meander_amplitude()
    dense = _meandered(amp)

    seg = np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    total = s_dense[-1]

    s_target = np.linspace(0.0, total, n_points, endpoint=False)
    closed = np.vstack([dense, dense[:1]])
    x = np.interp(s_target, s_dense, closed[:, 0])
    y = np.interp(s_target, s_dense, closed[:, 1])
    pts = np.stack([x, y], axis=1)

    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    assert np.all(np.abs(pts[:, 0]) <= SITE_W / 2), "loop exceeds site width"
    assert np.all(np.abs(pts[:, 1]) <= SITE_H / 2), "loop exceeds site height"

    return {
        "points": pts,
        "tangents": tang,
        "normals": norm,
        "s": s_target,
        "length": total,
    }


def garden_rooms(loop: dict, n_rooms: int = 10, offset: float = 13.0) -> np.ndarray:
    """Nominal centres of the ten garden rooms of the entry.

    Rooms sit at equal arc-length fractions around the loop, pushed off the
    path along the local normal (alternating sides), clamped to the site.
    """
    n = len(loop["points"])
    idx = (np.linspace(0.03, 1.03, n_rooms, endpoint=False) % 1.0 * n).astype(int)
    sides = np.where(np.arange(n_rooms) % 2 == 0, 1.0, -1.0)
    rooms = loop["points"][idx] + sides[:, None] * offset * loop["normals"][idx]
    rooms[:, 0] = np.clip(rooms[:, 0], -SITE_W / 2 + 4, SITE_W / 2 - 4)
    rooms[:, 1] = np.clip(rooms[:, 1], -SITE_H / 2 + 4, SITE_H / 2 - 4)
    return rooms


if __name__ == "__main__":
    loop = generate_loop()
    print(f"loop length : {loop['length']:.1f} m ({len(loop['points'])} samples)")
    print(f"x extent    : [{loop['points'][:,0].min():.1f}, {loop['points'][:,0].max():.1f}] m")
    print(f"y extent    : [{loop['points'][:,1].min():.1f}, {loop['points'][:,1].max():.1f}] m")
