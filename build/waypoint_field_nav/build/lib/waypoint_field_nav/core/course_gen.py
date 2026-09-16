"""Random course generator (pure Python).

Produces a dict describing a test course::

    {
      "bounds": [min_x, min_y, max_x, max_y],
      "start":  [0.0, 0.0],
      "obstacles": [ {type: circle, x, y, radius}, {type: box, x, y, hx, hy}, ... ],
      "waypoints":  [ [x, y], [x, y], ... ],
    }

Rejection sampling keeps: the start clear, obstacles spaced apart, and every
waypoint outside every obstacle (plus a margin). Shared by the SDF generator
and the headless test harness so both exercise identical courses.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from . import geometry


def load_course(path: str) -> Dict:
    """Load a course manifest (.yaml) written by generate_course.

    Returns the same dict shape sample_course produces: keys ``bounds``,
    ``start``, ``obstacles``, ``waypoints``. Imported lazily so the core
    package has no hard dependency on PyYAML unless this helper is used.
    """
    import yaml
    with open(path, "r") as f:
        course = yaml.safe_load(f)
    for key in ("bounds", "obstacles", "waypoints"):
        if key not in course:
            raise ValueError(f"course file {path!r} missing '{key}'")
    course.setdefault("start", [0.0, 0.0])
    return course


def _clear_of_obstacles(x: float, y: float, obstacles, margin: float) -> bool:
    return geometry.min_clearance(x, y, obstacles) > margin


def sample_course(
    seed: int = 0,
    bounds: Tuple[float, float, float, float] = (-8.0, -8.0, 8.0, 8.0),
    n_obstacles: int = 10,
    n_waypoints: int = 5,
    circle_radius: Tuple[float, float] = (0.4, 1.0),
    box_half: Tuple[float, float] = (0.3, 1.6),
    box_fraction: float = 0.35,
    start: Tuple[float, float] = (0.0, 0.0),
    start_clear: float = 1.5,
    obstacle_spacing: float = 1.2,
    waypoint_margin: float = 0.8,
    waypoint_spacing: float = 2.0,
    edge_pad: float = 1.0,
    max_tries: int = 4000,
) -> Dict:
    rng = random.Random(seed)
    min_x, min_y, max_x, max_y = bounds
    ax0, ay0, ax1, ay1 = (min_x + edge_pad, min_y + edge_pad,
                          max_x - edge_pad, max_y - edge_pad)

    obstacles: List[Dict] = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < max_tries:
        tries += 1
        x = rng.uniform(ax0, ax1)
        y = rng.uniform(ay0, ay1)
        if math.hypot(x - start[0], y - start[1]) < start_clear:
            continue
        if rng.random() < box_fraction:
            hx = rng.uniform(*box_half) if rng.random() < 0.5 else rng.uniform(0.3, 0.6)
            hy = rng.uniform(0.3, 0.6) if hx > 0.7 else rng.uniform(*box_half)
            cand = {"type": "box", "x": x, "y": y, "hx": round(hx, 3), "hy": round(hy, 3)}
        else:
            r = rng.uniform(*circle_radius)
            cand = {"type": "circle", "x": x, "y": y, "radius": round(r, 3)}
        # keep obstacles from overlapping / crowding
        ok = True
        for o in obstacles:
            if math.hypot(cand["x"] - o["x"], cand["y"] - o["y"]) < obstacle_spacing + \
                    _reach(cand) + _reach(o):
                ok = False
                break
        if ok:
            cand["x"], cand["y"] = round(x, 3), round(y, 3)
            obstacles.append(cand)

    waypoints: List[List[float]] = []
    tries = 0
    while len(waypoints) < n_waypoints and tries < max_tries:
        tries += 1
        x = rng.uniform(ax0, ax1)
        y = rng.uniform(ay0, ay1)
        if not _clear_of_obstacles(x, y, obstacles, waypoint_margin):
            continue
        if math.hypot(x - start[0], y - start[1]) < waypoint_spacing:
            continue
        if any(math.hypot(x - wx, y - wy) < waypoint_spacing for wx, wy in waypoints):
            continue
        waypoints.append([round(x, 3), round(y, 3)])

    return {
        "bounds": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "start": [float(start[0]), float(start[1])],
        "obstacles": obstacles,
        "waypoints": waypoints,
    }


def _reach(obs: Dict) -> float:
    return obs["radius"] if obs["type"] == "circle" else math.hypot(obs["hx"], obs["hy"])
