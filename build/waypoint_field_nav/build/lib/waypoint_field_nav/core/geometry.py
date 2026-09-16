"""Pure-Python geometry + obstacle helpers.

This module has NO ROS / Gazebo dependency on purpose, so the whole navigation
"brain" can be unit-tested in a headless simulator (see test/headless_sim.py).

Obstacles are plain dicts so they serialize straight to/from the course YAML:

    {"type": "circle", "x": 3.0, "y": 1.5, "radius": 0.6}
    {"type": "box",    "x": -2.0, "y": 4.0, "hx": 1.5, "hy": 0.3}   # axis-aligned, half-extents

Everything is 2-D (we navigate on the ground plane).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def closest_point(px: float, py: float, obs: Dict) -> Tuple[float, float]:
    """Closest point on the *surface* of an obstacle to (px, py)."""
    if obs["type"] == "circle":
        dx, dy = px - obs["x"], py - obs["y"]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            # Degenerate: sitting on the centre. Push +x arbitrarily.
            return obs["x"] + obs["radius"], obs["y"]
        return obs["x"] + obs["radius"] * dx / d, obs["y"] + obs["radius"] * dy / d
    # axis-aligned box
    minx, maxx = obs["x"] - obs["hx"], obs["x"] + obs["hx"]
    miny, maxy = obs["y"] - obs["hy"], obs["y"] + obs["hy"]
    if minx <= px <= maxx and miny <= py <= maxy:
        # Point is *inside* the box (possible once obstacles are inflated by the
        # car radius): snap to the nearest face so the repulsion still points
        # sensibly outward instead of hitting the degenerate zero-vector case.
        dl, dr = px - minx, maxx - px
        db, dt = py - miny, maxy - py
        m = min(dl, dr, db, dt)
        if m == dl:
            return minx, py
        if m == dr:
            return maxx, py
        if m == db:
            return px, miny
        return px, maxy
    cx = min(max(px, minx), maxx)
    cy = min(max(py, miny), maxy)
    return cx, cy


def inflate_obstacles(obstacles: Sequence[Dict], amount: float) -> List[Dict]:
    """Grow every obstacle outward by ``amount`` in all directions.

    This is the configuration-space ("Minkowski") trick: after inflating the
    obstacles by the robot's radius, the robot can be treated as a single point,
    and any point-vs-obstacle test (repulsion, look-ahead, clearance, grid
    occupancy) automatically accounts for the whole vehicle body. Circles grow
    their radius; boxes grow their half-extents (a slightly conservative,
    square-cornered over-approximation of the true rounded C-space obstacle).
    """
    out: List[Dict] = []
    for o in obstacles:
        g = dict(o)
        if o["type"] == "circle":
            g["radius"] = o["radius"] + amount
        else:  # box
            g["hx"] = o["hx"] + amount
            g["hy"] = o["hy"] + amount
        out.append(g)
    return out


def signed_distance(px: float, py: float, obs: Dict) -> float:
    """Distance from (px, py) to the obstacle surface.

    Positive outside, ~0 on the surface, negative if the point is inside.
    """
    if obs["type"] == "circle":
        return math.hypot(px - obs["x"], py - obs["y"]) - obs["radius"]
    # box: distance to an axis-aligned rectangle
    dx = abs(px - obs["x"]) - obs["hx"]
    dy = abs(py - obs["y"]) - obs["hy"]
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def min_clearance(px: float, py: float, obstacles: Sequence[Dict]) -> float:
    """Smallest signed distance to any obstacle (inf if there are none)."""
    if not obstacles:
        return math.inf
    return min(signed_distance(px, py, o) for o in obstacles)


def point_blocked(px: float, py: float, obstacles: Sequence[Dict], inflation: float) -> bool:
    """True if the point is inside any obstacle grown by ``inflation``."""
    return min_clearance(px, py, obstacles) < inflation


def segment_blocked(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    obstacles: Sequence[Dict],
    inflation: float,
    step: float = 0.1,
) -> bool:
    """Sample a segment and report whether any sample is blocked.

    Used by the look-ahead check. Sampling (rather than analytic segment/shape
    intersection) keeps a single, shared definition of "blocked" for circles
    and boxes alike, and is plenty accurate at a 0.1 m step.
    """
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(length / step))
    for i in range(n + 1):
        t = i / n
        if point_blocked(x0 + t * (x1 - x0), y0 + t * (y1 - y0), obstacles, inflation):
            return True
    return False


def obstacle_bounds(obs: Dict) -> Tuple[float, float, float, float]:
    """Axis-aligned (min_x, min_y, max_x, max_y) footprint of an obstacle."""
    if obs["type"] == "circle":
        r = obs["radius"]
        return obs["x"] - r, obs["y"] - r, obs["x"] + r, obs["y"] + r
    return obs["x"] - obs["hx"], obs["y"] - obs["hy"], obs["x"] + obs["hx"], obs["y"] + obs["hy"]
