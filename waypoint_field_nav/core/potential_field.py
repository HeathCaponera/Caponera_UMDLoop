"""Artificial Potential Field (APF) — step 2 of the plan.

    total = attractive(goal) + sum( repulsive(obstacle_i) )

* Attractive term points at the goal (unit vector, optionally scaled up when
  far away so the robot commits to travel, then eases off near the goal).
* Each repulsive term points *away* from the nearest surface of an obstacle and
  is weighted by 1 / d**2 inside an "influence radius" d0, exactly as in the
  plan. Outside d0 an obstacle exerts no force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from . import geometry


@dataclass
class FieldParams:
    k_att: float = 1.0        # attractive gain
    k_rep: float = 0.9        # repulsive gain
    influence: float = 1.6    # d0: obstacles beyond this exert no force (m)
    att_ramp: float = 2.0     # attractive magnitude saturates at this distance (m)
    eps: float = 0.08         # clamp on surface distance to avoid blow-ups (m)
    rep_cap: float = 6.0      # per-obstacle repulsion magnitude cap


def attractive(pos: np.ndarray, goal: np.ndarray, p: FieldParams) -> np.ndarray:
    """Unit-ish vector toward the goal, scaled by k_att and gently ramped."""
    d = goal - pos
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return np.zeros(2)
    direction = d / dist
    scale = min(dist / p.att_ramp, 1.0)   # ease off within att_ramp metres
    return p.k_att * scale * direction


def repulsive(pos: np.ndarray, obstacles: Sequence[Dict], p: FieldParams) -> np.ndarray:
    """Sum of 1/d**2 pushes away from every nearby obstacle surface."""
    force = np.zeros(2)
    px, py = float(pos[0]), float(pos[1])
    for obs in obstacles:
        d = geometry.signed_distance(px, py, obs)
        if d >= p.influence:
            continue
        cx, cy = geometry.closest_point(px, py, obs)
        away = np.array([px - cx, py - cy], dtype=float)
        n = float(np.linalg.norm(away))
        if n < 1e-9:
            # Inside / on the surface: bail outward along +x (rare in practice).
            away = np.array([1.0, 0.0])
            n = 1.0
        away /= n
        d_eff = max(d, p.eps)
        # 1/d**2 weighting, tapered smoothly to 0 at the influence boundary.
        mag = p.k_rep * (1.0 / d_eff - 1.0 / p.influence) / (d_eff * d_eff)
        mag = min(max(mag, 0.0), p.rep_cap)
        force += mag * away
    return force


def compute_field(
    pos: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Dict],
    p: FieldParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (total, attractive, repulsive) vectors."""
    fa = attractive(pos, goal, p)
    fr = repulsive(pos, obstacles, p)
    return fa + fr, fa, fr
