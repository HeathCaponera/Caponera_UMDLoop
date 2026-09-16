"""Navigation brain (pure Python, no ROS).

Implements the whole plan as a small two-state controller:

  FIELD  (default)  -> follow the artificial potential field (step 2)
  PLAN   (recovery) -> follow a weighted-Dijkstra path (step 3b)

Every tick:
  1. build the potential field vector -> a desired travel direction
  2. LOOK AHEAD along that direction (step 3). If clear -> drive along it.
  3. If blocked, or if we've stopped making progress (local-minimum trap),
     switch to PLAN: run the grid search once, then pure-pursuit the path.
     Recover to FIELD once the way to the goal is clear again.

The final step converts the chosen direction into (linear.x, angular.z) for a
differential-drive base. Waypoint sequencing (step 1) lives in ``Mission``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry, planner
from .potential_field import FieldParams, compute_field


# --------------------------------------------------------------------------- #
# Waypoint sequencing  (step 1)
# --------------------------------------------------------------------------- #
class Mission:
    def __init__(self, waypoints: Sequence[Tuple[float, float]], tol: float = 0.30) -> None:
        # tol = one car radius: the vehicle's *centre* must reach within a car
        # radius of the waypoint, so it visibly arrives onto the marker rather
        # than counting it "met" while still short of it.
        self.waypoints = [tuple(map(float, w)) for w in waypoints]
        self.tol = tol
        self.idx = 0

    @property
    def done(self) -> bool:
        return self.idx >= len(self.waypoints)

    def current_goal(self) -> Optional[np.ndarray]:
        if self.done:
            return None
        return np.array(self.waypoints[self.idx], dtype=float)

    def update(self, x: float, y: float) -> bool:
        """Advance if within tol of the active waypoint. Returns True on advance."""
        if self.done:
            return False
        gx, gy = self.waypoints[self.idx]
        if math.hypot(x - gx, y - gy) <= self.tol:
            self.idx += 1
            return True
        return False


# --------------------------------------------------------------------------- #
# Controller parameters
# --------------------------------------------------------------------------- #
@dataclass
class NavParams:
    # body / safety
    robot_radius: float = 0.30
    safety_margin: float = 0.12   # extra clearance baked into planning inflation
    # How much to inflate obstacles for the *soft* repulsion field. The hard
    # collision layer (look-ahead, planner grid) and the speed governor always
    # treat the car as a point of radius `inflation`; the repulsion field is a
    # smooth steering heuristic, so inflating it by the full body+margin makes
    # it over-repel and destabilises the field<->plan hand-off. A gentler
    # inflation keeps the body pushed off obstacles without that instability.
    field_inflation: float = 0.12
    # potential field
    field: FieldParams = field(default_factory=FieldParams)
    # look-ahead (step 3)
    lookahead: float = 1.2
    # final approach: if within this range AND the straight line to the goal is
    # clear, drive straight in (ignore side repulsion) -> kills APF orbits.
    approach_radius: float = 1.4
    # reactive safety nudge (all modes): blend in repulsion when this close
    safety_dist: float = 0.6
    safety_gain: float = 1.4
    # speed governor near obstacles (separate, gentler than the nudge so the
    # car keeps cruising in dense fields). clr is body clearance.
    govern_dist: float = 0.40
    govern_floor: float = 0.6
    # planner (step 3b)
    grid_res: float = 0.25
    goal_weight: float = 1.6
    replan_period: float = 1.5
    plan_dwell: float = 2.0        # min time to commit to a plan before recovering
    # pure pursuit along a planned path
    pursuit_dist: float = 0.8
    path_reach: float = 0.4
    off_path: float = 1.0
    # stuck / local-minimum detection
    stuck_time: float = 4.0
    stuck_eps: float = 0.15
    # differential-drive limits
    v_max: float = 0.8
    w_max: float = 2.0
    k_w: float = 2.5
    slow_radius: float = 0.8

    @property
    def inflation(self) -> float:
        return self.robot_radius + self.safety_margin


FIELD, PLAN = "FIELD", "PLAN"


class NavCore:
    def __init__(self, obstacles: Sequence[Dict],
                 bounds: Tuple[float, float, float, float],
                 params: Optional[NavParams] = None) -> None:
        self.obstacles = list(obstacles)
        self.p = params or NavParams()
        # Three views of the world, each the obstacles grown so the vehicle can
        # be treated as a POINT (configuration-space / Minkowski inflation):
        #   hard_obs - grown by robot_radius + safety_margin: the look-ahead
        #              "is my body path blocked?" test and the planner grid.
        #   body_obs - grown by exactly robot_radius: honest body clearance,
        #              used to slow down and steer away as the body nears a wall.
        #   soft_obs - grown by a small field_inflation: the smooth repulsion
        #              field, kept gentle so it doesn't destabilise steering.
        self.hard_obs = geometry.inflate_obstacles(self.obstacles, self.p.inflation)
        self.body_obs = geometry.inflate_obstacles(self.obstacles, self.p.robot_radius)
        self.soft_obs = geometry.inflate_obstacles(self.obstacles, self.p.field_inflation)

        # expand grid slightly beyond the course so edge goals are plannable
        pad = 1.0
        b = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
        self.grid = planner.OccupancyGrid(b, self.p.grid_res, self.obstacles, self.p.inflation)

        self.mode = FIELD
        self.path: List[Tuple[float, float]] = []
        self.goal_ref: Optional[Tuple[float, float]] = None
        self.last_plan_t = -1e9
        self.plan_enter_t = -1e9
        self.plan_start_dist = math.inf
        # stuck tracking
        self.best_dist = math.inf
        self.best_t = 0.0

    # ---- pre-flight reachability check ------------------------------------ #
    def reachability_report(
        self,
        start_xy: Tuple[float, float],
        waypoints: Sequence[Tuple[float, float]],
        tol: float,
    ) -> List[Dict]:
        """Classify each waypoint as reachable-or-not from ``start_xy`` BEFORE
        the vehicle moves, so an impossible waypoint (one inside an obstacle,
        or a pocket walled off on all sides) is caught up front instead of
        hanging the mission.

        Returns one record per waypoint::

            {"index": i, "waypoint": (x, y), "reachable": bool, "reason": str|None}

        A waypoint is reachable when BOTH hold:
          * arrival is physically possible - the car's centre can sit within
            ``tol`` of the point while its body stays off every obstacle
            surface (flat-wall bound: true clearance >= robot_radius - tol);
          * the point is connected to the start through the free space of the
            planner grid (flood-fill), i.e. not boxed in.
        The two failure modes are reported with distinct reasons.
        """
        g = self.grid
        start_cell = g.nearest_free(*g.world_to_cell(start_xy[0], start_xy[1]))
        mask = g.reachable_mask(start_cell) if start_cell is not None else None
        arrival_min = self.p.robot_radius - tol  # needed true clearance at the point

        report: List[Dict] = []
        for i, wp in enumerate(waypoints):
            wx, wy = float(wp[0]), float(wp[1])
            reason = None
            clr = geometry.min_clearance(wx, wy, self.obstacles)  # true, signed
            if clr < arrival_min:
                # Can't place the car centre within tol of the point without the
                # body overlapping an obstacle.
                reason = ("is inside an obstacle" if clr < 0.0
                          else "sits too close to an obstacle to reach within tolerance")
            else:
                snapped = g.nearest_free(*g.world_to_cell(wx, wy))
                if snapped is None:
                    reason = "is buried inside an obstacle (no free space nearby)"
                elif mask is None or not mask[snapped[0], snapped[1]]:
                    reason = "is walled off from the start (enclosed by obstacles)"
            report.append({"index": i, "waypoint": (wx, wy),
                           "reachable": reason is None, "reason": reason})
        return report

    # ---- helpers ---------------------------------------------------------- #
    def _blocked_ahead(self, pos: np.ndarray, direction: np.ndarray, dist: float) -> bool:
        end = pos + direction * dist
        # cobs is already inflated by robot radius + margin, so probe as a point.
        return geometry.segment_blocked(
            (pos[0], pos[1]), (end[0], end[1]), self.hard_obs, 0.0)

    def _update_stuck(self, dist_goal: float, t: float) -> bool:
        if dist_goal < self.best_dist - self.p.stuck_eps:
            self.best_dist = dist_goal
            self.best_t = t
        return (t - self.best_t) > self.p.stuck_time

    def _carrot(self, pos: np.ndarray) -> np.ndarray:
        """Pure-pursuit target: trim passed points, then pick the first path
        point farther than pursuit_dist (or the final one)."""
        while len(self.path) > 1 and \
                math.hypot(self.path[0][0] - pos[0], self.path[0][1] - pos[1]) < self.p.path_reach:
            self.path.pop(0)
        for wx, wy in self.path:
            if math.hypot(wx - pos[0], wy - pos[1]) >= self.p.pursuit_dist:
                return np.array([wx, wy], dtype=float)
        return np.array(self.path[-1], dtype=float) if self.path else pos

    def _dist_to_path(self, pos: np.ndarray) -> float:
        if not self.path:
            return math.inf
        return min(math.hypot(wx - pos[0], wy - pos[1]) for wx, wy in self.path)

    def _replan(self, pos: np.ndarray, goal: np.ndarray, t: float) -> None:
        p = planner.plan(self.grid, (pos[0], pos[1]), (goal[0], goal[1]), self.p.goal_weight)
        self.path = p or []
        self.last_plan_t = t

    def _enter_plan(self, pos: np.ndarray, goal: np.ndarray, t: float) -> None:
        self.mode = PLAN
        self.plan_enter_t = t
        self.plan_start_dist = float(np.linalg.norm(goal - pos))
        self._replan(pos, goal, t)

    # ---- main tick -------------------------------------------------------- #
    def compute(self, x: float, y: float, yaw: float,
                goal: np.ndarray, t: float) -> Dict:
        pos = np.array([x, y], dtype=float)
        goal = np.asarray(goal, dtype=float)
        dist_goal = float(np.linalg.norm(goal - pos))

        # New goal? reset stuck + planner state.
        if self.goal_ref is None or math.hypot(goal[0] - self.goal_ref[0],
                                               goal[1] - self.goal_ref[1]) > 1e-3:
            self.goal_ref = (float(goal[0]), float(goal[1]))
            self.best_dist, self.best_t = dist_goal, t
            self.mode, self.path = FIELD, []

        # --- step 2: potential field ---------------------------------------
        field_vec, fa, fr = compute_field(pos, goal, self.soft_obs, self.p.field)
        n = float(np.linalg.norm(field_vec))
        field_dir = field_vec / n if n > 1e-6 else (goal - pos) / max(dist_goal, 1e-6)

        goal_dir = (goal - pos) / max(dist_goal, 1e-6)
        probe = min(self.p.lookahead, dist_goal)
        field_blocked = self._blocked_ahead(pos, field_dir, probe)
        goal_clear = not self._blocked_ahead(pos, goal_dir, probe)
        stuck = self._update_stuck(dist_goal, t)

        # Final approach: close to the goal with a clear straight shot -> go
        # straight in, ignoring side obstacles. This dissolves the APF orbit
        # that otherwise forms around a waypoint sitting next to an obstacle.
        near_goal_straight = dist_goal < self.p.approach_radius and goal_clear

        # --- decide mode (with hysteresis so PLAN actually commits) --------
        if self.mode == FIELD:
            if not near_goal_straight and (field_blocked or stuck):
                self._enter_plan(pos, goal, t)
        else:  # PLAN
            dwelled = (t - self.plan_enter_t) > self.p.plan_dwell
            progressed = (self.plan_start_dist - dist_goal) > 0.5
            if goal_clear and (near_goal_straight or (not field_blocked and (dwelled or progressed))):
                self.mode, self.path = FIELD, []            # recovered / escaped
                self.best_dist, self.best_t = dist_goal, t
            elif (not self.path
                  or self._dist_to_path(pos) > self.p.off_path
                  or (t - self.last_plan_t) > self.p.replan_period):
                self._replan(pos, goal, t)

        # --- choose travel direction ---------------------------------------
        if near_goal_straight:
            direction = goal_dir
        elif self.mode == PLAN and self.path:
            direction = self._carrot(pos) - pos
            nd = float(np.linalg.norm(direction))
            direction = direction / nd if nd > 1e-6 else field_dir
        else:
            direction = field_dir

        # --- reactive safety nudge (every mode EXCEPT final approach) ------
        # Blend away-from-obstacle steering in when we get close, so neither a
        # tangential field pass nor a corner-cutting path can graze an obstacle.
        # Skipped during near_goal_straight: there the straight line to the goal
        # is already confirmed body-clear by the look-ahead, and nudging would
        # push the car off it and orbit a waypoint that sits next to an obstacle.
        clr = geometry.min_clearance(pos[0], pos[1], self.body_obs)
        rn = float(np.linalg.norm(fr))
        if not near_goal_straight and clr < self.p.safety_dist and rn > 1e-6:
            wgt = self.p.safety_gain * (1.0 - max(clr, 0.0) / self.p.safety_dist)
            direction = direction + wgt * (fr / rn)
            dn = float(np.linalg.norm(direction))
            if dn > 1e-6:
                direction = direction / dn

        v, w = self._to_diff_drive(direction, yaw, dist_goal, pos, clr)
        return {
            "v": v, "w": w, "mode": self.mode,
            "field": field_vec, "att": fa, "rep": fr,
            "direction": direction, "path": list(self.path),
            "dist_goal": dist_goal, "stuck": stuck,
        }

    # ---- step 3a: direction -> wheel commands ----------------------------- #
    def _to_diff_drive(self, direction: np.ndarray, yaw: float,
                       dist_goal: float, pos: np.ndarray, clr: float) -> Tuple[float, float]:
        desired_yaw = math.atan2(direction[1], direction[0])
        err = geometry.wrap_angle(desired_yaw - yaw)

        w = max(-self.p.w_max, min(self.p.w_max, self.p.k_w * err))

        align = math.cos(err)                      # only drive forward when aimed right
        v = self.p.v_max * max(0.0, align)
        # ease off approaching the goal (accurate stop) ...
        if self.p.slow_radius > 0:
            v *= min(1.0, dist_goal / self.p.slow_radius)
        # ... and slow near obstacles for safety. Uses its own (gentler)
        # distance + floor than the steering nudge, so the car keeps a useful
        # cruise speed threading dense fields instead of crawling everywhere.
        if clr < self.p.govern_dist:
            v *= max(self.p.govern_floor, clr / self.p.govern_dist)
        return v, w


def partition_waypoints(
    report: List[Dict],
    waypoints: Sequence[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Dict]]:
    """Split a reachability report into the reachable waypoints (in order, to
    keep as the mission) and the skipped records (for logging)."""
    keep = [tuple(waypoints[r["index"]]) for r in report if r["reachable"]]
    skipped = [r for r in report if not r["reachable"]]
    return keep, skipped
