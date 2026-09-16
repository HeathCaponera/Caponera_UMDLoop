"""Pre-flight waypoint-reachability check (Level 1).

Builds deliberately broken courses and confirms the check flags the bad
waypoints with the right reason, then confirms randomly generated courses
(whose generator keeps waypoints well clear of obstacles) produce zero false
positives.

Run:  python3 test/test_reachability.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from waypoint_field_nav.core.course_gen import sample_course  # noqa: E402
from waypoint_field_nav.core.nav_core import (  # noqa: E402
    NavCore, NavParams, partition_waypoints,
)

BOUNDS = (-8.0, -8.0, 8.0, 8.0)
TOL = 0.30


def check(obstacles, waypoints, start=(0.0, 0.0), tol=TOL):
    nav = NavCore(obstacles, BOUNDS, NavParams())
    return nav.reachability_report(start, waypoints, tol=tol)


def main() -> int:
    failures = 0

    # A sealed "room": four thick walls around (4, 4) with no body-sized gap.
    room = [
        {"type": "box", "x": 4.0, "y": 5.3, "hx": 1.8, "hy": 0.25},   # top
        {"type": "box", "x": 4.0, "y": 2.7, "hx": 1.8, "hy": 0.25},   # bottom
        {"type": "box", "x": 2.4, "y": 4.0, "hx": 0.25, "hy": 1.8},   # left
        {"type": "box", "x": 5.6, "y": 4.0, "hx": 0.25, "hy": 1.8},   # right
    ]

    cases = [
        # (name, obstacles, waypoint, tol, expect_reachable, reason_contains)
        ("open waypoint",
         [{"type": "circle", "x": 2.0, "y": 0.0, "radius": 0.6}],
         (-4.0, -4.0), TOL, True, None),
        ("waypoint just outside a wall",
         [{"type": "circle", "x": 0.0, "y": 3.0, "radius": 1.0}],
         (0.0, 4.2), TOL, True, None),               # 0.2 m clear, tol needs >=0
        ("waypoint inside a circle",
         [{"type": "circle", "x": 3.0, "y": 0.0, "radius": 0.8}],
         (3.0, 0.0), TOL, False, "inside an obstacle"),
        ("waypoint inside a box",
         [{"type": "box", "x": -3.0, "y": 2.0, "hx": 0.6, "hy": 0.6}],
         (-3.0, 2.0), TOL, False, "inside an obstacle"),
        ("waypoint too close (tight tol)",
         [{"type": "circle", "x": 0.0, "y": 3.0, "radius": 1.0}],
         (0.0, 4.1), 0.15, False, "too close"),      # 0.1 m clear, tol 0.15 -> need >=0.15
        ("waypoint boxed in on all sides",
         room, (4.0, 4.0), TOL, False, "walled off"),
    ]

    for name, obs, wp, tol, expect_reachable, needle in cases:
        rep = check(obs, [wp], tol=tol)[0]
        ok = (rep["reachable"] == expect_reachable) and (
            expect_reachable or (needle in (rep["reason"] or "")))
        failures += 0 if ok else 1
        verdict = "reachable" if rep["reachable"] else f"UNREACHABLE ({rep['reason']})"
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34} -> {verdict}")

    # Mixed course: partition should keep the good ones, drop the bad ones.
    obs = [{"type": "circle", "x": 3.0, "y": 0.0, "radius": 0.8}] + room
    wps = [(-4.0, -4.0), (3.0, 0.0), (4.0, 4.0), (-2.0, -1.0)]
    rep = check(obs, wps)
    keep, skipped = partition_waypoints(rep, wps)
    ok = len(keep) == 2 and len(skipped) == 2
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] partition mixed course      "
          f"-> keep {len(keep)}, skip {len(skipped)}")

    # No false positives on generated courses (waypoints kept clear by design).
    total_wp = bad = 0
    for seed in range(40):
        c = sample_course(seed=seed, n_obstacles=14, n_waypoints=6)
        rep = check(c["obstacles"], c["waypoints"], start=c["start"])
        total_wp += len(rep)
        bad += sum(0 if r["reachable"] else 1 for r in rep)
    ok = bad == 0
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] generated courses clean     "
          f"-> {bad} false positives across {total_wp} waypoints")

    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
