"""Headless verification of the navigation brain — NO ROS, NO Gazebo.

Runs NavCore + Mission against the same differential-drive kinematics Gazebo
would integrate:

    x   += v * cos(yaw) * dt
    y   += v * sin(yaw) * dt
    yaw += w * dt

For each random course it reports: waypoints reached, whether the body ever
hit an obstacle, sim time, and path length. This is how we confirm the plan
actually navigates before wiring it into Gazebo.
"""

from __future__ import annotations

import math
import os
import sys

# allow running straight from the package dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from waypoint_field_nav.core import geometry  # noqa: E402
from waypoint_field_nav.core.course_gen import sample_course  # noqa: E402
from waypoint_field_nav.core.nav_core import Mission, NavCore, NavParams, partition_waypoints  # noqa: E402


def run_course(course, dt=0.05, timeout=180.0, collide_r=0.18, verbose=False):
    obstacles = course["obstacles"]
    nav = NavCore(obstacles, tuple(course["bounds"]), NavParams())

    # pre-flight: drop waypoints that are provably unreachable (inside an
    # obstacle or walled off from the start) so the mission can't hang on one.
    report = nav.reachability_report(course["start"], course["waypoints"], tol=0.30)
    waypoints, skipped = partition_waypoints(report, course["waypoints"])
    mission = Mission(waypoints, tol=0.30)  # one car radius

    x, y = course["start"]
    yaw = 0.0
    t = 0.0
    path_len = 0.0
    min_clr = math.inf
    collided = False

    steps = int(timeout / dt)
    for _ in range(steps):
        mission.update(x, y)
        if mission.done:
            break
        goal = mission.current_goal()
        out = nav.compute(x, y, yaw, goal, t)
        v, w = out["v"], out["w"]

        px, py = x, y
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = geometry.wrap_angle(yaw + w * dt)
        t += dt
        path_len += math.hypot(x - px, y - py)

        clr = geometry.min_clearance(x, y, obstacles)
        min_clr = min(min_clr, clr)
        if clr < collide_r:
            collided = True
        if verbose and int(t / dt) % 40 == 0:
            print(f"  t={t:5.1f} wp={mission.idx} mode={out['mode']:5} "
                  f"pos=({x:5.2f},{y:5.2f}) v={v:.2f} w={w:+.2f} clr={clr:5.2f}")

    return {
        "reached": mission.idx,
        "total": len(mission.waypoints),
        "done": mission.done,
        "collided": collided,
        "min_clearance": min_clr,
        "time": t,
        "path_len": path_len,
        "skipped": len(skipped),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ok = 0
    clean = 0
    print(f"Running {n} random courses through the headless diff-drive sim...\n")
    print(f"{'seed':>4} {'reached':>9} {'done':>5} {'hit':>4} "
          f"{'minClr':>7} {'time':>6} {'len':>6}")
    for seed in range(n):
        course = sample_course(seed=seed, n_obstacles=12, n_waypoints=5)
        r = run_course(course)
        ok += 1 if r["done"] else 0
        clean += 1 if (r["done"] and not r["collided"]) else 0
        flag = "OK " if r["done"] and not r["collided"] else ("HIT" if r["collided"] else "INC")
        print(f"{seed:>4} {r['reached']:>4}/{r['total']:<4} "
              f"{str(r['done']):>5} {str(r['collided']):>4} "
              f"{r['min_clearance']:>7.2f} {r['time']:>6.1f} {r['path_len']:>6.1f}  {flag}")
    print(f"\nComplete (all waypoints): {ok}/{n}")
    print(f"Complete AND collision-free: {clean}/{n}")


if __name__ == "__main__":
    main()
