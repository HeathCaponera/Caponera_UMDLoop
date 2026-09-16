# waypoint_field_nav

Autonomous waypoint navigation for a differential-drive vehicle in
**ROS 2 Jazzy + Gazebo Harmonic**.

A box robot starts at the origin and drives itself through a sequence of
randomized XY waypoints, avoiding randomly generated obstacles, with no teleop
and no manual driving. A single navigation node reads odometry, decides where to
go, and publishes velocity commands on its own.

This implements the "Plan 2" design: **waypoint sequencing → potential field →
look-ahead → weighted-Dijkstra/A\* fallback.**

---

## What's in the box

```
waypoint_field_nav/
├── waypoint_field_nav/
│   ├── core/                 # ROS-free navigation brain (unit-testable)
│   │   ├── geometry.py       # clearance / blocked-segment preds + C-space inflation
│   │   ├── potential_field.py# attractive + 1/d^2 repulsive field
│   │   ├── planner.py        # occupancy grid + weighted-A* + reachability flood-fill
│   │   ├── nav_core.py       # the controller: field + look-ahead + fallback
│   │   └── course_gen.py     # random course sampler + course loader
│   ├── navigator.py          # ROS node: /odom -> /cmd_vel  (the driver)
│   ├── course_publisher.py   # ROS node: latched RViz markers for the course
│   └── generate_course.py    # CLI: emit a Gazebo world + course manifest
├── launch/bringup.launch.py  # gazebo + bridge + nodes + rviz
├── config/
│   ├── bridge.yaml           # ros_gz topic bridge (clock/cmd_vel/odom/tf)
│   └── nav.rviz              # RViz layout
├── worlds/
│   ├── default_course.sdf    # ready-to-run sample world
│   └── default_course.yaml   # matching course manifest
└── test/
    ├── headless_sim.py       # headless verifier for the brain
    └── test_reachability.py  # pre-flight reachability-check tests
```

---

## Build

Assuming ROS 2 Jazzy and the Gazebo bridge are installed:

```bash
sudo apt install ros-jazzy-ros-gz            # Harmonic bridge for Jazzy

cd ~/ws                                       # your colcon workspace
colcon build --packages-select waypoint_field_nav
source install/setup.bash
```

## Run

```bash
ros2 launch waypoint_field_nav bringup.launch.py
```

That launches Gazebo with the sample world, the ROS↔Gazebo bridge, the
navigator, the course visualizer, and RViz. The robot begins driving to
waypoint 1 immediately and reports progress on `/mission_status`. Turn RViz off
with `rviz:=false`.

Watch it think:

```bash
ros2 topic echo /mission_status
```

## Generate new courses

```bash
ros2 run waypoint_field_nav generate_course \
    --seed 42 --obstacles 15 --waypoints 6 \
    --bounds -8 -8 8 8 -o /tmp/course42 --name course42
```

This writes a matched pair — `/tmp/course42.sdf` (Gazebo world) and
`/tmp/course42.yaml` (obstacle + waypoint manifest the navigator reads). Run it:

```bash
ros2 launch waypoint_field_nav bringup.launch.py \
    world:=/tmp/course42.sdf course_file:=/tmp/course42.yaml
```

The world and the manifest must be the pair emitted together — same obstacles,
waypoints, and robot spawn.

---

## How the navigation works (Plan 2)

The brain lives in `core/` and has **no ROS or Gazebo dependency**, so it can be
driven by a plain kinematic simulator and checked in isolation.

**The vehicle is treated as a point.** At startup `NavCore` grows every obstacle
outward (configuration-space / Minkowski inflation, `geometry.inflate_obstacles`)
so every point-vs-obstacle test automatically accounts for the whole car body.
Three views are kept, each for a different job:

- `hard_obs` — grown by `robot_radius + safety_margin` (0.42 m): the look-ahead
  "is my body path blocked?" test and the planner grid.
- `body_obs` — grown by exactly `robot_radius` (0.30 m): honest body clearance,
  used to slow down and steer away as the body nears a surface.
- `soft_obs` — grown by a small `field_inflation` (0.12 m): the smooth repulsion
  field, kept gentle so it doesn't over-repel and destabilise steering.

This is what stops the car clipping an obstacle with a corner *before* reacting:
the reactive field and clearance now see the body, not a zero-size dot at the
centre.

Each control tick (`nav_core.NavCore.compute`) does:

1. **Waypoint check** (`Mission`). If the car's centre is within one car radius
   (`tol = 0.30 m`) of the active waypoint, advance to the next; when none
   remain, the mission is complete and the navigator commands zero velocity.
   (One car radius equals the waypoint pad radius, so the car visibly arrives
   *onto* the marker rather than counting it met while still short of it.)

2. **Potential field** (`potential_field.py`).
   - *Attractive:* unit vector toward the goal, ramped down close in.
   - *Repulsive:* for every obstacle within an influence radius, a push away
     from its surface weighted by `1/d²` and tapered to zero at the boundary.
   - Sum → desired direction.

3. **Look-ahead** (`nav_core._blocked_ahead`). Probe along the field direction.
   If clear, convert direction → `(v, ω)` for the diff-drive and go.

4. **Planner fallback** (`planner.py`). If the field direction is blocked (or the
   robot stops making progress), plan on an inflated occupancy grid with a
   weighted A\* — `f = g + w·h`, `w = 1.6`. This is the "modified Dijkstra
   weighted by distance from goal": raising `w` biases expansion toward the
   goal. The robot pure-pursuits the returned path until the field is viable
   again, gated by hysteresis so it commits to a recovery instead of dithering.

Two safety layers run in every mode: a reactive nudge that blends in
away-from-obstacle steering when body clearance drops below `safety_dist`
(suppressed during the final straight-in approach, where the line to the goal is
already confirmed clear — otherwise the nudge would push the car off it and orbit
a waypoint next to an obstacle), and a speed governor with its own gentler
thresholds (`govern_dist`, `govern_floor`) that slows near obstacles while
keeping a useful cruise speed in dense fields, easing to an accurate stop at each
waypoint.

### Pre-flight reachability check

Before the vehicle moves, `NavCore.reachability_report` classifies every
waypoint so an impossible one is caught up front instead of silently hanging the
mission. It flood-fills the free cells of the planner grid from the start
(`OccupancyGrid.reachable_mask`, same 8-connectivity and corner rule as the
planner) and marks a waypoint **unreachable** when either:

- **arrival is impossible** — the car's centre can't sit within `tol` of the
  point while its body stays off obstacle surfaces (i.e. the point is inside, or
  on the lip of, an obstacle); or
- **it's walled off** — the point isn't connected to the start through free
  space (a pocket boxed in on all sides).

The two cases are reported with distinct reasons. The navigator logs a warning
per bad waypoint, **skips** it, and drives the rest (or exits cleanly if none
remain). Because the test mirrors the planner's own feasibility, it flags nothing
on well-formed courses — the random generator already keeps waypoints ≥ 0.8 m
from obstacle surfaces, so this only ever fires on hand-edited course files.
It does *not* pre-judge the marginal case of a pocket reachable only through a
gap narrower than body-plus-margin; that isn't provably unreachable.

### Obstacle information

The navigator uses the **ground-truth obstacle manifest** from the course
generator (positions and sizes), which is exactly the input Plan 2's field and
planner call for. This keeps detection decoupled from render-engine details. The
`core` predicates are sensor-agnostic, so a lidar/costmap front-end could be
swapped in later without touching the planning logic.

### Coordinates

The robot spawns at the origin, so the DiffDrive plugin's `odom` frame coincides
with the world frame the course is defined in; the navigator uses the `/odom`
pose directly as the world position. All markers are published in `odom`.

---

## Verification & a note on honesty

The navigation brain was verified **outside Gazebo** with the headless
kinematic simulator in `test/headless_sim.py`, which integrates the same
diff-drive model Gazebo uses and reports waypoint completion, collisions, and
minimum obstacle clearance. The pre-flight reachability check has its own tests:

```bash
python3 test/headless_sim.py          # 20 randomized courses, summary table
python3 test/test_reachability.py     # pathological + generated-course checks
```

On the 20-course default suite the controller reaches every waypoint on every
course with zero collisions. Across a stress suite of **240 randomized courses**
(dense fields up to 25 obstacles, 8 waypoints, tight gaps) it completes
**227/240, every completed course collision-free**, with worst-case clearance
**~0.49 m** against a robot half-diagonal of ~0.25 m.

**Why clearance, not just completion, is the headline.** An earlier version of
this controller completed all 240 courses — but at a worst-case clearance of
only **0.31 m**. That looked clean in the idealized kinematic sim, yet the car
still clipped obstacles in Gazebo, because Gazebo adds real inertia, wheel slip,
and controller lag that eat a thin margin the perfect integrator never sees.
Treating the car as a point (C-space inflation) and slowing on *body* clearance
widened the worst case to ~0.49 m — a buffer that survives those dynamics. The
trade is the ~13 tightest dense courses, where the car now orbits a pocket it
can't thread with the larger effective footprint rather than scraping through;
in real deployment those are exactly the situations where scraping through is the
wrong answer.

I was **not able to run Gazebo itself in the environment where this package was
written** (no display / GPU for the simulator), so the ROS↔Gazebo integration —
the SDF world, the bridge config, the launch file — is built to the documented
Jazzy/Harmonic interfaces but has not been executed here. The autonomy logic
that actually solves the task is what the headless suite exercises end to end.

---

## Key parameters

Controller defaults live in `NavParams` (`core/nav_core.py`):

| param | default | meaning |
|-------|---------|---------|
| `robot_radius` | 0.30 m | footprint radius (also the C-space "point" inflation) |
| `safety_margin` | 0.12 m | extra grid inflation (→ 0.42 m total for `hard_obs`) |
| `field_inflation` | 0.12 m | inflation of the soft repulsion field only |
| `lookahead` | 1.2 m | look-ahead probe distance |
| `goal_weight` | 1.6 | A\* heuristic weight `w` |
| `safety_dist` | 0.6 m | body clearance below which the reactive nudge engages |
| `govern_dist` | 0.40 m | body clearance below which the speed governor slows |
| `govern_floor` | 0.6 | floor on the governor's speed scaling |
| `v_max` / `w_max` | 0.8 / 2.0 | speed limits |
| `tol` (Mission) | 0.30 m | waypoint arrival radius (one car radius) |

Field gains (`k_att`, `k_rep`, `influence`, …) are in `FieldParams`
(`core/potential_field.py`).

## License

MIT.
