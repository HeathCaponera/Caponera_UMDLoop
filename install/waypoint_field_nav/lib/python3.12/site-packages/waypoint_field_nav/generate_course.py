#!/usr/bin/env python3
"""Generate a randomized Gazebo world (SDF) + a course manifest (YAML).

Outputs two files that the launch file and nodes consume:

  <name>.sdf   - Gazebo Harmonic world: ground, sun, the diff-drive vehicle at
                 the origin, the random obstacles, and translucent waypoint pillars.
  <name>.yaml  - machine-readable course: bounds, start, obstacles, waypoints.
                 The navigator reads obstacles/waypoints from here; the course
                 publisher reads it to draw RViz markers.

Usage:
  ros2 run waypoint_field_nav generate_course --seed 7 --obstacles 12 --waypoints 5
  python3 generate_course.py --seed 7 -o /tmp/course        # (bare, no ROS)
"""

from __future__ import annotations

import argparse
import os

import yaml

try:
    from waypoint_field_nav.core.course_gen import sample_course
except ImportError:  # running the file directly, before install
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from waypoint_field_nav.core.course_gen import sample_course


# --------------------------------------------------------------------------- #
# SDF fragments
# --------------------------------------------------------------------------- #
def _robot_model(spawn_x: float, spawn_y: float) -> str:
    """A minimal differential-drive box: chassis + 2 driven wheels + 1 caster."""
    return f"""
    <model name="vehicle">
      <pose>{spawn_x} {spawn_y} 0.10 0 0 0</pose>
      <self_collide>false</self_collide>

      <link name="base_link">
        <inertial>
          <mass>3.0</mass>
          <inertia><ixx>0.05</ixx><iyy>0.08</iyy><izz>0.10</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <visual name="v_chassis">
          <geometry><box><size>0.40 0.30 0.15</size></box></geometry>
          <material><ambient>0.1 0.35 0.8 1</ambient><diffuse>0.1 0.4 0.9 1</diffuse></material>
        </visual>
        <collision name="c_chassis">
          <geometry><box><size>0.40 0.30 0.15</size></box></geometry>
        </collision>
        <!-- little nose marker so heading is obvious in the GUI -->
        <visual name="v_nose">
          <pose>0.20 0 0 0 0 0</pose>
          <geometry><box><size>0.06 0.10 0.10</size></box></geometry>
          <material><ambient>1 0.8 0 1</ambient><diffuse>1 0.85 0 1</diffuse></material>
        </visual>
      </link>

      <link name="left_wheel">
        <pose>-0.12 0.17 0 -1.5707 0 0</pose>
        <inertial><mass>0.2</mass>
          <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual name="v"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.15 0.15 0.15 1</diffuse></material></visual>
        <collision name="c"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface>
        </collision>
      </link>

      <link name="right_wheel">
        <pose>-0.12 -0.17 0 -1.5707 0 0</pose>
        <inertial><mass>0.2</mass>
          <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual name="v"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.15 0.15 0.15 1</diffuse></material></visual>
        <collision name="c"><geometry><cylinder><radius>0.10</radius><length>0.05</length></cylinder></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface>
        </collision>
      </link>

      <link name="caster">
        <pose>0.15 0 -0.05 0 0 0</pose>
        <inertial><mass>0.05</mass>
          <inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual name="v"><geometry><sphere><radius>0.05</radius></sphere></geometry>
          <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>
        <collision name="c"><geometry><sphere><radius>0.05</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface>
        </collision>
      </link>

      <joint name="left_wheel_joint" type="revolute">
        <parent>base_link</parent><child>left_wheel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>
      <joint name="right_wheel_joint" type="revolute">
        <parent>base_link</parent><child>right_wheel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>
      <joint name="caster_joint" type="ball">
        <parent>base_link</parent><child>caster</child>
      </joint>

      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.34</wheel_separation>
        <wheel_radius>0.10</wheel_radius>
        <topic>cmd_vel</topic>
        <odom_topic>odom</odom_topic>
        <tf_topic>tf</tf_topic>
        <frame_id>odom</frame_id>
        <child_frame_id>base_link</child_frame_id>
        <odom_publish_frequency>30</odom_publish_frequency>
      </plugin>
    </model>"""


def _obstacle_model(i: int, obs: dict) -> str:
    h = 1.0
    if obs["type"] == "circle":
        geom = f"<cylinder><radius>{obs['radius']}</radius><length>{h}</length></cylinder>"
    else:
        geom = f"<box><size>{2*obs['hx']} {2*obs['hy']} {h}</size></box>"
    return f"""
    <model name="obstacle_{i}"><static>true</static>
      <pose>{obs['x']} {obs['y']} {h/2} 0 0 0</pose>
      <link name="link">
        <visual name="v"><geometry>{geom}</geometry>
          <material><ambient>0.6 0.2 0.2 1</ambient><diffuse>0.75 0.25 0.2 1</diffuse></material></visual>
        <collision name="c"><geometry>{geom}</geometry></collision>
      </link>
    </model>"""


def _waypoint_marker(i: int, wp) -> str:
    return f"""
    <model name="waypoint_{i}"><static>true</static>
      <pose>{wp[0]} {wp[1]} 0.05 0 0 0</pose>
      <link name="link">
        <visual name="v"><transparency>0.55</transparency>
          <geometry><cylinder><radius>0.30</radius><length>0.10</length></cylinder></geometry>
          <material><ambient>0.1 0.8 0.2 1</ambient><diffuse>0.1 0.9 0.2 1</diffuse></material></visual>
      </link>
    </model>"""


def build_world(course: dict, world_name: str = "course") -> str:
    parts = [_robot_model(course["start"][0], course["start"][1])]
    for i, obs in enumerate(course["obstacles"]):
        parts.append(_obstacle_model(i, obs))
    for i, wp in enumerate(course["waypoints"]):
        parts.append(_waypoint_marker(i, wp))
    models = "\n".join(parts)
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <gravity>0 0 -9.8</gravity>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>

    <model name="ground_plane"><static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.35 0.38 0.4 1</ambient><diffuse>0.4 0.43 0.45 1</diffuse></material></visual>
      </link>
    </model>
{models}
  </world>
</sdf>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--obstacles", type=int, default=12)
    ap.add_argument("--waypoints", type=int, default=5)
    ap.add_argument("--bounds", type=float, nargs=4,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"), default=[-8, -8, 8, 8])
    ap.add_argument("-o", "--output", default="course",
                    help="output path prefix (writes <prefix>.sdf and <prefix>.yaml)")
    ap.add_argument("--name", default="course", help="Gazebo world name")
    args = ap.parse_args(argv)

    course = sample_course(seed=args.seed, bounds=tuple(args.bounds),
                           n_obstacles=args.obstacles, n_waypoints=args.waypoints)

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".yaml", "w") as f:
        yaml.safe_dump(course, f, sort_keys=False)
    with open(out + ".sdf", "w") as f:
        f.write(build_world(course, args.name))

    print(f"Wrote {out}.sdf and {out}.yaml")
    print(f"  seed={args.seed}  obstacles={len(course['obstacles'])}  "
          f"waypoints={len(course['waypoints'])}  bounds={args.bounds}")


if __name__ == "__main__":
    main()
