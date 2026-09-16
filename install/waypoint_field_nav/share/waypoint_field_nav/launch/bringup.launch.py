#!/usr/bin/env python3
"""bringup.launch.py - one command to run the whole thing.

Starts Gazebo (Harmonic) with the generated world, the ros_gz bridge, the
navigator, the course visualizer, and (optionally) RViz.

    ros2 launch waypoint_field_nav bringup.launch.py

Point it at a different course you generated with::

    ros2 launch waypoint_field_nav bringup.launch.py \\
        world:=/abs/path/my_course.sdf course_file:=/abs/path/my_course.yaml

The world .sdf and the course .yaml must be the matching pair emitted together
by generate_course (same obstacles / waypoints / robot spawn).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory("waypoint_field_nav")
    default_world = os.path.join(pkg, "worlds", "default_course.sdf")
    default_course = os.path.join(pkg, "worlds", "default_course.yaml")
    bridge_config = os.path.join(pkg, "config", "bridge.yaml")
    rviz_config = os.path.join(pkg, "config", "nav.rviz")

    world = LaunchConfiguration("world")
    course_file = LaunchConfiguration("course_file")
    use_rviz = LaunchConfiguration("rviz")

    args = [
        DeclareLaunchArgument(
            "world", default_value=default_world,
            description="Absolute path to the Gazebo world .sdf"),
        DeclareLaunchArgument(
            "course_file", default_value=default_course,
            description="Absolute path to the matching course .yaml"),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="Launch RViz with the course visualization"),
    ]

    # Gazebo Harmonic via ros_gz_sim. '-r' runs immediately; '-v 3' = info logs.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
        ),
        launch_arguments={"gz_args": ["-r -v 3 ", world]}.items(),
    )

    # ros <-> gz topic bridge (clock, cmd_vel, odom, tf).
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[{"config_file": bridge_config, "use_sim_time": True}],
        output="screen",
    )

    navigator = Node(
        package="waypoint_field_nav",
        executable="navigator",
        name="navigator",
        parameters=[{"course_file": course_file, "use_sim_time": True}],
        output="screen",
    )

    course_pub = Node(
        package="waypoint_field_nav",
        executable="course_publisher",
        name="course_publisher",
        parameters=[{"course_file": course_file, "use_sim_time": True}],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    return LaunchDescription(args + [gz_sim, bridge, navigator, course_pub, rviz])
