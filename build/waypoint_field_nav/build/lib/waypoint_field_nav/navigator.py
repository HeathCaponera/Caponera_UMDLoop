#!/usr/bin/env python3
"""navigator - the autonomous driver.

Wraps the (ROS-free, headless-verified) navigation brain in
``waypoint_field_nav.core`` and connects it to Gazebo through ROS 2:

  in :  /odom          (nav_msgs/Odometry)   -> current pose
  out:  /cmd_vel       (geometry_msgs/Twist) -> wheel commands
  out:  /nav/markers   (visualization_msgs/MarkerArray) -> goal + planned path
  out:  /mission_status(std_msgs/String)     -> human-readable status

The brain implements the user's Plan 2: waypoint sequencing, a potential
field (attractive-to-goal + 1/d^2 repulsion), a look-ahead check, and a
weighted-Dijkstra / A* fallback when the field direction is blocked. None of
that logic lives here - this file is purely the ROS <-> brain adapter.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from waypoint_field_nav.core.nav_core import NavCore, NavParams, Mission, partition_waypoints
from waypoint_field_nav.core.course_gen import load_course


def yaw_from_quaternion(x, y, z, w):
    """Extract planar heading (yaw) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Navigator(Node):
    def __init__(self):
        super().__init__("navigator")

        # --- parameters ---------------------------------------------------
        self.declare_parameter("course_file", "")
        self.declare_parameter("rate", 20.0)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("goal_tolerance", 0.30)  # one car radius

        course_file = self.get_parameter("course_file").value
        rate = float(self.get_parameter("rate").value)
        self.frame_id = self.get_parameter("frame_id").value
        tol = float(self.get_parameter("goal_tolerance").value)

        if not course_file:
            self.get_logger().fatal(
                "no 'course_file' parameter set - point it at a course .yaml")
            raise SystemExit(1)

        course = load_course(course_file)
        self.bounds = tuple(course["bounds"])
        self.obstacles = course["obstacles"]
        waypoints = [tuple(wp) for wp in course["waypoints"]]
        start_xy = tuple(course.get("start", (0.0, 0.0)))

        # --- the brain ----------------------------------------------------
        self.params = NavParams()
        self.core = NavCore(self.obstacles, self.bounds, self.params)

        # --- pre-flight reachability check --------------------------------
        # Catch waypoints that are inside an obstacle or walled off from the
        # start BEFORE driving, so they're skipped (with a warning) instead of
        # silently hanging the whole mission.
        report = self.core.reachability_report(start_xy, waypoints, tol)
        waypoints, skipped = partition_waypoints(report, waypoints)
        for r in skipped:
            wx, wy = r["waypoint"]
            self.get_logger().warn(
                f"waypoint {r['index']} at ({wx:+.2f}, {wy:+.2f}) {r['reason']}; "
                f"skipping it.")
        if not waypoints:
            self.get_logger().fatal(
                "no reachable waypoints in this course - nothing to do.")
            raise SystemExit(1)
        if skipped:
            self.get_logger().info(
                f"proceeding with {len(waypoints)} reachable waypoint(s), "
                f"{len(skipped)} skipped.")

        self.mission = Mission(waypoints, tol=tol)

        # --- state --------------------------------------------------------
        self.have_odom = False
        self.x = self.y = self.yaw = 0.0
        self.t0 = None
        self.finished = False
        self._last_status = ""

        # --- IO -----------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "mission_status", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "nav/markers", 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 20)

        self.dt = 1.0 / rate
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info(
            f"navigator up: {len(waypoints)} waypoints, "
            f"{len(self.obstacles)} obstacles, bounds={self.bounds}")

    # ---------------------------------------------------------------- odom
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.have_odom = True

    # --------------------------------------------------------------- clock
    def sim_time(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        return now - self.t0

    # -------------------------------------------------------------- 20 Hz
    def on_timer(self):
        if not self.have_odom or self.finished:
            return

        t = self.sim_time()

        # step 1: waypoint sequencing (advance when within tolerance)
        self.mission.update(self.x, self.y)
        if self.mission.done:
            self.stop()
            self.finished = True
            self.publish_status("MISSION COMPLETE - all waypoints reached")
            self.get_logger().info("mission complete")
            return

        goal = self.mission.current_goal()

        # steps 2 & 3: field + look-ahead + planner fallback -> (v, w)
        out = self.core.compute(self.x, self.y, self.yaw, goal, t)

        cmd = Twist()
        cmd.linear.x = float(out["v"])
        cmd.angular.z = float(out["w"])
        self.cmd_pub.publish(cmd)

        idx = self.mission.idx + 1
        n = len(self.mission.waypoints)
        self.publish_status(
            f"waypoint {idx}/{n}  mode={out['mode']}  "
            f"dist={out['dist_goal']:.2f}m")
        self.publish_markers(goal, out)

    # ------------------------------------------------------------- helpers
    def stop(self):
        self.cmd_pub.publish(Twist())

    def publish_status(self, text):
        if text != self._last_status:
            self.status_pub.publish(String(data=text))
            self._last_status = text

    def publish_markers(self, goal, out):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def base(mid, mtype):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "nav"
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        # current goal (red sphere)
        g = base(0, Marker.SPHERE)
        g.pose.position.x = float(goal[0])
        g.pose.position.y = float(goal[1])
        g.pose.position.z = 0.25
        g.scale.x = g.scale.y = g.scale.z = 0.5
        g.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9)
        arr.markers.append(g)

        # planned path (blue line strip) when the planner is active
        path = out.get("path") or []
        line = base(1, Marker.LINE_STRIP)
        line.scale.x = 0.08
        line.color = ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.9)
        if len(path) >= 2:
            for px, py in path:
                line.points.append(Point(x=float(px), y=float(py), z=0.1))
        else:
            line.action = Marker.DELETE
        arr.markers.append(line)

        # robot footprint (green box at current pose)
        r = base(2, Marker.CUBE)
        r.pose.position.x = self.x
        r.pose.position.y = self.y
        r.pose.position.z = 0.1
        r.scale.x, r.scale.y, r.scale.z = 0.40, 0.30, 0.15
        r.color = ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.6)
        arr.markers.append(r)

        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
