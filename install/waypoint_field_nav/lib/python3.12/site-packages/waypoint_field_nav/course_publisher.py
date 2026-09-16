#!/usr/bin/env python3
"""course_publisher - static course visualization for RViz.

Loads the same course manifest the navigator uses and publishes a latched
(transient-local) MarkerArray showing every obstacle and every waypoint, so
you can see the whole problem in RViz even without Gazebo's rendering. Purely
cosmetic - it publishes no commands and is not part of the control loop.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from waypoint_field_nav.core.course_gen import load_course


class CoursePublisher(Node):
    def __init__(self):
        super().__init__("course_publisher")
        self.declare_parameter("course_file", "")
        self.declare_parameter("frame_id", "odom")

        course_file = self.get_parameter("course_file").value
        self.frame_id = self.get_parameter("frame_id").value
        if not course_file:
            self.get_logger().fatal("no 'course_file' parameter set")
            raise SystemExit(1)

        course = load_course(course_file)
        self.obstacles = course["obstacles"]
        self.waypoints = course["waypoints"]

        # latched QoS so RViz sees the markers whenever it connects
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(MarkerArray, "course/markers", qos)

        self.publish_once()
        # re-publish periodically as a belt-and-braces for late subscribers
        self.create_timer(2.0, self.publish_once)
        self.get_logger().info(
            f"course_publisher up: {len(self.obstacles)} obstacles, "
            f"{len(self.waypoints)} waypoints")

    def publish_once(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        mid = 0

        for obs in self.obstacles:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "obstacles"
            m.id = mid
            mid += 1
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x = float(obs["x"])
            m.pose.position.y = float(obs["y"])
            m.pose.position.z = 0.5
            m.color = ColorRGBA(r=0.6, g=0.3, b=0.3, a=0.8)
            if obs["type"] == "circle":
                m.type = Marker.CYLINDER
                d = 2.0 * float(obs["radius"])
                m.scale.x = m.scale.y = d
                m.scale.z = 1.0
            else:  # box (axis-aligned half-extents)
                m.type = Marker.CUBE
                m.scale.x = 2.0 * float(obs["hx"])
                m.scale.y = 2.0 * float(obs["hy"])
                m.scale.z = 1.0
            arr.markers.append(m)

        for i, wp in enumerate(self.waypoints):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "waypoints"
            m.id = mid
            mid += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x = float(wp[0])
            m.pose.position.y = float(wp[1])
            m.pose.position.z = 0.5
            m.scale.x = m.scale.y = 0.3
            m.scale.z = 1.0
            m.color = ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.5)
            arr.markers.append(m)

            # number label floating above each waypoint
            t = Marker()
            t.header.frame_id = self.frame_id
            t.header.stamp = stamp
            t.ns = "waypoint_labels"
            t.id = mid
            mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(wp[0])
            t.pose.position.y = float(wp[1])
            t.pose.position.z = 1.3
            t.pose.orientation.w = 1.0
            t.scale.z = 0.5
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            t.text = str(i + 1)
            arr.markers.append(t)

        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = CoursePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
