import os
from glob import glob

from setuptools import setup

package_name = "waypoint_field_nav"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name, package_name + ".core"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Robotics Challenge",
    maintainer_email="dev@example.com",
    description="Autonomous potential-field waypoint navigation in Gazebo (ROS 2 Jazzy).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigator = waypoint_field_nav.navigator:main",
            "course_publisher = waypoint_field_nav.course_publisher:main",
            "generate_course = waypoint_field_nav.generate_course:main",
        ],
    },
)
