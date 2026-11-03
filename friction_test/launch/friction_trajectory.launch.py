#!/usr/bin/env python
"""Opens friction_test_node in its own terminal window (live keys and interface).

  ros2 launch friction_test friction_trajectory.launch.py

Same as running `ros2 run friction_test friction_test_node` in a terminal. Start it after
friction_estimate.launch.py; do not run keyboard_controller as well (both publish command/cmd_key).
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="friction_test",
            executable="friction_test_node",
            # New window with a real terminal: the node reads single keys from it. --wait keeps the
            # launch attached until the window is closed.
            prefix="gnome-terminal --wait --title=friction_test --",
            output="screen",
        ),
    ])
