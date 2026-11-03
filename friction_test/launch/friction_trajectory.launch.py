#!/usr/bin/env python
"""Opens friction_test_node in its own terminal window (live keys and interface).

  ros2 launch friction_test friction_trajectory.launch.py                    # hardware, after hw.launch.py
  ros2 launch friction_test friction_trajectory.launch.py use_sim_time:=true # Gazebo, after friction_estimate.launch.py

Same as running `ros2 run friction_test friction_test_node` in a terminal. Do not run
keyboard_controller as well (both publish command/cmd_key).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="false: hardware clock (default); true: Gazebo clock"),
        DeclareLaunchArgument("namespace", default_value="",
                              description="Same namespace as the robot controllers"),
        Node(
            package="friction_test",
            executable="friction_test_node",
            namespace=LaunchConfiguration("namespace"),
            parameters=[{"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}],
            # New window with a real terminal: the node reads single keys from it. --wait keeps the
            # launch attached until the window is closed.
            prefix="gnome-terminal --wait --title=friction_test --",
            output="screen",
        ),
    ])
