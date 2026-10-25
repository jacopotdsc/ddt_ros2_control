#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration("backend")
    use_cpp = IfCondition(PythonExpression(["'", backend, "' == 'cpp'"]))
    use_python = IfCondition(PythonExpression(["'", backend, "' == 'python'"]))
    source = {"inputs.base_state_source": LaunchConfiguration("base_state_source")}
    # The C++ nodes load mpx_node.yaml and tita/mpx.yaml by themselves
    # (include/mpx/default_params.hpp); the Python node needs both files.
    python_params = [
        os.path.join(get_package_share_directory("mpx"), "config", "mpx_node.yaml"),
        os.path.join(get_package_share_directory("rl_controller"), "config", "tita", "mpx.yaml"),
        source,
    ]
    return LaunchDescription([
        DeclareLaunchArgument(
            "backend", default_value="cpp", choices=["cpp", "python"],
            description="cpp: mpx_node_cpp + mpx_wbc_node; python: mpx_node_python.py",
        ),
        DeclareLaunchArgument(
            "base_state_source", default_value="filter",
            choices=["filter", "ground_truth", "odometry"],
            description="Base pose/twist source (ground_truth: Gazebo only)",
        ),
        # Only the chosen backend starts; both publish mpx/joint_command.
        Node(package="mpx", executable="mpx_node_cpp", output="screen",
             parameters=[source], condition=use_cpp),
        Node(package="mpx", executable="mpx_wbc_node", output="screen",
             parameters=[source], condition=use_cpp),
        Node(package="mpx", executable="mpx_node_python.py", output="screen",
             parameters=python_params, condition=use_python),
    ])
