#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    cpp = PythonExpression(["'", LaunchConfiguration("mpx_implementation"), "' == 'cpp'"])
    # The C++ nodes load mpx_node.yaml and rl_controller's tita/mpx.yaml by themselves
    # (include/mpx/default_params.hpp): passing mpx_node.yaml again here would override the
    # robot overlay (relative topics, FSM mode). mpx_node_python.py has no defaults: it gets
    # both.
    source = {"inputs.base_state_source": LaunchConfiguration("base_state_source")}
    python_params = [
        os.path.join(get_package_share_directory("mpx"), "config", "mpx_node.yaml"),
        os.path.join(get_package_share_directory("rl_controller"), "config", "tita", "mpx.yaml"),
        source,
    ]
    return LaunchDescription([
        DeclareLaunchArgument(
            "mpx_implementation", default_value="cpp", choices=["python", "cpp"],
            description="cpp: MPC, WBC and LLC nodes; python: the single mpx_node_python.py",
        ),
        DeclareLaunchArgument(
            "base_state_source", default_value="filter",
            choices=["filter", "ground_truth", "odometry"],
            description="Base pose/twist source (ground_truth: Gazebo only)",
        ),
        # Hierarchical MPC, WBC and LLC run in separate processes.
        Node(package="mpx", executable="mpx_node_cpp", output="screen",
             parameters=[source], condition=IfCondition(cpp)),
        Node(package="mpx", executable="mpx_wbc_node", output="screen",
             parameters=[source], condition=IfCondition(cpp)),
        Node(package="mpx", executable="mpx_llc_node", output="screen",
             condition=IfCondition(cpp)),
        Node(package="mpx", executable="mpx_node_python.py", output="screen",
             parameters=python_params, condition=UnlessCondition(cpp)),
    ])
