#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("mpx"), "config", "mpx_node.yaml")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file", default_value=default_params,
            description="YAML with every mpx_node parameter",
        ),
        DeclareLaunchArgument(
            "mpx_implementation", default_value="python", choices=["python", "cpp"],
            description="ROS front end; the numerical JAX solver is shared",
        ),
        Node(
            package="mpx",
            executable=PythonExpression([
                "'mpx_node_cpp' if '", LaunchConfiguration("mpx_implementation"),
                "' == 'cpp' else 'mpx_node.py'",
            ]),
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
        # Hierarchical MPC, WBC and LLC run in separate processes.
        Node(
            package="mpx",
            executable="mpx_wbc_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("mpx_implementation"), "' == 'cpp'"])),
        ),
        Node(
            package="mpx",
            executable="mpx_llc_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("mpx_implementation"), "' == 'cpp'"])),
        ),
    ])
