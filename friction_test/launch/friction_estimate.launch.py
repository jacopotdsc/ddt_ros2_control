#!/usr/bin/env python
"""Gazebo with the TITA hung from a fixed hook (wheels off the ground), for joint friction tests.

Spawns only the robot and the same ros2_control controllers as sim_gazebo.launch.py (joint_state and IMU
broadcasters, tita_rl_controller with its controllers.yaml). The FSM starts in idle (legs hanging, only
damping); the test node (friction_trajectory_node.py), started by hand, drives the FSM state joint_cmd.

  ros2 launch friction_test friction_estimate.launch.py              # robot_friction.xacro
  ros2 launch friction_test friction_estimate.launch.py friction:=false   # robot.xacro, no friction
  ros2 launch friction_test friction_estimate.launch.py noise:=true       # sensor noise measured on the real robot
Sensor noise (noise:=true): values of tita_description/xacro/sensor_noise.xacro, applied by the Gazebo bridge.
"""
import os
import sys
import tempfile

import launch
import xacro
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def expand(xacro_path, noise="false"):
    """Expanded URDF, as sim_gazebo.launch.py (tita_rl_controller and its controllers.yaml)."""
    description = xacro.process_file(
        xacro_path,
        mappings={"hw_env": "gazebo", "sim_env": "gazebo", "ctrl_mode": "wbc", "yaml_path": "gazebo_bridge",
                  "sensor_noise": noise},
    ).toxml()
    description = description.replace(
        "package://tita_description", "file://" + get_package_share_directory("tita_description"))
    return description.replace(
        get_package_share_directory("gazebo_bridge") + "/config/controllers.yaml",
        get_package_share_directory("rl_controller") + "/config/tita/controllers.yaml",
    )


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("friction_test")
    friction = LaunchConfiguration("friction").perform(context).lower() == "true"
    xacro_path = os.path.join(share, "urdf", "robot_friction.xacro" if friction else "robot.xacro")
    noise = LaunchConfiguration("noise").perform(context).lower()
    if noise not in ("true", "false"):
        raise RuntimeError(f"noise:={noise}: use true or false")
    robot_description = expand(xacro_path, noise)
    fd, urdf_path = tempfile.mkstemp(prefix="tita_friction_test_", suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(robot_description)

    # Same world as the MPX simulation (500 Hz, 0.002 s, ODE).
    world_file = os.path.join(get_package_share_directory("gazebo_bridge"), "worlds", "empty_world.world")
    gazebo_ros_launch = os.path.join(get_package_share_directory("gazebo_ros"), "launch")
    nodes = [
        launch.actions.LogInfo(msg=f"friction_test: Gazebo simulates {xacro_path} (hung), sensor noise {noise}"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros_launch, "gzserver.launch.py")),
            launch_arguments={"world": world_file, "pause": "false", "verbose": "false",
                              "params_file": os.path.join(share, "config", "gazebo.yaml")}.items(),
        ),
    ]
    if LaunchConfiguration("gui").perform(context).lower() == "true":
        # Same GUI setup as sim_gazebo.launch.py (lower priority, render rate capped).
        render_rate_plugin = os.path.join(get_package_prefix("gazebo_bridge"), "lib", "libtita_gui_render_rate.so")
        sys.path.insert(0, get_package_share_directory("gazebo_ros"))
        from scripts import GazeboRosPaths
        model, plugin, media = GazeboRosPaths.get_paths()
        env = {"TITA_GUI_FPS": "20"}
        for key, value in (("GAZEBO_MODEL_PATH", model), ("GAZEBO_PLUGIN_PATH", plugin),
                           ("GAZEBO_RESOURCE_PATH", media)):
            env[key] = value + (os.pathsep + os.environ[key] if key in os.environ else "")
        nodes.append(launch.actions.ExecuteProcess(
            cmd=["nice", "-n", "10", "gzclient", "--gui-client-plugin=libgazebo_ros_eol_gui.so",
                 "-g", render_rate_plugin],
            additional_env=env, output="screen",
        ))
    nodes += [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[{"robot_description": robot_description}, {"use_sim_time": True},
                        {"publish_frequency": 15.0}],
        ),
        # Height comes from the fixed hook joint in the URDF (world -> base_link).
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=["-file", urdf_path, "-entity", "tita", "-x", "0.", "-y", "0.", "-z", "0."],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "imu_sensor_broadcaster", "tita_rl_controller",
                       "--controller-manager", "/controller_manager"],
        ),
    ]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            "friction", default_value="true",
            description="true: robot_friction.xacro (joint friction), false: robot.xacro (none)"),
        launch.actions.DeclareLaunchArgument("gui", default_value="true", description="Start gzclient"),
        launch.actions.DeclareLaunchArgument(
            "noise", default_value="false",
            description="true: sensor noise of the Gazebo bridge, as measured on the real robot"),
        OpaqueFunction(function=launch_setup),
    ])
