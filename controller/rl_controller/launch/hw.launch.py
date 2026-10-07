import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import OpaqueFunction, DeclareLaunchArgument
import xacro


def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration("robot").perform(context)
    # Same namespace as hardware_bridge.launch.py. The estimator and MPX use relative
    # topics, so they follow it (the keyboard and MPX must run in the same namespace).
    nn = LaunchConfiguration("namespace").perform(context)
    command_timeout = float(LaunchConfiguration("command_timeout").perform(context))

    robot_xacro_path = os.path.join(
        get_package_share_directory(robot_name + "_description"),
        "xacro",
        "robot.xacro",
    )

    robot_description = xacro.process_file(
        robot_xacro_path, mappings={"hw_env": "hw"}
    ).toxml()

    robot_controllers = os.path.join(
        get_package_share_directory("rl_controller"),
        "config",
        robot_name,
        "controllers.yaml",
    )
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="both",
        parameters=[{"robot_description": robot_description}, robot_controllers],
        # remappings=[("~/robot_description", "/tita/robot_description")],
        namespace=nn,
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        # output="both",
        executable="robot_state_publisher",
        parameters=[
            {"robot_description": robot_description},
            {"frame_prefix": nn + "/"},
        ],
        namespace=nn,
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            nn + "/controller_manager",
        ],
    )

    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager",
            nn + "/controller_manager",
        ],
    )
    rl_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            robot_name + "_rl_controller",
            "--controller-manager",
            nn + "/controller_manager",
        ],
    )
    nodes = [
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        rl_controller_spawner,
    ]

    # Lost WiFi / keyboard / remote: fold the robot instead of keeping the last command.
    # Needs this launch to survive the disconnection (tmux, nohup or systemd, not a bare ssh).
    if command_timeout > 0:
        nodes.append(Node(
            package="rl_controller", executable="command_watchdog", output="screen",
            namespace=nn, parameters=[{"timeout": command_timeout}],
        ))

    # TITA only (the filter is written for its 8 joints and wheels). Always runs, as in
    # sim_gazebo.launch.py, so the filter can be checked on the robot without MPX.
    if robot_name == "tita":
        nodes.append(Node(
            package="tita_state_estimator", executable="state_estimator_node", output="screen",
            namespace=nn,
        ))

    # MPX is not started here: run it by hand in another terminal, in the same namespace
    # (mpx_node_cpp with base_state_source: filter, see MPX_INTEGRATION_CHANGES.md).

    return nodes


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot",
            default_value="tita",
            description="Path to the robot description file",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "command_timeout",
            default_value="1.0",
            description="s without commands (keyboard/remote twist) before transform_down; 0 disables",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "namespace",
            # Empty like the MPX nodes and the keyboard started with ros2 run (relative topics).
            default_value="",
            description="namespace of robot",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
