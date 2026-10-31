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
    
    watchdog_node = Node(
        package="rl_controller",
        executable="command_watchdog",
        output="screen",
        parameters=[os.path.join(get_package_share_directory("rl_controller"), "config", "watchdog.yaml")],
        namespace=nn,
    )
    
    state_estimator_node = Node(
        package="tita_state_estimator",
        executable="state_estimator_node",
        output="screen",
        namespace=nn,
    )

    fsm_audio_node = Node(
        package="fsm_audio",
        executable="fsm_audio_node",
        output="screen",
        namespace=nn,
    )
    
    nodes = [
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        rl_controller_spawner,
        watchdog_node,
        state_estimator_node,
        fsm_audio_node,
    ]

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
            "namespace",
            # Empty like the MPX nodes and the keyboard started with ros2 run (relative topics).
            default_value="",
            description="namespace of robot",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
