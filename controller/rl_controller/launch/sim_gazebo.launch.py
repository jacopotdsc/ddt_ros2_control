#!/usr/bin/env python
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import xacro
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import OpaqueFunction


def launch_setup(context, *args, **kwargs):
    robot_name = LaunchConfiguration("robot").perform(context)
    # Same namespace argument as hw.launch.py (empty by default, like the MPX nodes).
    ns = LaunchConfiguration("namespace").perform(context)
    # Gazebo entity name: also the model MPX looks up in /model_states (inputs.model_name).
    entity = ns.strip("/") or robot_name

    # Share sim_bringup physics settings (500 Hz, 0.002 s, 300 ODE iterations).
    world_file = os.path.join(
        FindPackageShare("gazebo_bridge").find("gazebo_bridge"),
        "worlds",
        "empty_world.world",
    )

    world = ET.parse(world_file)
    physics = world.find("./world/physics")
    rtf = float(LaunchConfiguration("rtf").perform(context))
    if rtf != 1.0:
        # Same world, slower wall clock: MPX (not in lockstep with Gazebo) gets more
        # wall time per simulated control period. Controllers see the same sim time.
        rate = physics.find("real_time_update_rate")
        rate.text = f"{float(rate.text) * rtf:g}"
        world_file = os.path.join(tempfile.gettempdir(), f"tita_empty_world_rtf{rtf:g}.world")
        world.write(world_file)
    physics_info = launch.actions.LogInfo(msg=(
        f"Shared Gazebo world: {world_file}; "
        f"physics={physics.get('type')}, "
        f"step={physics.findtext('max_step_size')} s, "
        f"rate={physics.findtext('real_time_update_rate')} Hz, "
        f"ODE iterations={physics.findtext('ode/solver/iters')}"
    ))

    gazebo_ros_launch = os.path.join(get_package_share_directory("gazebo_ros"), "launch")
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_launch, "gzserver.launch.py")),
        launch_arguments={"world": world_file, "pause": "false", "verbose": "false"}.items(),
    )
    gazebo = [gzserver]
    if LaunchConfiguration("gui").perform(context).lower() == "true":
        # The GUI competes with gzserver and MPX for the CPU: lower its priority and
        # cap its render rate (gazebo_bridge plugin). Physics rates are unchanged.
        gui_fps = LaunchConfiguration("gui_fps").perform(context)
        gui_nice = LaunchConfiguration("gui_nice").perform(context)
        render_rate_plugin = os.path.join(
            get_package_prefix("gazebo_bridge"), "lib", "libtita_gui_render_rate.so")
        sys.path.insert(0, get_package_share_directory("gazebo_ros"))
        from scripts import GazeboRosPaths  # same paths as gazebo_ros gzclient.launch.py
        model, plugin, media = GazeboRosPaths.get_paths()
        env = {"TITA_GUI_FPS": gui_fps}
        for key, value in (("GAZEBO_MODEL_PATH", model), ("GAZEBO_PLUGIN_PATH", plugin),
                           ("GAZEBO_RESOURCE_PATH", media)):
            env[key] = value + (os.pathsep + os.environ[key] if key in os.environ else "")
        gazebo.append(launch.actions.ExecuteProcess(
            # -g loads a gzclient system plugin; --gui-client-plugin is only for overlay widgets.
            cmd=["nice", "-n", gui_nice, "gzclient",
                 "--gui-client-plugin=libgazebo_ros_eol_gui.so", "-g", render_rate_plugin],
            additional_env=env, output="screen",
        ))

    description_share = get_package_share_directory(robot_name + "_description")
    # TITA uses a robot subdirectory; the DDT descriptions use xacro/ directly.
    robot_xacro_path = os.path.join(description_share, robot_name, "xacro", "robot.xacro")
    if not os.path.isfile(robot_xacro_path):
        robot_xacro_path = os.path.join(description_share, "xacro", "robot.xacro")

    def expand(xacro_path):
        description = xacro.process_file(
            xacro_path,
            mappings={
                "hw_env": "gazebo",
                "sim_env": "gazebo",
                "ctrl_mode": "wbc",
                "yaml_path": "gazebo_bridge",
            },
        ).toxml()

        # Replace package:// URIs for all *_description packages
        description_packages = ["d1_description", "d1h_description", "tita_description", "titatit_description"]
        for desc_pkg in description_packages:
            package_uri = "package://" + desc_pkg
            if package_uri in description:
                description = description.replace(
                    package_uri,
                    "file://" + get_package_share_directory(desc_pkg),
                )

        return description.replace(
            get_package_share_directory("gazebo_bridge")+ "/config/controllers.yaml",
            get_package_share_directory("rl_controller")+ "/config/" + robot_name + "/controllers.yaml",
        )

    # Controller model (MPX, state estimator, robot_state_publisher): robot.xacro, as on the robot.
    robot_description = expand(robot_xacro_path)
    # Plant model simulated by Gazebo: robot_gazebo.xacro next to robot.xacro when present (a copy
    # of robot.xacro changed step by step towards the real robot), else the controller model.
    gazebo_xacro_path = os.path.join(os.path.dirname(robot_xacro_path), "robot_gazebo.xacro")
    if os.path.isfile(gazebo_xacro_path):
        fd, gazebo_urdf_path = tempfile.mkstemp(prefix="tita_gazebo_", suffix=".urdf")
        with os.fdopen(fd, "w") as f:
            f.write(expand(gazebo_xacro_path))
        spawn_source = ["-file", gazebo_urdf_path]
        plant = gazebo_xacro_path
    else:
        spawn_source = ["-topic", f"{ns}/robot_description"]
        plant = robot_xacro_path
    models_info = launch.actions.LogInfo(
        msg=f"Gazebo simulates {plant}; controllers use robot_description from {robot_xacro_path}")

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            *spawn_source,
            "-entity",
            entity,
            "-robot_namespace",
            f"{ns}",
            "-x",
            "0.",
            "-y",
            "0.",
            "-z",
            "0.65",
        ],
        output="screen",
    )


    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": True},
            {"publish_frequency": 15.0},
            {"frame_prefix": ns + "/"},
        ],
        namespace=ns,
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            ns + "/controller_manager",
        ],
    )

    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager",
            ns + "/controller_manager",
        ],
    )
    rl_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            robot_name + "_rl_controller",
            "--controller-manager",
            ns + "/controller_manager",
        ],
    )
    # Lost keyboard / remote: fold the robot, as in hw.launch.py.
    watchdog_node = Node(
        package="rl_controller",
        executable="command_watchdog",
        output="screen",
        parameters=[os.path.join(
            get_package_share_directory("rl_controller"), "config", "watchdog.yaml")],
        namespace=ns,
    )
    # Always runs, so it can be compared with /model_states even when MPX uses the
    # ground truth (written for TITA: its 8 joints and wheels).
    state_estimator_node = Node(
        package="tita_state_estimator",
        executable="state_estimator_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        namespace=ns,
    )
    nodes = [
        physics_info,
        models_info,
        robot_state_pub_node,
        *gazebo,
        spawn_entity,
        joint_state_broadcaster_spawner,
        imu_sensor_broadcaster_spawner,
        rl_controller_spawner,
        #watchdog_node,
        state_estimator_node,
    ]

    # MPX is not started here, as on the robot: run mpx_node_cpp and mpx_wbc_node
    # by hand (base from the filter; ground truth with
    # --ros-args -p inputs.base_state_source:=ground_truth on MPC and WBC).
    return nodes


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "robot",
            default_value="tita",
            description="Path to the robot description file",
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="namespace of robot",
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "gui", default_value="true", description="Start the Gazebo GUI"
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "gui_fps", default_value="20",
            description="GUI render rate limit (FPS), not simulation Hz",
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "gui_nice", default_value="10",
            description="nice level of gzclient (0-19), so the GUI yields CPU to gzserver and MPX",
        )
    )
    declared_arguments.append(
        launch.actions.DeclareLaunchArgument(
            "rtf", default_value="1.0",
            description="Target real-time factor. Below 1 MPX gets more wall time per control "
                        "period (useful with the GUI or a loaded CPU)",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
