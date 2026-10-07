import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    ns = arg("namespace")
    return [Node(
        package="tita_rerun_logger",
        executable="rerun_logger",
        name="tita_rerun_logger",
        namespace=ns,
        output="screen",
        # Lower priority than MPX, gzserver and the controllers: under CPU contention the
        # logger may drop samples (queues of 50 messages), the controller keeps its rate.
        prefix=f"nice -n {arg('nice')}",
        parameters=[
            arg("config"),
            {
                "use_sim_time": arg("use_sim_time").lower() == "true",
                # Same Gazebo entity as sim_gazebo.launch.py: the namespace or "tita"
                "model_name": ns.strip("/") or "tita",
                "open_viewer": arg("open_viewer").lower() == "true",
                "viewer_url": arg("viewer_url"),
                "save_rrd": arg("save_rrd").lower() == "true",
                "rrd_dir": arg("rrd_dir"),
            },
        ],
    )]


def generate_launch_description():
    default_cfg = os.path.join(
        get_package_share_directory("tita_rerun_logger"), "config", "rerun_logger.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_cfg),
        DeclareLaunchArgument("namespace", default_value="",
                              description="robot namespace (same as the sim/hw launch)"),
        DeclareLaunchArgument("nice", default_value="10",
                              description="nice level of the logger (0-19)"),
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="true in Gazebo, false on the robot"),
        DeclareLaunchArgument("open_viewer", default_value="false",
                              description="also stream live to a Rerun viewer"),
        DeclareLaunchArgument("viewer_url", default_value="rerun+http://127.0.0.1:9876/proxy",
                              description="viewer address, e.g. rerun+http://<IP>:9876/proxy"),
        DeclareLaunchArgument("save_rrd", default_value="true"),
        DeclareLaunchArgument("rrd_dir", default_value="",
                              description='"" = src/tita_rerun_logger/rerun_log'),
        OpaqueFunction(function=launch_setup),
    ])
