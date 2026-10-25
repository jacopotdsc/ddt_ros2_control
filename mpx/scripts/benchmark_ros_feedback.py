#!/usr/bin/env python3
"""Synthetic feedback for the isolated ROS benchmark; no hardware access."""
import argparse
from pathlib import Path
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray
from mpx_benchmark import robot_description
import mpx_cpp_solver as backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params-file', required=True)
    parser.add_argument('--joint-rate', type=float, default=500.,
                        help='synthetic joint_states rate, Hz (controller manager: 500)')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.params_file).read_text())['/**']['ros__parameters']
    inputs, outputs = cfg['inputs'], cfg['outputs']
    rclpy.init()
    node = Node('feedback', namespace='/mpx/benchmark')
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    fast = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    description = node.create_publisher(String, inputs['robot_description_topic'], latched)
    fsm = node.create_publisher(String, inputs['fsm_state_topic'], latched)
    mode = node.create_publisher(String, inputs['mode_command_topic'], latched)
    joints = node.create_publisher(JointState, inputs['joint_states_topic'], fast)
    base = node.create_publisher(Odometry, inputs['filtered_state_topic'], fast)
    description.publish(String(data=robot_description()))
    fsm.publish(String(data='transform_up'))
    mode.publish(String(data='mpc'))
    active = False
    plan_ready = False

    def on_plan(msg):
        nonlocal plan_ready
        plan_ready = len(msg.data) >= 5 and msg.data[4] == 1

    def on_wbc(msg):
        nonlocal active
        if plan_ready and not active and msg.effort:
            active = True
            fsm.publish(String(data='mpc'))

    node.create_subscription(Float64MultiArray, outputs['trajectory_topic'], on_plan, fast)

    node.create_subscription(JointState, outputs['joint_command_topic'], on_wbc, fast)
    stamp_ns = node.get_clock().now().nanoseconds
    period_ns = round(1e9 / args.joint_rate)

    def feedback():
        nonlocal stamp_ns
        # Exact synthetic sampling intervals, independent of Python timer jitter.
        stamp_ns += period_ns
        stamp = rclpy.time.Time(nanoseconds=stamp_ns).to_msg()
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = list(backend.config.joints_name)
        joint.position = [float(v) for v in backend.config.q0]
        joint.velocity = [0.] * len(joint.name)
        joint.effort = [0.] * len(joint.name)
        state = Odometry()
        state.header.stamp = stamp
        state.header.frame_id = 'odom'
        state.child_frame_id = 'base_link'
        state.pose.pose.position.z = float(backend.config.robot_height)
        state.pose.pose.orientation.w = 1.
        base.publish(state)
        joints.publish(joint)

    node.create_timer(1. / args.joint_rate, feedback)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
