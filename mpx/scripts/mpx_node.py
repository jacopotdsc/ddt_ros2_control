#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent / "python"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "mpx_tita"))

# The TITA controller uses small, latency-sensitive solves (n_env=1). Force JAX
# to initialize only the CPU backend; this must be set before importing JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
# Small matrices, one robot: avoid BLAS/Eigen thread-pool contention with Gazebo.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import gc
import time
import threading
import signal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import rclpy

from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Twist, Vector3Stamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from mpx.utils.ros_model import model_from_description

import mpx.config.config_tita as config
from mpx.utils.mpc_wrapper_dfcip import BatchedMPCControllerWrapper

jax.config.update("jax_enable_x64", True)
# This node already owns a dedicated solver thread; a second dispatch queue
# adds latency for these small CPU solves.
jax.config.update("jax_cpu_enable_async_dispatch", False)

# Max rate (m/s) at which the WBC COM height target moves toward the command.
# The WBC tracks the target directly, so a step would kick the robot.
HEIGHT_RAMP_RATE = 0.1

# Solves on the live state before the first published torque of every activation
# (mode-controlled only; the previous controller keeps the robot meanwhile). The first
# live solves after compilation are slow, and a late torque at takeover shakes the legs.
WARMUP_STEPS = 10


def rotation_matrix(quaternion):
    """Return the world-from-body matrix for a ROS xyzw quaternion."""
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class EstimatePublisher:
    """Publishes the state MPX believes (CoM, CoM velocity, wheel centres, base RPY).

    Designed to stay off the control path: the solver thread only calls
    submit(), which stores a reference under a lock and sets an event (no
    message building, no ROS calls). A separate daemon thread builds and
    publishes the messages. Only the latest sample is kept: if publishing
    ever falls behind, samples are dropped, never queued.
    """

    def __init__(self, node, prefix, frame_id="world"):
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._frame = frame_id
        self._com = node.create_publisher(PointStamped, f"{prefix}/com", qos)
        self._com_vel = node.create_publisher(Vector3Stamped, f"{prefix}/com_vel", qos)
        self._rpy_pub = node.create_publisher(Vector3Stamped, f"{prefix}/base_rpy", qos)
        self._wheels = [
            node.create_publisher(PointStamped, f"{prefix}/left_wheel", qos),
            node.create_publisher(PointStamped, f"{prefix}/right_wheel", qos),
        ]
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending = None
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="mpx-estimate-pub", daemon=True
        )
        self._thread.start()

    def submit(self, stamp, com, com_vel, wheel_centers, base_quat_wxyz):
        """Called by the solver thread: O(1), no allocation beyond a tuple."""
        with self._lock:
            self._pending = (stamp, com, com_vel, wheel_centers, base_quat_wxyz)
        self._event.set()

    def stop(self):
        self._stop = True
        self._event.set()
        self._thread.join(timeout=1.0)

    def _vector(self, stamp, v):
        msg = Vector3Stamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame
        msg.vector.x, msg.vector.y, msg.vector.z = float(v[0]), float(v[1]), float(v[2])
        return msg

    @staticmethod
    def _rpy(q):
        """Roll, pitch, yaw [rad] from a MuJoCo wxyz quaternion (ZYX convention)."""
        w, x, y, z = (float(v) for v in q)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

    def _point(self, stamp, v):
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame
        msg.point.x, msg.point.y, msg.point.z = float(v[0]), float(v[1]), float(v[2])
        return msg

    def _run(self):
        while True:
            self._event.wait()
            self._event.clear()
            if self._stop:
                return
            with self._lock:
                item, self._pending = self._pending, None
            if item is None:
                continue
            stamp, com, com_vel, wheel_centers, quat = item
            try:
                self._com.publish(self._point(stamp, com))
                self._com_vel.publish(self._vector(stamp, com_vel))
                self._rpy_pub.publish(self._vector(stamp, self._rpy(quat)))
                for pub, center in zip(self._wheels, wheel_centers):
                    pub.publish(self._point(stamp, center))
            except Exception:  # noqa: BLE001
                # Diagnostics must never affect control (e.g. during shutdown).
                pass


class MPXNode(Node):
    def __init__(self):
        # Parameters come from config/mpx_node.yaml; there are no defaults in code.
        super().__init__(
            "controller", namespace="mpx",
            automatically_declare_parameters_from_overrides=True,
        )

        self.get_logger().info(
            f"JAX backend: {jax.default_backend()} ({jax.devices()})"
        )

        self._mode_controlled = bool(self.get_parameter("inputs.mode_command_topic").value)
        self._mode_enabled = not self._mode_controlled
        self.h_mpc = self.get_parameter("mpc_rl_handoff.h_mpc").value
        self.h_rl = self.get_parameter("mpc_rl_handoff.h_rl").value
        self.rl_modes = list(self.get_parameter("mpc_rl_handoff.rl_modes").value)
        self.handoff_height_tol = self.get_parameter("mpc_rl_handoff.height_tol").value
        self.handoff_vz_tol = self.get_parameter("mpc_rl_handoff.vz_tol").value
        self.handoff_settle_ns = self.get_parameter("mpc_rl_handoff.settle_time").value * 1e9
        self.handoff_timeout = self.get_parameter("mpc_rl_handoff.timeout").value

        description = self.wait_for_description()
        self.model, self.config = model_from_description(description, config)
        self.data = mujoco.MjData(self.model)
        if self._mode_controlled:
            # The RL spawn posture is folded. It is not an upright MPC reference.
            # Keep the imported physical model, but use MPX's standing posture
            # for posture regularization, nominal wheel spacing, and warm-up.
            self.config.q0 = np.asarray(config.q0, dtype=float).copy()
            self.data.qpos[7:] = self.config.q0
            mujoco.mj_forward(self.model, self.data)
            contacts = [self.model.geom(name).id for name in self.config.contact_frame]
            self.config.d = float(np.linalg.norm(self.data.geom_xpos[contacts[0]] - self.data.geom_xpos[contacts[1]]))

        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY,
            self.config.base_body_name,
        )
        self.contact_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            for name in self.config.contact_frame
        ]

        self.joint_addresses = []
        for name in self.config.joints_name:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            self.joint_addresses.append((
                self.model.jnt_qposadr[joint_id],
                self.model.jnt_dofadr[joint_id],
            ))

        self.mpc = BatchedMPCControllerWrapper(self.config, n_env=1, model=self.model)
        self.mpc_state = self.mpc.init_state()
        self.theta_prev = np.asarray(0.0, dtype=np.float64)

        self._step_controller_jit = jax.jit(self.step_controller)
        self.period_ns = round(1e9 / self.config.mpc_frequency)
        self.model_name = self.get_parameter("inputs.model_name").value
        self._condition = threading.Condition()
        self._stop = False
        self._base_sample = None
        self._joint_sample = None
        self._command = (0.0, 0.0, 0.0, self.h_mpc)
        self._last_pose_z = None
        self._last_mode = None  # last mode command received
        self._fsm_state = None  # current RL FSM state (None: unknown, no fsm_state_topic)
        self._compiling = False
        self._activating = False  # "mpc" requested, COM not yet settled at the ramped height
        self._activation_start = None
        self._keys_locked = None  # last value published on keys_locked_topic
        self._generation = 0
        self._took_over = False  # an effort was published in this activation
        self._handoff_target = None  # requested rl_N / transform_down while lowering the COM
        self._handoff_start = None
        self._handoff_settled = None
        self._last_received_stamp = None
        self.initial_effort = None
        self._reported_nan_diagnostics = False
        self.last_control_error = 0.0

        # This node is never added to an executor. DDS keeps only the latest
        # feedback; the solver takes raw bytes and decodes them when needed.
        self._feedback_node = Node(
            "mpx_feedback", namespace=self.get_namespace(), context=self.context,
            use_global_arguments=False, start_parameter_services=False,
        )
        self._feedback_subscriptions = []
        feedback_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        base_state_source = self.get_parameter("inputs.base_state_source").value
        if base_state_source == "ground_truth":
            self._base_topic = self.get_parameter("inputs.model_states_topic").value
            self.create_feedback_subscription(
                ModelStates, self._base_topic,
                self.on_model_states, feedback_qos,
            )
        elif base_state_source in ("filter", "odometry"):
            self._base_topic = self.get_parameter("inputs.filtered_state_topic").value
            self.create_feedback_subscription(
                Odometry, self._base_topic,
                self.on_filtered_state, feedback_qos,
            )
        else:
            raise ValueError(
                "inputs.base_state_source must be 'ground_truth', 'filter' or 'odometry', "
                f"got '{base_state_source}'")
        # "odometry": the chassis odometry has no z. It gives x, y and the horizontal
        # velocity; the filter gives z, vz, orientation and angular velocity (IMU).
        self._odom_topic = None
        self._odom_sample = None  # (Odometry, time.monotonic())
        if base_state_source == "odometry":
            self._odom_topic = self.get_parameter("inputs.odometry_topic").value
            self.create_feedback_subscription(
                Odometry, self._odom_topic, self.on_chassis_odometry, feedback_qos,
                first=True)
        self.get_logger().info(f"Base state source: {base_state_source}")
        self._joints_topic = self.get_parameter("inputs.joint_states_topic").value
        self.create_feedback_subscription(
            JointState, self._joints_topic,
            self.on_joints, feedback_qos,
        )
        self.cmd_vel_topic = self.get_parameter("inputs.cmd_vel_topic").value
        self.create_subscription(Twist, self.cmd_vel_topic, self.on_command, 1)
        self.create_subscription(
            Float64, self.get_parameter("inputs.com_height_topic").value, self.on_com_height,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.default_command_pub = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.default_command_timer = self.create_timer(0.1, self.publish_default_command)
        if not self.get_parameter("behavior.publish_default_commands").value:
            self.default_command_timer.cancel()
        if self._mode_controlled:
            self.create_subscription(
                String, self.get_parameter("inputs.mode_command_topic").value, self.on_mode_command,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            )
        if self.get_parameter("inputs.fsm_state_topic").value:
            self.create_subscription(
                String, self.get_parameter("inputs.fsm_state_topic").value, self.on_fsm_state,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            )
        if self.get_parameter("inputs.pose_command_topic").value:
            self.create_subscription(
                PoseStamped, self.get_parameter("inputs.pose_command_topic").value,
                self.on_pose_command, 1,
            )
        self.effort_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("outputs.effort_command_topic").value, 1,
        )
        self.keys_locked_pub = self.create_publisher(
            Bool, self.get_parameter("outputs.keys_locked_topic").value,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.publish_keys_locked()
        self.handoff_pub = self.create_publisher(
            String, self.get_parameter("outputs.handoff_topic").value, 1,
        )
        self.estimate_pub = None
        if self.get_parameter("behavior.publish_estimate").value:
            self.estimate_pub = EstimatePublisher(
                self, prefix=self.get_parameter("outputs.estimate_topic_prefix").value)

        # Both cold compilation and a second call with returned JAX state must
        # finish before taking control. The live path has the same shapes and dtypes.
        self._precompile_pending = bool(self.get_parameter("behavior.precompile").value)
        if self._precompile_pending and not self._mode_controlled:
            self.precompile_controller()
            self._precompile_pending = False
        if self._mode_controlled:
            self.get_logger().info(
                "MPX standby: first key 4 will compile MPC. RL stays active while waiting; no timeout."
            )
        self._worker = threading.Thread(target=self.control_loop, name="mpx-solver")
        self._worker.start()
        status = "MPX standby" if self._mode_controlled else "MPX ready on CPU"
        self.get_logger().info(
            f"{status}: MPC + WBC {self.config.mpc_frequency} Hz in simulation time; "
            "ros2_control holds the latest effort at 500 Hz."
        )

    def wait_for_description(self):
        topic = self.get_parameter("inputs.robot_description_topic").value
        timeout = float(self.get_parameter("inputs.robot_description_timeout").value)
        if timeout <= 0:
            raise ValueError("robot_description_timeout must be positive")
        received = []
        subscription = self.create_subscription(
            String, topic, lambda msg: received.append(msg.data),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.get_logger().info(f"Waiting for robot description on {topic}")
        deadline = time.monotonic() + timeout
        try:
            while rclpy.ok() and not received and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0., deadline - time.monotonic())))
            if not received:
                raise RuntimeError(f"No robot description received on {topic} within {timeout}s")
            self.get_logger().info(f"Using robot description from {topic}")
            return received[0]
        finally:
            self.destroy_subscription(subscription)

    def create_feedback_subscription(self, msg_type, topic, callback, qos, first=False):
        # Resolve on the controller first, preserving its namespace/remappings.
        sub = self._feedback_node.create_subscription(
            msg_type, self.resolve_topic_name(topic), lambda _: None, qos, raw=True,
        )
        item = (sub, callback)
        if first:
            self._feedback_subscriptions.insert(0, item)
        else:
            self._feedback_subscriptions.append(item)

    def poll_feedback(self):
        """Take one latest sample per depth-one queue, without executor callbacks.

        Bound the work even if publishers run faster than the solver. DDS receipt
        time rejects a sample left in the queue during standby/compilation after
        its publisher stopped; taking it now must not make old data fresh again.
        """
        for sub, callback in self._feedback_subscriptions:
            with sub.handle:
                taken = sub.handle.take_message(sub.msg_type, True)
            if taken is None:
                continue
            data, info = taken
            received = info["received_timestamp"]
            if received > 0 and time.time_ns() - received >= 100_000_000:
                continue
            callback(deserialize_message(data, sub.msg_type))

    def on_model_states(self, msg):
        try:
            index = msg.name.index(self.model_name)
        except ValueError:
            return
        with self._condition:
            # ModelStates velocities are world-frame velocities. Store one
            # immutable message snapshot; transformations stay inside MPX.
            self._base_sample = (msg.pose[index], msg.twist[index], time.monotonic())
            self._condition.notify()

    def on_filtered_state(self, msg):
        # tita_state_estimator publishes the twist in the base frame; store it
        # world-frame like ModelStates, so update_mujoco_state is unchanged.
        o = msg.pose.pose.orientation
        world_from_body = rotation_matrix([o.x, o.y, o.z, o.w])
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = (
            float(c) for c in world_from_body @ np.array([v.x, v.y, v.z]))
        twist.angular.x, twist.angular.y, twist.angular.z = (
            float(c) for c in world_from_body @ np.array([w.x, w.y, w.z]))
        pose, stamp = msg.pose.pose, time.monotonic()
        with self._condition:
            if self._odom_topic is not None:
                if self._odom_sample is None:
                    return
                odom, odom_stamp = self._odom_sample
                pose = self.merge_odometry(pose, twist, world_from_body, odom)
                stamp = min(stamp, odom_stamp)  # stale if either source is stale
            self._base_sample = (pose, twist, stamp)
            self._condition.notify()

    def on_chassis_odometry(self, msg):
        with self._condition:
            self._odom_sample = (msg, time.monotonic())

    @staticmethod
    def merge_odometry(pose, twist, world_from_body, odom):
        """Filter pose/twist (world twist, modified in place) with x, y and the horizontal
        velocity from the chassis odometry. Its twist is planar, in the robot heading
        frame (nav_msgs: child frame), rotated to the world with the filter (IMU) yaw."""
        yaw = np.arctan2(world_from_body[1, 0], world_from_body[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        v = odom.twist.twist.linear
        twist.linear.x = float(c * v.x - s * v.y)
        twist.linear.y = float(s * v.x + c * v.y)
        merged = Pose()
        merged.position.x = odom.pose.pose.position.x
        merged.position.y = odom.pose.pose.position.y
        merged.position.z = pose.position.z
        merged.orientation = pose.orientation
        return merged

    def on_joints(self, msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self._condition:
            if self._last_received_stamp is not None and stamp < self._last_received_stamp:
                self._generation += 1  # Gazebo reset: discard the previous horizon.
                self._handoff_start = self._handoff_settled = None
            self._last_received_stamp = stamp
            self._joint_sample = (msg, time.monotonic(), stamp)
            self._condition.notify()

    def on_command(self, msg):
        with self._condition:
            if self.commands_blocked():
                return
            self._command = (
                msg.linear.x, msg.linear.z, msg.angular.z, self._command[3],
            )

    def commands_blocked(self):
        """Velocity/height commands are ignored while compiling for a pending "mpc"
        request, while moving to h_mpc after it, and during the handoff. Call with
        self._condition held."""
        return ((self._compiling and self._mode_enabled) or self._activating
                or self._handoff_target is not None)

    def publish_keys_locked(self, force=False):
        """Publish commands_blocked() on change, so the keyboard locks motion keys.
        force: publish anyway, e.g. to correct the lock the keyboard sets on a mode key."""
        with self._condition:  # held while publishing: keeps messages in order
            locked = self.commands_blocked()
            if force or locked != self._keys_locked:
                self._keys_locked = locked
                self.keys_locked_pub.publish(Bool(data=locked))

    def on_fsm_state(self, msg):
        with self._condition:
            self._fsm_state = msg.data
            # The FSM left "mpc" on its own (rl_controller torque guard): stop computing.
            stop = self._mode_enabled and self._took_over and self.fsm_blocks_mpc()
            if stop:
                self._mode_enabled = self._took_over = self._activating = False
                self._handoff_target = None
                self._generation += 1
                self._condition.notify_all()
        if stop:
            self.get_logger().warning(
                f"RL FSM switched from mpc to {msg.data} (rl_controller torque guard?): "
                "MPX standby")
            self.publish_keys_locked()

    def fsm_blocks_mpc(self):
        """True when the RL FSM is in a state that never enters "mpc" (idle, transform_down):
        MPX torques would not be applied. Call with self._condition held."""
        return self._fsm_state in ("idle", "transform_down")

    def on_mode_command(self, msg):
        # The keyboard republishes the mode every 10 ms: act on every message, but
        # confirm the lock state to it once per mode change.
        changed = msg.data != self._last_mode
        self._last_mode = msg.data
        self.apply_mode_command(msg.data)
        self.publish_keys_locked(force=changed)

    def apply_mode_command(self, mode):
        '''
        Handle mode change commands from external sources.
        This method updates the internal state of the node based on the requested mode.
        Supported modes include "mpc", "idle", "joint_pd", "transform_down", and the RL
        modes in mpc_rl_handoff.rl_modes (one per RL policy of the FSM).
        When switching to MPX mode, it goes smoothly to h_mpc. Before handing off to an RL
        mode or to transform_down it lowers the COM to h_rl at rest; idle and joint_pd are
        the immediate safety commands.
        '''
        # Ignore unsupported requests, just like the RL FSM. An unknown rl_N must not be
        # handed off: the FSM would stay in "mpc" with no torque updates.
        rl = mode in self.rl_modes
        if mode == "mpc":
            enabled = True
        elif rl or mode in ("idle", "joint_pd", "transform_down", "transform_up"):
            enabled = False
        else:
            return
        with self._condition:
            if enabled and not self._mode_enabled and self.fsm_blocks_mpc():
                return  # checked again on the next (repeated) mode command
            # MPX drives the robot: it published a torque and the FSM can be in "mpc".
            in_control = self._mode_enabled and self._took_over and not self.fsm_blocks_mpc()
            if mode == "transform_up" and in_control:
                return  # the FSM does not leave "mpc" for transform_up
            if (rl or mode == "transform_down") and in_control:
                # MPX keeps control and moves to the handoff height; the RL FSM switches
                # only after check_handoff publishes the mode on handoff_topic.
                self._activating = False
                h = self.h_rl
                if self._handoff_target is None or h != self._command[3]:
                    self._command = (0.0, 0.0, 0.0, h)
                    self._handoff_start = self._handoff_settled = None
                    self.get_logger().info(
                        f"{mode} requested: MPX moves the COM to {h:.3f} m before handing off")
                self._handoff_target = mode
                return
            cancelled = enabled and self._handoff_target is not None
            self._handoff_target = None
            if cancelled:
                # Back to MPC during the handoff: MPX is still active, just reset commands.
                h_start = self.h_mpc
                self._command = (0.0, 0.0, 0.0, h_start)
                self._activating, self._activation_start = True, None
                self.get_logger().info(
                    f"Handoff cancelled: zero velocity, h_start={h_start:.3f} m")
            elif self._mode_enabled != enabled:
                self._mode_enabled = enabled
                self._took_over = False
                self._activating, self._activation_start = enabled, None
                if enabled:
                    # Every activation starts at rest at h_start; later pose commands
                    # override the height only when the requested height changes.
                    h_start = self.h_mpc
                    self._command = (0.0, 0.0, 0.0, h_start)
                    self.get_logger().info(
                        f"MPC command reset: zero velocity, h_start={h_start:.3f} m")
                self._generation += 1
                self._condition.notify_all()
                self.get_logger().info("MPC computation " + ("enabled" if enabled else "standby"))

    def check_handoff(self, generation, stamp, raw):
        """Release control to the requested mode once the COM holds the handoff height."""
        with self._condition:
            target = self._handoff_target
            if target is None or generation != self._generation:
                return
            if self._handoff_start is None:
                self._handoff_start = stamp
            h = self._command[3]
            settled = (
                raw is not None
                and abs(raw[0][2] - h) < self.handoff_height_tol
                and abs(raw[1][2]) < self.handoff_vz_tol
            )
            if not settled:
                self._handoff_settled = None
            elif self._handoff_settled is None:
                self._handoff_settled = stamp
            timed_out = stamp - self._handoff_start >= self.handoff_timeout * 1e9
            if not timed_out and (
                self._handoff_settled is None
                or stamp - self._handoff_settled < self.handoff_settle_ns
            ):
                return
            self._handoff_target = None
            self._mode_enabled = False
            self._took_over = False
            self._generation += 1
            self._condition.notify_all()
        self.handoff_pub.publish(String(data=target))
        self.publish_keys_locked()
        if timed_out:
            self.get_logger().warning(
                f"COM {h:.3f} m not reached within {self.handoff_timeout:g} s; "
                f"handing off to {target} anyway")
        else:
            self.get_logger().info(f"COM at {h:.3f} m: handing off to {target}")
        self.get_logger().info("MPC computation standby")

    def check_activation(self, generation, stamp, raw, target, height_ref):
        """Unlock commands once MPX drives the robot and the COM holds the ramped height."""
        with self._condition:
            if (not self._activating or generation != self._generation
                    or not self._took_over or raw is None):
                return
            if self._activation_start is None:
                self._activation_start = stamp
            settled = (
                height_ref == target
                and abs(raw[0][2] - target) < self.handoff_height_tol
                and abs(raw[1][2]) < self.handoff_vz_tol
            )
            timed_out = stamp - self._activation_start >= self.handoff_timeout * 1e9
            if not (settled or timed_out):
                return
            self._activating = False
            self.publish_keys_locked()
        if settled:
            self.get_logger().info(f"COM at h_mpc={target:.3f} m: MPC commands unlocked")
        else:
            self.get_logger().warning(
                f"COM not settled at {target:.3f} m (z={raw[0][2]:.3f}) within "
                f"{self.handoff_timeout:g} s; MPC commands unlocked anyway")

    def on_pose_command(self, msg):
        # The keyboard republishes its pose every 10 ms. Follow only changes, so
        # a repeated stale height does not overwrite h_start after activation.
        z = msg.pose.position.z
        if z == self._last_pose_z:
            return
        self._last_pose_z = z
        self.on_com_height(Float64(data=z))

    def on_com_height(self, msg):
        if not np.isfinite(msg.data):
            self.get_logger().warning("Ignoring non-finite COM height command")
            return
        height = float(msg.data)
        with self._condition:
            if self.commands_blocked():
                return
            self._command = (*self._command[:3], height)

    def publish_default_command(self):
        """Publish zero velocity until an external command source connects."""
        publishers = self.get_publishers_info_by_topic(self.cmd_vel_topic)
        external_publishers = [
            endpoint for endpoint in publishers
            if not (
                endpoint.node_name == "controller"
                and endpoint.node_namespace.rstrip("/") == "/mpx"
            )
        ]
        if external_publishers:
            self.default_command_timer.cancel()
            self.destroy_timer(self.default_command_timer)
            self.destroy_publisher(self.default_command_pub)
            self.get_logger().info(
                "External /mpx/cmd_vel publisher detected; stopped publishing default zero commands."
            )
            return

        self.default_command_pub.publish(Twist())

    def update_mujoco_state(self, pose, twist, joints):
        # MuJoCo free-joint position: xyz, then quaternion wxyz.
        self.data.qpos[:3] = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]
        self.data.qpos[3:7] = [
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]

        # Gazebo reports world linear/angular velocity. MuJoCo's free joint
        # expects world linear velocity and body angular velocity.
        world_from_body = rotation_matrix([
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        ])
        self.data.qvel[:3] = [twist.linear.x, twist.linear.y, twist.linear.z]
        self.data.qvel[3:6] = world_from_body.T @ np.array([
            twist.angular.x, twist.angular.y, twist.angular.z,
        ])

        joint_values = {
            name: (position, velocity)
            for name, position, velocity in zip(
                joints.name,
                joints.position,
                joints.velocity,
            )
        }
        for name, (qpos_addr, qvel_addr) in zip(
            self.config.joints_name, self.joint_addresses
        ):
            if name not in joint_values:
                raise ValueError(f"Missing joint: {name}")
            position, velocity = joint_values[name]
            self.data.qpos[qpos_addr] = position
            self.data.qvel[qvel_addr] = velocity

        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            raise ValueError("Non-finite feedback state")
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_subtreeVel(self.model, self.data)

    def raw_state_from(self, data):
        centers = data.geom_xpos[self.contact_ids].copy()
        rotations = data.geom_xmat[
            self.contact_ids
        ].reshape(2, 3, 3).copy()
        radii = self.model.geom_size[
            self.contact_ids, 0
        ].copy()

        wheel_velocities = np.zeros((2, 3))
        spatial_velocity = np.zeros(6)
        for i, geom_id in enumerate(self.contact_ids):
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(geom_id),
                spatial_velocity,
                0,  # world frame
            )
            wheel_velocities[i] = spatial_velocity[3:6]

        return (
            data.subtree_com[self.base_id].copy(),
            data.subtree_linvel[self.base_id].copy(),
            centers,
            rotations,
            radii,
            wheel_velocities,
        )

    def raw_state(self):
        return self.raw_state_from(self.data)

    def step_controller(self, mpc_state, theta_prev, raw, qpos, qvel, command):
        """One MPC + WBC step, following mjx_tita.step_controller.

        Inputs qpos/qvel/command are already batched for one robot. This pure
        numerical sequence stays in one CPU JIT call. The ROS worker schedules
        steps and commits the returned state/torque only after validation;
        Gazebo advances physics independently and holds the last valid effort.
        """
        tita_state, x0, theta_prev = self.mpc.process_state(*raw, theta_prev)

        mpc_state, reference = self.mpc.run(mpc_state, x0[None, :], command)
        mpc_ok = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(mpc_state)
        ]))

        mpc_state, tau_wbc, qddot, fl, fr, desired = self.mpc.whole_body_run(
            mpc_state,
            x0,
            qpos,
            qvel,
            tita_state[6:9][None, :],
            tita_state[9:12][None, :],
            tita_state[12:15][None, :],
            tita_state[15:18][None, :],
            com_height_target=command[:, 3],
        )
        flags = jnp.array([
            mpc_ok, jnp.all(jnp.isfinite(desired)), jnp.all(jnp.isfinite(qddot)),
            jnp.all(jnp.isfinite(fl)) & jnp.all(jnp.isfinite(fr)),
        ])
        return mpc_state, theta_prev, tau_wbc, flags

    def precompile_controller(self):
        self.get_logger().info("Compiling MPX on CPU; the previous control remains active until valid MPC output."
                               if self._mode_controlled else
                               "Precompiling MPX on CPU; keep Gazebo paused until MPX ready.")
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [0.0, 0.0, self.config.robot_height]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = self.config.q0
        mujoco.mj_forward(self.model, data)
        mpc_state, theta_prev = self.mpc_state, self.theta_prev
        start = time.perf_counter()
        for i in range(3):
            mpc_state, theta_prev, tau_wbc, flags = self._step_controller_jit(
                mpc_state, theta_prev, self.raw_state_from(data),
                data.qpos.copy()[None, :], data.qvel.copy()[None, :],
                np.asarray([self._command], dtype=np.float64),
            )
            jax.block_until_ready((mpc_state, theta_prev, tau_wbc, flags))
            if not np.all(np.isfinite(tau_wbc)) or not np.all(flags):
                raise RuntimeError("MPX warm-up produced invalid control")
            if i == 0:
                self.initial_effort = np.asarray(tau_wbc[0]).copy()
        _, x0, _ = self.mpc.process_state(*self.raw_state_from(data), self.theta_prev)
        jax.block_until_ready(self.mpc.init_state(x0))
        # Like mjx_tita.py: compilation must not advance the live MPC horizon.
        self.mpc_state = self.mpc.init_state()
        self.theta_prev = np.asarray(0.0, dtype=np.float64)
        self.get_logger().info(
            f"MPX precompilation complete in {time.perf_counter() - start:.2f} s; "
            "warm-up state discarded."
        )

    def publish_effort_command(self, effort):
        effort = np.asarray(effort, dtype=float)
        if effort.shape != (8,) or not np.all(np.isfinite(effort)):
            raise ValueError(f"Refusing to publish invalid torque: {effort}")
        effort = np.clip(
            effort, -self.config.effort_limits, self.config.effort_limits
        )
        msg = Float64MultiArray()
        msg.data = effort.tolist()
        self.effort_pub.publish(msg)

    def control_loop(self):
        """One serial solve per 10 ms of JointState time; skip missed periods.

        The physics/effort loop is independent and holds the last command.
        While paused no new timestamp arrives, so no MPC horizon is advanced.
        """
        generation = -1
        next_due = None
        first_step = True
        warmup_left = 0
        height_ref = None
        prev_stamp = None
        durations = []
        missed = invalid = 0
        window_start = None
        last_report = time.monotonic()
        last_wait_log = 0.0
        while True:
            with self._condition:
                while not self._stop:
                    if self._mode_enabled:
                        self.poll_feedback()
                    now = time.monotonic()
                    base, joint = self._base_sample, self._joint_sample
                    if self._mode_enabled and base is not None and joint is not None:
                        stamp = joint[2]
                        reset = generation != self._generation
                        if (reset or next_due is None or stamp >= next_due) and (
                            now - base[2] < 0.1 and now - joint[1] < 0.1
                        ):
                            break
                    # Tell why MPX is not running instead of waiting silently.
                    if self._mode_enabled and now - last_wait_log >= 2.0:
                        missing = [
                            topic for topic, sample, t_idx in (
                                (self._base_topic, base, 2), (self._joints_topic, joint, 1),
                                (self._odom_topic, self._odom_sample, 1))
                            if topic is not None and (sample is None or now - sample[t_idx] >= 0.1)
                        ]
                        if missing:
                            self.get_logger().warning(
                                f"MPX waiting for data on: {', '.join(missing)}")
                            last_wait_log = now
                    # Feedback no longer wakes the executor at 500 Hz. Sleep
                    # toward the next simulation-time deadline (bounded for
                    # pause/reset/slow simulation); mode commands still wake us.
                    wait = 0.1
                    if self._mode_enabled:
                        wait = 0.01
                        if joint is not None and next_due is not None:
                            wait = min(wait, max(0.001, (next_due - joint[2]) / 1e9))
                    self._condition.wait(timeout=wait)
                if self._stop:
                    return
                command = self._command
                if generation != self._generation:
                    generation = self._generation
                    next_due = None
                    self.mpc_state = self.mpc.init_state()
                    self.theta_prev = np.asarray(0.0, dtype=np.float64)
                    window_start = stamp
                    first_step = True
                if next_due is None:
                    next_due = stamp
                skipped = (stamp - next_due) // self.period_ns
                missed += skipped
                next_due += (skipped + 1) * self.period_ns

            start = time.perf_counter()
            raw = qpos = None
            try:
                if self._precompile_pending:
                    self.get_logger().warning(
                        "Compiling MPX after key 4. Keep the current RL control active; "
                        "waiting without timeout. The first valid torque will trigger the switch."
                    )
                    with self._condition:
                        self._compiling = True
                    self.publish_keys_locked()
                    try:
                        self.precompile_controller()
                    finally:
                        with self._condition:
                            self._compiling = False
                        self.publish_keys_locked()
                    self._precompile_pending = False
                    # Collect the compilation garbage now and keep it out of later GC
                    # passes, so no long collection pause lands on a control step.
                    gc.collect()
                    gc.freeze()
                    generation = -1
                    next_due = None
                    durations.clear()
                    missed = invalid = 0
                    self.get_logger().info("MPX ready on CPU: compilation complete; preparing live torque.")
                    continue  # reacquire current feedback and honor commands changed during compilation
                self.update_mujoco_state(base[0], base[1], joint[0])
                raw = self.raw_state()
                if first_step:
                    # Start from the measured state: warm-start the MPC horizon
                    # at the current COM and ramp the height target from here.
                    _, x0, _ = self.mpc.process_state(*raw, self.theta_prev)
                    self.mpc_state = self.mpc.init_state(x0)
                    height_ref = float(raw[0][2])
                    prev_stamp = None
                    first_step = False
                    warmup_left = WARMUP_STEPS if self._mode_controlled else 0
                    nominal = (
                        abs(self.data.qpos[2] - self.config.robot_height) < 0.05
                        and np.max(np.abs(self.data.qpos[7:] - np.asarray(self.config.q0))) < 0.05
                    )
                    if self.initial_effort is not None and nominal and not self._mode_controlled:
                        self.publish_effort_command(self.initial_effort)
                    self.get_logger().info(
                        "First feedback received; starting simulation-time control"
                        + (f" ({warmup_left} warm-up solves before the first torque)."
                           if warmup_left else "."))
                dt = self.period_ns / 1e9 if prev_stamp is None else (stamp - prev_stamp) / 1e9
                prev_stamp = stamp
                max_delta = HEIGHT_RAMP_RATE * dt
                height_ref += float(np.clip(command[3] - height_ref, -max_delta, max_delta))
                qpos = self.data.qpos.copy()  # same copy the solver receives
                mpc_state, theta_prev, tau_wbc, flags = self._step_controller_jit(
                    self.mpc_state, self.theta_prev, raw,
                    qpos[None, :], self.data.qvel.copy()[None, :],
                    np.asarray([(*command[:3], height_ref)], dtype=np.float64),
                )
                # NumPy materialization waits for actual CPU completion; these
                # timings include the solve, not merely asynchronous dispatch.
                effort = np.asarray(tau_wbc[0])
                flags = np.asarray(flags)
                if not np.all(np.isfinite(effort)) or not np.all(flags):
                    if not self._reported_nan_diagnostics:
                        self._reported_nan_diagnostics = True
                        self.get_logger().error(
                            f"First invalid control: qpos={self.data.qpos.tolist()} "
                            f"qvel={self.data.qvel.tolist()} "
                            f"finite [mpc, desired, qddot, forces]={flags.tolist()}"
                        )
                    raise ValueError("Invalid MPC/WBC result; holding last valid effort")
                # Commit only validated results. On failure the previous MPC state
                # and the last effort held by ros2_control remain in use.
                self.mpc_state, self.theta_prev = mpc_state, theta_prev
                if warmup_left:
                    warmup_left -= 1  # horizon warmed on the live state, nothing published
                    continue
                with self._condition:
                    # Never publish a result from an old activation or stale feedback.
                    if (self._mode_enabled and generation == self._generation
                            and time.monotonic() - joint[1] < 0.15):
                        self.publish_effort_command(effort)
                        self._took_over = True
            except Exception as exc:
                invalid += 1
                if time.monotonic() - self.last_control_error >= 1.0:
                    self.get_logger().error(f"MPX step failed: {exc}")
                    self.last_control_error = time.monotonic()
            self.check_handoff(generation, stamp, raw)
            self.check_activation(generation, stamp, raw, command[3], height_ref)
            durations.append(1000 * (time.perf_counter() - start))
            if raw is not None and self.estimate_pub is not None:
                # After the effort is published and outside the timed solve.
                # raw holds fresh copies (see raw_state_from) that nothing mutates.
                self.estimate_pub.submit(
                    joint[0].header.stamp, raw[0], raw[1], raw[2], qpos[3:7]
                )
            now = time.monotonic()
            if now - last_report >= 5.0:
                sim_seconds = (stamp - window_start) / 1e9
                hz = len(durations) / sim_seconds if sim_seconds > 0 else 0.0
                p50, p95, maximum = np.percentile(durations, [50, 95, 100])
                self.get_logger().info(
                    f"Control: {hz:.1f} Hz sim, solve ms median={p50:.2f} "
                    f"p95={p95:.2f} max={maximum:.2f}, "
                    f"missed periods={missed}, invalid={invalid}."
                )
                durations.clear()
                missed = invalid = 0
                window_start = stamp
                last_report = now

    def stop_control(self):
        with self._condition:
            self._stop = True
            self._condition.notify()
        self._worker.join()
        self._feedback_node.destroy_node()
        if self.estimate_pub is not None:
            self.estimate_pub.stop()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # A launch started in the background can inherit SIGINT ignored from bash.
    # Always make Ctrl+C interrupt the main thread, including with idle DDS.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = None
    executor = None
    spin_thread = None
    try:
        node = MPXNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, name="mpx-executor")
        spin_thread.start()
        # Python handles SIGINT in this thread, independently of DDS activity.
        idle_wait = threading.Event()
        while spin_thread.is_alive():
            idle_wait.wait(timeout=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # ros2 launch may deliver a second SIGINT during solver shutdown.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            node.stop_control()
        if executor is not None:
            executor.shutdown()
        if spin_thread is not None:
            spin_thread.join()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()