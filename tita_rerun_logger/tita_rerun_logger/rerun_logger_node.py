#!/usr/bin/env python3
"""ROS 2 -> Rerun logger for TITA (Gazebo or robot) with MPX, RL and the state filter.

Every compared quantity is logged as <quantity>/<axis>/<source>, one colour per source:
  gt      ground truth from Gazebo (/link_states + URDF masses), simulation only
  filter  tita_state_estimator (filtered_state) + forward kinematics of joint_states
  imu     robot IMU (imu_sensor_broadcaster/imu)
  odom    robot chassis odometry (robot only)
  mpx     what MPX believes (mpx/estimate/*), the state its MPC/WBC receives
  cmd     keyboard / remote commands (command/cmd_twist, command/cmd_pose)
Errors against the ground truth: errors/<source>/<quantity>/<axis|norm>.
Also: joints (position, velocity, measured torque, MPX torque, |tau|/limit), FSM state,
MPX rate and latency, key lock, handoffs, any extra topic, 3D scene.

Timeline "sim_time": message stamps; unstamped messages take the stamp of the latest
joint_states (Gazebo /clock is only 10 Hz). On the robot the first stamp is t = 0.
"""

import array
import math
import os
import struct
import time
from collections import OrderedDict, deque

import numpy as np
import rclpy
import yaml
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message

import rerun as rr
from rerun.components import InterpolationMode

from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float64MultiArray, String

from tita_rerun_logger.blueprint import (APPLICATION_ID, BALANCE, COMPARED,
                                         FSM_CODES, SOURCES, TITA_JOINTS, WHEELS,
                                         make_blueprint)
from tita_rerun_logger.kinematics import (UrdfModel, cross, make_T, quat_xyzw_to_matrix,
                                          quat_xyzw_to_rpy, yaw_rotation)

try:  # Gazebo messages are not needed on the robot
    from gazebo_msgs.msg import LinkStates
except ImportError:  # pragma: no cover
    LinkStates = None

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None

try:
    from rosidl_runtime_py.utilities import get_message
except ImportError:  # pragma: no cover
    get_message = None


NAN3 = np.full(3, np.nan)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cdr_stamp(data):
    """header.stamp of a serialized message whose first field is a std_msgs/Header.

    Read straight from the CDR bytes, so high-rate topics can be timed and throttled
    without deserializing them in Python.
    """
    if len(data) < 12:
        return None
    fmt = "<iI" if data[1] == 1 else ">iI"  # encapsulation: CDR little / big endian
    sec, nsec = struct.unpack_from(fmt, data, 4)
    return sec + nsec * 1e-9


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


SKIP_FIELDS = {"header", "layout"}
MAX_ARRAY_ELEMS = 64


def flatten_numeric(value, prefix, out):
    if hasattr(value, "get_fields_and_field_types"):
        for f in value.get_fields_and_field_types():
            if f in SKIP_FIELDS:
                continue
            flatten_numeric(getattr(value, f), f"{prefix}/{f}" if prefix else f, out)
    elif isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float, np.floating, np.integer)):
        out[prefix] = float(value)
    elif isinstance(value, (list, tuple, array.array, np.ndarray)):
        if len(value) <= MAX_ARRAY_ELEMS:
            for i, v in enumerate(value):
                flatten_numeric(v, f"{prefix}/{i}", out)


def resolve_mesh(filename):
    if filename.startswith("package://"):
        pkg, _, sub = filename[len("package://"):].partition("/")
        if get_package_share_directory is None:
            return None
        try:
            return os.path.join(get_package_share_directory(pkg), sub)
        except Exception:  # noqa: BLE001
            return None
    if filename.startswith("file://"):
        return filename[len("file://"):]
    return filename if os.path.isabs(filename) else None


def resolve_log_dir(rrd_dir):
    """Explicit dir if given, else <ws>/src/**/tita_rerun_logger/rerun_log."""
    if rrd_dir:
        return os.path.expanduser(rrd_dir)
    fallback = os.path.expanduser("~/rerun_logs")
    if get_package_share_directory is None:
        return fallback
    try:
        share = get_package_share_directory("tita_rerun_logger")
    except Exception:  # noqa: BLE001
        return fallback
    # <ws>/install/tita_rerun_logger/share/tita_rerun_logger -> <ws>
    ws = os.path.abspath(os.path.join(share, "..", "..", "..", ".."))
    src = os.path.join(ws, "src")
    direct = os.path.join(src, "tita_rerun_logger")
    if os.path.isfile(os.path.join(direct, "package.xml")):
        return os.path.join(direct, "rerun_log")
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and d not in ("build", "install", "log", "jax_cache", "__pycache__")]
        if os.path.basename(root) == "tita_rerun_logger" and "package.xml" in files:
            return os.path.join(root, "rerun_log")
    return fallback


def state_values(p, R, v_world, w_world, com=None, com_vel=None, wheels=None):
    """Compared quantities of one source from its base pose/twist (+ CoM and wheels)."""
    rpy = np.asarray(quat_from_R_rpy(R))
    Rh = yaw_rotation(rpy[2])
    v_h = Rh.T @ v_world
    v_h[2] = v_world[2]  # vertical velocity in the world frame
    values = {
        "base/position": np.asarray(p, dtype=float),
        "base/rpy": rpy,
        "base/vel_heading": v_h,
        "base/ang_vel_body": R.T @ w_world,
    }
    if com is not None:
        values["com"] = com
    if com_vel is not None:
        values["com_vel"] = com_vel
    if wheels is not None and len(wheels) == 2:
        values["wheels/left"], values["wheels/right"] = wheels
        mid = 0.5 * (wheels[0] + wheels[1])
        values["balance/base_height_over_wheels"] = p[2] - mid[2]
        if com is not None:
            off = Rh.T @ (com - mid)
            values["balance/com_ahead_of_wheels"] = off[0]
            values["balance/com_lateral"] = off[1]
            values["balance/com_height_over_wheels"] = off[2]
    return values


def quat_from_R_rpy(R):
    """Roll, pitch, yaw (ZYX) of a rotation matrix."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
class RerunLogger(Node):
    def __init__(self):
        super().__init__("tita_rerun_logger")

        dp = self.declare_parameter
        dp("application_id", APPLICATION_ID)
        dp("recording_id", "")
        dp("open_viewer", False)
        dp("viewer_url", "rerun+http://127.0.0.1:9876/proxy")
        dp("save_rrd", True)
        dp("rrd_dir", "")
        dp("model_name", "tita")
        dp("base_link", "base_link")
        dp("wheel_links", ["left_leg_4", "right_leg_4"])
        dp("wheel_radius", 0.0925)
        dp("controllers_yaml", "")
        dp("controller_name", "tita_rl_controller")
        for key in ("robot_description", "joint_states", "imu", "link_states",
                    "filtered_state", "chassis_odometry", "mpx_effort", "mpx_handoff",
                    "mpx_keys_locked", "mpx_estimate_prefix", "fsm_state", "cmd_key",
                    "cmd_twist", "cmd_pose"):
            dp(f"topics.{key}", "")
        dp("extra_topics", [""])
        dp("max_gt_dt_s", 0.003)
        dp("gt_buffer_s", 0.5)
        for key in ("joints", "gt", "filter", "imu", "odom", "extra", "scene"):
            dp(f"rates_hz.{key}", 50.0)
        dp("flush_period_s", 0.5)
        dp("poll_period_s", 0.02)
        dp("trail_length", 600)
        dp("log_meshes", True)

        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.topic = lambda k: gp(f"topics.{k}")  # noqa: E731
        self.model_prefix = f"{gp('model_name')}::"
        self.base_link = gp("base_link")
        self.wheel_links = list(gp("wheel_links"))
        self.wheel_radius = float(gp("wheel_radius"))
        self.rates = {k: float(gp(f"rates_hz.{k}"))
                      for k in ("joints", "gt", "filter", "imu", "odom", "extra", "scene")}
        self.max_gt_dt = float(gp("max_gt_dt_s"))
        self.log_meshes = bool(gp("log_meshes"))
        self.extra_topics = [t for t in gp("extra_topics") if t]
        self.joints, self.torque_limit = self._load_controller_config(
            gp("controllers_yaml"), gp("controller_name"))

        # State
        self.urdf = None
        self.static_scene_done = False
        self.anchor_cache = {}
        self.buckets = {}
        self.clock_t = None          # latest joint_states stamp
        self.t_offset = None
        self.t_rel = 0.0             # current timeline value [s]
        self.series = {}             # entity path -> ([t], [value]), sent in batches
        self.points_3d = {}          # latest filter / MPX points for the 3D view
        # Recent joint_states: (DDS source timestamp [ns], stamp [s], bytes). Maps the
        # publication time of unstamped messages (link_states, mpx/effort) to sim time.
        self.joint_hist = deque(maxlen=250)
        self.joint_msg = None        # (stamp, JointState) deserialized lazily
        self.pending_links = []      # link_states newer than the latest joint_states
        self.last_effort_meas = {}
        self.gt_raw = deque(maxlen=max(10, int(float(gp("gt_buffer_s")) * 500)))
        self.gt_cache = OrderedDict()
        self.fsm_state = self.fsm_requested = None
        self.keys_locked = None
        self.cmd_twist = None
        self.cmd_pose = None
        self.mpx_parts = {}          # MPX estimate stamp -> {quantity: value}
        self.mpx_stamps = deque(maxlen=20)
        self.effort_times = deque(maxlen=20)
        trail = int(gp("trail_length"))
        self.trails = {k: deque(maxlen=trail) for k in ("base", "com")}
        self.extra_subs = {}

        self._init_rerun(gp)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        def sub(key, mtype, cb, qos):
            if self.topic(key):
                self.create_subscription(mtype, self.topic(key), cb, qos)

        # Low-rate event topics: normal callbacks
        sub("robot_description", String, self._on_robot_description, latched)
        sub("mpx_handoff", String, lambda m: self._event("handoff", m.data, "WARN"), reliable)
        sub("mpx_keys_locked", Bool, self._on_keys_locked, latched)
        sub("fsm_state", String, self._on_fsm_state, latched)
        sub("cmd_key", String, self._on_cmd_key, latched)

        # Streams (500 Hz sensors, ~100 Hz MPX): not in the executor. A timer takes
        # every queued message in one go; one rclpy wake-up per message cost ~1 core.
        self.stream_node = rclpy.create_node(
            "tita_rerun_logger_streams", namespace=self.get_namespace(),
            use_global_arguments=False, start_parameter_services=False)
        self.streams = []  # (subscription, handler(msg_or_bytes, source_timestamp_ns))
        self._stream("joint_states", JointState, self._on_joint_states, raw=True)
        if LinkStates is not None:
            self._stream("link_states", LinkStates, self._on_link_states, raw=True)
        else:
            self.get_logger().warn("gazebo_msgs not available: no ground truth")
        self._stream("imu", Imu, self._stamped(Imu, "imu", self._on_imu), raw=True)
        self._stream("filtered_state", Odometry,
                     self._stamped(Odometry, "filter", self._on_filter), raw=True)
        self._stream("chassis_odometry", Odometry,
                     self._stamped(Odometry, "odom", self._on_chassis_odom), raw=True)
        self._stream("mpx_effort", Float64MultiArray, self._on_mpx_effort)
        self._stream("cmd_twist", Twist, self._on_cmd_twist)
        self._stream("cmd_pose", PoseStamped, self._on_cmd_pose)
        prefix = self.topic("mpx_estimate_prefix")
        if prefix:
            for name, mtype, quantity in (("com", PointStamped, "com"),
                                          ("com_vel", Vector3Stamped, "com_vel"),
                                          ("base_rpy", Vector3Stamped, "base/rpy"),
                                          ("left_wheel", PointStamped, "wheels/left"),
                                          ("right_wheel", PointStamped, "wheels/right")):
                self._stream(None, mtype, lambda m, src, q=quantity: self._on_mpx_estimate(m, q, src),
                             topic=f"{prefix}/{name}")

        # Wall-clock timers: keep working while Gazebo is paused.
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0, self._discover_extra_topics, clock=steady)
        self.create_timer(float(gp("flush_period_s")), self.flush, clock=steady)
        self.create_timer(float(gp("poll_period_s")), self._poll_streams, clock=steady)
        self.get_logger().info(
            f"Rerun logger started; joints (mpx/effort order): {', '.join(self.joints)}")

    # ------------------------------------------------------------ setup
    def _load_controller_config(self, path, controller):
        """Joint order of mpx/effort and torque_limit from rl_controller's controllers.yaml."""
        if not path and get_package_share_directory is not None:
            try:
                path = os.path.join(get_package_share_directory("rl_controller"),
                                    "config", "tita", "controllers.yaml")
            except Exception:  # noqa: BLE001
                path = ""
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            for scope in cfg.values():
                params = scope.get(controller, {}).get("ros__parameters")
                if params:
                    joints = list(params["joints"])
                    limits = dict(zip(joints, params.get("torque_limit", [])))
                    return joints, limits
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Cannot read {controller} from '{path}': {e}")
        return list(TITA_JOINTS), {}

    def _stream(self, key, mtype, handler, raw=False, topic=None):
        topic = topic or self.topic(key)
        if not topic:
            return
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        sub = self.stream_node.create_subscription(mtype, topic, lambda _: None, qos, raw=raw)
        self.streams.append((sub, handler))

    def _poll_streams(self):
        """Take every queued stream message; joint_states first (it drives the clock)."""
        for sub, handler in self.streams:
            while True:
                with sub.handle:
                    taken = sub.handle.take_message(sub.msg_type, sub.raw)
                if taken is None:
                    break
                handler(taken[0], taken[1]["source_timestamp"])
            if handler == self._on_joint_states and self.pending_links:
                pending, self.pending_links = self.pending_links, []
                for data, src in pending:
                    self._on_link_states(data, src)

    def _stamped(self, mtype, rate_key, cb):
        """Stream handler: throttle on the header stamp, then deserialize."""
        def _cb(data, _src):
            t = cdr_stamp(data)
            if t is not None and self._due(rate_key, t):
                cb(deserialize_message(data, mtype), t)
        return _cb

    def _sim_time(self, src):
        """Sim time of the joint_states published closest to DDS time src (ns)."""
        best = None
        for s_src, t, _ in reversed(self.joint_hist):
            if best is None or abs(s_src - src) < abs(best[0] - src):
                best = (s_src, t)
            if s_src < src - 20_000_000:
                break
        return best[1] if best is not None else self._now()

    def _init_rerun(self, gp):
        self.rrd_path = None
        rr.init(gp("application_id"), recording_id=gp("recording_id") or None)
        sinks = []
        if gp("save_rrd"):
            rrd_dir = resolve_log_dir(gp("rrd_dir"))
            os.makedirs(rrd_dir, exist_ok=True)
            # Local wall-clock time when recording starts, not simulation time.
            path = os.path.join(rrd_dir, time.strftime("tita_rerun:%y%m%d_%H%M%S.rrd"))
            sinks.append(rr.FileSink(path))
            self.rrd_path = path
            self.get_logger().info(f"Offline log: {path}")
        if gp("open_viewer"):
            url = gp("viewer_url")
            if "127.0.0.1" in url or "localhost" in url:
                rr.spawn(connect=False)  # reuses a running local viewer
            sinks.append(rr.GrpcSink(url))
            self.get_logger().info(f"Live streaming to {url}")
        if not sinks:
            self.get_logger().warn("save_rrd and open_viewer are both false: nothing is recorded")
            sinks.append(rr.FileSink(os.devnull))
        rr.set_sinks(*sinks)
        rr.send_blueprint(make_blueprint(self.joints))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._log_styles()

    def _log_styles(self):
        """Fixed colour and legend name per source in every plot."""
        def style(path, src, step=False):
            color, name = SOURCES[src]
            rr.log(path, rr.SeriesLines(
                colors=[color], names=name,
                interpolation_mode=InterpolationMode.StepAfter if step else None), static=True)

        for quantity, (axes, sources) in COMPARED.items():
            for a in axes:
                for src in sources:
                    style(f"{quantity}/{a}/{src}", src, step=src == "cmd")
        for quantity, sources in BALANCE.items():
            for src in sources:
                style(f"{quantity}/{src}", src)
        for joint in self.joints:
            rr.log(f"joints/effort/{joint}/measured", rr.SeriesLines(
                colors=[SOURCES["imu"][0]], names="measured (joint_states)"), static=True)
            rr.log(f"joints/effort/{joint}/mpx_cmd", rr.SeriesLines(
                colors=[SOURCES["mpx"][0]], names="MPX command (mpx/effort)"), static=True)
        for path, name in (("fsm/state", "FSM state (fsm_state)"),
                           ("fsm/requested", "requested (command/cmd_key)"),
                           ("mpx/keys_locked", "MPX keys locked")):
            rr.log(path, rr.SeriesLines(names=name, interpolation_mode=InterpolationMode.StepAfter),
                   static=True)
        rr.log("mpx/rate_hz/estimate", rr.SeriesLines(names="mpx/estimate (solves)"), static=True)
        rr.log("mpx/rate_hz/effort", rr.SeriesLines(names="mpx/effort (published torques)"),
               static=True)
        codes = ", ".join(f"{k}={v}" for k, v in FSM_CODES.items())
        rr.log("events", rr.TextLog(f"FSM codes: {codes}"), static=True)

    # ------------------------------------------------------------ time
    def _now(self):
        if self.clock_t is not None:
            return self.clock_t
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_time(self, t):
        if self.t_offset is None:
            self.t_offset = t if t > 1e8 else 0.0  # wall-clock stamps (robot) start at 0
        self.t_rel = max(0.0, t - self.t_offset)

    def _log(self, path, entity, **kw):
        """Non-scalar data (3D, text), logged right away at the current time."""
        rr.set_time("sim_time", duration=self.t_rel)
        rr.log(path, entity, **kw)

    def _due(self, key, t):
        """True once per 1/rate of message time (same instants for same stamps)."""
        rate = self.rates.get(key, 0.0)
        if rate <= 0:
            return True
        bucket = int(math.floor(t * rate))
        if self.buckets.get(key) != bucket:
            self.buckets[key] = bucket
            return True
        return False

    # ------------------------------------------------------------ logging
    def _scalar(self, path, value):
        """Buffered: one rr.log per scalar costs too much CPU and file size at ~10k/s."""
        if value is not None and math.isfinite(value):
            ts, vs = self.series.setdefault(path, ([], []))
            ts.append(self.t_rel)
            vs.append(float(value))

    def flush(self):
        series, self.series = self.series, {}
        for path, (ts, vs) in series.items():
            rr.send_columns(path, indexes=[rr.TimeColumn("sim_time", duration=ts)],
                            columns=rr.Scalars.columns(scalars=vs))

    def _log_values(self, src, values):
        for quantity, vec in values.items():
            if quantity in COMPARED:
                for a, v in zip(COMPARED[quantity][0], vec):
                    self._scalar(f"{quantity}/{a}/{src}", v)
            elif quantity in BALANCE and src in BALANCE[quantity]:
                self._scalar(f"{quantity}/{src}", vec)

    def _log_errors(self, src, t, values):
        gt = self._gt_at(t)
        if gt is None:
            return
        for quantity, vec in values.items():
            if quantity not in COMPARED or quantity not in gt:
                continue
            err = np.asarray(vec, dtype=float) - gt[quantity]
            if quantity == "base/rpy":
                err = wrap_angle(err)
            finite = np.isfinite(err)
            if not finite.any():
                continue
            for a, v in zip(COMPARED[quantity][0], err):
                self._scalar(f"errors/{src}/{quantity}/{a}", v)
            self._scalar(f"errors/{src}/{quantity}/norm", float(np.linalg.norm(err[finite])))

    def _event(self, what, text, level="INFO"):
        self._set_time(self._now())
        self._log("events", rr.TextLog(f"{what}: {text}", level=level))

    # ------------------------------------------------------------ URDF
    def _on_robot_description(self, msg):
        try:
            self.urdf = UrdfModel(msg.data, resolve_mesh)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"Cannot parse robot_description: {e}")
            return
        self.static_scene_done = False
        self.anchor_cache.clear()
        self.gt_cache.clear()
        self.get_logger().info(
            f"URDF loaded: {len(self.urdf.links)} links, total mass {self.urdf.total_mass:.3f} kg")

    # ------------------------------------------------------------ joints (robot)
    def _joints_at(self, t):
        """JointState with stamp t (or the latest), deserialized once."""
        if not self.joint_hist:
            return None
        entry = next((e for e in reversed(self.joint_hist) if e[1] <= t), self.joint_hist[-1])
        if self.joint_msg is None or self.joint_msg[0] != entry[1]:
            self.joint_msg = (entry[1], deserialize_message(entry[2], JointState))
        return self.joint_msg[1]

    def _on_joint_states(self, data, src):
        t = cdr_stamp(data)
        if t is None:
            return
        if self.clock_t is not None and t < self.clock_t:
            self.joint_hist.clear()  # simulation reset
        self.clock_t = t
        self.joint_hist.append((src, t, data))
        if not self._due("joints", t):
            return
        msg = self._joints_at(t)
        self._set_time(t)
        for i, n in enumerate(msg.name):
            if i < len(msg.position):
                self._scalar(f"joints/position/{n}", msg.position[i])
            if i < len(msg.velocity):
                self._scalar(f"joints/velocity/{n}", msg.velocity[i])
            if i < len(msg.effort):
                self.last_effort_meas[n] = msg.effort[i]
                self._scalar(f"joints/effort/{n}/measured", msg.effort[i])
                if self.torque_limit.get(n):
                    self._scalar(f"joints/effort_ratio/{n}",
                                 abs(msg.effort[i]) / self.torque_limit[n])
        self._log_held_state()

    def _log_held_state(self):
        """Commands and modes, sampled and held on the joint_states ticks."""
        if self.fsm_state is not None:
            self._scalar("fsm/state", self._fsm_code(self.fsm_state))
        if self.fsm_requested is not None:
            self._scalar("fsm/requested", self._fsm_code(self.fsm_requested))
        if self.keys_locked is not None:
            self._scalar("mpx/keys_locked", float(self.keys_locked))
        if self.cmd_twist is not None:
            tw = self.cmd_twist
            for k, v in (("vx", tw.linear.x), ("vy", tw.linear.y), ("vz", tw.linear.z),
                         ("wz", tw.angular.z)):
                self._scalar(f"command/twist/{k}", v)
            self._scalar("base/vel_heading/x/cmd", tw.linear.x)
            self._scalar("base/vel_heading/y/cmd", tw.linear.y)
            self._scalar("base/ang_vel_body/z/cmd", tw.angular.z)
        if self.cmd_pose is not None:
            p = self.cmd_pose.pose
            q = p.orientation
            rpy = quat_xyzw_to_rpy(q.x, q.y, q.z, q.w) if (q.x or q.y or q.z or q.w) else (0, 0, 0)
            self._scalar("command/pose/z", p.position.z)
            self._scalar("com/z/cmd", p.position.z)
            for a, v in zip(("roll", "pitch", "yaw"), rpy):
                self._scalar(f"command/pose/{a}", v)
                self._scalar(f"base/rpy/{a}/cmd", v)

    def _fsm_code(self, name):
        if name not in FSM_CODES:
            FSM_CODES[name] = max(FSM_CODES.values()) + 1
            self._log("events", rr.TextLog(f"FSM code {FSM_CODES[name]} = {name}"))
        return FSM_CODES[name]

    # ------------------------------------------------------------ ground truth
    def _on_link_states(self, data, src):
        if self.joint_hist and src > self.joint_hist[-1][0] + 1_000_000:
            # Published after the latest joint_states taken: wait for its step
            self.pending_links = self.pending_links[-50:] + [(data, src)]
            return
        t = self._sim_time(src)
        self.gt_raw.append((t, data))
        if not self._due("gt", t):
            return
        gt = self._gt_at(t)
        if gt is None:
            return
        self._set_time(t)
        self._log_values("gt", gt)
        if self._due("scene", t):
            self._log_scene(gt)

    def _gt_at(self, t):
        """Ground truth nearest to stamp t (within max_gt_dt_s), computed once per sample."""
        best = None
        for tb, data in reversed(self.gt_raw):
            if best is None or abs(tb - t) < abs(best[0] - t):
                best = (tb, data)
            if tb < t - self.max_gt_dt:
                break
        if best is None or abs(best[0] - t) > self.max_gt_dt:
            return None
        key = (best[0], id(best[1]))
        if key not in self.gt_cache:
            self.gt_cache[key] = self._gt_from(best[1])
            while len(self.gt_cache) > 64:
                self.gt_cache.popitem(last=False)
        return self.gt_cache[key]

    def _gt_from(self, data):
        msg = deserialize_message(data, LinkStates)
        link_T, link_v, link_w = {}, {}, {}
        for name, pose, twist in zip(msg.name, msg.pose, msg.twist):
            if not name.startswith(self.model_prefix):
                continue
            link = name[len(self.model_prefix):]
            q = pose.orientation
            link_T[link] = make_T(quat_xyzw_to_matrix(q.x, q.y, q.z, q.w),
                                  [pose.position.x, pose.position.y, pose.position.z])
            link_v[link] = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
            link_w[link] = np.array([twist.angular.x, twist.angular.y, twist.angular.z])
        if self.base_link not in link_T:
            return None
        gazebo_links = set(link_T)
        com = com_vel = None
        if self.urdf is not None:
            # Links lumped by Gazebo (fixed joints) move with their anchor.
            for link in self.urdf.links:
                if link in gazebo_links:
                    continue
                anchor = self.anchor_cache.get(link) or self.urdf.anchor_of(link, gazebo_links)
                if anchor is None:
                    continue
                self.anchor_cache[link] = anchor
                a, T_al = anchor
                link_T[link] = link_T[a] @ T_al
                link_w[link] = link_w[a]
                link_v[link] = link_v[a] + cross(link_w[a],
                                                    link_T[link][:3, 3] - link_T[a][:3, 3])
            com, com_vel = self.urdf.com(link_T, link_v, link_w)
        T = link_T[self.base_link]
        wheels = [link_T[w][:3, 3] for w in self.wheel_links if w in link_T]
        values = state_values(T[:3, 3], T[:3, :3], link_v[self.base_link],
                              link_w[self.base_link], com, com_vel, wheels)
        values["_links"] = {k: link_T[k] for k in gazebo_links}
        return values

    def _log_scene(self, gt):
        link_T = gt["_links"]  # links published by Gazebo; meshes hang below them
        if not self.static_scene_done and self.urdf is not None and self.log_meshes:
            for link, entry in self.urdf.links.items():
                anchor = self.urdf.anchor_of(link, link_T)
                if anchor is None:
                    continue
                a, T_al = anchor
                for i, (path, T_vis, scale) in enumerate(entry["visuals"]):
                    T = T_al @ T_vis
                    ent = f"world/robot/{a}/visual_{link}_{i}"
                    self._log(ent, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3],
                                                  scale=scale), static=True)
                    self._log(ent, rr.Asset3D(path=path), static=True)
            self.static_scene_done = True
        for link, T in link_T.items():
            self._log(f"world/robot/{link}",
                      rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        green = [SOURCES["gt"][0]]
        self.trails["base"].append(gt["base/position"].copy())
        self._log("world/trails/base", rr.LineStrips3D([list(self.trails["base"])],
                                                        colors=[[120, 120, 255]]))
        if gt.get("com") is not None:
            self.trails["com"].append(gt["com"].copy())
            self._log("world/gt/com", rr.Points3D([gt["com"]], radii=0.02, colors=green))
            self._log("world/gt/com_ground", rr.Points3D([[gt["com"][0], gt["com"][1], 0.0]],
                                                         radii=0.015, colors=green))
            self._log("world/trails/com", rr.LineStrips3D([list(self.trails["com"])],
                                                          colors=green))
        if "wheels/left" in gt:
            contacts = [gt[f"wheels/{w}"] - [0, 0, self.wheel_radius] for w in WHEELS]
            self._log("world/gt/support", rr.LineStrips3D([contacts], colors=green))
        # Filter (odom frame) and MPX (its base source frame) next to the GT robot
        for (src, what), pts in self.points_3d.items():
            self._log(f"world/{src}/{what}", rr.Points3D(pts, radii=0.018,
                                                         colors=[SOURCES[src][0]]))

    # ------------------------------------------------------------ filter
    def _on_filter(self, msg, t):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        R = quat_xyzw_to_matrix(q.x, q.y, q.z, q.w)
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        v_w = R @ np.array([v.x, v.y, v.z])  # twist is in the base frame
        w_w = R @ np.array([w.x, w.y, w.z])
        pos = np.array([p.x, p.y, p.z])
        com = com_vel = wheels = None
        joints = self._joints_at(t)
        if self.urdf is not None and joints is not None:
            qd = dict(zip(joints.name, joints.velocity))
            link_T, link_v, link_w = self.urdf.forward(
                make_T(R, pos), v_w, w_w, dict(zip(joints.name, joints.position)), qd)
            com, com_vel = self.urdf.com(link_T, link_v, link_w)
            wheels = [link_T[l][:3, 3] for l in self.wheel_links if l in link_T]
        values = state_values(pos, R, v_w, w_w, com, com_vel, wheels)
        self._set_time(t)
        self._log_values("filter", values)
        self._log_errors("filter", t, values)
        if com is not None:
            self.points_3d[("filter", "com")] = [com]
            self.points_3d[("filter", "wheels")] = wheels

    # ------------------------------------------------------------ robot sensors
    def _on_imu(self, msg, t):
        self._set_time(t)
        g = msg.angular_velocity
        a = msg.linear_acceleration
        q = msg.orientation
        gyro = np.array([g.x, g.y, g.z])
        for k, gv, av in zip("xyz", gyro, (a.x, a.y, a.z)):
            self._scalar(f"robot/imu/ang_vel/{k}", gv)
            self._scalar(f"robot/imu/lin_acc/{k}", av)
        values = {"base/rpy": quat_xyzw_to_rpy(q.x, q.y, q.z, q.w), "base/ang_vel_body": gyro}
        self._log_values("imu", values)
        self._log_errors("imu", t, values)

    def _on_chassis_odom(self, msg, t):
        # Planar odometry: twist in the heading frame, no z (see mpx_node merge_odometry).
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        values = {
            "base/position": np.array([p.x, p.y, np.nan]),
            "base/vel_heading": np.array([v.x, v.y, np.nan]),
            "base/ang_vel_body": np.array([np.nan, np.nan, msg.twist.twist.angular.z]),
        }
        self._set_time(t)
        self._log_values("odom", values)
        self._log_errors("odom", t, values)

    # ------------------------------------------------------------ MPX
    def _on_mpx_estimate(self, msg, quantity, src):
        t = stamp_to_sec(msg.header.stamp)
        vec = msg.point if hasattr(msg, "point") else msg.vector
        vec = np.array([vec.x, vec.y, vec.z])
        self._set_time(t)
        values = {quantity: vec}
        self._log_values("mpx", values)
        self._log_errors("mpx", t, values)
        if quantity == "com":
            self.mpx_stamps.append(t)
            if len(self.mpx_stamps) >= 2 and self.mpx_stamps[-1] > self.mpx_stamps[0]:
                self._scalar("mpx/rate_hz/estimate", (len(self.mpx_stamps) - 1)
                             / (self.mpx_stamps[-1] - self.mpx_stamps[0]))
            # sim time at publication - stamp of the joint sample used by the solve
            self._scalar("mpx/latency_ms", 1000.0 * (self._sim_time(src) - t))
            self.points_3d[("mpx", "com")] = [vec]
        # Balance once com, rpy and both wheels of the same solve have arrived
        # (topics are read one after the other, so several solves are open at once)
        parts = self.mpx_parts.setdefault(t, {})
        parts[quantity] = vec
        while len(self.mpx_parts) > 20:
            self.mpx_parts.pop(next(iter(self.mpx_parts)))
        needed = ("com", "base/rpy", "wheels/left", "wheels/right")
        if all(k in parts for k in needed):
            mid = 0.5 * (parts["wheels/left"] + parts["wheels/right"])
            off = yaw_rotation(parts["base/rpy"][2]).T @ (parts["com"] - mid)
            self._scalar("balance/com_ahead_of_wheels/mpx", off[0])
            self._scalar("balance/com_lateral/mpx", off[1])
            self._scalar("balance/com_height_over_wheels/mpx", off[2])
            self.points_3d[("mpx", "wheels")] = [parts["wheels/left"], parts["wheels/right"]]
            del self.mpx_parts[t]

    def _on_mpx_effort(self, msg, src):
        t = self._sim_time(src)
        self._set_time(t)
        names = self.joints if len(self.joints) == len(msg.data) else \
            [f"cmd_{i}" for i in range(len(msg.data))]
        for n, tau in zip(names, msg.data):
            self._scalar(f"joints/effort/{n}/mpx_cmd", tau)
            if n in self.last_effort_meas:
                self._scalar(f"joints/effort_clamp/{n}", tau - self.last_effort_meas[n])
        self.effort_times.append(t)
        if len(self.effort_times) >= 2 and self.effort_times[-1] > self.effort_times[0]:
            self._scalar("mpx/rate_hz/effort", (len(self.effort_times) - 1)
                         / (self.effort_times[-1] - self.effort_times[0]))

    def _on_keys_locked(self, msg):
        if msg.data != self.keys_locked:
            self._event("MPX keys locked", str(msg.data))
        self.keys_locked = msg.data

    def _on_fsm_state(self, msg):
        if msg.data != self.fsm_state:
            self._event("FSM state", msg.data, "WARN")
        self.fsm_state = msg.data

    def _on_cmd_key(self, msg):
        self._event("key command", msg.data)
        self.fsm_requested = msg.data

    def _on_cmd_twist(self, msg, _src):
        self.cmd_twist = msg

    def _on_cmd_pose(self, msg, _src):
        self.cmd_pose = msg

    # ------------------------------------------------------------ extra topics
    def _discover_extra_topics(self):
        if get_message is None:
            return
        pending = [t for t in self.extra_topics if t not in self.extra_subs]
        if not pending:
            return
        ns = self.get_namespace().rstrip("/")
        available = dict(self.get_topic_names_and_types())
        for topic in pending:
            full = topic if topic.startswith("/") else f"{ns}/{topic}"
            types = available.get(full)
            if not types:
                continue
            try:
                msg_type = get_message(types[0])
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Cannot import {types[0]} for {full}: {e}")
                self.extra_subs[topic] = None
                continue
            prefix = "extra" + full
            self.rates[prefix] = self.rates["extra"]
            self.extra_subs[topic] = True
            self._stream(None, msg_type, lambda d, src, p=prefix, m=msg_type:
                         self._on_extra(d, src, p, m), raw=True, topic=full)
            self.get_logger().info(f"Logging extra topic {full} [{types[0]}]")

    def _on_extra(self, data, src, prefix, msg_type):
        t = self._sim_time(src)
        if not self._due(prefix, t):
            return
        msg = deserialize_message(data, msg_type)
        values = {}
        flatten_numeric(msg, "", values)
        self._set_time(t)
        for k, v in values.items():
            self._scalar(f"{prefix}/{k}", v)


def report_saved_file(path):
    """Print where the recording was written (the ROS logger may be gone at shutdown)."""
    if not path:
        return
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[tita_rerun_logger] Recording saved: {path} ({size_mb:.1f} MB)", flush=True)
        print("[tita_rerun_logger] Open it with: ros2 run tita_rerun_logger view_logs "
              f"'{path}'", flush=True)
    else:
        print(f"[tita_rerun_logger] WARNING: expected recording not found: {path}", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = RerunLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():  # errors after a signal shut the context down are expected
            raise
    finally:
        node.flush()
        rr.disconnect()  # flushes and closes the .rrd file
        report_saved_file(node.rrd_path)
        node.stream_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
