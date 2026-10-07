"""Entity layout, source colours and viewer tabs, shared by the ROS node and the
offline viewer script (no ROS imports)."""

import rerun.blueprint as rrb

APPLICATION_ID = "tita_mpx"

TITA_JOINTS = (
    "joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3", "joint_left_leg_4",
    "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3", "joint_right_leg_4",
)

# Every compared quantity is logged as <quantity>/<axis>/<source>, so that one plot per
# axis shows all sources on the same axes. Colour and legend name are fixed per source.
SOURCES = {
    "gt":     ([40, 170, 40],  "GT (Gazebo)"),
    "filter": ([40, 110, 230], "filter (tita_state_estimator)"),
    "imu":    ([160, 60, 200], "robot: IMU"),
    "odom":   ([200, 50, 120], "robot: chassis odometry"),
    "mpx":    ([240, 130, 20], "MPX belief"),
    "cmd":    ([90, 90, 90],   "command"),
}

XYZ = ("x", "y", "z")
RPY = ("roll", "pitch", "yaw")
WHEELS = ("left", "right")

# quantity -> (axes, sources that can provide it)
COMPARED = {
    "base/position":     (XYZ, ("gt", "filter", "odom")),
    "base/rpy":          (RPY, ("gt", "filter", "imu", "mpx", "cmd")),
    "base/vel_heading":  (XYZ, ("gt", "filter", "odom", "cmd")),
    "base/ang_vel_body": (XYZ, ("gt", "filter", "imu", "odom", "cmd")),
    "com":               (XYZ, ("gt", "filter", "mpx", "cmd")),
    "com_vel":           (XYZ, ("gt", "filter", "mpx")),
    "wheels/left":       (XYZ, ("gt", "filter", "mpx")),
    "wheels/right":      (XYZ, ("gt", "filter", "mpx")),
}
BALANCE = {
    # CoM - midpoint of the wheel centres, in the heading frame of the same source
    "balance/com_ahead_of_wheels": ("gt", "filter", "mpx"),
    "balance/com_lateral": ("gt", "filter", "mpx"),
    "balance/com_height_over_wheels": ("gt", "filter", "mpx"),
    "balance/base_height_over_wheels": ("gt", "filter"),
}
# Estimate - ground truth, logged as errors/<source>/<quantity>/<axis|norm>
ERROR_SOURCES = ("filter", "imu", "odom", "mpx")

# Numeric codes of the FSM states, to plot them next to the signals
FSM_CODES = {
    "idle": 0, "transform_down": 1, "transform_up": 2, "joint_pd": 3,
    "rl_0": 4, "rl_flat": 4,  # TITA: key 0 = rl_0 runs the policy "rl_flat"
    "rl_1": 5, "rl_2": 6, "rl_3": 7, "mpc": 8, "residual": 9, "jump": 10,
}


def _ts(origin, name, **kw):
    return rrb.TimeSeriesView(origin=origin, name=name, **kw)


def _axis_row(quantity, title, axes):
    return rrb.Horizontal(*[_ts(f"/{quantity}/{a}", f"{title} {a}") for a in axes])


def _joint_grid(prefix, joints, title):
    return rrb.Grid(*[_ts(f"/{prefix}/{j}", f"{title} {j.replace('joint_', '')}") for j in joints],
                    grid_columns=4)


def make_blueprint(joints=TITA_JOINTS):
    joints = tuple(joints)
    left = [j for j in joints if "left" in j] or list(joints)
    right = [j for j in joints if "right" in j]

    overview = rrb.Vertical(
        _ts("/", "FSM state (code) / requested / MPX keys locked",
            contents=["+ /fsm/**", "+ /mpx/keys_locked"]),
        _ts("/base/vel_heading/x", "Forward velocity [m/s]"),
        _ts("/base/ang_vel_body/z", "Yaw rate [rad/s]"),
        _ts("/com/z", "CoM height [m]"),
        _ts("/base/rpy/pitch", "Base pitch [rad]"),
        _ts("/balance/com_ahead_of_wheels", "CoM ahead of the wheel axis [m]"),
        name="Overview",
    )
    base_pose = rrb.Vertical(
        _axis_row("base/position", "Base position", XYZ),
        _axis_row("base/rpy", "Base", RPY),
        name="Base pose",
    )
    base_vel = rrb.Vertical(
        _axis_row("base/vel_heading", "Base velocity (heading frame; z world)", XYZ),
        _axis_row("base/ang_vel_body", "Angular velocity (body)", XYZ),
        _axis_row("robot/imu/lin_acc", "IMU acceleration", XYZ),
        name="Base velocity",
    )
    com = rrb.Vertical(
        _axis_row("com", "CoM", XYZ),
        _axis_row("com_vel", "CoM velocity", XYZ),
        _axis_row("wheels/left", "Left wheel", XYZ),
        _axis_row("wheels/right", "Right wheel", XYZ),
        name="CoM & wheels",
    )
    balance = rrb.Vertical(
        *[_ts(f"/{k}", k.split("/")[1].replace("_", " ") + " [m]") for k in BALANCE],
        name="Balance",
    )
    errors = rrb.Horizontal(
        *[rrb.Vertical(*[_ts(f"/errors/{src}/{q}", f"{src} - GT: {q}")
                         for q, (_, srcs) in COMPARED.items() if src in srcs],
                       name=f"{src} - GT")
          for src in ERROR_SOURCES],
        name="Errors vs GT",
    )
    joints_tab = rrb.Vertical(
        rrb.Horizontal(_ts("/joints/position", "Joint position [rad]",
                           contents=[f"+ /joints/position/{j}" for j in left]),
                       _ts("/joints/position", "Joint position [rad]",
                           contents=[f"+ /joints/position/{j}" for j in right])),
        rrb.Horizontal(_ts("/joints/velocity", "Joint velocity [rad/s]",
                           contents=[f"+ /joints/velocity/{j}" for j in left]),
                       _ts("/joints/velocity", "Joint velocity [rad/s]",
                           contents=[f"+ /joints/velocity/{j}" for j in right])),
        name="Joints",
    )
    torques = rrb.Vertical(
        _joint_grid("joints/effort", joints, "Torque [Nm]"),
        _ts("/joints/effort_ratio", "|measured torque| / torque_limit (1 = clamped)"),
        _ts("/joints/effort_clamp", "MPX command - measured torque [Nm]"),
        row_shares=[3, 1, 1],
        name="Torques",
    )
    mpx = rrb.Vertical(
        _ts("/mpx/rate_hz", "MPX control rate [Hz, sim time]"),
        _ts("/mpx/latency_ms", "MPX latency: joint sample -> estimate published [ms, sim time]"),
        _ts("/command/twist", "Velocity command (command/cmd_twist)"),
        _ts("/command/pose", "Pose command (command/cmd_pose)"),
        rrb.TextLogView(origin="/events", name="Events"),
        name="MPX & commands",
    )
    robot = rrb.Vertical(
        _axis_row("robot/imu/ang_vel", "IMU gyro", XYZ),
        _ts("/extra", "Other topics (extra_topics)"),
        name="Robot / extra",
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="/world", name="3D (GT robot)"),
                rrb.TextLogView(origin="/events", name="Events"),
                row_shares=[3, 1],
            ),
            rrb.Tabs(overview, base_pose, base_vel, com, balance, errors,
                     joints_tab, torques, mpx, robot),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )
