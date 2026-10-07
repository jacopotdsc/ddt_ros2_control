# MPX integration: changes to the upstream workspace

This document lists every change made to the upstream code (commit
`63e9e6e fix: urdf joint frictions and dampings`) so that TITA can switch live
between the RL policies and the MPX controller (MPC + WBC, package `mpx`), in
Gazebo and on the robot. Nothing is committed.

Target setup: **TITA, Gazebo Classic 11, ROS 2 Humble**. Default namespace is
empty; every topic below is relative, so a namespace moves all of them.

Snapshots: `checkpoints/mpx_baseline_2026-09-28/` (first working integration) and
`checkpoints/checkpoint2/` (this state, 2026-09-30). See `checkpoint_claude_workspace.md`.

---

## 1. Overview

```
keyboard_controller ──command/cmd_key──────┬──> rl_controller (FSM: idle, transform_up, rl_*, mpc, ...)
                    ──command/cmd_twist────┤
                    ──command/cmd_pose─────┤
                                           └──> mpx_node (node /controller)
rl_controller ──fsm_state──────────────────────> mpx_node, keyboard_controller  (current FSM state)
gazebo (gazebo_ros_state) ──/model_states──────> mpx_node      (base_state_source: ground_truth)
tita_state_estimator ──filtered_state──────────> mpx_node      (filter / odometry)
robot chassis ──/tita4267305/chassis/odometry──> mpx_node      (odometry: x, y, horizontal velocity)
ros2_control ──joint_states, imu_sensor_broadcaster/imu, robot_description──> estimator, mpx_node
mpx_node ──mpx/effort──────> rl_controller: the "mpc" state applies the torques
mpx_node ──mpx/handoff─────> rl_controller: leaves "mpc" for rl_N / transform_down
mpx_node ──mpx/keys_locked─> keyboard_controller: locks the motion keys
```

Shared topic names are in `ros_utils/include/ros_utils/topic_names.hpp` (C++) and in
`rl_controller/config/tita/mpx.yaml` (MPX). No launch file sets topic names.

### 1.1 Transitions

| From | Key | What happens |
|---|---|---|
| `rl_N`, `transform_up`, `joint_pd` | `4` | The keyboard zeroes the twist, sets the height to 0.4 m and locks the motion keys. The first time MPX JIT-compiles (~25–45 s) while the current state keeps the robot. MPX then runs 10 warm-up solves on the live state, publishes its first torque, and the FSM switches to `mpc`. MPX ramps the COM from its measured height to `h_mpc = 0.4` m at 0.1 m/s and unlocks the keys once it holds there. |
| `idle`, `transform_down` | `4` | Refused: the FSM never enters `mpc` from these states. MPX stays in standby and the keys are not locked. |
| `mpc` | `0` (`rl_0`) | MPX zeroes the velocity, lowers the COM to `h_rl = 0.31` m, publishes `rl_0` on `mpx/handoff`; only then the FSM switches. Keys locked meanwhile. |
| `mpc` | `8` | Same handoff, to `h_transform_down = 0.28` m (≈ the lowest COM MPC reaches), then `transform_down` folds the legs. Folding from the MPC height without this made the robot fall. |
| `mpc` | `6`, `9` | Immediate (emergency exits). |
| `mpc` | `1`–`3` | Ignored when the policy does not exist (`mpc_rl_handoff.rl_modes`); on TITA only `rl_0` exists. |
| `mpc` | `7`, `5` | Ignored (the FSM never leaves `mpc` for them). |
| during a handoff | `4` | Cancels it; MPX goes back to `h_mpc`. |
| during a handoff | `0` / `8` | Changes the target and its height. |
| during compilation | `0` | MPX goes to standby, the keys unlock at once; compilation finishes in the background and the next `4` does not recompile. |

---

## 2. Changes per package

### 2.1 `controller/rl_controller`

| File | Change |
|---|---|
| `include/rl_controller/fsm/FSMState_MPC.h`, `src/fsm/FSMState_MPC.cpp` | **New** FSM state `mpc`. Applies the latest MPX torque (pure torque control, `kp = kd = 0`, clamped to `torque_limit`), holds the previous command until a valid one arrives. Leaves at once for `idle` / `joint_pd`; for `rl_N` and `transform_down` only after MPX publishes that mode on `mpx/handoff`. `FSMState_MPC::ready()` is true once a valid torque was received. |
| `include/rl_controller/fsm/ControlFSMData.h` | New `MpcCommand` (mutex, torque vector, `valid`, `handoff`) in `ControlFSMData`. |
| `include/rl_controller/fsm/FSM.h`, `src/fsm/FSM.cpp` | Registers the `mpc` state. |
| `src/fsm/FSMState_RL.cpp`, `FSMState_JointPD.cpp`, `FSMState_TransformUp.cpp` | Go to `mpc` when requested **and** `FSMState_MPC::ready()`. `TransformUp` also waits 100 iterations after its stand-up ramps. |
| `src/rl_controller_node.cpp`, `include/rl_controller/rl_controller_node.hpp` | Subscribes to `ros_topic::mpx_effort` (`Float64MultiArray`; wrong size or non-finite values are rejected) and `ros_topic::mpx_handoff` (`String`). A new `mpc` request clears `valid` and `handoff`. `residual` only logs a warning (not implemented). **Publishes the current FSM state** on `ros_topic::fsm_state` (`String`, reliable + transient local, on change, `realtime_tools::RealtimePublisher` in `update()`). Generated-parameters include changed to `rl_controller/rl_controller_parameters.hpp` (removes a deprecation note). |
| `CMakeLists.txt` | Adds `src/fsm/FSMState_MPC.cpp` and the `command_watchdog` executable. |
| `src/command_watchdog.cpp` | **New** robot-side node. Armed by the first `command/cmd_twist` (keyboard or remote); when no twist arrives for `timeout` s (default 1.0) it publishes `transform_down` on `command/cmd_key` every 100 ms, so the robot folds through the normal path (from MPC via the MPX handoff). Stops as soon as commands come back. |
| `package.xml` | `depend realtime_tools`; `exec_depend mpx`, `tita_state_estimator`. |
| `config/tita/mpx.yaml` | **New.** MPX parameters for the switching, loaded on top of `mpx/config/mpx_node.yaml`: `publish_default_commands: false`, all topics relative (`joint_states`, `robot_description`, `filtered_state`, `command/*`, `fsm_state`, `mpx/*`), `mpc_rl_handoff.rl_modes: ["rl_0"]` (one per `rl_policy_names` entry in `controllers.yaml`). |
| `launch/sim_gazebo.launch.py` | See 2.1.1. |
| `launch/hw.launch.py` | See 2.1.2. |

#### 2.1.1 `sim_gazebo.launch.py`

- **Namespace** `ns` (default empty, as upstream). The Gazebo entity is `ns` or, if empty, the robot name (`tita`); MPX looks it up in `/model_states`.
- **`gzserver` and `gzclient` started separately.** `gzclient` runs as `nice -n <gui_nice> gzclient -g libtita_gui_render_rate.so` with `TITA_GUI_FPS=<gui_fps>`, same model/plugin/resource paths as `gazebo_ros/gzclient.launch.py`.
- **`tita_state_estimator`** always started for TITA (`use_sim_time`), so it can be compared with `/model_states` even when MPX uses the ground truth.
- **MPX** started in standby when `enable_mpx:=true` (default for TITA), in the robot namespace, with `mpx_node.yaml` + `config/tita/mpx.yaml` and only two overrides: `inputs.model_name` and `inputs.base_state_source`.
- **`rtf`**: below 1 the launch writes `/tmp/tita_empty_world_rtf<rtf>.world` with `real_time_update_rate` scaled, so MPX (not in lockstep with Gazebo) gets more wall time per control period. Controllers see the same simulated time.
- **xacro:** looks for `<robot>/xacro/robot.xacro`, then `xacro/robot.xacro`; passes `sim_env`, `ctrl_mode`, `yaml_path` (ignored by the current descriptions). `package://` URIs resolved only for installed description packages. Logs the physics settings.

| Argument | Default | Meaning |
|---|---|---|
| `gui` | `true` | start `gzclient` |
| `gui_fps` | `20` | GUI render-rate cap (FPS) |
| `gui_nice` | `10` | `nice` level of `gzclient` |
| `enable_mpx` | `true` if `robot:=tita` | start MPX (TITA only) |
| `mpx_base_state` | `ground_truth` | `ground_truth`, `filter` or `odometry` |
| `rtf` | `1.0` | target real-time factor |

#### 2.1.2 `hw.launch.py`

Aligned with `sim_gazebo.launch.py` and with `hardware/hardware_bridge/launch/hardware_bridge.launch.py`:

- `namespace` argument, default `$ROBOT_NS` (as `hardware_bridge.launch.py`); the keyboard must run in the same namespace (`--ros-args -r __ns:=/<ns>`).
- `tita_state_estimator` always started for TITA; MPX started when `enable_mpx:=true` (default for TITA) with the same parameter files as in simulation.
- `mpx_base_state`: `filter` (default) or `odometry` (no ground truth on the robot).
- `command_timeout` (default `1.0` s, `0` disables): starts `command_watchdog`. **The launch must
  survive the disconnection** (run it on the robot in tmux / nohup / systemd): if it is started
  from a bare ssh session, losing the link kills it and the torques go to zero (the robot collapses).
- Unused imports removed.

### 2.2 `mpx` (separate package, its own git repository)

The solver (`mpx_tita`) is unchanged; only the ROS node and its config.

| File | Change |
|---|---|
| `scripts/mpx_node.py` | See below. |
| `config/mpx_node.yaml` | All parameters (the node has no defaults): `inputs.base_state_source`, `filtered_state_topic`, `odometry_topic`, `fsm_state_topic`, `cmd_vel_topic`, `com_height_topic`, `outputs.keys_locked_topic`, `estimate_topic_prefix`, `mpc_rl_handoff.{h_mpc, h_rl, h_transform_down, rl_modes, height_tol, vz_tol, settle_time, timeout}`. |
| `package.xml` | `depend nav_msgs`. |

`mpx_node.py`:
- **Base state source** (`inputs.base_state_source`): `ground_truth` (Gazebo `ModelStates`), `filter` (`nav_msgs/Odometry` from `tita_state_estimator`, twist rotated from the base frame to the world), `odometry` (x, y and horizontal velocity from the chassis odometry, twist taken as planar in the heading frame and rotated with the IMU yaw; z, vz, orientation and angular velocity from the filter). A sample is used only while every source is fresher than 0.1 s; otherwise `MPX waiting for data on: <topics>` every 2 s.
- **All topics from the config**, none written in the code.
- **Mode handling:** unknown `rl_N` ignored (`rl_modes`); `mpc` refused while the FSM (`fsm_state`) is in `idle` / `transform_down`; `transform_up` ignored while MPX drives the robot; handoff to `rl_N` at `h_rl` and to `transform_down` at `h_transform_down`; a handoff can be cancelled (`4`) or retargeted.
- **Key lock** (`keys_locked`): `(compiling and mpc requested) or ramping to h_mpc or handing off`; published on change and re-published on every mode change, so the keyboard lock set on a key press is always confirmed or cleared.
- **Takeover:** after compilation `gc.collect()` + `gc.freeze()`; on every activation `WARMUP_STEPS = 10` solves on the live state before the first published torque.
- **Wheel distance** `config.d` computed from `q0` (the posture MPC tracks, 0.567 m), not from the spawn posture.

### 2.3 `tita_state_estimator` (new package)

Kalman filter `StateFilter_no_bias.hpp` from
[TITA-dynamic-obstacle-avoidance](https://github.com/Emilianogith/TITA-dynamic-obstacle-avoidance).
State `[p_base, v_base, p_contact_L, p_contact_R]`; prediction from the accelerometer and
wheel rolling kinematics, correction from base→contact kinematics with the contacts at
z = 0 (so **z and vz are estimated**); orientation and angular velocity from the IMU.
Node: steps on every `joint_states` message using message stamps; publishes
`filtered_state` (`nav_msgs/Odometry`, pose in `odom`, twist in `base_link`); topics from
`ros_topic::`; parameters `max_input_age` (0.1 s) and `gyro_offset` ([0, 0, 0], to calibrate
on the robot); service `~/reset`. Gravity in the filter 9.81. Details: `filter_implementation_recap.md`.

### 2.4 `interaction/keyboard_controller`

| Change | Details |
|---|---|
| Key mapping | `4: mpc` (was `rl_4`), `5: residual` (was `jump`). |
| `4` | Zeroes the twist, height `MPC_START_HEIGHT = 0.4` (= `h_mpc`), locks the motion keys (if MPX runs). |
| `0`–`3`, `8` from MPC | Zero the twist; the lock stays until MPX hands off. "From MPC" means the goal is `mpc` or the FSM is in `mpc`. |
| `7`, `5` while the FSM is in `mpc` | Ignored. |
| Lock | `mpx/keys_locked` (`Bool`, reliable, transient local). While locked, motion keys are read (escape sequences consumed) and undone; mode keys, `Space`, `Ctrl+C` work. Red line in the UI. Cleared if nobody publishes the topic (MPX not running). |
| FSM state | Subscribes to `fsm_state`; the UI shows `(FSM: x)` when it differs from the requested mode. |
| Controller gone | If the `*_rl_controller` subscriber of `command/cmd_key` disappears, the mode goes back to `idle`. |
| New keys | `PgUp` / `PgDn` height ± `STEP_HEIGHT`; `Space` zeroes velocities and rpy, keeps the height. |
| Limits | `STEP_HEIGHT 0.1 → 0.01`, `MIN_HEIGHT 0.1 → 0.25`, `MAX_HEIGHT 0.5 → 0.44`. MPC does not go below COM ≈ 0.278 m. |

### 2.5 `ros_utils`

`topic_names.hpp`: `fsm_state`, `robot_description`, `filtered_state`, `mpx_effort` (`mpx/effort`),
`mpx_handoff`, `mpx_keys_locked`.

### 2.6 `simulation/gazebo_bridge`

| File | Change |
|---|---|
| `worlds/empty_world.world` | ODE QuickStep **300 iterations** (step 0.002 s, 500 Hz unchanged); **`libgazebo_ros_state.so`** publishing `/model_states` at 500 Hz; `<gravity>0 0 -9.81</gravity>`. |
| `src/gui_render_rate.cpp`, `CMakeLists.txt` | **New** gzclient system plugin `libtita_gui_render_rate.so`: caps the GUI render rate to `TITA_GUI_FPS` (default 20). Load it with `-g` (with `--gui-client-plugin` gzclient crashes). |

### 2.7 Other

- `hardware/hardware_bridge/src/hardware_bridge_node.cpp`: the destructor sends zero joint torques on exit (made outside this integration). Note: it dereferences `robot_` without a null check.
- `simulation/mujoco_bridge/COLCON_IGNORE`, `TITA-dynamic-obstacle-avoidance/COLCON_IGNORE` (reference clone, not needed for the build).

---

## 3. Build and run

```bash
cd ~/ddt_ros2_ws && colcon build --symlink-install && source install/setup.bash

# simulation (GUI 20 FPS, nice 10) + rl_controller + estimator + MPX in standby
ros2 launch rl_controller sim_gazebo.launch.py                      # mpx_base_state:=ground_truth
ros2 launch rl_controller sim_gazebo.launch.py rtf:=0.8             # loaded CPU / GUI visible

# robot
ros2 launch rl_controller hw.launch.py                              # mpx_base_state:=filter, namespace:=$ROBOT_NS
ros2 launch rl_controller hw.launch.py mpx_base_state:=odometry

# keyboard (same namespace as the launch)
ros2 run keyboard_controller keyboard_controller_node
```

Do not also start `ros2 launch mpx mpx.launch.py`: it would start a second MPX node.

---

## 4. Validation (Gazebo, GUI, ground truth)

| Test | Conditions | Result |
|---|---|---|
| `7 → 4 →` forward 0.4, turn, back, PgDn/PgUp `→ 8` | `rtf 0.8` | ok, invalid=0, tilt ≤ 4.8° in MPC, fold 32.5° |
| `7 → 0 → 4 →` loco `→ 0` (turning at 0.4) `→` RL `→ 4 →` loco `→ 8` | `rtf 0.8` | ok, invalid=0, tilt ≤ 5.1° in MPC, fold 35° |
| `4` from idle, `7 → 4` (compile), loco, `8`, `7 → 4` (compiled), `0`, `8` from RL | `rtf 0.8` | ok, invalid=0, 99–100 Hz |
| `0` during compilation, `1`/`5`/`7` in MPC, `0` then `8` during handoff, `4` cancel, `4` during the fold | `rtf 0.8` | all as in 1.1, invalid=0 |
| RL reference `7 → 0 → 8` | | fold 34° |

Handoff and unlock log lines: `COM at h_mpc=0.400 m: MPC commands unlocked`,
`COM at 0.310 m: handing off to rl_0`, `COM at 0.280 m: handing off to transform_down`.

---

## 5. Known limits

- **CPU / control rate.** MPX is not in lockstep with Gazebo; below ~75 Hz MPC diverges.
  On this laptop (i7-10510U) with the GUI visible, `rtf 1` gave 53–64 Hz right after the
  takeover in the afternoon of 2026-09-30: `7 → 4` fell or shook badly in 5 of 5 runs
  (in the morning, at 100 Hz, it worked 3 of 3); at `rtf 0.8` it worked every time.
  The MPX process itself uses ~1–2 cores (Python callbacks for ~1000 messages/s compete with
  the solver thread). Use `rtf:=0.8` or `7 → 0 → 4`; power profile "performance" helps but is not enough.
- **Takeover from `transform_up`** starts from a crouched posture: the hip pitch swings at
  2–3 Hz for ~1.5 s even at 100 Hz (damped), and diverges at low rates.
- **Filter as MPX source:** in earlier tests MPC fell 2 of 2 times while moving with `filter`
  (0 of 3 with the ground truth); the filter velocity diverges when the wheels bounce.
- **`odometry` source not tested** (no chassis odometry in Gazebo; `checkpoints/test_tools/fake_chassis_odom.py`
  simulates it). Check on the robot that `twist.linear.x` is along the heading.
- If MPX stops while the FSM is in `mpc`, `FSMState_MPC` keeps applying the last torque until another mode is chosen.
- `FSM::run` clamps `tau_cmd` with a discarded `std::clamp` (upstream, no effect); `FSMState_MPC` clamps its own torques.
