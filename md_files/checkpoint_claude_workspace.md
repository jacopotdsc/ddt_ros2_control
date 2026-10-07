# Claude workspace checkpoints — MPX integration

This file lets you (or Claude) check and restore the frozen snapshots of
`~/ddt_ros2_ws/src`. Experiments made after a snapshot are temporary and are
checked against, or reverted to, it.

What the code contains is described in [`MPX_INTEGRATION_CHANGES.md`](MPX_INTEGRATION_CHANGES.md)
(changes to upstream `63e9e6e`). Nothing is committed.

---

## 1. Snapshots

Both are under `~/ddt_ros2_ws/checkpoints/` (the folder has a `COLCON_IGNORE`).

| Name | Date | Content |
|---|---|---|
| `mpx_baseline_2026-09-28` | 2026-09-28 | first working RL ↔ MPC switching (724 files) |
| `checkpoint2` | 2026-09-30 | current state: estimator, relative topics, `fsm_state`, handoff to `transform_down`, transition fixes, `odometry` source, `rtf`, hw namespace |

Each snapshot folder has:

| File | Content |
|---|---|
| `src.tar.gz` | `src/`, excluding `.git`, `jax_cache`, `__pycache__` (and, for `checkpoint2`, the reference clone `TITA-dynamic-obstacle-avoidance`) |
| `sha256.txt` | hash of every file, paths relative to `src/` |
| `git_heads.txt` | HEAD and number of uncommitted changes of each nested repo |

Nested git repositories (all changes uncommitted, same HEADs in both snapshots):

| Repo (under `src/`) | HEAD |
|---|---|
| `.` | `63e9e6e` |
| `interaction` | `40b825f` |
| `mpx` | `70c3e6f` |
| `mpx/mpx_tita` | `78bdc59` |
| `controller/rl_controller` | `6d42dda` |

### What changed from `mpx_baseline_2026-09-28` to `checkpoint2`

- New package `tita_state_estimator`; MPX `base_state_source` `ground_truth` / `filter` / `odometry`.
- Topics: no hardcoded names, relative names from `ros_utils/topic_names.hpp` and
  `rl_controller/config/tita/mpx.yaml` (new); MPX and estimator in the robot namespace, MPX node `/controller`.
- `rl_controller` publishes `fsm_state`; MPX and keyboard use it.
- `8` from MPC: handoff at COM 0.28 m before `transform_down`.
- Transition fixes: unknown `rl_N` in MPC, `4` from idle / transform_down, stuck keyboard lock,
  lock during compilation after `0`.
- MPX warm-up solves + `gc.freeze()` at takeover; `config.d` from `q0` restored.
- Keyboard refactor (`ReadKeyThread`), no yaml reading.
- `sim_gazebo.launch.py`: `rtf`; `hw.launch.py`: `namespace`, `mpx_base_state`, estimator always on.
- Gravity 9.81 in the world file and in the filter.

---

## 2. Check whether anything changed

```bash
cd ~/ddt_ros2_ws/src
C=../checkpoints/checkpoint2            # or ../checkpoints/mpx_baseline_2026-09-28

# modified or deleted files (no output = identical)
sha256sum -c --quiet $C/sha256.txt

# files added after the snapshot
comm -13 <(cut -c67- $C/sha256.txt | sort) \
         <(find . \( -name .git -o -name jax_cache -o -name __pycache__ \
                     -o -path ./TITA-dynamic-obstacle-avoidance \) -prune -o -type f -print | sort)
```

---

## 3. Restore a snapshot

This overwrites `src/` with the snapshot. `.git`, `jax_cache`, `__pycache__`,
the reference clone and this file are left untouched. Files added after the snapshot are deleted.

```bash
cd ~/ddt_ros2_ws
C=checkpoints/checkpoint2               # or checkpoints/mpx_baseline_2026-09-28
TMP=$(mktemp -d)
tar -xzf $C/src.tar.gz -C $TMP
EXCL="--exclude .git --exclude jax_cache --exclude __pycache__ \
      --exclude checkpoint_claude_workspace.md --exclude TITA-dynamic-obstacle-avoidance"

rsync -a --delete --dry-run --itemize-changes $EXCL $TMP/src/ src/   # dry run first
rsync -a --delete $EXCL $TMP/src/ src/                              # then apply
rm -rf $TMP

(cd src && sha256sum -c --quiet ../$C/sha256.txt && echo "snapshot restored")
colcon build --symlink-install && source install/setup.bash
```

To restore one file only: `tar -xzf $C/src.tar.gz -C /tmp src/<path>` and copy it back.

---

## 4. How to run

```bash
cd ~/ddt_ros2_ws && colcon build --symlink-install && source install/setup.bash   # alias: goc

# terminal 1 — Gazebo GUI + rl_controller + estimator + MPX standby
ros2 launch rl_controller sim_gazebo.launch.py                    # alias: sim
#   gui:=false | gui_fps:=N | gui_nice:=N | enable_mpx:=false
#   mpx_base_state:=ground_truth|filter|odometry | rtf:=0.8

# terminal 2 — keyboard
ros2 run keyboard_controller keyboard_controller_node             # alias: joystick_tita
```

**Keys:** `7` transform_up, `0`–`3` rl_N, `4` mpc, `6` idle, `8` transform_down, `9` joint_pd;
`w`/`s` forward velocity, `a`/`d` yaw rate, arrows y/z velocity; `PgUp`/`PgDn` height;
`i`/`k`/`j`/`l`/`u`/`o` orientation; `Space` stops everything and keeps the height; `r` resets the velocity.

**Sequences:** `7 → 4`, `7 → 0 → 4`, then `0` (RL) or `8` (fold) from MPC. The first `4`
compiles MPX (~25–45 s) while the current state keeps the robot; motion keys are locked
during compilation, the ramp to 0.4 m and every handoff. Transition table:
`MPX_INTEGRATION_CHANGES.md` §1.1.

Do not start `ros2 launch mpx mpx.launch.py` together with the launch above.

---

## 5. Key facts to resume quickly

- **Topics (empty namespace):** keyboard `/command/cmd_key|cmd_twist|cmd_pose`; FSM `/fsm_state`;
  MPX `/mpx/effort`, `/mpx/handoff`, `/mpx/keys_locked`, `/mpx/estimate/*`; estimator `/filtered_state`;
  Gazebo `/model_states` (500 Hz), `/joint_states`, `/robot_description`; robot chassis
  `/tita4267305/chassis/odometry` (absolute). **MPX node name: `/controller`.**
- **Heights (COM):** `h_mpc = 0.4` (= keyboard `MPC_START_HEIGHT`), `h_rl = 0.31`,
  `h_transform_down = 0.28`; ramp 0.1 m/s; `height_tol 0.03`, `vz_tol 0.05`, timeout 5 s.
- **CPU margin:** each MPX solve must fit in 10 ms of wall time; below ~75 Hz MPC diverges.
  With the GUI visible use `rtf:=0.8` (100 Hz, invalid=0 in every test). Power profile
  "performance" helps; power saver makes it much worse.
- **GUI plugin:** load `libtita_gui_render_rate.so` with `gzclient -g`, not `--gui-client-plugin`.
- **Logs:** MPX `Control: X Hz sim, solve ms median=..`, `First invalid control`, handoff/unlock
  lines; FSM `FSM state changed from ... to ...`. Per-node logs in `~/.ros/log/` (MPX is `python3_<pid>_*.log`).
- **Test tools** (`checkpoints/test_tools/`, not part of the workspace):
  - `sim.sh <name> [launch args]` restarts the sim; run it alone in a command (its `pkill -f` matches its own command line).
  - `drive.py <name> <steps>` drives the real keyboard through a pty and records GT, filter, joints (env `LAUNCH_LOG`, `FALL_DEG`).
  - `record.py <name>` records joints at 500 Hz, MPX effort, GT and FSM state until SIGINT.
  - `fake_chassis_odom.py` publishes a chassis odometry from `/model_states`.
  - To stop the sim use self-safe patterns (`pkill -f 'gzserve[r]'`); never `pkill -f spawner` (it matches gvfsd).

---

## 6. Rules for Claude

- Snapshots are **read-only**. Small modifications and tests are experiments: say which
  files change, verify against `sha256.txt`, and offer to restore when done.
- Do not commit anything unless explicitly asked.
- Test MPX in Gazebo with the ground truth unless asked otherwise; close Gazebo and every sim process at the end.
