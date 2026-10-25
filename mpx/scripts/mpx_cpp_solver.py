"""Numerical backend of the C++ MPX nodes (mpx_node_cpp = MPC, mpx_wbc_node = WBC).

ROS callbacks, scheduling and mode handling live in C++. JAX/XLA and MuJoCo
retain the exact numerical implementation of mpx_tita. This is deliberately not
a native C++ rewrite of the MPC/WBC optimizer. Independent of the Python node
(mpx_node_python.py).
"""
import gc
import os
import sys
from pathlib import Path

# mpx_tita: the copy installed next to this file, else the source tree.
sys.path.insert(0, str(Path(__file__).absolute().parent / "python"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "mpx_tita"))

# Small solves for one robot: CPU backend only (set before importing JAX), and no
# BLAS/Eigen thread pools competing with Gazebo. The C++ nodes set the same.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import mpx.config.config_tita as config
from mpx.utils.mpc_wrapper_dfcip import BatchedMPCControllerWrapper, ControlSol, MPCState
from mpx.utils.ros_model import model_from_description

jax.config.update("jax_enable_x64", True)
# The callers own their solver threads; a second dispatch queue adds latency for
# these small CPU solves.
jax.config.update("jax_cpu_enable_async_dispatch", False)

# The C++ node calls the MPC and the WBC from two threads. JAX releases the GIL
# while XLA runs; a short switch interval keeps the WBC thread from waiting up to
# the default 5 ms for the GIL while the MPC thread runs Python dispatch code.
sys.setswitchinterval(5e-4)


class Solver:
    def __init__(self, description, mode_controlled, height, log, mpc_frequency=0.):
        # log: logger callback of the C++ node (unused here, kept for its constructor call).
        self._command = (0., 0., 0., height)
        self.model, self.config = model_from_description(description, config)
        if mpc_frequency > 0:
            # The warm start shifts the previous solution by one MPC period.
            shift = 1. / (mpc_frequency * self.config.dt_mpc)
            if not np.isfinite(shift) or abs(shift - round(shift)) > 1e-6 or not 1 <= round(shift) <= self.config.N:
                raise ValueError('MPC period must span a whole number of prediction nodes within the horizon')
            self.config.mpc_frequency = mpc_frequency
            self.config.mpc_shift_nodes = int(round(shift))
        self.data = mujoco.MjData(self.model)
        if mode_controlled:
            self.config.q0 = np.asarray(config.q0, dtype=float).copy()
            self.data.qpos[7:] = self.config.q0
            mujoco.mj_forward(self.model, self.data)
        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.base_body_name)
        self.contact_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                            for n in self.config.contact_frame]
        self.joint_addresses = []
        for n in self.config.joints_name:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.joint_addresses.append((int(self.model.jnt_qposadr[i]),
                                         int(self.model.jnt_dofadr[i])))
        self.mpc = BatchedMPCControllerWrapper(self.config, n_env=1, model=self.model)
        self.mpc_state = self.mpc.init_state()
        self.theta_prev = np.asarray(0., dtype=np.float64)
        self.period_ns = round(1e9 / self.config.mpc_frequency)
        self.joint_names = list(self.config.joints_name)
        self.effort_limits = np.asarray(self.config.effort_limits, dtype=float).tolist()
        self.nq, self.nv = self.model.nq, self.model.nv
        # Split MPC / WBC (C++ node): separate MjData and heading unwrap per thread.
        self.data_wbc = mujoco.MjData(self.model)
        self.theta_wbc = np.asarray(0., dtype=np.float64)
        self.com_err_integral = np.zeros(3)  # integral of the COM error, Ki term of the WBC QP
        self.base_err_integral = np.zeros(3)  # integral of the base roll/pitch error, Ki_base term
        self.vel_integral = 0.  # integral of the forward velocity error [m], outer loop of the MPC
        self.vel_profile = np.zeros(2)  # expected response to the command: [velocity, acceleration]
        self._wbc_template = self.mpc.init_state()
        self._plan_jit = jax.jit(self.plan_controller)
        # MPC weights: W normally, W_takeover while standing up from transform_up.
        self.W = jnp.asarray(self.config.W)
        self.W_takeover = jnp.asarray(self.config.W_takeover)
        self._wbc_jit = jax.jit(self.wbc_controller)
        self.horizon = int(self.config.N)
        self.nu = int(self.config.nu)
        self.dt_mpc_ns = int(round(self.config.dt_mpc * 1e9))
        self.dof_addresses = [dof for _, dof in self.joint_addresses]

    def raw_state_from(self, data):
        """COM, COM velocity and wheel contacts (centers, rotations, radii, velocities)."""
        centers = data.geom_xpos[self.contact_ids].copy()
        rotations = data.geom_xmat[self.contact_ids].reshape(2, 3, 3).copy()
        radii = self.model.geom_size[self.contact_ids, 0].copy()
        wheel_velocities = np.zeros((2, 3))
        spatial_velocity = np.zeros(6)
        for i, geom_id in enumerate(self.contact_ids):
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id),
                                     spatial_velocity, 0)  # world frame
            wheel_velocities[i] = spatial_velocity[3:6]
        return (data.subtree_com[self.base_id].copy(), data.subtree_linvel[self.base_id].copy(),
                centers, rotations, radii, wheel_velocities)

    def prepare(self, qpos, qvel, reset):
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (self.nq,) or qvel.shape != (self.nv,):
            raise ValueError('Invalid generalized state dimensions')
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
            raise ValueError('Non-finite feedback state')
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_subtreeVel(self.model, self.data)
        self.raw = self.raw_state_from(self.data)
        if reset:
            self.theta_prev = np.asarray(0., dtype=np.float64)
            self.mpc_state = self.mpc.init_state()
            _, x0, _ = self.mpc.process_state(*self.raw, self.theta_prev)
            self.mpc_state = self.mpc.init_state(x0)
            self.vel_integral = 0.
            self.vel_profile = np.array([float(np.ravel(x0)[11]), 0.])  # starts from the measured velocity
        nominal = (abs(qpos[2] - self.config.robot_height) < .05 and
                   np.max(np.abs(qpos[7:] - np.asarray(self.config.q0))) < .05)
        return np.concatenate((self.raw[0], self.raw[1], self.raw[2].ravel(),
                               qpos[3:7], [float(nominal)]))

    # ── Split MPC / WBC (hierarchical C++ node) ─────────────────────────────
    # MPC and WBC stay the MPX JAX functions; only their scheduling is split.

    def plan_controller(self, mpc_state, theta_prev, raw, command, W):
        """MPC only: same computation as BatchedMPCControllerWrapper.run, also returning U.

        run() keeps only U[:, 0] (state.sol) and the shifted warm start. Publish the
        whole plan in the trajectory message; the WBC currently consumes its first input,
        matching mjx_tita.py, while recomputing on each fresh measured state.
        """
        mpc = self.mpc
        _, x0, theta_prev = mpc.process_state(*raw, theta_prev)
        x0 = x0[None, :]
        x_ref, u_ref, rates = mpc._ref_gen(x0, command, mpc_state.ref_rate)
        reference = jnp.concatenate([x_ref, u_ref], axis=-1)
        W = jnp.tile(W, (mpc.n_env, 1, 1))
        X, U, D = mpc._solve(reference, None, W, x0, mpc_state.X0_shifted, mpc_state.U0_shifted)
        D = jnp.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        shifted = mpc.shift_warm_start
        new_state = MPCState(
            sol=ControlSol(a=U[:, 0, 0], ac_z=U[:, 0, 1], alpha=U[:, 0, 2], grf=U[:, 0, 3:]),
            X0_shifted=shifted(X), U0_shifted=shifted(U), D0_shifted=shifted(D),
            ref_rate=rates[:, mpc.shift],  # the next solve continues the reference from here
        )
        ok = jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(U))
        return new_state, theta_prev, U[0], ok

    def wbc_controller(self, template, theta_prev, raw, qpos, qvel, u, height, com_err_integral, base_err_integral):
        """WBC on measured state, using the held first input from the latest MPC plan."""
        tita_state, x0, theta_prev = self.mpc.process_state(*raw, theta_prev)
        state = template.replace(sol=ControlSol(
            a=u[None, 0], ac_z=u[None, 1], alpha=u[None, 2], grf=u[None, 3:]))
        _, tau, qddot, fl, fr, desired = self.mpc.whole_body_run(
            state, x0, qpos, qvel,
            tita_state[6:9][None, :], tita_state[9:12][None, :],
            tita_state[12:15][None, :], tita_state[15:18][None, :],
            com_height_target=height,
            com_err_integral=com_err_integral[None, :],
            base_err_integral=base_err_integral[None, :],
        )
        flags = jnp.array([
            jnp.all(jnp.isfinite(desired)), jnp.all(jnp.isfinite(qddot)),
            jnp.all(jnp.isfinite(fl)) & jnp.all(jnp.isfinite(fr)),
        ])
        return tau[0], qddot[0], theta_prev, flags

    def _nominal_raw(self):
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [0.0, 0.0, self.config.robot_height]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = self.config.q0
        mujoco.mj_forward(self.model, data)
        mujoco.mj_subtreeVel(self.model, data)
        return data, self.raw_state_from(data), np.asarray([self._command], dtype=np.float64)

    def compile_plan(self):
        """JIT-compile the MPC on the nominal posture; the live MPC state is untouched."""
        _, raw, command = self._nominal_raw()
        state, theta = self.mpc.init_state(), np.asarray(0., dtype=np.float64)
        for _ in range(3):
            state, theta, U, ok = self._plan_jit(state, theta, raw, command, self.W)
            jax.block_until_ready(state)
            if not bool(ok):
                raise RuntimeError('MPX MPC warm-up produced an invalid trajectory')
        self.mpc_state = self.mpc.init_state()
        self.theta_prev = np.asarray(0., dtype=np.float64)
        gc.collect()
        gc.freeze()
        return np.asarray(U)

    def compile_wbc(self):
        """JIT-compile the WBC on the nominal posture with the nominal MPC input."""
        data, raw, command = self._nominal_raw()
        u = np.asarray(self.config.u_ref, dtype=np.float64)
        for _ in range(3):
            tau, qddot, _, flags = self._wbc_jit(
                self._wbc_template, np.asarray(0., dtype=np.float64), raw,
                data.qpos.copy()[None, :], data.qvel.copy()[None, :], u, command[:, 3], np.zeros(3), np.zeros(3))
            jax.block_until_ready(tau)
            if not np.all(np.asarray(flags)) or not np.all(np.isfinite(tau)):
                raise RuntimeError('MPX WBC warm-up produced invalid control')
        self.theta_wbc = np.asarray(0., dtype=np.float64)
        gc.collect()
        gc.freeze()

    def compile_split(self):
        """Both (single-process use, benchmark)."""
        U = self.compile_plan()
        self.compile_wbc()
        return U

    def plan(self, command, standup=False):
        """One MPC solve on the state of the last prepare(); returns U flattened (N*nu).

        standup: taking over from transform_up, use the stand-up weights W_takeover.
        """
        command = np.asarray(command, dtype=np.float64).copy()
        cfg = self.config
        if cfg.ki_vel > 0.:
            # Outer integral loop on the forward velocity. The MPC reference restarts from the
            # measured velocity at every solve, so a constant model mismatch (e.g. the real COM
            # behind the model one) leaves a steady velocity error. The error is taken against
            # the expected response to the command (the same critically damped profile, tau_v,
            # the MPC reference follows), so normal accelerations do not wind it up. With a zero
            # command I is the displacement since the takeover, and the MPC gets
            # v_cmd - ki_vel * I, which stops the robot and brings it back.
            w, x, y, z = self.data.qpos[3:7]
            yaw = np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
            v = self.raw[1][0] * np.cos(yaw) + self.raw[1][1] * np.sin(yaw)
            dt, tau = cfg.dt_mpc, cfg.tau_v
            v_prof, acc = self.vel_profile
            acc += dt * ((command[0] - v_prof) / tau ** 2 - 2. * acc / tau)
            self.vel_profile = np.array([v_prof + acc * dt, acc])
            limit = cfg.vel_correction_max / cfg.ki_vel
            self.vel_integral = float(np.clip(self.vel_integral + (v - self.vel_profile[0]) * dt, -limit, limit))
            command[0] -= cfg.ki_vel * self.vel_integral
        state, theta, U, ok = self._plan_jit(
            self.mpc_state, self.theta_prev, self.raw,
            command[None, :],
            self.W_takeover if standup else self.W)
        U = np.asarray(U)
        if not bool(ok) or U.shape != (self.horizon, self.nu):
            raise ValueError('Invalid MPC result; keeping the previous trajectory')
        self.mpc_state, self.theta_prev = state, theta
        return U.ravel()

    def prepare_wbc(self, qpos, qvel, reset):
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (self.nq,) or qvel.shape != (self.nv,):
            raise ValueError('Invalid generalized state dimensions')
        if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
            raise ValueError('Non-finite feedback state')
        data = self.data_wbc
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(self.model, data)
        mujoco.mj_subtreeVel(self.model, data)
        self.raw_wbc = self.raw_state_from(data)
        if reset:
            self.theta_wbc = np.asarray(0., dtype=np.float64)
            self.com_err_integral = np.zeros(3)
            self.base_err_integral = np.zeros(3)

    def wbc(self, u, height):
        """WBC on the state of the last prepare_wbc(); returns [tau (8), qddot (nv)]."""
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.nu,):
            raise ValueError('Invalid MPC input dimension')
        data = self.data_wbc
        cfg = self.config
        # Integral of the COM height error for the QP's Ki term (only z: the horizontal
        # COM target comes from the MPC). No wind-up during large transients.
        error = height - data.subtree_com[self.base_id, 2]
        if abs(error) < cfg.com_integral_band:
            self.com_err_integral[2] = np.clip(self.com_err_integral[2] + error * cfg.dt_wbc,
                                               -cfg.com_integral_max, cfg.com_integral_max)
        # Integral of the base roll/pitch error for the QP's Ki_base term: same angle-axis error
        # as the QP (mpc_utils err_rotation), with the target = the base levelled at its own yaw,
        # so the yaw component is ~0 (heading is the MPC's job).
        if cfg.Ki_base > 0.:
            Rb = data.xmat[self.base_id].reshape(3, 3)
            yaw = np.arctan2(Rb[1, 0], Rb[0, 0])
            Ra = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
            Rd = Rb.T @ Ra
            angle = np.arccos(np.clip((np.trace(Rd) - 1.) / 2., -1., 1.))
            axis = 0.5 * np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])
            n = np.linalg.norm(axis)
            err = angle * (Ra @ (axis / n)) if angle > 1e-7 and n > 1e-7 else np.zeros(3)
            err[2] = 0.
            if np.linalg.norm(err) < cfg.base_integral_band:
                self.base_err_integral = np.clip(self.base_err_integral + err * cfg.dt_wbc,
                                                 -cfg.base_integral_max, cfg.base_integral_max)
        tau, qddot, theta, flags = self._wbc_jit(
            self._wbc_template, self.theta_wbc, self.raw_wbc,
            data.qpos.copy()[None, :], data.qvel.copy()[None, :], u,
            np.asarray([height], dtype=np.float64), self.com_err_integral.copy(), self.base_err_integral.copy())
        out = np.concatenate((np.asarray(tau), np.asarray(qddot)))
        if out.shape != (8 + self.nv,) or not np.all(np.isfinite(out)) or not np.all(np.asarray(flags)):
            raise ValueError('Invalid WBC result; holding the last valid command')
        self.theta_wbc = theta
        return out
