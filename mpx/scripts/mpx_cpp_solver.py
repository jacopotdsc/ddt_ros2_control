"""Numerical backend for the experimental rclcpp node.

ROS callbacks, scheduling and mode handling live in C++. JAX/XLA and MuJoCo
retain the exact numerical implementation of the Python reference node. This
is deliberately not a native C++ rewrite of the MPC/WBC optimizer.
"""
import gc
import sys
from types import SimpleNamespace

from mpx_node import (
    MPXNode, config, model_from_description, jax, jnp, mujoco, np,
    BatchedMPCControllerWrapper,
)
from mpx.utils.mpc_wrapper_dfcip import ControlSol, MPCState

# The C++ node calls the MPC and the WBC from two threads. JAX releases the GIL
# while XLA runs; a short switch interval keeps the WBC thread from waiting up to
# the default 5 ms for the GIL while the MPC thread runs Python dispatch code.
sys.setswitchinterval(5e-4)


class Solver:
    step_controller = MPXNode.step_controller
    raw_state_from = MPXNode.raw_state_from
    precompile_controller = MPXNode.precompile_controller

    def __init__(self, description, mode_controlled, height, log, mpc_frequency=0.):
        self._mode_controlled = mode_controlled
        self._command = (0., 0., 0., height)
        self._logger = SimpleNamespace(info=log)
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
            contacts = [self.model.geom(n).id for n in self.config.contact_frame]
            self.config.d = float(np.linalg.norm(
                self.data.geom_xpos[contacts[0]] - self.data.geom_xpos[contacts[1]]))
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
        self._step_controller_jit = jax.jit(self.step_controller)
        self.initial_effort = None
        self.period_ns = round(1e9 / self.config.mpc_frequency)
        self.joint_names = list(self.config.joints_name)
        self.effort_limits = np.asarray(self.config.effort_limits, dtype=float).tolist()
        self.nq, self.nv = self.model.nq, self.model.nv
        # Split MPC / WBC (C++ node): separate MjData and heading unwrap per thread.
        self.data_wbc = mujoco.MjData(self.model)
        self.theta_wbc = np.asarray(0., dtype=np.float64)
        self._wbc_template = self.mpc.init_state()
        self._plan_jit = jax.jit(self.plan_controller)
        self._wbc_jit = jax.jit(self.wbc_controller)
        self.horizon = int(self.config.N)
        self.nu = int(self.config.nu)
        self.dt_mpc_ns = int(round(self.config.dt_mpc * 1e9))
        self.dof_addresses = [dof for _, dof in self.joint_addresses]

    def get_logger(self):
        return self._logger

    def compile(self):
        self.precompile_controller()
        gc.collect()
        gc.freeze()

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
        nominal = (abs(qpos[2] - self.config.robot_height) < .05 and
                   np.max(np.abs(qpos[7:] - np.asarray(self.config.q0))) < .05)
        return np.concatenate((self.raw[0], self.raw[1], self.raw[2].ravel(),
                               qpos[3:7], [float(nominal)]))

    def step(self, command):
        state, theta, tau, flags = self._step_controller_jit(
            self.mpc_state, self.theta_prev, self.raw,
            self.data.qpos.copy()[None, :], self.data.qvel.copy()[None, :],
            np.asarray(command, dtype=np.float64)[None, :])
        effort, flags = np.asarray(tau[0]), np.asarray(flags)
        if effort.shape != (8,) or not np.all(np.isfinite(effort)) or not np.all(flags):
            if not getattr(self, '_reported_invalid', False):
                self._reported_invalid = True
                self._logger.info(f"First invalid: qpos={self.data.qpos.tolist()} qvel={self.data.qvel.tolist()} command={list(command)} flags={flags.tolist()} tau={effort.tolist()}")
            raise ValueError('Invalid MPC/WBC result; holding last valid effort')
        self.mpc_state, self.theta_prev = state, theta
        return effort

    # ── Split MPC / WBC (hierarchical C++ node) ─────────────────────────────
    # MPC and WBC stay the MPX JAX functions; only their scheduling is split.

    def plan_controller(self, mpc_state, theta_prev, raw, command):
        """MPC only: same computation as BatchedMPCControllerWrapper.run, also returning U.

        run() keeps only U[:, 0] (state.sol) and the shifted warm start. Publish the
        whole plan in the trajectory message; the WBC currently consumes its first input,
        matching mjx_tita.py, while recomputing on each fresh measured state.
        """
        mpc, cfg = self.mpc, self.config
        _, x0, theta_prev = mpc.process_state(*raw, theta_prev)
        x0 = x0[None, :]
        x_ref, u_ref = mpc._ref_gen(x0, command)
        reference = jnp.concatenate([x_ref, u_ref], axis=-1)
        W = jnp.tile(cfg.W, (mpc.n_env, 1, 1))
        X, U, D = mpc._solve(reference, None, W, x0, mpc_state.X0_shifted, mpc_state.U0_shifted)
        D = jnp.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        shifted = mpc.shift_warm_start
        new_state = MPCState(
            sol=ControlSol(a=U[:, 0, 0], ac_z=U[:, 0, 1], alpha=U[:, 0, 2], grf=U[:, 0, 3:]),
            X0_shifted=shifted(X), U0_shifted=shifted(U), D0_shifted=shifted(D),
        )
        ok = jnp.all(jnp.isfinite(X)) & jnp.all(jnp.isfinite(U))
        return new_state, theta_prev, U[0], ok

    def wbc_controller(self, template, theta_prev, raw, qpos, qvel, u, height):
        """WBC on measured state, using the held first input from the latest MPC plan."""
        tita_state, x0, theta_prev = self.mpc.process_state(*raw, theta_prev)
        state = template.replace(sol=ControlSol(
            a=u[None, 0], ac_z=u[None, 1], alpha=u[None, 2], grf=u[None, 3:]))
        _, tau, qddot, fl, fr, desired = self.mpc.whole_body_run(
            state, x0, qpos, qvel,
            tita_state[6:9][None, :], tita_state[9:12][None, :],
            tita_state[12:15][None, :], tita_state[15:18][None, :],
            com_height_target=height,
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
            state, theta, U, ok = self._plan_jit(state, theta, raw, command)
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
                data.qpos.copy()[None, :], data.qvel.copy()[None, :], u, command[:, 3])
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

    def plan(self, command):
        """One MPC solve on the state of the last prepare(); returns U flattened (N*nu)."""
        state, theta, U, ok = self._plan_jit(
            self.mpc_state, self.theta_prev, self.raw,
            np.asarray(command, dtype=np.float64)[None, :])
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

    def wbc(self, u, height):
        """WBC on the state of the last prepare_wbc(); returns [tau (8), qddot (nv)]."""
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.nu,):
            raise ValueError('Invalid MPC input dimension')
        data = self.data_wbc
        tau, qddot, theta, flags = self._wbc_jit(
            self._wbc_template, self.theta_wbc, self.raw_wbc,
            data.qpos.copy()[None, :], data.qvel.copy()[None, :], u,
            np.asarray([height], dtype=np.float64))
        out = np.concatenate((np.asarray(tau), np.asarray(qddot)))
        if out.shape != (8 + self.nv,) or not np.all(np.isfinite(out)) or not np.all(np.asarray(flags)):
            raise ValueError('Invalid WBC result; holding the last valid command')
        self.theta_wbc = theta
        return out
