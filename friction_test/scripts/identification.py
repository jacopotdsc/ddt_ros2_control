#!/usr/bin/env python3
"""Joint inertia and friction identification from the friction_test CSVs (robot hung in Gazebo).

Imitates roahmlab/ConstrainedSysID (Zhang, Haugk, Vasudevan 2024), restricted to the motor and
friction parameters of each joint (link inertias from the URDF, as for Digit's legs):
  dataProcessing.m / centralDifference.m -> clear_data() (simple first-order low-pass, not Butterworth)
  SysIDAlgFull.m (motor + friction part)  -> identify()
  validation of Fig. 4 / eq. (33)         -> simulate(), plot_identification()
Dynamic model: the MuJoCo model of the WBC (mpx.utils.ros_model.model_from_description) built from
tita_description/xacro/robot.xacro (manufacturer URDF, no friction), base fixed (robot hung).

Per joint:  tau_meas - tau_URDF(q, qd, qdd) = Im qdd + Fc sgn(qd) + Fv qd (+ beta)

Approximations:
- each CSV holds only the joints under test (both sides of a test are merged on t_sim); the other
  joints are taken at the default pose of config/friction_test.yaml, where the PD holds them;
- tau_applied is the torque Gazebo applies (PD command): friction is not in it, Gazebo adds it.
Constraints of the fit: Im, Fc, Fv >= 0 (no negative inertia or friction), as lb in the MATLAB code.

usage: identification.py [--fs 50] [--cf-tau 33 --cf-vel 18 --cf-acc 15] [--k 3] [--seg 0.5] ...
Outputs next to this script: data_identification/ (parameters, clean and simulation CSVs) and
plot_identification/ (one figure per joint, L2 table).
"""
import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from scipy.optimize import least_squares, lsq_linear
from scipy.signal import butter as scipy_butter, filtfilt, savgol_filter

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                      # src/friction_test
WS_SRC = PKG.parent                    # src
sys.path.insert(0, str(WS_SRC / "mpx" / "mpx_tita"))  # as src/mpx/tests/closed_loop_regression.py

JOINTS = [f"joint_{s}_leg_{i}" for s in ("left", "right") for i in (1, 2, 3, 4)]
BASE_QPOS = [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]   # hung: base fixed, level


def node_config():
    """tests, default pose of friction_test_node (config/friction_test.yaml)."""
    with open(PKG / "config" / "friction_test.yaml") as f:
        p = yaml.safe_load(f)["/**"]["ros__parameters"]
    tests = {k.replace("key_", ""): v for k, v in p["tests"].items()}
    default = {j: p["default_pose"][int(j[-1]) - 1] for j in JOINTS}
    return tests, default


# ── 1. data ──────────────────────────────────────────────────────────────────
def load_data(csv_dir):
    """Each joint_*.csv (not *_aborted) -> one record, as SysIDDigit.m step 1 reads one data file.

    Returns {joint: {"t": (N,) from 0, "joints": [joint], "state_joints", "q"/"dq"/"tau"/"q_des"/"dq_des":
    (N, 8) over JOINTS, "level", "segment"}}. If the CSV holds the state of every joint (q_<joint>,
    dq_<joint>, tau_<joint>, as their data files hold all joints), all of them are used ("state_joints" =
    JOINTS); older CSVs hold only the tested joint and the others are taken at the default pose.
    """
    _, default = node_config()
    data = {}
    for path in sorted(Path(csv_dir).glob("joint_*.csv")):
        if path.stem.endswith("_aborted") or path.stem not in JOINTS:
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        col = lambda k: np.array([float(r[k]) for r in rows])
        t = col("t_sim")
        n, i = len(t), JOINTS.index(path.stem)
        rec = {"t": t - t[0], "joints": [path.stem], "level": col("speed_level"),
               "segment": np.array([r["segment"] for r in rows])}
        for name in ("q", "dq", "tau", "q_des", "dq_des"):
            rec[name] = np.zeros((n, len(JOINTS)))
        for k, j in enumerate(JOINTS):
            rec["q"][:, k] = rec["q_des"][:, k] = default[j]
        for name, column in (("q", "q"), ("dq", "dq"), ("tau", "tau_applied"), ("q_des", "q_des"), ("dq_des", "dq_des")):
            rec[name][:, i] = col(column)
        rec["state_joints"] = [path.stem]
        if all(f"q_{j}" in rows[0] for j in JOINTS):
            for k, j in enumerate(JOINTS):
                rec["q"][:, k], rec["dq"][:, k], rec["tau"][:, k] = col(f"q_{j}"), col(f"dq_{j}"), col(f"tau_{j}")
            rec["state_joints"] = list(JOINTS)
        data[path.stem] = rec
    return data


def low_pass(order, Wn, butter):
    """[b, a] of a low-pass (always low-pass), with the arguments of butter(order, Wn, 'low') of
    MATLAB/SciPy minus the filter type.

    butter=True: the Butterworth filter itself (scipy.signal.butter, as their MATLAB code).
    butter=False: a simple low-pass, `order` first-order
    low-passes in series, y[n] = alpha x[n] + (1 - alpha) y[n-1] each. Wn = fc / (fs/2), the cutoff over
    half the sampling frequency, as for butter.
    One stage: alpha = dt / (tau + dt), tau = 1 / (2 pi fc_stage), i.e. alpha = pi Wn_s / (1 + pi Wn_s).
    The stage cutoff is raised to fc / sqrt(2^(1/order) - 1) so that the whole filter is at -3 dB at fc,
    as butter (order 1: the stage is the filter). Coefficients of the series: b = [alpha^order],
    a = product of the stages [1, -(1 - alpha)]."""
    if butter:
        return scipy_butter(order, Wn, "low")
    Wn_s = Wn / np.sqrt(2.0 ** (1.0 / order) - 1.0)
    alpha = np.pi * Wn_s / (1.0 + np.pi * Wn_s)
    b = np.array([alpha ** order])
    a = np.poly([1.0 - alpha] * order)   # (1 - (1 - alpha) z^-1)^order
    return b, a


def central_difference(y, dt, cfA, orderA, butter):
    """centralDifference.m: derivative of y (one joint) after the low-pass at cfA (low_pass + filtfilt
    as their butter + filtfilt). The 4th order central difference [1/12, -2/3, 0, 2/3, -1/12] / dt is
    scipy.signal.savgol_filter with a 5-point window and a 4th degree polynomial (exact through the 5
    points). Same as theirs inside and at the first and last point; at the second and second-to-last
    point savgol uses the first / last 5-point window instead of their shifted forward / backward one."""
    # measurement frequency in Hz
    fs = 1.0 / dt
    # design the filter
    b, a = low_pass(orderA, cfA / (fs / 2), butter)
    # apply the filter using filtfilt to achieve zero-phase filtering
    yfilt = filtfilt(b, a, y)
    # 4th order central difference (5 points)
    return savgol_filter(yfilt, window_length=5, polyorder=4, deriv=1, delta=dt, mode="interp")


def clear_data(data, order, butter, cfT=33.0, cfV=18.0, cfA=15.0, cut_time_before=0.0, cut_time_after=np.inf,
               point_number=None, qd_min=0.0, downsample=None):
    """dataProcessing.m for one data file (one joint record of load_data), same steps and names.

    order: low-pass order (theirs: orderT = orderV = orderA = 4), passed to every filter.
    butter: True = Butterworth (as theirs), False = simple low-pass; passed to every filter.
    point_number: max number of data points (their pointNumber), ds = ceil(n / point_number); None: no limit.
    downsample: fixed factor ds (keep one sample every ds) when point_number is None; None or 1: all samples.
    Differences: low_pass(..., butter) gives the Butterworth (as theirs) or a simple low-pass.
    Returns the dict out: t, q, dq (qdot), ddq (qddot), tau (torq), i (number of points), plus the
    columns of the file (q_des, dq_des, level, segment), fs and used (|qd| >= qd_min, for the fit;
    the paper keeps all, qd_min = 0)."""
    time = data["t"]
    pos, vel, torque = data["q"].copy(), data["dq"].copy(), data["tau"].copy()   # (samples, 8 joints)
    # joints of this data file
    nj = [JOINTS.index(j) for j in data["state_joints"]]   # every joint with a measured state

    # remove time after "cutTimeAfter" seconds: number of points left
    i = int(np.count_nonzero(time <= cut_time_after))
    # remove time before "cutTimeBefore" seconds: index of the first element kept, points left
    j = int(np.count_nonzero(time[:i] < cut_time_before))
    n = i - j

    # time between 2 measurements. 1/dt is the frequency.
    dt = float(time[i - 1] - time[i - 2])
    # measurement frequency in Hz
    fs = 1.0 / dt
    # downsample factor. Approximates the number of points someone wants to keep (None: all samples).
    # fixed factor (the paper: "downsampled by a factor of 10"); max points per file (their pointNumber) wins
    ds = 1 if downsample is None else max(1, int(downsample))
    if point_number is not None:
        ds = int(np.ceil(n / point_number))

    # approximate acceleration and filter it.
    qddots = np.zeros_like(vel)
    for k in nj:
        qddots[:, k] = central_difference(vel[:, k], dt, cfA, order, butter)
    # set the initial acceleration to 0.
    qddots[0, :] = 0.0

    # low-pass filter coefficients for torque (T) and velocity (V)
    b, a = low_pass(order, cfT / (fs / 2), butter)
    b2, a2 = low_pass(order, cfV / (fs / 2), butter)
    # apply the filter using filtfilt to achieve zero-phase filtering
    for k in nj:
        torque[:, k] = filtfilt(b, a, torque[:, k])
        vel[:, k] = filtfilt(b2, a2, vel[:, k])

    # downsample the data by the factor ds
    idx = np.arange(j, i)[::ds]
    out = {"t": time[idx], "q": pos[idx], "dq": vel[idx], "ddq": qddots[idx], "tau": torque[idx],
           "q_des": data["q_des"][idx], "dq_des": data["dq_des"][idx], "level": data["level"][idx],
           "segment": data["segment"][idx], "joints": data["joints"], "fs": fs / ds}
    # number of downsampled datapoints
    out["i"] = len(idx)
    out["used"] = {jn: np.abs(out["dq"][:, JOINTS.index(jn)]) >= qd_min for jn in data["joints"]}
    return out


# ── 2. dynamic model (WBC) ────────────────────────────────────────────────────
class DynamicModel:
    """MuJoCo model of the WBC from the manufacturer URDF; base fixed at BASE_QPOS."""

    def __init__(self):
        import mpx.config.config_tita as config
        from mpx.utils.ros_model import model_from_description
        share = get_package_share_directory("tita_description")
        path = os.path.join(share, "tita", "xacro", "robot.xacro")
        if not os.path.isfile(path):
            path = os.path.join(share, "xacro", "robot.xacro")
        description = xacro.process_file(path, mappings={
            "hw_env": "gazebo", "sim_env": "gazebo", "ctrl_mode": "wbc", "yaml_path": "gazebo_bridge"}).toxml()
        description = description.replace("package://tita_description", "file://" + share)
        self.model, _ = model_from_description(description, config)
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(j).qposadr[0] for j in JOINTS]
        self.vadr = [self.model.joint(j).dofadr[0] for j in JOINTS]
        self.data.qpos[:7] = BASE_QPOS

    def _set(self, q, dq, ddq=None):
        d = self.data
        d.qpos[:7] = BASE_QPOS
        d.qvel[:6] = 0.0
        d.qpos[self.qadr] = q
        d.qvel[self.vadr] = dq
        if ddq is not None:
            d.qacc[:6] = 0.0
            d.qacc[self.vadr] = ddq

    def inverse_dynamics(self, Q, DQ, DDQ):
        """Joint torques (N, 8) for the given motion, as the WBC (mpc_utils.py, section 2 and the
        torque line): fwd_position + fwd_velocity, M = qM, c = qfrc_bias, Ma, ca = M[6:, :], c[6:],
        tau = Ma @ qddot + ca - J^T f with f = 0 (robot hung) and the base at rest (base fixed)."""
        nv = self.model.nv
        tau = np.zeros_like(Q)
        for k in range(len(Q)):
            self._set(Q[k], DQ[k])
            mujoco.mj_fwdPosition(self.model, self.data)   # mjx.fwd_position
            mujoco.mj_fwdVelocity(self.model, self.data)   # mjx.fwd_velocity
            M = np.zeros((nv, nv))
            mujoco.mj_fullM(self.model, M, self.data.qM)
            c = self.data.qfrc_bias                         # Coriolis + gravity (RNEA with qddot = 0)
            Ma, ca = M[6:, :], c[6:]
            qddot = np.zeros(nv)
            qddot[6:] = DDQ[k]                              # base (first 6 dofs) at rest
            tau[k] = Ma @ qddot + ca
        return tau

    def joint_terms(self, q, dq, i):
        """Diagonal mass term and bias (Coriolis + gravity) of joint i."""
        self._set(q, dq)
        mujoco.mj_forward(self.model, self.data)
        dof = self.vadr[i]
        return self.data.qM[self.model.dof_Madr[dof]], self.data.qfrc_bias[dof]


# ── 3. identification ─────────────────────────────────────────────────────────
def identify(clean, model, k=0.3, includeOffset=False, loss="cauchy"):
    """AlgNoAlphaFMOnly of SysIDAlgFMParamOnly.m (alpha = 1), with one SciPy least_squares per joint.

    T_hat = tau_meas - tau of the WBC model (their T_hat = T - W_ip*X_ip), Wfm = [qdd, sgn(qd), qd, (1)]
    for [Im, Fc, Fv, (beta)], bounds Im, Fc, Fv >= 0. Only SciPy as it is: a plain bounded linear fit
    (lsq_linear) gives the starting point and the residual scale sigma; then least_squares with the robust
    loss and f_scale = k * sigma (their k = 0.3 on the residual normalized by its spread), which lowers the
    weight of large residuals smoothly. Their 0/1 "hard redescender" weights are not a SciPy loss, so they
    are not used. No MultiStart: the problem is linear, its solution unique."""
    params = {}
    for key, c in clean.items():
        c["tau_urdf"] = model.inverse_dynamics(c["q"], c["dq"], c["ddq"])
        for j in c["joints"]:
            i, used = JOINTS.index(j), c["used"][j]
            qd, qdd = c["dq"][:, i], c["ddq"][:, i]
            T_hat = c["tau"][:, i] - c["tau_urdf"][:, i]
            Wfm = np.column_stack([qdd, np.sign(qd), qd] + ([np.ones_like(qd)] if includeOffset else []))
            lb = [0.0, 0.0, 0.0] + ([-np.inf] if includeOffset else [])
            Ws, Ts = Wfm[used], T_hat[used]
            # plain bounded linear fit (exact): starting point and residual scale; then the robust fit
            x0 = lsq_linear(Ws, Ts, bounds=(lb, np.inf)).x
            sigma = np.std(Ts - Ws @ x0) or 1.0
            X = least_squares(lambda x: Ws @ x - Ts, x0, bounds=(lb, np.inf), loss=loss, f_scale=k * sigma).x
            names = ["Im", "Fc", "Fv"] + (["beta"] if includeOffset else [])
            params[j] = {n: float(v) for n, v in zip(names, X)}
            params[j].setdefault("beta", 0.0)
            params[j].update(
                samples=int(len(T_hat)), samples_used=int(used.sum()),
                samples_within_k=int((np.abs(Ts - Ws @ X) <= k * sigma).sum()),
                rms_residual_before=float(np.sqrt(np.mean(T_hat**2))),
                rms_residual_after=float(np.sqrt(np.mean((T_hat - Wfm @ X) ** 2))))
            c.setdefault("residual", {})[j] = T_hat
    return params


# ── 4. validation: forward simulation (Fig. 4, eq. 33) ───────────────────────
def simulate(clean, params, model, seg=0.5, dt=0.001, truth=None):
    """Per joint, segments of `seg` s restarting from the measured state; only that joint is
    integrated (other joints at their recorded / held state), with the recorded torque held between
    samples (ZOH): (M_jj + Im) qdd = tau - bias - Fv qd - Fc sgn(qd) - beta. Viscous and Coulomb
    friction are integrated implicitly (stick when the torque cannot overcome Fc), instead of ode15s,
    to avoid chattering at qd = 0. Cases: manufacturer (no friction, Im = 0), identified and, with
    truth (Gazebo <dynamics> per joint), oracle (the true Fc, Fv, Im = 0: the best this model can do).
    L2 (eq. 33) on the position for the legs, on the velocity for the wheels (continuous joints: their
    angle has no meaning and its error only accumulates); the model torque does not depend on the wheel
    angle (checked: 8e-5 Nm over a turn)."""
    sim, l2 = {}, {}
    for key, c in clean.items():
        t = c["t"]
        for j in c["joints"]:
            i = JOINTS.index(j)
            out = {"t": t, "q_meas": c["q"][:, i], "dq_meas": c["dq"][:, i], "wheel": j.endswith("_4")}
            cases = [("manufacturer", dict(Im=0.0, Fc=0.0, Fv=0.0, beta=0.0)), ("identified", params[j])]
            if truth and j in truth:
                cases.append(("oracle", dict(Im=0.0, Fc=truth[j]["Fc"], Fv=truth[j]["Fv"], beta=0.0)))
            for case, p in cases:
                q_sim, dq_sim = np.empty(len(t)), np.empty(len(t))
                seg_start = 0
                for k in range(len(t)):
                    if k == 0 or t[k] - t[seg_start] >= seg - 1e-9:
                        seg_start = k
                        q, v = c["q"][k, i], c["dq"][k, i]
                        q_sim[k], dq_sim[k] = q, v
                        continue
                    # integrate from t[k-1] to t[k] with the inputs of sample k-1 (ZOH)
                    qa, dqa = c["q"][k - 1].copy(), c["dq"][k - 1].copy()
                    tau = c["tau"][k - 1, i]
                    n = max(int(round((t[k] - t[k - 1]) / dt)), 1)
                    h = (t[k] - t[k - 1]) / n
                    for _ in range(n):
                        qa[i], dqa[i] = q, v
                        m_jj, bias = model.joint_terms(qa, dqa, i)
                        mass = m_jj + p["Im"] + h * p["Fv"]
                        v_free = ((m_jj + p["Im"]) * v + h * (tau - bias - p["beta"])) / mass
                        coul = h * p["Fc"] / mass
                        v = 0.0 if abs(v_free) <= coul else v_free - np.sign(v_free) * coul
                        q += h * v
                    q_sim[k], dq_sim[k] = q, v
                out[f"q_{case}"], out[f"dq_{case}"] = q_sim, dq_sim
            out["segment_start"] = np.array([0] + [k for k in range(1, len(t))
                                                   if (t[k] - t[0]) // seg != (t[k - 1] - t[0]) // seg])
            sim[j] = out
            var = "dq" if out["wheel"] else "q"
            l2[j] = {case: float(np.sqrt(np.sum((out[f"{var}_{case}"] - out[f"{var}_meas"]) ** 2)))
                     for case, _ in cases}
    return sim, l2


def friction_torque(p, qd, qdd):
    """Im qdd + Fc sgn(qd) + Fv qd + beta (the identified model, as in identify)."""
    return p["Im"] * qdd + p["Fc"] * np.sign(qd) + p["Fv"] * qd + p.get("beta", 0.0)


def torque_rmse(clean, params, model):
    """Fig. 3 check on any data: RMS of measured torque - (tau_URDF + friction model), per joint."""
    out = {}
    for c in clean.values():
        if "tau_urdf" not in c:
            c["tau_urdf"] = model.inverse_dynamics(c["q"], c["dq"], c["ddq"])
        for j in c["joints"]:
            i = JOINTS.index(j)
            pred = c["tau_urdf"][:, i] + friction_torque(params[j], c["dq"][:, i], c["ddq"][:, i])
            out[j] = float(np.sqrt(np.mean((c["tau"][:, i] - pred) ** 2)))
    return out


# ── 5. outputs ────────────────────────────────────────────────────────────────
def gazebo_truth():
    """Joint friction simulated by Gazebo: <dynamics friction damping> of urdf/robot_friction.xacro."""
    truth = {}
    root = ET.parse(PKG / "urdf" / "robot_friction.xacro").getroot()
    for joint in root.iter("joint"):
        dyn = joint.find("dynamics")
        if joint.get("name") in JOINTS and dyn is not None:
            truth[joint.get("name")] = {"Fc": float(dyn.get("friction", 0)), "Fv": float(dyn.get("damping", 0))}
    return truth


def save_data(params, clean, sim, out_dir=HERE / "data_identification"):
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = gazebo_truth()
    for j in params:
        params[j]["Fc_gazebo"] = truth.get(j, {}).get("Fc", float("nan"))
        params[j]["Fv_gazebo"] = truth.get(j, {}).get("Fv", float("nan"))
    with open(out_dir / "parameters.yaml", "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    fields = ["joint", "Im", "Fc", "Fv", "beta", "Fc_gazebo", "Fv_gazebo", "samples", "samples_used",
              "rms_residual_before", "rms_residual_after"]
    with open(out_dir / "parameters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for j, p in params.items():
            w.writerow([j] + [p[k] for k in fields[1:]])
    for key, c in clean.items():
        for j in c["joints"]:
            i = JOINTS.index(j)
            with open(out_dir / f"{j}_clean.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "q", "dq", "ddq", "tau", "tau_urdf", "residual", "used", "speed_level", "segment"])
                for k in range(len(c["t"])):
                    w.writerow([c["t"][k], c["q"][k, i], c["dq"][k, i], c["ddq"][k, i], c["tau"][k, i],
                                c["tau_urdf"][k, i], c["residual"][j][k], int(c["used"][j][k]),
                                int(c["level"][k]), c["segment"][k]])
    for j, s in sim.items():
        with open(out_dir / f"{j}_sim.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "q_meas", "q_manufacturer", "q_identified", "dq_meas", "dq_manufacturer", "dq_identified"])
            for k in range(len(s["t"])):
                w.writerow([s["t"][k], s["q_meas"][k], s["q_manufacturer"][k], s["q_identified"][k],
                            s["dq_meas"][k], s["dq_manufacturer"][k], s["dq_identified"][k]])


def plot_reconstruction(sim, l2, clean, params, model, truth, out_dir, title=""):
    """Signal reconstruction per joint (on the data of `clean`, the validation set when given):
    1) measured q (wheels: dq) and the forward simulation with the manufacturer, identified and oracle
       model (segments of 0.5 s restarting from the measured state, Fig. 4 of the paper);
    2) error measured - simulated;
    3) measured torque and the torque predicted from the measured motion (Fig. 3): tau_URDF alone
       (manufacturer), tau_URDF + identified friction, tau_URDF + true friction (oracle)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"manufacturer": "tab:blue", "identified": "tab:red", "oracle": "tab:green"}
    for c in clean.values():
        if "tau_urdf" not in c:
            c["tau_urdf"] = model.inverse_dynamics(c["q"], c["dq"], c["ddq"])
        for j in c["joints"]:
            s, i = sim[j], JOINTS.index(j)
            var, unit = ("dq", "[rad/s]") if s["wheel"] else ("q", "[rad]")
            fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            ax[0].plot(s["t"], s[f"{var}_meas"], "k", lw=1.2, label="measured")
            for case in l2[j]:
                ax[0].plot(s["t"], s[f"{var}_{case}"], color=colors[case], lw=0.8, label=f"{case} (L2 {l2[j][case]:.3f})")
                ax[1].plot(s["t"], s[f"{var}_meas"] - s[f"{var}_{case}"], color=colors[case], lw=0.8, label=case)
            for a in ax[:2]:
                for k in s["segment_start"][1:]:
                    a.axvline(s["t"][k], color="0.88", lw=0.4, ls=":")
            ax[1].axhline(0.0, color="k", lw=0.6)
            ax[0].set_ylabel(f"{'velocity' if s['wheel'] else 'position'} {unit}")
            ax[1].set_ylabel(f"measured − simulated {unit}")
            tau_m = c["tau"][:, i]
            ax[2].plot(c["t"], tau_m, "k", lw=1.0, label="measured (filtered)")
            preds = {"manufacturer": c["tau_urdf"][:, i],
                     "identified": c["tau_urdf"][:, i] + friction_torque(params[j], c["dq"][:, i], c["ddq"][:, i])}
            if truth and j in truth:
                preds["oracle"] = c["tau_urdf"][:, i] + friction_torque(
                    dict(Im=0.0, Fc=truth[j]["Fc"], Fv=truth[j]["Fv"]), c["dq"][:, i], c["ddq"][:, i])
            for case, pr in preds.items():
                ax[2].plot(c["t"], pr, color=colors[case], lw=0.7,
                           label=f"{case} (RMS {np.sqrt(np.mean((tau_m - pr) ** 2)):.3f} N·m)")
            ax[2].set_ylabel("torque [N·m]")
            ax[2].set_xlabel("time [s]")
            p = params[j]
            ax[0].set_title(f"{j} {title}: identified Im {p['Im']:.4f}, Fc {p['Fc']:.3f}, Fv {p['Fv']:.3f}"
                            + (f"  (true Fc {truth[j]['Fc']}, Fv {truth[j]['Fv']})" if truth and j in truth else ""))
            for a in ax:
                a.grid(alpha=0.3)
                a.legend(loc="upper right", fontsize=7, ncol=4)
            fig.tight_layout()
            fig.savefig(out_dir / f"reconstruction_{j}.png", dpi=110)
            plt.close(fig)


def plot_identification(sim, l2, fs, out_dir=HERE / "plot_identification"):
    """One figure per joint: error of the reconstructed trajectory, measured (CSV) - simulated, with
    the manufacturer model (blue) and with the identified model (red), at fs; dotted lines between
    the simulation segments (each restarts from the measured state, error 0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, s in sim.items():
        fig, ax = plt.subplots(figsize=(14, 4.5))
        for k in s["segment_start"][1:]:
            ax.axvline(s["t"][k], color="0.85", lw=0.4, ls=":")
        ax.axhline(0.0, color="k", lw=0.8)
        # wheels: velocity error (the wheel angle has no meaning); legs: position error
        var, unit = ("dq", "velocity error [rad/s]") if s["wheel"] else ("q", "position error [rad]")
        ax.plot(s["t"], s[f"{var}_meas"] - s[f"{var}_manufacturer"], "b", lw=0.9,
                label=f"measured - manufacturer model (L2 {l2[j]['manufacturer']:.3f})")
        ax.plot(s["t"], s[f"{var}_meas"] - s[f"{var}_identified"], "r", lw=0.9,
                label=f"measured - identified model (L2 {l2[j]['identified']:.3f})")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(unit)
        ax.set_title(f"{j}: error of the forward simulation with the recorded torque, {fs:.0f} Hz")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{j}.png", dpi=120)
        plt.close(fig)
    cases = list(next(iter(l2.values())))
    with open(out_dir / "l2_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint"] + [f"L2_{c}" for c in cases])
        for j, v in l2.items():
            w.writerow([j] + [v[c] for c in cases])
    with open(out_dir / "l2_table.md", "w") as f:
        f.write("| joint | " + " | ".join(f"L2 {c}" for c in cases) + " |\n" + "|---" * (len(cases) + 1) + "|\n")
        for j, v in l2.items():
            f.write(f"| {j} | " + " | ".join(f"{v[c]:.4f}" for c in cases) + " |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv-dir", default=str(PKG / "friction_test_csv" / "csv_simulazione"))
    ap.add_argument("--point-number", type=int, default=None,
                    help="max data points per test file (their pointNumber); default: max_points_per_test of "
                         "config/identification.yaml")
    ap.add_argument("--cf-tau", type=float, default=33.0)
    ap.add_argument("--cf-vel", type=float, default=18.0)
    ap.add_argument("--cf-acc", type=float, default=15.0)
    ap.add_argument("--order", type=int, default=4, help="low-pass order (theirs: 4)")
    ap.add_argument("--butter", action=argparse.BooleanOptionalAction, default=True,
                    help="Butterworth (as theirs, default); --no-butter: simple low-pass")
    ap.add_argument("--downsample", type=int, default=None,
                    help="keep one sample every N for the fit (paper: 10); default: downsample of "
                         "config/identification.yaml")
    ap.add_argument("--qd-min", type=float, default=0.0)
    ap.add_argument("--k", type=float, default=0.3, help="their k: robust scale on the normalized residual")
    ap.add_argument("--loss", default="cauchy", help="least_squares loss (cauchy, arctan, soft_l1, huber, linear)")
    ap.add_argument("--offset", action="store_true", help="also identify a torque offset beta")
    ap.add_argument("--seg", type=float, default=0.5, help="simulation segment length [s]")
    ap.add_argument("--val-csv-dir", default=None,
                    help="validation data (e.g. excitation B): forward simulation, L2 and figures on it instead "
                         "of on the identification data")
    ap.add_argument("--no-sim", action="store_true", help="skip the forward simulation (parameters only)")
    ap.add_argument("--data-dir", default=str(HERE / "data_identification"))
    ap.add_argument("--plot-dir", default=str(HERE / "plot_identification"))
    args = ap.parse_args()
    with open(PKG / "config" / "identification.yaml") as f:
        settings = yaml.safe_load(f) or {}
    if args.downsample is None:
        args.downsample = int(settings.get("downsample", 1) or 1)
    if args.point_number is None:
        args.point_number = settings.get("max_points_per_test")
    print(f"sampling of the fit: downsample {args.downsample}, max points per test {args.point_number}")

    data = load_data(args.csv_dir)
    print("data files:", list(data))
    # one clear_data per data file, as SysIDData.m calls dataProcessing.m for each file
    clean = {j: clear_data(rec, args.order, args.butter, cfT=args.cf_tau, cfV=args.cf_vel, cfA=args.cf_acc, point_number=args.point_number,
                           qd_min=args.qd_min, downsample=args.downsample) for j, rec in data.items()}
    model = DynamicModel()
    params = identify(clean, model, k=args.k, includeOffset=args.offset, loss=args.loss)
    truth = gazebo_truth()
    val = clean
    if args.val_csv_dir:
        vdata = load_data(args.val_csv_dir)
        print("validation files:", list(vdata))
        # validation always at the full rate (no downsampling), so its L2 does not depend on the fit's sampling
        val = {j: clear_data(rec, args.order, args.butter, cfT=args.cf_tau, cfV=args.cf_vel, cfA=args.cf_acc,
                             qd_min=0.0) for j, rec in vdata.items()}
    rmse = torque_rmse(val, params, model)
    if args.no_sim:
        save_data(params, clean, {}, Path(args.data_dir))
        for j, p in params.items():
            print(f"| {j} | {p['Im']:.4f} | {p['Fc']:.3f} | {p['Fv']:.3f} | torque RMS {rmse[j]:.3f} |")
        return
    sim, l2 = simulate(val, params, model, seg=args.seg, truth=truth)
    save_data(params, clean, sim, Path(args.data_dir))
    plot_identification(sim, l2, next(iter(val.values()))["fs"], Path(args.plot_dir))
    plot_reconstruction(sim, l2, val, params, model, truth, Path(args.plot_dir),
                        title="(validation)" if args.val_csv_dir else "")
    print("| joint | Im | Fc (gazebo) | Fv (gazebo) | used | torque RMS | " + " | ".join(f"L2 {c}" for c in next(iter(l2.values()))) + " |")
    for j, p in params.items():
        print(f"| {j} | {p['Im']:.4f} | {p['Fc']:.3f} ({p['Fc_gazebo']}) | {p['Fv']:.3f} ({p['Fv_gazebo']}) | "
              f"{p['samples_used']}/{p['samples']} | {rmse[j]:.3f} | " + " | ".join(f"{v:.3f}" for v in l2[j].values()) + " |")


if __name__ == "__main__":
    main()
