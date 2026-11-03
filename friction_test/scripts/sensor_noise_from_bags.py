#!/usr/bin/env python3
"""Sensor noise of the TITA from bags: /joint_states and the IMU (real robot or Gazebo with noise:=...).

usage: sensor_noise_from_bags.py <bag> [<bag> ...]

Per joint (rows = joint names of /joint_states):
- q quant      smallest non-zero step of the position [rad] (encoder resolution);
- sd q rest    std of the position in 1-s windows with the legs at rest, after removing a line [rad];
- sd dq rest   std of the velocity in the same windows [rad/s];
- dq quant     smallest non-zero step of the velocity [rad/s];
- sd dq move   std of (velocity - its 11-sample centred moving average) while |dq| > 0.3 rad/s: high
               frequency part, an upper bound of the noise (some signal is left in it) [rad/s];
- sd tau rest  std of the effort in the rest windows [N·m];
- tau quant    smallest non-zero step of the effort [N·m].
IMU, rest windows: std of the angular velocity [rad/s], its mean (bias) [rad/s], std of the linear
acceleration [m/s²] and of roll/pitch/yaw after removing a line [rad].
Rest window: every leg joint moves less than 2e-3 rad and every wheel less than 0.5 rad/s for 1 s.
"""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation


def read(uri):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    js = [k for k in types if k.endswith("/joint_states") and "dynamic" not in k]
    imu = [k for k in types if k.endswith("imu_sensor_broadcaster/imu")]
    r.set_filter(rosbag2_py.StorageFilter(topics=js + imu))
    TJ, TI = get_message("sensor_msgs/msg/JointState"), get_message("sensor_msgs/msg/Imu")
    J, I, names = [], [], None
    while r.has_next():
        topic, raw, t = r.read_next()
        if topic in js:
            m = deserialize_message(raw, TJ)
            names = names or list(m.name)
            J.append([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, *m.position, *m.velocity, *m.effort])
        else:
            m = deserialize_message(raw, TI)
            o, w, a = m.orientation, m.angular_velocity, m.linear_acceleration
            I.append([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, o.x, o.y, o.z, o.w, w.x, w.y, w.z, a.x, a.y, a.z])
    return names, np.array(J), np.array(I)


def min_step(x):
    d = np.abs(np.diff(x))
    d = d[d > 1e-7]
    return d.min() if len(d) else 0.0


def detrended_std(t, x):
    return np.std(x - np.polyval(np.polyfit(t - t[0], x, 1), t - t[0]))


def main():
    for uri in sys.argv[1:]:
        names, J, I = read(uri)
        n = len(names)
        t, q, dq, tau = J[:, 0], J[:, 1:1 + n], J[:, 1 + n:1 + 2 * n], J[:, 1 + 2 * n:1 + 3 * n]
        legs = [k for k in range(n) if not names[k].endswith("_4")]
        wheels = [k for k in range(n) if names[k].endswith("_4")]
        rest, i = [], 0
        while i < len(t):
            j = np.searchsorted(t, t[i] + 1.0)
            if j >= len(t):
                break
            if np.ptp(q[i:j][:, legs], 0).max() < 2e-3 and np.abs(dq[i:j][:, wheels]).max() < 0.5:
                rest.append((i, j))
            i = j
        print(f"\n## {uri}\n/joint_states {1 / np.median(np.diff(t)):.0f} Hz, {t[-1] - t[0]:.0f} s, "
              f"{len(rest)} rest windows of 1 s\n")
        print("| joint | q quant [rad] | sd q rest [rad] | sd dq rest [rad/s] | dq quant [rad/s] | sd dq move [rad/s] "
              "| sd tau rest [N·m] | tau quant [N·m] |")
        print("|---|---|---|---|---|---|---|---|")
        ma = lambda x: np.convolve(x, np.ones(11) / 11, "same")
        for k in range(n):
            med = lambda f: np.median([f(a, b) for a, b in rest]) if rest else np.nan
            moving = np.abs(np.convolve(dq[:, k], np.ones(51) / 51, "same")) > 0.3
            sd_move = np.std((dq[:, k] - ma(dq[:, k]))[moving]) if moving.sum() > 50 else np.nan
            print(f"| {names[k]} | {min_step(q[:, k]):.3g} | {med(lambda a, b: detrended_std(t[a:b], q[a:b, k])):.2g} | "
                  f"{med(lambda a, b: np.std(dq[a:b, k])):.3f} | {min_step(dq[:, k]):.3g} | {sd_move:.3f} | "
                  f"{med(lambda a, b: np.std(tau[a:b, k])):.3f} | {min_step(tau[:, k]):.3g} |")
        if len(I) and rest:
            ti = I[:, 0]
            rpy = np.unwrap(Rotation.from_quat(I[:, 1:5]).as_euler("xyz"), axis=0)
            win = [((ti >= t[a]) & (ti < t[b])) for a, b in rest]
            win = [w for w in win if w.sum() > 50]
            sel = np.any(win, axis=0)
            sd_rpy = np.median([[detrended_std(ti[w], rpy[w, c]) for c in range(3)] for w in win], 0)
            print(f"\nIMU (rest): sd gyro {np.round(np.median([I[w, 5:8].std(0) for w in win], 0), 4)} rad/s, "
                  f"mean gyro {np.round(I[sel, 5:8].mean(0), 4)} rad/s, "
                  f"sd acc {np.round(np.median([I[w, 8:11].std(0) for w in win], 0), 3)} m/s², "
                  f"sd roll/pitch/yaw {np.array2string(sd_rpy, precision=2)} rad")


if __name__ == "__main__":
    main()
