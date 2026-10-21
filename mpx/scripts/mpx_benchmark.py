#!/usr/bin/env python3
"""Offline MPC/WBC timing benchmark; runs JAX and MuJoCo directly, with no ROS.

The benchmark uses the same ``Solver.plan`` and ``Solver.wbc`` functions as
production. It measures MPC first, then WBC with a fixed valid MPC input.
Both loops are unpaced; requested rates define budgets, not execution cadence.
No commands are published and no robot or simulator is controlled.

    source ~/ddt_ros2_ws/install/setup.bash
    python3 ~/ddt_ros2_ws/src/mpx/scripts/mpx_benchmark.py
    python3 ~/ddt_ros2_ws/src/mpx/scripts/mpx_benchmark.py --mpc-hz 40 --wbc-hz 100

JAX compiles both functions on the first run; compile time is reported separately
and excluded from steady-state measurements. ``--steps`` counts solves per component (MPC and WBC).
"""
import argparse
import os
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import mpx_cpp_solver as backend  # noqa: E402 (sets CPU/JAX policy via mpx_node)

import jax  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import xacro  # noqa: E402
from ament_index_python.packages import get_package_share_directory  # noqa: E402


def robot_description():
    try:
        share = get_package_share_directory('tita_description')
    except LookupError:
        raise SystemExit('tita_description not found: source install/setup.bash first')
    return xacro.process_file(
        os.path.join(share, 'xacro', 'robot.xacro'), mappings={'hw_env': 'hw'}
    ).toxml()


def distribution(samples):
    values = np.asarray(samples, dtype=float)
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)), max_ms=float(np.max(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=2000,
                        help='number of solves per component to time (default: 2000)')
    parser.add_argument('--mpc-hz', type=float, default=40.,
                        help='MPC budget rate (default: 40 Hz)')
    parser.add_argument('--wbc-hz', type=float, default=100.,
                        help='WBC budget rate (default: 100 Hz)')
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('--steps must be at least 2')
    if not all(np.isfinite(v) and v > 0 for v in (args.mpc_hz, args.wbc_hz)):
        parser.error('--mpc-hz and --wbc-hz must be positive')
    print('NODE NAME: nessuno (benchmark Python offline)', flush=True)
    print('  Publishes to topics:\n    - Nessuno', flush=True)
    print('  Subscribes to topics:\n    - Nessuno', flush=True)
    model_desc = robot_description()
    solver = backend.Solver(model_desc, mode_controlled=True,
                            height=float(backend.config.com_z_to_track),
                            log=lambda _message: None, mpc_frequency=args.mpc_hz)
    model, cfg = solver.model, solver.config
    qpos = np.zeros(model.nq, dtype=float)
    qvel = np.zeros(model.nv, dtype=float)
    qpos[:3] = [0., 0., cfg.robot_height]
    qpos[3:7] = [1., 0., 0., 0.]
    qpos[7:] = np.asarray(cfg.q0, dtype=float)
    command = np.asarray([0., 0., 0., cfg.com_z_to_track], dtype=float)

    print(f'JAX backend: {jax.default_backend()} ({jax.devices()})', flush=True)
    print('Compiling MPC and WBC independently; cold-start compilation is excluded...',
          flush=True)
    compile_start = time.perf_counter()
    solver.compile_split()
    compile_seconds = time.perf_counter() - compile_start
    # Warm-up methods leave both compiled functions initialized. Start with a fresh
    # receding-horizon state, and prime measured-state data just as the live nodes do.
    solver.prepare(qpos, qvel, reset=True)
    solver.prepare_wbc(qpos, qvel, reset=True)

    mpc_ms, wbc_ms, prepare_mpc_ms, prepare_wbc_ms = [], [], [], []
    invalid = 0
    print(f'Timing MPC separately: {args.steps} plans...', flush=True)
    for _ in range(args.steps):
        start = time.perf_counter()
        solver.prepare(qpos, qvel, reset=False)
        prepare_mpc_ms.append((time.perf_counter() - start) * 1000.)
        start = time.perf_counter()
        trajectory = solver.plan(command)
        mpc_ms.append((time.perf_counter() - start) * 1000.)
    # Reuse the last valid plan throughout the isolated WBC measurement.
    latest_u = np.asarray(trajectory[:solver.nu], dtype=float).copy()
    print(f'Timing WBC separately: {args.steps} solves with fixed U[0]...', flush=True)
    for tick in range(args.steps):
        start = time.perf_counter()
        solver.prepare_wbc(qpos, qvel, reset=False)
        prepare_wbc_ms.append((time.perf_counter() - start) * 1000.)
        start = time.perf_counter()
        try:
            command_out = solver.wbc(latest_u, cfg.com_z_to_track)
        except ValueError:
            invalid += 1
            raise
        wbc_ms.append((time.perf_counter() - start) * 1000.)
        if not np.all(np.isfinite(command_out)):
            invalid += 1
            raise RuntimeError(f'non-finite WBC result at tick {tick}')

    mpc = distribution(mpc_ms)
    wbc = distribution(wbc_ms)
    mpc_period_ms, wbc_period_ms = 1000. / args.mpc_hz, 1000. / args.wbc_hz
    mpc['requested_hz'] = args.mpc_hz
    mpc['mean_capacity_hz'] = 1000. / mpc['mean_ms']
    mpc['p95_capacity_hz'] = 1000. / mpc['p95_ms']
    mpc['requested_period_ms'] = mpc_period_ms
    mpc['p95_budget_percent'] = 100. * mpc['p95_ms'] / mpc_period_ms
    mpc['samples_over_period'] = int(np.count_nonzero(np.asarray(mpc_ms) > mpc_period_ms))
    wbc['requested_hz'] = args.wbc_hz
    wbc['mean_capacity_hz'] = 1000. / wbc['mean_ms']
    wbc['p95_capacity_hz'] = 1000. / wbc['p95_ms']
    wbc['requested_period_ms'] = wbc_period_ms
    wbc['p95_budget_percent'] = 100. * wbc['p95_ms'] / wbc_period_ms
    wbc['samples_over_period'] = int(np.count_nonzero(np.asarray(wbc_ms) > wbc_period_ms))
    aggregate_cpu_percent = (mpc['mean_ms'] * args.mpc_hz +
                             wbc['mean_ms'] * args.wbc_hz) / 10.

    print(f'Compilation MPC + WBC: {compile_seconds:.2f} s (excluded from timing)')
    print(f'\nMPC: {len(mpc_ms)} plans; budget rate {args.mpc_hz:g} Hz '
          f'(period {mpc_period_ms:.2f} ms)')
    print('  compute ms: mean {mean_ms:.3f} | median {median_ms:.3f} | '
          'p95 {p95_ms:.3f} | p99 {p99_ms:.3f} | max {max_ms:.3f}'.format(**mpc))
    print(f"  p95 uses {mpc['p95_budget_percent']:.1f}% of period; "
          f"over-period samples {mpc['samples_over_period']}/{len(mpc_ms)}; "
          f"capacity mean/p95 {mpc['mean_capacity_hz']:.1f}/{mpc['p95_capacity_hz']:.1f} Hz")
    print(f'\nWBC: {len(wbc_ms)} solves; budget rate {args.wbc_hz:g} Hz '
          f'(period {wbc_period_ms:.2f} ms)')
    print('  compute ms: mean {mean_ms:.3f} | median {median_ms:.3f} | '
          'p95 {p95_ms:.3f} | p99 {p99_ms:.3f} | max {max_ms:.3f}'.format(**wbc))
    print(f"  p95 uses {wbc['p95_budget_percent']:.1f}% of period; "
          f"over-period samples {wbc['samples_over_period']}/{len(wbc_ms)}; "
          f"capacity mean/p95 {wbc['mean_capacity_hz']:.1f}/{wbc['p95_capacity_hz']:.1f} Hz")
    print(f'\nMuJoCo/state preparation (separate): MPC {np.median(prepare_mpc_ms):.3f} ms, '
          f'WBC {np.median(prepare_wbc_ms):.3f} ms median')
    print(f'Estimated aggregate CPU demand at requested rates: {aggregate_cpu_percent:.1f}% '
          '(one core = 100%, compute only; extrapolated from separate runs)')
    print(f'Invalid MPC/WBC results: {invalid}')
    print('Unpaced static-posture benchmark: reports solver compute time, not ROS/DDS, '
          'scheduler jitter, simulation tracking, or end-to-end latency.')


if __name__ == '__main__':
    main()
