#!/usr/bin/env python3
"""Passive MPX benchmark. Run on the same machine as MPX; never sends commands."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone

CONTROL = re.compile(
    r'Control: ([\d.]+) Hz sim, solve ms median=([\d.]+) '
    r'p95=([\d.]+) max=([\d.]+), missed periods=(\d+), invalid=(\d+)')
MPC_WINDOW = re.compile(
    r'MPC: ([\d.]+) Hz sim, plan ms median=([\d.]+) p95=([\d.]+) '
    r'max=([\d.]+), missed periods=(\d+), invalid=(\d+)')
WBC_WINDOW = re.compile(
    r'WBC: ([\d.]+) Hz sim, ms median=([\d.]+) p95=([\d.]+) '
    r'max=([\d.]+), MPC trajectory age ms median=([\d.]+) max=([\d.]+), '
    r'missed periods=(\d+), invalid=(\d+)')
COMPILE = re.compile(r'precompilation complete in ([\d.]+)')
WBC_COMPILE = re.compile(r'MPX WBC ready on CPU \(compiled in ([\d.]+) s\)')


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)
    return dict(n=len(values), mean=statistics.mean(values), median=percentile(.5),
                p95=percentile(.95), max=values[-1], min=values[0])


def parse_control(message):
    match = CONTROL.search(message)
    if not match:
        return None
    hz, median, p95, maximum, missed, invalid = match.groups()
    return dict(hz_state_clock=float(hz), median_ms=float(median), p95_ms=float(p95),
                max_ms=float(maximum), missed=int(missed), invalid=int(invalid))


def parse_mpc_window(message):
    match = MPC_WINDOW.search(message)
    if not match:
        return None
    hz, median, p95, maximum, missed, invalid = match.groups()
    return dict(hz_state_clock=float(hz), median_ms=float(median), p95_ms=float(p95),
                max_ms=float(maximum), missed=int(missed), invalid=int(invalid))


def parse_wbc_window(message):
    match = WBC_WINDOW.search(message)
    if not match:
        return None
    hz, median, p95, maximum, age_median, age_max, missed, invalid = match.groups()
    return dict(hz_state_clock=float(hz), median_ms=float(median), p95_ms=float(p95),
                max_ms=float(maximum), trajectory_age_median_ms=float(age_median),
                trajectory_age_max_ms=float(age_max), missed=int(missed), invalid=int(invalid))


def summarize_windows(windows):
    # Ignore the first window after activation/JIT warm-up, separately per node.
    steady = windows[1:]
    return dict(
        windows_received=len(windows), windows_used=len(steady),
        hz_state_clock=statistics.mean(w['hz_state_clock'] for w in steady) if steady else None,
        median_ms=statistics.mean(w['median_ms'] for w in steady) if steady else None,
        p95_ms=statistics.mean(w['p95_ms'] for w in steady) if steady else None,
        max_ms=max((w['max_ms'] for w in steady), default=None),
        missed_periods=sum(w['missed'] for w in windows) if windows else None,
        invalid_solves=sum(w['invalid'] for w in windows) if windows else None,
        trajectory_age_median_ms=(statistics.mean(w['trajectory_age_median_ms'] for w in steady)
                                  if steady and 'trajectory_age_median_ms' in steady[0] else None),
        trajectory_age_max_ms=(max((w['trajectory_age_max_ms'] for w in steady), default=None)
                               if steady and 'trajectory_age_max_ms' in steady[0] else None))


def parameter_block(document):
    """Read ROS 2 params from wildcard, node-specific, or flat YAML layouts."""
    if not isinstance(document, dict):
        return {}
    root = document.get('/**')
    if isinstance(root, dict):
        if isinstance(root.get('ros__parameters'), dict):
            return root['ros__parameters']
        for value in root.values():
            if isinstance(value, dict) and isinstance(value.get('ros__parameters'), dict):
                return value['ros__parameters']
        return root
    for value in document.values():
        if isinstance(value, dict) and isinstance(value.get('ros__parameters'), dict):
            return value['ros__parameters']
    return document


def merge_dicts(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def process_sample(pid):
    # /proc/PID/stat: names can contain spaces and parentheses.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(ticks=int(fields[11]) + int(fields[12]), start_ticks=int(fields[19]),
                rss_mib=int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / 2**20)


def find_mpx():
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            names = [Path(os.fsdecode(arg)).name for arg in args[:2] if arg]
            if any(name in ('mpx_node_cpp', 'mpx_node_python.py') for name in names):
                matches.append(int(path.parent.name))
        except (OSError, ValueError):
            pass
    return sorted(matches)


def topic_name(namespace, name):
    if name.startswith('/'):
        return name
    return '/' + '/'.join(p for p in (namespace.strip('/'), name) if p)


def logger_matches(actual, expected):
    if actual == expected:
        return True
    return actual.endswith('.' + expected) or actual.endswith('/' + expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60,
                        help='Seconds after first torque (default 60); pauses count too')
    parser.add_argument('--wait-timeout', type=float, default=180,
                        help='Maximum seconds waiting for first torque (default 180)')
    parser.add_argument('--label', default=platform.node())
    parser.add_argument('--output', type=Path, help='New output directory (must not exist)')
    parser.add_argument('--namespace', default=os.environ.get('ROBOT_NS', ''))
    parser.add_argument('--fsm-topic', help='Override configured FSM topic')
    parser.add_argument('--rosout-topic', default='/rosout')
    parser.add_argument('--logger-name', help='Exact MPX logger name; default namespace.controller')
    parser.add_argument('--wbc-logger-name', help='Exact WBC logger name; default wbc')
    parser.add_argument('--params-file', type=Path,
                        help='Topic configuration; default rl_controller/config/tita/mpx.yaml')
    parser.add_argument('--mpc-topic', help='Override MPC trajectory topic')
    parser.add_argument('--wbc-topic', help='Override WBC joint-command topic')
    parser.add_argument('--pid', type=int, help='Local MPX PID; otherwise discover a unique process')
    parser.add_argument('--expected-hz', type=float, default=100,
                        help='Wall-clock torque rate for gap comparison (default 100)')
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in
               (args.duration, args.wait_timeout, args.expected_hz)):
        parser.error('duration, wait-timeout and expected-hz must be positive')

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import ExternalShutdownException
    from rcl_interfaces.msg import Log
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from ament_index_python.packages import get_package_share_directory
    import yaml

    try:
        mpx_share = Path(get_package_share_directory('mpx'))
    except LookupError:
        # Also support running this script directly from an unbuilt source tree.
        mpx_share = Path(__file__).resolve().parents[1]
    base_config_path = mpx_share / 'config/mpx_node.yaml'
    base_config = parameter_block(yaml.safe_load(base_config_path.read_text()))
    if args.params_file:
        config_path = args.params_file
    else:
        try:
            controller_share = Path(get_package_share_directory('rl_controller'))
        except LookupError:
            controller_share = mpx_share.parent / 'controller/rl_controller'
        config_path = controller_share / 'config/tita/mpx.yaml'
    config_overlay = parameter_block(yaml.safe_load(config_path.read_text()))
    config = merge_dicts(base_config, config_overlay)
    mpc_topic = topic_name(args.namespace, args.mpc_topic or config['outputs']['trajectory_topic'])
    wbc_topic = topic_name(args.namespace, args.wbc_topic or config['outputs']['joint_command_topic'])
    fsm_topic = topic_name(args.namespace, args.fsm_topic or config['inputs']['fsm_state_topic'])
    logger = args.logger_name or 'controller'
    wbc_logger = args.wbc_logger_name or 'wbc'
    output = args.output or Path('benchmark_results') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    files = []
    def csv_file(name, columns):
        stream = (output / name).open('w', newline='')
        files.append(stream)
        writer = csv.writer(stream)
        writer.writerow(columns)
        return writer
    torques = csv_file('torques.csv', ['elapsed_s', 'fsm', 'segment', 'interval_ms', 'valid', 'tau_ff_json'])
    windows_csv = csv_file('solver_windows.csv', ['elapsed_s', 'fsm', 'component', 'hz_state_clock', 'median_ms', 'p95_ms', 'max_ms', 'trajectory_age_median_ms', 'trajectory_age_max_ms', 'missed', 'invalid'])
    cpu_csv = csv_file('resources.csv', ['elapsed_s', 'phase', 'pid', 'cpu_percent', 'rss_mib', 'observer_cpu_percent'])
    logs = (output / 'mpx_logs.jsonl').open('w')
    files.append(logs)
    metadata = dict(label=args.label, utc_start=datetime.now(timezone.utc).isoformat(),
                    host=platform.node(), platform=platform.platform(), machine=platform.machine(),
                    cpu_count=os.cpu_count(), ros_distro=os.environ.get('ROS_DISTRO'),
                    ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'),
                    rmw_implementation=os.environ.get('RMW_IMPLEMENTATION', 'default'),
                    namespace=args.namespace, mpc_topic=mpc_topic,
                    wbc_topic=wbc_topic, fsm_topic=fsm_topic,
                    mpc_logger=logger, wbc_logger=wbc_logger,
                    duration_s=args.duration, expected_hz=args.expected_hz,
                    params_file=str(config_path), pid=None)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    class Observer(Node):
        def __init__(self):
            super().__init__('mpx_benchmark', enable_rosout=False)
            self.first = self.last = self.previous = None
            self.measurement_start = None
            self.state = 'unknown'
            self.segment = 0
            self.count = self.bad = 0
            self.intervals = []
            self.windows = []
            self.mpc_windows = []
            self.wbc_windows = []
            self.compilations = []
            self.wbc_compilations = []
            self.compiling = False
            self.cpu = []
            self.pid = args.pid
            self.proc_previous = None
            self.observer_previous = (time.monotonic(), time.process_time())
            self.ambiguous = False
            self.live_previous = (time.monotonic(), 0, 0)
            qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
            state_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
            # MPX output (WBC of the C++ or Python backend): tau_ff, q_des, qd_des.
            self.create_subscription(JointState, wbc_topic, self.torque, qos)
            self.create_subscription(String, fsm_topic, self.fsm, state_qos)
            self.create_subscription(Log, args.rosout_topic, self.log, qos)
            self.create_timer(1., self.resources)

        def torque(self, msg):
            now = time.monotonic()
            if self.first is None:
                self.first = now
                if self.measurement_start is None:
                    self.measurement_start = now
                print(f'Prima coppia ricevuta: misuro per {args.duration:g} s.', flush=True)
            interval = None if self.previous is None else (now - self.previous) * 1000
            if interval is not None:
                self.intervals.append(interval)
            self.previous = self.last = now
            valid = len(msg.effort) == 8 and all(math.isfinite(v) for v in msg.effort)
            self.count += 1
            self.bad += not valid
            torques.writerow([now - started, self.state, self.segment, interval, int(valid),
                              json.dumps(list(msg.effort))])

        def fsm(self, msg):
            if msg.data != self.state:
                self.segment += 1
                self.previous = None  # Never compare separate control activations.
                self.state = msg.data

        def log(self, msg):
            is_mpc = logger_matches(msg.name, logger)
            is_wbc = logger_matches(msg.name, wbc_logger)
            if not is_mpc and not is_wbc:
                return
            now = time.monotonic() - started
            logs.write(json.dumps(dict(elapsed_s=now, stamp_sec=msg.stamp.sec,
                                      stamp_nanosec=msg.stamp.nanosec, level=msg.level,
                                      logger=msg.name, message=msg.msg)) + '\n')
            component = 'WBC' if is_wbc and not is_mpc else 'MPC'
            row = parse_wbc_window(msg.msg) if component == 'WBC' else parse_mpc_window(msg.msg)
            if row:
                row['component'] = component
                (self.wbc_windows if component == 'WBC' else self.mpc_windows).append(row)
            elif is_mpc:
                row = parse_control(msg.msg)
                if row:
                    row['component'] = 'combined'
            if row:
                row.update(elapsed_s=now, fsm=self.state)
                self.windows.append(row)
                windows_csv.writerow([now, self.state, row['component'], row['hz_state_clock'],
                                      row['median_ms'], row['p95_ms'], row['max_ms'],
                                      row.get('trajectory_age_median_ms'),
                                      row.get('trajectory_age_max_ms'), row['missed'], row['invalid']])
                print(f"[{row['component']}] {row['hz_state_clock']:.1f} Hz clock stato | "
                      f"mediana/p95/max: {row['median_ms']:.2f}/"
                      f"{row['p95_ms']:.2f}/{row['max_ms']:.2f} ms | "
                      f"periodi saltati: {row['missed']} | invalidi: {row['invalid']}"
                      + (f" | età MPC: {row['trajectory_age_median_ms']:.1f}/"
                         f"{row['trajectory_age_max_ms']:.1f} ms"
                         if component == 'WBC' else ''), flush=True)
            if 'Compiling MPX' in msg.msg or 'Compiling the WBC' in msg.msg:
                self.compiling = True
                print(f'[{component}] Compilazione in corso...', flush=True)
            match = COMPILE.search(msg.msg)
            if match:
                self.compilations.append(float(match.group(1)))
                self.compiling = False
                print(f'[{component}] Compilazione completata in {match.group(1)} s.', flush=True)
            match = WBC_COMPILE.search(msg.msg)
            if match:
                self.wbc_compilations.append(float(match.group(1)))
                self.compiling = False
                print(f'[WBC] Compilazione completata in {match.group(1)} s.', flush=True)
            if 'compilation complete' in msg.msg:
                self.compiling = False

        def resources(self):
            now = time.monotonic()
            observer_cpu = time.process_time()
            ot, oc = self.observer_previous
            observer_percent = 100 * (observer_cpu - oc) / (now - ot)
            self.observer_previous = (now, observer_cpu)
            if self.pid is None:
                candidates = find_mpx()
                if len(candidates) == 1:
                    self.pid = candidates[0]
                    metadata['pid'] = self.pid
                    try:
                        metadata['process_command'] = Path(f'/proc/{self.pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    except OSError:
                        pass
                    print(f'CPU/memoria: processo MPX locale PID {self.pid}', flush=True)
                elif len(candidates) > 1 and not self.ambiguous:
                    print(f'Più processi MPX: {candidates}. CPU non attribuita; usare --pid.', flush=True)
                    self.ambiguous = True
            phase = 'compilation' if self.compiling else self.state
            percent = rss = None
            if self.pid is not None:
                try:
                    sample = process_sample(self.pid)
                    rss = sample['rss_mib']
                    if self.proc_previous:
                        pt, previous = self.proc_previous
                        if previous['start_ticks'] != sample['start_ticks']:
                            raise ValueError('PID reused')
                        percent = 100 * (sample['ticks'] - previous['ticks']) / os.sysconf('SC_CLK_TCK') / (now - pt)
                    self.proc_previous = now, sample
                except (OSError, ValueError):
                    self.proc_previous = None
            row = dict(elapsed_s=now-started, phase=phase, pid=self.pid,
                       cpu_percent=percent, rss_mib=rss, observer_cpu_percent=observer_percent)
            self.cpu.append(row)
            cpu_csv.writerow(row.values())
            previous_time, previous_count, previous_intervals = self.live_previous
            command_count = self.count
            hz = (command_count - previous_count) / (now - previous_time)
            received_intervals = self.intervals
            intervals = stats(received_intervals[previous_intervals:])
            interval_text = (
                f"{intervals['median']:.2f}/{intervals['p95']:.2f}/{intervals['max']:.2f} ms"
                if intervals else 'n/d')
            cpu_text = f'{percent:.1f}%' if percent is not None else 'n/d'
            rss_text = f'{rss:.1f} MiB' if rss is not None else 'n/d'
            progress = (f'{now - self.measurement_start:.0f}/{args.duration:g} s'
                        if self.measurement_start is not None else 'attesa attivazione MPC')
            print(
                f'[{progress}] fase={phase} | comandi torque ricevuti: {hz:.1f} Hz | '
                f'intervallo med/p95/max: {interval_text} | '
                f'CPU MPX: {cpu_text} | RAM: {rss_text} | '
                f'messaggi invalidi totali: {self.bad}', flush=True)
            self.live_previous = now, command_count, len(received_intervals)
            for stream in files:
                stream.flush()

    rclpy.init(args=[])
    node = Observer()
    reason = 'duration_complete'
    print(f'Osservo comandi WBC {wbc_topic}, '
          f'log MPC={logger}/WBC={wbc_logger}. Attendo MPC attivo.\n'
          f'Report: {output.resolve()}', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            if node.measurement_start is not None and now - node.measurement_start >= args.duration:
                break
            if node.measurement_start is None and now - started >= args.wait_timeout:
                reason = 'no_mpc_activity_timeout'
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        reason = 'interrupted'
    finally:
        ended = time.monotonic()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for stream in files:
            stream.close()

    windows = node.windows
    combined_windows = [w for w in windows if w.get('component') == 'combined']
    steady = combined_windows[1:]
    mpc_summary = summarize_windows(node.mpc_windows)
    wbc_summary = summarize_windows(node.wbc_windows)
    warnings = []
    if not node.count:
        warnings.append('Nessun comando torque ricevuto: verificare FSM, namespace e topic joint_command.')
    if not steady and not node.mpc_windows and not node.wbc_windows:
        warnings.append('Finestre solver insufficienti: nessuna statistica stabile disponibile.')
    if node.count and not node.mpc_windows:
        warnings.append('Nessuna finestra MPC: controllare --logger-name e /rosout.')
    if node.count and not node.wbc_windows:
        warnings.append('Nessuna finestra WBC: controllare --wbc-logger-name e /rosout.')
    if not any(row['cpu_percent'] is not None for row in node.cpu):
        warnings.append('CPU MPX non disponibile: eseguire sul computer di MPX e/o specificare --pid.')
    if node.state == 'unknown':
        warnings.append('FSM non osservata: separazione delle attivazioni non disponibile.')
    metadata['pid'] = node.pid
    metadata['stop_reason'] = reason
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    span = node.last - node.first if node.count > 1 else 0
    report = dict(
        metadata=metadata, observation_s=ended-started, messages=node.count,
        mpc=mpc_summary, wbc=wbc_summary,
        invalid_torque_messages=node.bad,
        torque_hz_over_first_last_span=(node.count-1)/span if span else None,
        torque_span_s=span, silence_after_last_torque_s=ended-node.last if node.last else None,
        torque_intervals_ms=stats(node.intervals),
        intervals_over_1_5_target_period=sum(v > 1500/args.expected_hz for v in node.intervals),
        compilation_seconds=node.compilations, wbc_compilation_seconds=node.wbc_compilations,
        solver_windows_received=len(combined_windows), solver_windows_used=len(steady),
        solver_mean_window_median_ms=statistics.mean(w['median_ms'] for w in steady) if steady else None,
        solver_mean_window_p95_ms=statistics.mean(w['p95_ms'] for w in steady) if steady else None,
        solver_max_ms=max((w['max_ms'] for w in steady), default=None),
        solver_mean_window_hz_state_clock=statistics.mean(w['hz_state_clock'] for w in steady) if steady else None,
        missed_periods_reported=sum(w['missed'] for w in combined_windows) if combined_windows else None,
        invalid_solves_reported=sum(w['invalid'] for w in combined_windows) if combined_windows else None,
        component_missed_periods={'mpc': mpc_summary['missed_periods'],
                                  'wbc': wbc_summary['missed_periods']},
        component_invalid_solves={'mpc': mpc_summary['invalid_solves'],
                                  'wbc': wbc_summary['invalid_solves']},
        cpu_percent_by_phase={phase: stats([r['cpu_percent'] for r in node.cpu
                                          if r['phase'] == phase and r['cpu_percent'] is not None])
                              for phase in sorted({r['phase'] for r in node.cpu})},
        rss_mib=stats([r['rss_mib'] for r in node.cpu if r['rss_mib'] is not None]),
        observer_cpu_percent=stats([r['observer_cpu_percent'] for r in node.cpu]), warnings=warnings)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def fmt(value):
        return 'n/d' if value is None else f'{value:.3f}'
    interval = report['torque_intervals_ms'] or {}
    lines = [f'Benchmark MPX: {args.label}', f'Esito raccolta: {reason}',
             f'Coppie ricevute: {node.count}; messaggi invalidi: {node.bad}',
             f"Frequenza ricezione, prima-ultima coppia: {fmt(report['torque_hz_over_first_last_span'])} Hz",
             f"Intervallo coppie mediana/p95/max: {fmt(interval.get('median'))} / {fmt(interval.get('p95'))} / {fmt(interval.get('max'))} ms",
             f"Intervalli > 1.5 periodi attesi: {report['intervals_over_1_5_target_period']}",
             f"Solver: media mediane finestre {fmt(report['solver_mean_window_median_ms'])} ms; media p95 finestre {fmt(report['solver_mean_window_p95_ms'])} ms; massimo {fmt(report['solver_max_ms'])} ms",
             f"Frequenza solver sul clock dello stato: {fmt(report['solver_mean_window_hz_state_clock'])} Hz",
             f"Finestre solver: {len(windows)} ricevute, {len(steady)} usate (prima esclusa)",
             f"Periodi saltati / solve invalidi nei log: {report['missed_periods_reported']} / {report['invalid_solves_reported']}",
             f"MPC indipendente: {mpc_summary['hz_state_clock']} Hz sim; "
             f"calcolo med/p95/max {fmt(mpc_summary['median_ms'])}/"
             f"{fmt(mpc_summary['p95_ms'])}/{fmt(mpc_summary['max_ms'])} ms; "
             f"periodi persi/invalidi {mpc_summary['missed_periods']}/"
             f"{mpc_summary['invalid_solves']}",
             f"WBC indipendente: {wbc_summary['hz_state_clock']} Hz sim; "
             f"calcolo med/p95/max {fmt(wbc_summary['median_ms'])}/"
             f"{fmt(wbc_summary['p95_ms'])}/{fmt(wbc_summary['max_ms'])} ms; "
             f"periodi persi/invalidi {wbc_summary['missed_periods']}/"
             f"{wbc_summary['invalid_solves']}; età MPC med/max "
             f"{fmt(wbc_summary['trajectory_age_median_ms'])}/"
             f"{fmt(wbc_summary['trajectory_age_max_ms'])} ms",
             f"Compilazioni osservate MPC/WBC: {node.compilations} / {node.wbc_compilations} s"]
    for phase, data in report['cpu_percent_by_phase'].items():
        lines.append(f"CPU MPX fase {phase}: media {fmt(data['mean'] if data else None)}% (100%=un core)")
    lines += [f"RSS massima: {fmt((report['rss_mib'] or {}).get('max'))} MiB",
              f"CPU media osservatore: {fmt((report['observer_cpu_percent'] or {}).get('mean'))}%",
              'Intervalli misurati alla ricezione, non durata publish() né latenza sensore-motore.',
              'solve ms include preparazione e controlli; non è il solo kernel JAX.',
              'Le medie dei percentili delle finestre non sono percentili globali.',
              'Pause/standby dentro la misura abbassano la frequenza globale; consultare CSV/FSM.',
              'QoS best effort: ritardi o perdite del monitor non provano perdite del controller.']
    lines += ['ATTENZIONE: ' + warning for warning in warnings]
    rendered = '\n'.join(lines) + '\n'
    (output / 'report.txt').write_text(rendered)
    print(rendered)
    print(f'File salvati in {output.resolve()}')
    return 2 if not node.count else 0


if __name__ == '__main__':
    sys.exit(main())
