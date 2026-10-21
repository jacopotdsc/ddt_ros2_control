#!/usr/bin/env python3
"""Passively measure MPC, WBC and LLC topic rates while the FSM is in MPC."""

import argparse
import math
import re
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from rcl_interfaces.msg import Log


MPC_SOLVE = re.compile(
    r'MPC: ([\d.]+) Hz sim, plan ms median=([\d.]+) p95=([\d.]+) '
    r'max=([\d.]+), missed periods=(\d+), invalid=(\d+)')
WBC_SOLVE = re.compile(
    r'WBC: ([\d.]+) Hz sim, ms median=([\d.]+) p95=([\d.]+) '
    r'max=([\d.]+), MPC trajectory age ms median=([\d.]+) max=([\d.]+), '
    r'missed periods=(\d+), invalid=(\d+)')


def parse_solve_window(message, component):
    match = (MPC_SOLVE if component == 'MPC' else WBC_SOLVE).search(message)
    if not match:
        return None
    values = [float(value) for value in match.groups()]
    row = dict(hz_sim=values[0], median_ms=values[1], p95_ms=values[2],
               max_ms=values[3])
    if component == 'WBC':
        row.update(trajectory_age_median_ms=values[4], trajectory_age_max_ms=values[5],
                   missed=int(values[6]), invalid=int(values[7]))
    else:
        row.update(missed=int(values[4]), invalid=int(values[5]))
    return row


class Rate:
    def __init__(self, label, topic, target_hz):
        self.label = label
        self.topic = topic
        self.target_hz = target_hz
        self.count = 0
        self.invalid = 0
        self.intervals = []
        self.previous = None

    def reset_segment(self):
        self.previous = None

    def add(self, now):
        self.count += 1
        if self.previous is not None:
            dt = now - self.previous
            if dt > 0:
                self.intervals.append(dt)
        self.previous = now

    @property
    def hz(self):
        span = sum(self.intervals)
        return len(self.intervals) / span if span > 0 else 0.0

    @property
    def p95_ms(self):
        return 1000.0 * percentile(self.intervals, 0.95) if self.intervals else math.nan

    @property
    def max_ms(self):
        return 1000.0 * max(self.intervals) if self.intervals else math.nan

    @property
    def missed(self):
        period = 1.0 / self.target_hz
        return sum(max(0, round(dt / period) - 1) for dt in self.intervals)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def topic(namespace, name):
    if name.startswith("/"):
        return name
    return "/" + "/".join(part for part in (namespace, name) if part.strip("/"))


class RateMonitor(Node):
    def __init__(self, args):
        super().__init__("mpx_rate_check")
        ns = args.namespace.strip("/")
        self.fsm_topic = topic(ns, args.fsm_topic) if args.fsm_topic else ""
        self.state = "mpc" if args.no_fsm else "unknown"
        self.rates = [
            Rate("MPC", topic(ns, args.mpc_topic), args.mpc_hz),
            Rate("WBC", topic(ns, args.wbc_topic), args.wbc_hz),
            Rate("LLC", topic(ns, args.llc_topic), args.llc_hz),
        ]
        self.solve_windows = {'MPC': [], 'WBC': []}
        self.duration = args.duration
        self.active_elapsed = 0.0
        self.segment_start = time.monotonic() if args.no_fsm else None
        self.started = self.segment_start
        self.last_print = self.segment_start or 0.0
        self.previous_print = self.segment_start
        self.previous_counts = [0, 0, 0]
        self.done = False
        self.qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.joint_count = args.joint_count
        self.create_subscription(Float64MultiArray, self.rates[0].topic,
                                 self.on_mpc, self.qos)
        self.create_subscription(JointState, self.rates[1].topic,
                                 self.on_wbc, self.qos)
        self.create_subscription(Float64MultiArray, self.rates[2].topic,
                                 self.on_llc, self.qos)
        if self.fsm_topic:
            reliable_latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                          durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, self.fsm_topic, self.on_fsm, reliable_latched)
        self.create_subscription(Log, '/rosout', self.on_rosout, self.qos)
        print("Misura passiva delle frequenze ROS (nessun comando viene inviato).", flush=True)
        print(f"FSM: {'disabilitata' if args.no_fsm else self.fsm_topic}; "
              f"durata: {self.duration:.0f} s effettivi nello stato mpc", flush=True)
        for rate in self.rates:
            print(f"  {rate.label}: {rate.topic}, target {rate.target_hz:.0f} Hz", flush=True)
        if not args.no_fsm:
            print("Attesa attivazione FSM del benchmark." if self.fsm_topic.startswith("/mpx/benchmark/")
                  else "Porta il robot in MPC con il joystick; la misura parte quando FSM=mpc.",
                  flush=True)
        else:
            print("FSM disabilitata; misura avviata subito.", flush=True)

    def on_fsm(self, msg):
        new_state = msg.data.strip()
        if new_state == self.state:
            return
        now = time.monotonic()
        if self.state == "mpc" and self.segment_start is not None:
            self.active_elapsed += now - self.segment_start
            self.segment_start = None
            for rate in self.rates:
                rate.reset_segment()
            print(f"FSM={new_state}; pausa misura, accumulati {self.active_elapsed:.1f} s in MPC.",
                  flush=True)
        self.state = new_state
        if self.state == "mpc":
            self.segment_start = now
            if self.started is None:
                self.started = now
                self.previous_print = now
                self.last_print = now
                print("FSM=mpc; misura avviata.", flush=True)
            else:
                print("FSM=mpc; misura ripresa.", flush=True)

    def record(self, index, valid):
        if self.state == "mpc":
            rate = self.rates[index]
            rate.add(time.monotonic())
            if not valid:
                rate.invalid += 1

    def on_mpc(self, msg):
        valid = (len(msg.data) >= 6 and all(math.isfinite(x) for x in msg.data)
                 and msg.data[4] > 0.5)
        self.record(0, valid)

    def on_wbc(self, msg):
        n = len(msg.name)
        valid = (n > 0 and len(msg.position) == n and len(msg.velocity) == n
                 and len(msg.effort) == n
                 and all(math.isfinite(x) for x in msg.position + msg.velocity + msg.effort))
        self.record(1, valid)

    def on_llc(self, msg):
        valid = (len(msg.data) == self.joint_count
                 and all(math.isfinite(x) for x in msg.data))
        self.record(2, valid)

    def on_rosout(self, msg):
        if msg.name.endswith('.controller') or msg.name == 'controller':
            component = 'MPC'
        elif msg.name.endswith('.wbc') or msg.name == 'wbc':
            component = 'WBC'
        else:
            return
        row = parse_solve_window(msg.msg, component)
        if row is None:
            return
        self.solve_windows[component].append(row)
        median_hz = 1000. / row['median_ms'] if row['median_ms'] > 0 else math.inf
        p95_hz = 1000. / row['p95_ms'] if row['p95_ms'] > 0 else math.inf
        detail = (f" | età reference MPC med/max="
                  f"{row['trajectory_age_median_ms']:.1f}/{row['trajectory_age_max_ms']:.1f} ms"
                  if component == 'WBC' else '')
        print(f"[{component} solver] solve singolo ms med/p95/max="
              f"{row['median_ms']:.2f}/{row['p95_ms']:.2f}/{row['max_ms']:.2f}; "
              f"capacità equivalente 1000/solve={median_hz:.1f} Hz (mediana), "
              f"{p95_hz:.1f} Hz (p95); cadenza MPC={row['hz_sim']:.1f} Hz sim; "
              f"missed={row['missed']}, invalid={row['invalid']}" + detail, flush=True)

    def tick(self):
        now = time.monotonic()
        if self.segment_start is not None:
            elapsed = self.active_elapsed + now - self.segment_start
            if now - self.last_print >= 1.0:
                dt = now - self.previous_print if self.previous_print else 1.0
                live = []
                for i, rate in enumerate(self.rates):
                    count = rate.count - self.previous_counts[i]
                    live.append(f"{rate.label} {count / dt:6.1f} Hz")
                    self.previous_counts[i] = rate.count
                self.previous_print = now
                print(f"[{elapsed:5.1f}/{self.duration:.0f} s] " + " | ".join(live), flush=True)
                self.last_print = now
            if elapsed >= self.duration:
                self.active_elapsed = elapsed
                self.segment_start = None
                self.done = True

    def report(self, tolerance, max_missed_percent):
        print("\nRisultati (frequenza ricevuta dal monitor; jitter include trasporto ROS/rete):")
        passed = True
        for rate in self.rates:
            missed_limit = max(1, rate.count * max_missed_percent / 100.0)
            ok = (rate.count >= 2 and rate.invalid == 0
                  and abs(rate.hz - rate.target_hz) <= tolerance * rate.target_hz
                  and rate.missed <= missed_limit)
            passed &= ok
            print(f"  {rate.label:3s}: {rate.hz:7.2f} Hz / target {rate.target_hz:.1f}; "
                  f"messaggi={rate.count}, intervallo p95={rate.p95_ms:.2f} ms, "
                  f"max={rate.max_ms:.2f} ms, periodi persi stimati={rate.missed}, "
                  f"messaggi incoerenti={rate.invalid} "
                  f"[{ 'OK' if ok else 'FUORI TARGET' }]")
        mpc, wbc, llc = (rate.hz for rate in self.rates)
        for label, measured, expected in (("WBC/MPC", wbc / mpc if mpc else 0.0,
                                           self.rates[1].target_hz / self.rates[0].target_hz),
                                          ("LLC/WBC", llc / wbc if wbc else 0.0,
                                           self.rates[2].target_hz / self.rates[1].target_hz)):
            ok = measured > 0 and abs(measured - expected) <= tolerance * expected
            passed &= ok
            print(f"  rapporto {label}: {measured:.2f} / atteso {expected:.2f} "
                  f"[{ 'OK' if ok else 'FUORI TARGET' }]")
        for component in ('MPC', 'WBC'):
            windows = self.solve_windows[component]
            if not windows:
                print(f"  {component} solve: nessuna finestra timing ricevuta da /rosout")
                continue
            median_ms = sum(w['median_ms'] for w in windows) / len(windows)
            p95_ms = sum(w['p95_ms'] for w in windows) / len(windows)
            max_ms = max(w['max_ms'] for w in windows)
            print(f"  {component} solve: {len(windows)} finestre; media mediane/p95 "
                  f"{median_ms:.2f}/{p95_ms:.2f} ms, max {max_ms:.2f} ms; "
                  f"capacità equivalente med/p95 {1000./median_ms:.1f}/"
                  f"{1000./p95_ms:.1f} Hz")
        print("ESITO: " + ("OK" if passed else "VERIFICA NECESSARIA"))
        return passed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="", help="namespace ROS del robot, per es. /tita")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="secondi effettivi di MPC da misurare (default: 20)")
    parser.add_argument("--startup-timeout", type=float, default=180.0,
                        help="attesa massima dello stato FSM mpc (default: 180 s)")
    parser.add_argument("--tolerance", type=float, default=0.20,
                        help="tolleranza relativa sulle frequenze/rapporti (default: 0.20)")
    parser.add_argument("--max-missed-percent", type=float, default=1.0,
                        help="massima percentuale di periodi persi stimati (default: 1%)")
    parser.add_argument("--joint-count", type=int, default=8,
                        help="numero di giunti atteso nel comando LLC (default: 8)")
    parser.add_argument("--mpc-hz", type=float, default=50.0)
    parser.add_argument("--wbc-hz", type=float, default=100.0)
    parser.add_argument("--llc-hz", type=float, default=500.0)
    parser.add_argument("--fsm-topic", default="fsm_state")
    parser.add_argument("--no-fsm", action="store_true",
                        help="misura subito senza filtrare sullo stato FSM")
    parser.add_argument("--mpc-topic", default="mpx/trajectory")
    parser.add_argument("--wbc-topic", default="mpx/joint_command")
    parser.add_argument("--llc-topic", default="mpx/llc_command")
    args = parser.parse_args()
    if (args.duration <= 0 or args.startup_timeout <= 0 or not 0 < args.tolerance < 1
            or not 0 <= args.max_missed_percent < 100 or args.joint_count <= 0):
        parser.error("duration/startup-timeout devono essere >0; tolerance tra 0 e 1; "
                     "max-missed-percent tra 0 e 100 e joint-count >0")
    if min(args.mpc_hz, args.wbc_hz, args.llc_hz) <= 0:
        parser.error("le frequenze attese devono essere >0")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = RateMonitor(args)
    wait_start = time.monotonic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
            node.tick()
            if node.started is None and time.monotonic() - wait_start >= args.startup_timeout:
                print(f"Timeout: FSM non è entrata in mpc entro "
                      f"{args.startup_timeout:.0f} s.", flush=True)
                return 2
        if not node.started:
            print("Misura interrotta prima dell'avvio.", flush=True)
            return 2
        return 0 if node.report(args.tolerance, args.max_missed_percent) else 1
    except KeyboardInterrupt:
        print("\nMisura interrotta dall'utente.", flush=True)
        if node.started:
            node.report(args.tolerance, args.max_missed_percent)
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
