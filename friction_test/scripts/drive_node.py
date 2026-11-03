#!/usr/bin/env python3
"""Run friction_test_node in a pseudo-terminal and send keys, waiting for its messages.

usage: drive_node.py <step> [<step> ...], step = 'key:<c>' | 'wait:<text>[@<timeout s>]' | 'sleep:<s>'
'wait' looks for <text> in the node's screen output (ANSI codes removed). Exit 1 on a timeout."""
import os
import pty
import re
import subprocess
import sys
import time

ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")


def main():
    master, slave = pty.openpty()
    proc = subprocess.Popen(["ros2", "run", "friction_test", "friction_test_node",
                             "--ros-args", "-p", "use_sim_time:=true"],  # Gazebo clock
                            stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.set_blocking(master, False)
    buf = b""

    def pump():
        nonlocal buf
        try:
            while True:
                chunk = os.read(master, 65536)
                if not chunk:
                    break
                buf += ANSI.sub(b"", chunk)
        except BlockingIOError:
            pass
        buf = buf[-200000:]

    ok = True
    for step in sys.argv[1:]:
        kind, _, arg = step.partition(":")
        if kind == "key":
            pump()
            buf = b""
            os.write(master, arg.encode())
            print(f"[drive] key {arg!r}", flush=True)
        elif kind == "sleep":
            end = time.time() + float(arg)
            while time.time() < end:
                pump(); time.sleep(0.1)
        elif kind == "wait":
            text, _, timeout = arg.partition("@")
            end = time.time() + float(timeout or 120)
            while time.time() < end and text.encode() not in buf:
                pump(); time.sleep(0.1)
            found = text.encode() in buf
            print(f"[drive] wait {text!r}: {'ok' if found else 'TIMEOUT'}", flush=True)
            if not found:
                ok = False
                break
    pump()
    lines = [l for l in buf.decode(errors="ignore").splitlines() if "->" in l or "rows" in l or "test started" in l]
    print("\n".join(dict.fromkeys(lines[-40:])), flush=True)
    os.write(master, b"0")
    time.sleep(1)
    os.write(master, b"\x03")
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
