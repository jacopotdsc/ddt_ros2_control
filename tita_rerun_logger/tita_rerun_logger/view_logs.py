#!/usr/bin/env python3
"""Open TITA Rerun recordings offline (ROS is not needed).

Examples:
    view_logs              # latest recording
    view_logs --list       # list recordings
    view_logs -n 3         # the 3 latest, together in the viewer
    view_logs --all        # every recording in rerun_log/
    view_logs path/a.rrd   # specific file(s)
"""

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from tita_rerun_logger.blueprint import APPLICATION_ID, make_blueprint


def default_log_dir():
    """src/tita_rerun_logger/rerun_log (also from an install with --symlink-install)."""
    here = Path(__file__).resolve().parent.parent
    if (here / "rerun_log").is_dir():
        return here / "rerun_log"
    try:
        from ament_index_python.packages import get_package_share_directory
        ws = Path(get_package_share_directory("tita_rerun_logger")).parents[3]
        candidate = ws / "src" / "tita_rerun_logger" / "rerun_log"
        if candidate.is_dir():
            return candidate
    except Exception:  # noqa: BLE001
        pass
    return Path.home() / "rerun_logs"


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help=".rrd files to open")
    ap.add_argument("--dir", type=Path, default=None, help="log folder (default rerun_log/)")
    ap.add_argument("-n", type=int, default=1, help="open the N most recent recordings")
    ap.add_argument("--all", action="store_true", help="open all recordings")
    ap.add_argument("--list", action="store_true", help="only list recordings")
    ap.add_argument("--memory-limit", default="50%", help="viewer RAM limit (e.g. 4GB, 50%%)")
    args = ap.parse_args()

    log_dir = args.dir or default_log_dir()
    recordings = sorted(log_dir.glob("*.rrd"), key=lambda p: p.stat().st_mtime)

    if args.list:
        if not recordings:
            print(f"No recordings in {log_dir}")
        for i, p in enumerate(reversed(recordings)):
            ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{i}] {ts}  {human_size(p.stat().st_size):>8}  {p.name}")
        return

    selected = args.files or (recordings if args.all else recordings[-args.n:])
    missing = [p for p in selected if not p.is_file()]
    if missing:
        sys.exit(f"File not found: {', '.join(map(str, missing))}")
    if not selected:
        sys.exit(f"No recordings in {log_dir}. Run the logger first.")

    # Same tabs and plots as during logging
    blueprint = Path(tempfile.gettempdir()) / "tita_mpx_layout.rbl"
    make_blueprint().save(APPLICATION_ID, str(blueprint))

    print("Opening:\n  " + "\n  ".join(str(p) for p in selected))
    cmd = [sys.executable, "-m", "rerun", "--memory-limit", args.memory_limit,
           *map(str, selected), str(blueprint)]
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
