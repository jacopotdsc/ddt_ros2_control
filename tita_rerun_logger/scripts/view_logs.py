#!/usr/bin/env python3
"""Offline viewer without ROS or colcon: python3 scripts/view_logs.py [--list | -n N | files]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tita_rerun_logger.view_logs import main  # noqa: E402

if __name__ == "__main__":
    main()
