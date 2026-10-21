#!/usr/bin/env bash
# Synthetic feedback; computes locally without sending commands to motors.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd -- "$script_dir/../../.." && pwd -P)"
if [[ -f "$workspace/install/setup.bash" ]]; then
  source "$workspace/install/setup.bash"
fi
if ! command -v ros2 >/dev/null; then
  echo 'Source ROS and the workspace install/setup.bash first.' >&2
  exit 1
fi
exec ros2 run mpx benchmark_mpx_cpp "$@"
