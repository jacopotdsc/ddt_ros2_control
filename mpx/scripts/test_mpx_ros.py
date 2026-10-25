#!/usr/bin/env python3
"""Benchmark isolated production ROS MPC/WBC/LLC nodes with synthetic feedback.
All inputs and outputs use /mpx/benchmark/. No hardware/controller launch.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def benchmark_parameters():
    """Override every input/output topic; production YAMLs remain untouched."""
    import yaml
    from ament_index_python.packages import get_package_share_directory
    cfg = yaml.safe_load((Path(get_package_share_directory('mpx')) /
                          'config/mpx_node.yaml').read_text())
    params = cfg['/**']['ros__parameters']
    prefix = '/mpx/benchmark/'
    for group in ('inputs', 'outputs'):
        for key in params[group]:
            if key.endswith('_topic') or key.endswith('_topic_prefix'):
                params[group][key] = prefix + key.removesuffix('_topic').removesuffix('_topic_prefix')
    params['inputs']['base_state_source'] = 'filter'
    params['behavior']['publish_default_commands'] = False
    params['behavior']['publish_estimate'] = False
    return cfg


def print_topic_summary(params):
    """Show configured application topics before starting any child process."""
    inputs, outputs = params['inputs'], params['outputs']
    feedback = [inputs[key] for key in ('joint_states_topic', 'filtered_state_topic')]
    description = inputs['robot_description_topic']
    fsm = inputs['fsm_state_topic']
    trajectory = outputs['trajectory_topic']
    wbc = outputs['joint_command_topic']
    rows = [
        ('/mpx/benchmark/controller',
         [trajectory, outputs['handoff_topic'], outputs['keys_locked_topic']],
         [description, *feedback, fsm, *[inputs[key] for key in
          ('mode_command_topic', 'cmd_vel_topic', 'com_height_topic', 'pose_command_topic')]]),
        ('/mpx/benchmark/wbc', [wbc], [description, *feedback, trajectory]),
        ('/mpx/benchmark/feedback',
         [description, *feedback, fsm, inputs['mode_command_topic']], [trajectory, wbc]),
        ('/mpx_rate_check', [], [trajectory, wbc, fsm, '/rosout']),
    ]
    print('Topic applicativi configurati (esclusi log e notifiche parametri ROS):', flush=True)
    for name, publishers, subscribers in rows:
        print(f'NODE NAME: {name}', flush=True)
        for label, topics in (('Publishes to topics', publishers),
                              ('Subscribes to topics', subscribers)):
            print(f'  {label}:', flush=True)
            for topic in topics or ['Nessuno']:
                print(f'    - {topic}', flush=True)


def command(args, ros2, params_file, params):
    ns = '/mpx/benchmark'
    nodes = [[ros2, 'run', 'mpx', name, '--ros-args', '-r', f'__ns:={ns}',
              '--params-file', str(params_file)]
             for name in ('mpx_node_cpp', 'mpx_wbc_node')]
    feeder = [sys.executable, str(Path(__file__).resolve().with_name('benchmark_ros_feedback.py')),
              '--params-file', str(params_file)]
    monitor = [sys.executable, str(Path(__file__).resolve().with_name('check_mpx_rates.py')),
               '--duration', str(args.duration), '--startup-timeout', str(args.startup_timeout),
               '--fsm-topic', params['inputs']['fsm_state_topic'],
               '--mpc-topic', params['outputs']['trajectory_topic'],
               '--wbc-topic', params['outputs']['joint_command_topic']]
    for component in ('mpc', 'wbc'):
        monitor += [f'--{component}-hz', str(params['timing'][component + '_frequency'])]
    return nodes, feeder, monitor


def stop(proc, label, timeout=20):
    if proc is None or proc.poll() is not None:
        return
    print(f'Chiusura {label}...', flush=True)
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    except ProcessLookupError:
        pass


def find_workspace(script_path):
    """Find the overlay root for source, symlink and copied colcon installs."""
    script_path = Path(script_path).resolve()
    for parent in script_path.parents:
        if (parent / 'install/setup.bash').is_file():
            return parent
        # Also support a script inside a merged or isolated install without src.
        if parent.name == 'install' and (parent / 'setup.bash').is_file():
            return parent.parent
    raise RuntimeError(
        f'Non trovo il workspace da {script_path}; specificare --workspace /percorso/ddt_ros2_ws.'
    )


def workspace_environment(workspace, ros_setup):
    """Source ROS and this workspace for child ros2 commands, even if caller did not."""
    workspace = Path(workspace).expanduser().resolve()
    install_setup = workspace / 'install/setup.bash'
    if not install_setup.is_file():
        raise RuntimeError(
            f'Non trovo {install_setup}. Compila prima dalla root del workspace: '
            '`colcon build`.'
        )
    setup_files = []
    if ros_setup and Path(ros_setup).expanduser().is_file():
        setup_files.append(str(Path(ros_setup).expanduser()))
    setup_files.append(str(install_setup))
    shell_script = 'for setup_file in "$@"; do source "$setup_file" || exit; done; env -0'
    completed = subprocess.run(
        ['bash', '-c', shell_script, 'bash', *setup_files],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors='replace').strip()
                           or 'Impossibile caricare gli setup ROS/workspace')
    env = {}
    for item in completed.stdout.split(b'\0'):
        if b'=' in item:
            key, value = item.split(b'=', 1)
            env[os.fsdecode(key)] = os.fsdecode(value)
    return env


def main():
    import tempfile
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
                        help='workspace root; detected from source or installed script by default')
    parser.add_argument('--ros-setup', default='/opt/ros/humble/setup.bash')
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--startup-timeout', type=float, default=240.)
    args = parser.parse_args()
    if args.duration <= 0 or args.startup_timeout <= 0:
        parser.error('duration and startup-timeout must be positive')
    try:
        workspace = args.workspace if args.workspace is not None else find_workspace(__file__)
        env = workspace_environment(workspace, args.ros_setup)
    except RuntimeError as exc:
        parser.error(str(exc))
    ros2 = shutil.which('ros2', path=env.get('PATH'))
    if not ros2:
        parser.error('ros2 not found')
    # Read package configs inside the sourced environment, also when called unsourced.
    config_cmd = [sys.executable, '-c',
                  'import sys,yaml;sys.path.insert(0,sys.argv[1]);'
                  'from test_mpx_ros import benchmark_parameters;'
                  'print(yaml.safe_dump(benchmark_parameters()))', str(Path(__file__).resolve().parent)]
    cfg = yaml.safe_load(subprocess.check_output(config_cmd, env=env, text=True))
    params = cfg['/**']['ros__parameters']
    print_topic_summary(params)
    children = []
    print('Benchmark ROS isolato: MPC/WBC/LLC + feedback sintetico, topic /mpx/benchmark/.', flush=True)
    print('Nessun hardware, rl_controller o tastiera avviato; coppie solo sui topic benchmark.', flush=True)
    with tempfile.TemporaryDirectory(prefix='mpx_ros_benchmark_') as directory:
        params_file = Path(directory) / 'params.yaml'
        params_file.write_text(yaml.safe_dump(cfg))
        nodes, feeder, monitor = command(args, ros2, params_file, params)
        try:
            for cmd in [*nodes, feeder, monitor]:
                children.append(subprocess.Popen(cmd, env=env, start_new_session=True))
            while children[-1].poll() is None:
                if any(proc.poll() is not None for proc in children[:-1]):
                    print('Un processo benchmark è terminato prima del monitor.', flush=True)
                    return 2
                time.sleep(.25)
            return children[-1].returncode
        except KeyboardInterrupt:
            return 130
        finally:
            for proc in reversed(children):
                stop(proc, 'processo benchmark')


if __name__ == '__main__':
    raise SystemExit(main())
