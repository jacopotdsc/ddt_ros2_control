# fsm_audio

`fsm_audio` announces state changes from `rl_controller` using TITA's
`PlayAudioWithFilePath` service. It subscribes to the relative `fsm_state` topic and
plays `config/fsm_mp3/{state_name}.mp3` when that file exists. It also announces the current
state once when it starts, including the latched initial state. It does not listen to
`joint_command`: that topic carries joint setpoints, while `fsm_state` carries the actual
FSM state published by `rl_controller`.

All custom MP3 clips belong to this package under `config/fsm_mp3/`. At build time,
`setup.py` installs them into the package share directory. The node passes their absolute
installed paths to TITA's audio service. It does not read from, copy to, or modify
`/opt/tita/sound` or the factory audio assets.

The filename must match the FSM state exactly, for example `transform_up.mp3` for the
`transform_up` state. States without a matching MP3 are logged and not announced.

Transfer the package, including the generated files, to the robot:

```bash
rsync -av --progress \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  ~/ddt_ros2_ws/src/interaction/fsm_audio/ \
  robot@tita:~/ddt_ros2_ws/src/interaction/fsm_audio/
```

## Build and run on the robot

```bash
source /opt/tita/ros2/setup.bash
cd ~/ddt_ros2_ws
colcon build --packages-select fsm_audio
source install/setup.bash
ros2 run fsm_audio fsm_audio_node --ros-args -r __ns:=/tita4267305
```

The node loads each MP3 from its installed `fsm_audio` package path. No files need to be
copied to the factory sound directory. It announces the first transient-local FSM message
and each subsequent state change. Keep each custom MP3 in `config/fsm_mp3/`; `setup.py`
includes those files in the installed package.
