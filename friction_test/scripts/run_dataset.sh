#!/bin/bash
# usage: run_dataset.sh <name> [key] [noise]
#   name   output folder friction_test_csv/csv_simulazione/<name>/
#   key    friction_test_node test key: t (default, every test in sequence) or one of 2-9 (one joint)
#   noise  true (default, sensor noise measured on the robot) or false
# Robot hung in Gazebo (GUI, joint friction), friction_test_node driven by drive_node.py (key 1, then the test
# key), bag of /joint_states and the IMU (check the noise with sensor_noise_from_bags.py). The node writes
# friction_test_csv/csv_simulazione/joint_*.csv; they are moved to friction_test_csv/csv_simulazione/<name>/.
# Never closes a simulation it did not start.
name=${1:?name}; key=${2:-t}; noise=${3:-true}
WS=/home/ubuntu/ddt_ros2_ws
CSV=$WS/src/friction_test/friction_test_csv/csv_simulazione
OUT=$CSV/$name
export DISPLAY=${DISPLAY:-:0}
cd $WS && source install/setup.bash
P='gzserve[r]|gzclien[t]|friction_estimate[.]launch|friction_test_nod[e]|ros2 bag recor[d]|robot_state_publishe[r]|spawn_entit[y]|controller_manager/spawne[r]'
if pgrep -f "gzserve[r]|gzclien[t]|ros2_control_nod[e]" > /dev/null; then
  echo "ABORT: a Gazebo is already running:"; pgrep -af "gzserve[r]|gzclien[t]"; exit 1
fi
if ls $CSV/joint_*.csv > /dev/null 2>&1; then
  echo "ABORT: $CSV/joint_*.csv exist (move them first, the node would overwrite them)"; exit 1
fi
rm -rf $OUT; mkdir -p $OUT
setsid ros2 launch friction_test friction_estimate.launch.py noise:=$noise > $OUT/launch.log 2>&1 &
for i in $(seq 120); do grep -q "activated tita_rl_controller" $OUT/launch.log && break; sleep 1; done
sleep 5
grep -E "SENSOR NOISE|sensor noise off|friction_test: Gazebo" $OUT/launch.log | sed 's/^/  /'
setsid ros2 bag record -o $OUT/bag /joint_states /imu_sensor_broadcaster/imu /fsm_state > $OUT/bag.log 2>&1 &
sleep 3
if [ "$key" = t ]; then done_text="sequence finished"; timeout=3600; else done_text="rows ->"; timeout=1200; fi
python3 $WS/src/friction_test/scripts/drive_node.py key:1 "wait:holding the default pose@60" sleep:3 key:$key \
  "wait:$done_text@$timeout" sleep:5 > $OUT/drive.log 2>&1
status=$?
cat $OUT/drive.log
grep -h "test started" $OUT/drive.log $OUT/launch.log 2>/dev/null | head -20
pkill -INT -f 'ros2 bag recor[d]'; sleep 3
pkill -INT -f "$P"; sleep 5; pkill -9 -f "$P"; sleep 1
pgrep -af "$P|ros2_control_nod[e]" || echo "all closed"
mv $CSV/joint_*.csv $OUT/ 2>/dev/null
ls $OUT
exit $status
