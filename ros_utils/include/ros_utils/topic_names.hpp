// Copyright (c) 2023 Direct Drive Technology Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ROS_UTILS__TOPIC_NAMES_HPP_
#define ROS_UTILS__TOPIC_NAMES_HPP_

#include <string>

namespace ros_topic
{
// command
inline const std::string joy = "joy";
inline const std::string manager_twist_command = "command/cmd_twist";
inline const std::string manager_pose_command = "command/cmd_pose";
inline const std::string manager_key_command = "command/cmd_key";
inline const std::string fsm_state = "fsm_state";  // current rl_controller FSM state, on change

// locomotion
inline const std::string body_imu = "imu_sensor_broadcaster/imu";
inline const std::string joint_states = "joint_states";
inline const std::string robot_description = "robot_description";
inline const std::string filtered_state = "filtered_state";  // tita_state_estimator

// mpx
inline const std::string mpx_effort = "mpx/effort";
inline const std::string mpx_joint_command = "mpx/joint_command";  // WBC: tau_ff, q_des, qd_des
inline const std::string mpx_llc_command = "mpx/llc_command";      // LLC tracked torque
inline const std::string mpx_handoff = "mpx/handoff";
inline const std::string mpx_keys_locked = "mpx/keys_locked";

}  // namespace ros_topic

#endif  // ROS_UTILS__TOPIC_NAMES_HPP_
