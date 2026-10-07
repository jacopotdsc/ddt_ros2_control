#include "rl_controller/fsm/FSMState_MPC.h"
#include <algorithm>
#include <iostream>

void FSMState_MPC::run()
{
  std::lock_guard<std::mutex> lock(_data->mpc_command.mutex);
  // No timeout: retain the previous command while waiting for valid MPC output.
  if (!_data->mpc_command.valid) return;
  // Pure torque control: no PD contribution from the previous RL state.
  _data->low_cmd->zero();
  for (size_t i = 0; i < _data->mpc_command.effort.size(); ++i) {
    const auto limit = _data->params->torque_limit[i];
    _data->low_cmd->tau_cmd(i) = std::clamp(_data->mpc_command.effort[i], -limit, limit);
  }
}

std::string FSMState_MPC::checkTransition()
{
  const auto & command = _data->rc_data->fsm_name_;
  if (command == "idle" || command == "joint_pd" || command == "transform_down") return command;
  for (size_t i = 0; i < _data->params->rl_policy_names.size(); ++i) {
    if (command != "rl_" + std::to_string(i)) continue;
    // MPX first brings the robot to h_rl, then publishes the handed-off mode.
    std::lock_guard<std::mutex> lock(_data->mpc_command.mutex);
    if (_data->mpc_command.handoff == command) return _data->params->rl_policy_names[i];
  }
  return "mpc";
}
