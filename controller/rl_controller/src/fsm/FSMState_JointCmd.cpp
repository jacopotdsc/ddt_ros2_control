#include "rl_controller/fsm/FSMState_JointCmd.h"
#include <algorithm>
#include <chrono>
#include <iostream>

void FSMState_JointCmd::enter()
{
  this->_nextStateName = this->_stateName;
  stale_ = true;
  _data->low_cmd->zero();
}

void FSMState_JointCmd::run()
{
  // Never block the control loop on the ROS callback: keep the previous command for this tick.
  std::unique_lock<std::mutex> lock(_data->joint_command.mutex, std::try_to_lock);
  if (!lock.owns_lock()) return;
  const auto & cmd = _data->joint_command;
  const size_t n = _data->low_cmd->tau_cmd.size();
  const double age =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - cmd.received).count();
  const bool stale = !cmd.valid || cmd.tau_ff.size() != n ||
                     age > _data->params->joint_cmd_params.command_timeout;
  if (stale != stale_) {
    std::cerr << (stale ? "[FSMState_JointCmd] no joint_cmd: zero torque"
                        : "[FSMState_JointCmd] joint_cmd received: applying it") << std::endl;
    stale_ = stale;
  }
  _data->low_cmd->zero();
  if (stale_) return;
  for (size_t i = 0; i < n; ++i) {
    const auto limit = _data->params->torque_limit[i];
    _data->low_cmd->qd(i) = cmd.q_des[i];
    _data->low_cmd->qd_dot(i) = cmd.qd_des[i];
    _data->low_cmd->tau_cmd(i) = std::clamp(cmd.tau_ff[i], -limit, limit);
    _data->low_cmd->kp(i) = cmd.kp[i];
    _data->low_cmd->kd(i) = cmd.kd[i];
  }
}

std::string FSMState_JointCmd::checkTransition()
{
  // The safety and recovery states are always reachable.
  const auto & desire_state = _data->rc_data->fsm_name_;
  if (desire_state == "idle" || desire_state == "joint_pd" || desire_state == "transform_down") {
    return desire_state;
  }
  return this->_stateName;
}
