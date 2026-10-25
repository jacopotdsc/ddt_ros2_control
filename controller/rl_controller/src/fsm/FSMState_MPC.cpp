#include "rl_controller/fsm/FSMState_MPC.h"
#include <algorithm>
#include <chrono>
#include <iostream>

void FSMState_MPC::run()
{
  const auto start = std::chrono::steady_clock::now();
  // Never block the control loop on the ROS callbacks: if they hold the command,
  // keep the previous low-level command for this tick.
  std::unique_lock<std::mutex> lock(_data->mpc_command.mutex, std::try_to_lock);
  if (!lock.owns_lock()) return;
  auto & mpc = _data->mpc_command;
  const auto & effort = mpc.effort;
  const auto & kp = _data->params->mpc_params.joint_kp;
  const auto & kd = _data->params->mpc_params.joint_kd;
  // One PD gain and one target per joint (mpx/joint_command: tau_ff, q_des, qd_des).
  const bool sizes_ok = !effort.empty() && kp.size() == effort.size() && kd.size() == effort.size() &&
                        mpc.q_des.size() == effort.size() && mpc.qd_des.size() == effort.size();
  // MPX stopped or dead: never hold its last torque. checkTransition folds the robot.
  const double age = std::chrono::duration<double>(start - mpc.received).count();
  const bool stale = !sizes_ok || age > _data->params->mpc_params.command_timeout;
  if (stale != stale_) {
    std::cerr << (stale ? "[FSMState_MPC] no MPX torque for " : "[FSMState_MPC] MPX torque back after ")
              << age << " s" << (stale ? ": damping, then transform_down" : "") << std::endl;
    stale_ = stale;
  }
  if (stale_) {
    _data->low_cmd->zero();
    for (int i = 0; i < _data->low_cmd->kd.size(); i++) _data->low_cmd->kd(i) = 5;
    for (auto i : _data->params->wheel_indices) _data->low_cmd->kd(i) = 0;
    return;
  }
  if (!mpc.valid) return;
  _data->low_cmd->zero();
  // PD around the WBC feedforward on this cycle's joints (read just before run()),
  // computed here rather than in the hardware bridge so torque_limit bounds the total.
  const auto & q = _data->low_state->q;
  const auto & dq = _data->low_state->dq;
  for (size_t i = 0; i < effort.size(); ++i) {
    const double tau = effort[i] + kp[i] * (mpc.q_des[i] - q(i)) + kd[i] * (mpc.qd_des[i] - dq(i));
    const auto limit = _data->params->torque_limit[i];
    _data->low_cmd->tau_cmd(i) = std::clamp(tau, -limit, limit);
  }
  if (mpc.run_us.size() < 100000) {
    mpc.run_us.push_back(
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
  }
}

std::string FSMState_MPC::checkTransition()
{
  const auto & command = _data->rc_data->fsm_name_;
  // idle and joint_pd are the safety commands: immediate, in any situation.
  if (command == "idle" || command == "joint_pd") return command;
  // No MPX command for mpc.command_timeout: fold now, without waiting for the MPX handoff.
  if (stale_) return "transform_down";
  // For rl_N and transform_down MPX first lowers the robot to h_rl at rest, then
  // publishes the handed-off mode.
  std::lock_guard<std::mutex> lock(_data->mpc_command.mutex);
  if (_data->mpc_command.handoff != command) return "mpc";
  if (command == "transform_down") return command;
  for (size_t i = 0; i < _data->params->rl_policy_names.size(); ++i) {
    if (command == "rl_" + std::to_string(i)) return _data->params->rl_policy_names[i];
  }
  return "mpc";
}
