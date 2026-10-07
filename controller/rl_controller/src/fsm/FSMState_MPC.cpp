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
  // Pure Python MPX still sends torque directly; the hierarchical path waits for
  // the independent LLC node to publish its tracked torque.
  const bool use_llc = !mpc.q_des.empty() && !mpc.llc_effort.empty();
  const auto & effort = use_llc ? mpc.llc_effort : mpc.effort;
  // MPX stopped or dead: never hold its last torque. checkTransition folds the robot.
  // Hierarchical path: the LLC is the last stage, so its torque must stay fresh (the
  // LLC only publishes in "mpc": count from the entry). Python path: mpx/effort.
  const auto last = mpc.q_des.empty() ? mpc.received : std::max(mpc.llc_received, entered_);
  const double age = std::chrono::duration<double>(start - last).count();
  const bool stale = effort.empty() || age > _data->params->mpc_command_timeout;
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
  if (!mpc.valid || effort.size() != mpc.effort.size()) return;
  _data->low_cmd->zero();
  // The standalone LLC publishes the already tracked torque. The legacy Python
  // path has no q_des and continues to use its direct torque vector.
  for (size_t i = 0; i < effort.size(); ++i) {
    const auto limit = _data->params->torque_limit[i];
    _data->low_cmd->tau_cmd(i) = std::clamp(effort[i], -limit, limit);
  }
  if (!mpc.q_des.empty() && mpc.llc_us.size() < 100000) {
    mpc.llc_us.push_back(
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
  }
}

std::string FSMState_MPC::checkTransition()
{
  const auto & command = _data->rc_data->fsm_name_;
  if (command == "idle" || command == "joint_pd") return command;
  // No MPX torque for mpc_command_timeout: fold now, without waiting for the MPX handoff.
  if (stale_) return "transform_down";
  // For rl_N and transform_down MPX first brings the robot to h_rl at rest, then
  // publishes the handed-off mode.
  std::lock_guard<std::mutex> lock(_data->mpc_command.mutex);
  if (_data->mpc_command.handoff != command) return "mpc";
  if (command == "transform_down") return command;
  for (size_t i = 0; i < _data->params->rl_policy_names.size(); ++i) {
    if (command == "rl_" + std::to_string(i)) return _data->params->rl_policy_names[i];
  }
  return "mpc";
}
