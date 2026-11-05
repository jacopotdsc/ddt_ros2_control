#include "rl_controller/fsm/FSMState_Residual.h"

#include <algorithm>
#include <chrono>
#include <iostream>

FSMState_Residual::FSMState_Residual(std::shared_ptr<ControlFSMData> data)
: FSMState(data, "residual")
{
}

void FSMState_Residual::enter()
{
  stale_ = false;
  delta_tau_.assign(_data->low_cmd->tau_cmd.size(), 0.0);
}

void FSMState_Residual::exit() {}

void FSMState_Residual::update_delta()
{
  // Dummy: the network thread will run the policy here and write delta_tau_.
}

std::vector<double> FSMState_Residual::delta() const
{
  // Dummy: no network yet.
  return std::vector<double>(delta_tau_.size(), 0.0);
}

void FSMState_Residual::run()
{
  // Same as FSMState_MPC::run(), with the network offset on tau_ff.
  const auto start = std::chrono::steady_clock::now();
  // Never block the control loop on the ROS callbacks: if they hold the command,
  // keep the previous low-level command for this tick.
  std::unique_lock<std::mutex> lock(_data->mpc_command.mutex, std::try_to_lock);
  if (!lock.owns_lock()) return;
  auto & mpc = _data->mpc_command;
  const auto & effort = mpc.effort;
  const auto & kp = _data->params->mpc_params.joint_kp;
  const auto & kd = _data->params->mpc_params.joint_kd;
  const bool sizes_ok = !effort.empty() && kp.size() == effort.size() && kd.size() == effort.size() &&
                        mpc.q_des.size() == effort.size() && mpc.qd_des.size() == effort.size();
  // MPX stopped or dead: never hold its last torque. checkTransition folds the robot.
  const double age = std::chrono::duration<double>(start - mpc.received).count();
  const bool stale = !sizes_ok || age > _data->params->mpc_params.command_timeout;
  if (stale != stale_) {
    std::cerr << (stale ? "[FSMState_Residual] no MPX torque for "
                        : "[FSMState_Residual] MPX torque back after ")
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
  const std::vector<double> dtau = delta();
  _data->low_cmd->zero();
  const auto & q = _data->low_state->q;
  const auto & dq = _data->low_state->dq;
  for (size_t i = 0; i < effort.size(); ++i) {
    const double tau_ff = effort[i] + (i < dtau.size() ? dtau[i] : 0.0);
    const double tau = tau_ff + kp[i] * (mpc.q_des[i] - q(i)) + kd[i] * (mpc.qd_des[i] - dq(i));
    const auto limit = _data->params->torque_limit[i];
    _data->low_cmd->tau_cmd(i) = std::clamp(tau, -limit, limit);
  }
}

std::string FSMState_Residual::checkTransition()
{
  const auto & command = _data->rc_data->fsm_name_;
  // idle and joint_pd are the safety commands: immediate, in any situation.
  if (command == "idle" || command == "joint_pd") return command;
  // No MPX command for mpc.command_timeout: fold now.
  if (stale_) return "transform_down";
  // MPX is already in control: back to plain mpc at once.
  if (command == "mpc") return "mpc";
  // rl_N and transform_down: as from mpc, wait for the MPX handoff (COM at h_rl, at rest).
  std::lock_guard<std::mutex> lock(_data->mpc_command.mutex);
  if (_data->mpc_command.handoff != command) return "residual";
  if (command == "transform_down") return command;
  for (size_t i = 0; i < _data->params->rl_policy_names.size(); ++i) {
    if (command == "rl_" + std::to_string(i)) return _data->params->rl_policy_names[i];
  }
  return "residual";
}
