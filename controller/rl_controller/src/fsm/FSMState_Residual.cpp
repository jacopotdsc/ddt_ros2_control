#include "rl_controller/fsm/FSMState_Residual.h"

#include <algorithm>
#include <chrono>
#include <iostream>

FSMState_Residual::FSMState_Residual(std::shared_ptr<ControlFSMData> data, FSMState_MPC * mpc)
: FSMState(data, "residual"), mpc_(mpc)
{
}

void FSMState_Residual::enter()
{
  mpc_->enter();
  offset_.clear();
  offset_stale_ = false;  // so the first tick logs the offset state
  std::lock_guard<std::mutex> lock(_data->residual_command.mutex);
  _data->residual_command.active = true;
}

void FSMState_Residual::run()
{
  const auto start = std::chrono::steady_clock::now();
  // Both commands under lock, read in the same instant. Never block the control loop: if a
  // callback holds either one, keep the previous low_cmd for this tick (std::try_lock
  // releases the one it got).
  auto & res = _data->residual_command;
  std::unique_lock<std::mutex> mpc_lock(_data->mpc_command.mutex, std::defer_lock);
  std::unique_lock<std::mutex> res_lock(res.mutex, std::defer_lock);
  if (std::try_lock(mpc_lock, res_lock) != -1) return;
  // MPX stale (damping) or not valid yet: no network on top.
  if (!mpc_->compute(start)) return;
  auto & tau = _data->low_cmd->tau_cmd;
  const size_t n = static_cast<size_t>(tau.size());
  // Latest offset from residual_node, 0 if older than residual.command_timeout.
  const bool fresh = res.tau.size() == n &&
                     std::chrono::duration<double>(start - res.received).count() <=
                       _data->params->residual_params.command_timeout;
  if (fresh == offset_stale_) {
    std::cerr << (fresh ? "[FSMState_Residual] residual offset received: added to tau"
                        : "[FSMState_Residual] no fresh residual offset: offset = 0")
              << std::endl;
    offset_stale_ = !fresh;
  }
  if (fresh) offset_ = res.tau;
  else offset_.assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const auto limit = _data->params->torque_limit[i];
    tau(i) = std::clamp(tau(i) + offset_[i], -limit, limit);
  }
}

void FSMState_Residual::exit()
{
  std::lock_guard<std::mutex> lock(_data->residual_command.mutex);
  _data->residual_command.active = false;
}

std::string FSMState_Residual::checkTransition()
{
  // Transitions of mpc: residual kept, mpc at once, rl_flat and transform_down after the
  // MPX handoff, fold when MPX stops, idle/joint_pd at once. While waiting for the handoff
  // mpc answers "mpc": stay in residual instead (only the key mpc leaves to mpc).
  const std::string next = mpc_->checkTransition();
  if (next == "mpc" && _data->rc_data->fsm_name_ != "mpc") return "residual";
  return next;
}
