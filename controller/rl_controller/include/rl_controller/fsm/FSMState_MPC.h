#ifndef RL_CONTROLLER__FSM__FSMSTATE_MPC_H_
#define RL_CONTROLLER__FSM__FSMSTATE_MPC_H_
#include "FSMState.h"

class FSMState_MPC : public FSMState
{
public:
  explicit FSMState_MPC(std::shared_ptr<ControlFSMData> data) : FSMState(data, "mpc") {}
  void enter() override
  {
    stale_ = false;
    entered_ = std::chrono::steady_clock::now();
  }
  void run() override;
  void exit() override {}
  std::string checkTransition() override;
  static bool ready(const std::shared_ptr<ControlFSMData> & data)
  {
    std::lock_guard<std::mutex> lock(data->mpc_command.mutex);
    return data->mpc_command.valid;
  }

private:
  bool stale_ = false;  // no MPX torque for params->mpc_command_timeout
  std::chrono::steady_clock::time_point entered_{};
};
#endif
