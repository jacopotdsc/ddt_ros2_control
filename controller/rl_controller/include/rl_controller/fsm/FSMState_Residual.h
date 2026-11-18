#ifndef RL_CONTROLLER__FSM__FSMSTATE_RESIDUAL_H_
#define RL_CONTROLLER__FSM__FSMSTATE_RESIDUAL_H_

#include <memory>
#include <string>
#include <vector>

#include "FSMState.h"
#include "FSMState_MPC.h"

// Residual learning on top of MPX. The network runs in its own node (residual_node) and
// publishes a torque offset on residual/joint_command. The sum is done here: with both command
// locks (MPX and residual, std::try_lock; tick skipped if either is busy) FSMState_MPC::compute()
// writes the clamped tau_ff + PD in low_cmd; if MPX is not stale, this state adds the latest fresh
// offset and clamps again:
//   tau = clamp(clamp(tau_ff + kp (q_des - q) + kd (qd_des - dq)) + offset)
// Transitions are those of mpc (same MPX, same handoff and fold).
class FSMState_Residual : public FSMState
{
public:
  FSMState_Residual(std::shared_ptr<ControlFSMData> data, FSMState_MPC * mpc);
  void enter() override;
  void run() override;
  void exit() override;
  std::string checkTransition() override;

private:
  FSMState_MPC * mpc_;  // MPX command + PD (owned by FSM)
  std::vector<double> offset_;  // last fresh offset, 0 when missing or stale
  bool offset_stale_ = false;   // for the log on change only
};

#endif  // RL_CONTROLLER__FSM__FSMSTATE_RESIDUAL_H_
