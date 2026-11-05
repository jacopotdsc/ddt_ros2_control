#ifndef RL_CONTROLLER__FSM__FSMSTATE_RESIDUAL_H_
#define RL_CONTROLLER__FSM__FSMSTATE_RESIDUAL_H_
#include <vector>

#include "FSMState.h"

// Residual learning on top of MPX: the MPX command (mpx/joint_command from the WBC:
// tau_ff, q_des, qd_des, in _data->mpc_command) plus an offset delta on tau_ff from a
// neural network. WBC and network write separate buffers at their own rates; run()
// combines the latest of each at the FSM rate:
//   tau = (tau_ff + delta) + kp (q_des - q) + kd (qd_des - dq)
// Entered only from mpc. Skeleton: the network is not there yet, delta = 0, so the
// state behaves exactly like mpc.
class FSMState_Residual : public FSMState
{
public:
  explicit FSMState_Residual(std::shared_ptr<ControlFSMData> data);
  void enter() override;
  void run() override;
  void exit() override;
  std::string checkTransition() override;

private:
  // Network side (dummy, to be filled): updates delta_tau_ at the network rate.
  void update_delta();
  // Latest network offset on tau_ff, one per joint (dummy: zeros).
  std::vector<double> delta() const;

  bool stale_ = false;              // no MPX command for params->mpc_params.command_timeout
  std::vector<double> delta_tau_;   // latest network offset on tau_ff
};
#endif
