#ifndef RL_CONTROLLER__FSM__FSMSTATE_JOINTCMD_H_
#define RL_CONTROLLER__FSM__FSMSTATE_JOINTCMD_H_
#include "FSMState.h"

// joint_cmd: applies command/joint_cmd as it is (per joint q_des, dq_des, tau_ff, kp, kd) to the
// low-level command; the hardware/Gazebo bridge computes tau_ff + kp (q_des - q) + kd (dq_des - dq).
// No command yet, or none for joint_cmd.command_timeout: zero torque.
class FSMState_JointCmd : public FSMState
{
public:
  explicit FSMState_JointCmd(std::shared_ptr<ControlFSMData> data) : FSMState(data, "joint_cmd") {}
  void enter() override;
  void run() override;
  void exit() override {}
  std::string checkTransition() override;

private:
  bool stale_ = true;
};
#endif
