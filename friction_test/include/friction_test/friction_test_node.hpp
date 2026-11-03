// friction_test_node: keyboard + joint friction test trajectories on the TITA hung in Gazebo
// (friction_estimate.launch.py). Same structure as keyboard_controller (live keys, interface,
// command/cmd_key at 100 Hz following the FSM state), plus the trajectories:
//   0  idle      FSM idle (safety), always available
//   1  joint_pd  FSM joint_cmd + ramp to the default pose, then hold it; always available
//   2-9          tests, one joint each (generate_joint_trajectory: smooth back-and-forth strokes for the legs,
//                speed steps for the wheels), only while holding the default pose
//   t            every test in sequence, 3 s of joint_pd between two tests
// Commands go to tita_rl_controller, FSM state joint_cmd, on command/joint_cmd (per joint
// [q_des, dq_des, tau_ff, kp, kd]). Do not run keyboard_controller at the same time: both publish
// command/cmd_key.
#ifndef FRICTION_TEST__FRICTION_TEST_NODE_HPP_
#define FRICTION_TEST__FRICTION_TEST_NODE_HPP_

#include <termios.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "urdf/model.h"
#include "ros_utils/topic_names.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"

using namespace std::chrono_literals;

// Sampled trajectory of one joint (DT spacing) and the rows logged while it runs.
struct JointTrajectory
{
  static constexpr double DT = 0.001;  // s
  struct Row
  {
    double t, q_des, dq_des, ddq_des, q, dq, tau, kp, kd;
    int level;
    uint8_t segment;
    std::vector<double> all_q, all_dq, all_tau;  // state of every joint (joint_states), joints_ order
  };
  std::string joint;
  std::vector<double> q, dq, ddq;
  std::vector<int> level;
  std::vector<uint8_t> segment;  // index in SEGMENTS
  std::vector<Row> rows;
  double duration() const { return (q.size() - 1) * DT; }
  size_t index(double t) const;
  std::string write_csv(
    const std::string & dir, const std::string & suffix, const std::vector<std::string> & joints) const;
};

class FrictionTestNode : public rclcpp::Node
{
public:
  explicit FrictionTestNode(const rclcpp::NodeOptions & options);

  // Trajectory of joint_name from its held position and back, for friction identification.
  JointTrajectory generate_joint_trajectory(const std::string & joint_name, double q_lim_low, double q_lim_high);

private:
  enum class Mode { ZERO, WAIT_JOINT_CMD, RAMP, HOLD, TEST };

  void print_interface();
  int get_key();
  void ReadKeyThread();
  void PubFsmGoalCallBack();  // command/cmd_key at 100 Hz, as keyboard_controller
  void ControlCallBack();     // command/joint_cmd at command_rate, simulated clock
  void on_joint_states(const sensor_msgs::msg::JointState & msg);
  void on_fsm_state(const std::string & state);
  void on_robot_description(const std::string & xml);

  // Called with mutex_ held.
  void note(const std::string & text, bool warn = false);
  void zero_torque(const std::string & reason);
  void start_joint_pd(const std::string & reason);
  void abort_test();
  bool start_test(char k);
  std::string sequence_status();
  void publish_zero();
  void set_command(size_t i, double q_des, double dq_des, double ddq_des, int level, uint8_t segment);
  double now_s() { return this->get_clock()->now().seconds(); }

  // parameters
  std::vector<std::string> joints_;
  std::vector<double> kp_, kd_, default_pose_;  // default_pose_: per leg [abduction, hip, knee, wheel]
  double command_rate_, home_ramp_time_;
  std::string csv_dir_;
  std::vector<double> speed_fractions_;
  int sweeps_per_level_;
  std::vector<double> acc_fractions_;  // legs: levels of peak acceleration, fractions of acc_max_
  // test range per joint (yaml), inside the URDF limits: legs positions [rad], wheels speeds [rad/s]
  std::vector<double> lower_bound_, upper_bound_;
  double wheel_hold_time_, pause_time_;
  std::map<char, std::vector<std::string>> tests_;        // key -> joints moved together
  std::map<std::string, double> acc_max_;  // joint -> peak acceleration (legs: strokes, wheels: speed ramps)
  std::string trajectory_info_;                             // levels of the last generated trajectory
  std::map<std::string, std::array<double, 2>> joint_limits_;  // joint -> URDF [lower, upper]
  std::map<std::string, double> joint_velocity_limits_;        // joint -> URDF velocity limit

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fsm_goal_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsm_state_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_subscription_;
  rclcpp::TimerBase::SharedPtr fsm_goal_timer_, control_timer_;

  std::mutex mutex_;
  std::string fsm_goal_{"idle"};  // published on command/cmd_key
  std::string fsm_state_;         // seen on fsm_state
  Mode mode_{Mode::ZERO};
  std::vector<double> q_, dq_, effort_;  // latest joint_states, joints_ order
  bool have_state_{false};
  std::vector<double> hold_;
  double ramp_t0_{0.};
  std::vector<double> ramp_start_, ramp_target_;
  char test_key_{0};
  double test_t0_{0.};
  std::map<size_t, JointTrajectory> test_;  // joint index -> trajectory
  // Completed tests own their data while a worker writes them; control never waits for disk I/O.
  std::deque<std::future<std::vector<std::string>>> csv_jobs_;
  // per joint: command of this tick [q_des, dq_des, tau_ff, kp, kd] and its trajectory info
  std::vector<std::array<double, 5>> cmd_;
  std::vector<std::tuple<double, int, uint8_t>> cmd_info_;  // ddq_des, speed level, segment
  std::string message_;  // last message, shown under the state lines (red if warn)
  bool message_warn_{false};
  double last_print_{0.};
  // key t: every test in sequence (keys of tests_ in order), SEQUENCE_PAUSE s holding the default pose
  // between two tests; stopped by 0, 1, Ctrl+C or the FSM leaving joint_cmd.
  bool sequence_{false};
  size_t sequence_index_{0};
  double sequence_next_{-1.};

  // {key, FSM state requested on command/cmd_key, name}
  static const std::vector<std::tuple<char, std::string, std::string>> key_mapping;
};

#endif  // FRICTION_TEST__FRICTION_TEST_NODE_HPP_
