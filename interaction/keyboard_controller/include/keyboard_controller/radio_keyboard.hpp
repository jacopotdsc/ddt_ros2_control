// Keyboard radio for the simulation: a bridge to the real teleop_command code.
//
// radio_keyboard opens a pseudo-terminal, starts teleop_command::TeleopCmdNode in the same process
// (teleop_command/config/param.yaml, uart_interface = the pty) and writes to the pty the CRSF RC
// frames an ELRS receiver would send, with the channels set from the keyboard. Everything after
// the channels (CRSF parser, radio FSM, arming, limits, cmd_key / cmd_twist / cmd_pose) is the
// teleop_command code itself: a change there applies here at the next build.
//
// Live screen: keys, channel positions, FSM state and the last log lines of the process (ROS
// logs of teleop_command and radio_keyboard, captured by an rcutils output handler).
#ifndef KEYBOARD_CONTROLLER__RADIO_KEYBOARD_HPP_
#define KEYBOARD_CONTROLLER__RADIO_KEYBOARD_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "gazebo_msgs/msg/model_states.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "urdf/model.h"
#include "std_msgs/msg/string.hpp"

namespace teleop_command
{
class TeleopCmdNode;
}

namespace keyboard_controller
{
class RadioKeyboard : public rclcpp::Node
{
public:
  explicit RadioKeyboard(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~RadioKeyboard() override;

  // Slave side of the pty: the uart_interface of the teleop_command node.
  const std::string & device() const { return device_; }
  // Write the frames, read the keyboard and draw the screen (after teleop_command opened the pty).
  void start();
  // Stop the threads and restore the terminal (before the teleop_command node goes away).
  void stop();
  // A log line of this process (rcutils output handler): kept among the last kLogLines.
  void add_log(const std::string & time, const std::string & text);
  // The teleop_command node of this process: armed state (red channels when not armed).
  void set_teleop(const teleop_command::TeleopCmdNode * teleop) { teleop_ = teleop; }

  static constexpr size_t kLogLines = 5;

private:
  void write_loop();
  void key_loop();
  void draw_loop();
  bool handle_key(int key);  // false = quit
  void draw();
  // Stick position in [-1, 1] -> channel value (teleop_command normalizes it back).
  static uint16_t stick(double position);
  // Ground-truth COM height (world z): Gazebo base pose + joint_states + URDF masses (all links,
  // base_link included). NaN without fresh data (mutex_ held).
  double ground_truth_com_z() const;

  int master_ = -1, slave_ = -1;
  std::string device_;
  std::mutex mutex_;  // channels, sticks, FSM state
  std::array<uint16_t, 16> channels_{};
  double forward_ = 0., turn_ = 0.;  // left stick, [-1, 1]
  std::chrono::steady_clock::time_point right_stick_until_{};  // right stick pushed (safety)
  std::chrono::steady_clock::time_point jump_until_{};         // JUMP pulled (safety)
  std::string fsm_state_;
  const teleop_command::TeleopCmdNode * teleop_ = nullptr;
  std::vector<float> axes_;  // joy.axes as teleop_command reads the channels
  double height_command_ = std::nan("");  // cmd_pose z (COM height command), m
  double com_height_ = std::nan("");      // MPX COM estimate z (ground truth state in Gazebo), m
  std::chrono::steady_clock::time_point com_received_{};
  std::shared_ptr<urdf::Model> urdf_;                     // robot_description
  std::unordered_map<std::string, double> joint_positions_;  // joint_states
  Eigen::Isometry3d base_pose_ = Eigen::Isometry3d::Identity();  // Gazebo model pose
  std::chrono::steady_clock::time_point joints_received_{}, base_received_{};
  std::string gazebo_model_;  // entity name in ModelStates (parameter gazebo_model)
  struct LogLine
  {
    std::string time, text;
    int count;  // identical consecutive lines are counted, not repeated
  };
  std::mutex log_mutex_;
  std::deque<LogLine> logs_;
  std::atomic<bool> stop_{false}, dirty_{true};
  std::thread writer_, reader_, drawer_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsm_state_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr com_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr description_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_subscription_;
  rclcpp::Subscription<gazebo_msgs::msg::ModelStates>::SharedPtr model_subscription_;
  rclcpp::TimerBase::SharedPtr com_timer_;
};
}  // namespace keyboard_controller

#endif  // KEYBOARD_CONTROLLER__RADIO_KEYBOARD_HPP_
