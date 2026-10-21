// Low-level joint tracker. It consumes the latest WBC setpoint and joint feedback,
// tracks the setpoint at the joint-state rate, and publishes the final torque vector
// for rl_controller to write through ros2_control's hardware interfaces.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpx/default_params.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

class LowLevelController final : public rclcpp::Node {
  using JointState = sensor_msgs::msg::JointState;
  using Steady = std::chrono::steady_clock;

 public:
  explicit LowLevelController(const rclcpp::NodeOptions &options)
      : Node("llc", options) {
    joint_states_topic_ = get_parameter("inputs.joint_states_topic").as_string();
    wbc_topic_ = get_parameter("outputs.joint_command_topic").as_string();
    fsm_topic_ = get_parameter("inputs.fsm_state_topic").as_string();
    output_topic_ = get_parameter("outputs.llc_command_topic").as_string();
    names_ = get_parameter("llc.joint_names").as_string_array();
    kp_ = get_parameter("llc.joint_kp").as_double_array();
    kd_ = get_parameter("llc.joint_kd").as_double_array();
    max_torque_ = get_parameter("llc.max_torque").as_double_array();
    frequency_ = get_parameter("timing.llc_frequency").as_double();
    wbc_timeout_ns_ = get_parameter("timing.wbc_command_timeout").as_double() * 1e9;
    if (names_.empty() || kp_.size() != names_.size() || kd_.size() != names_.size() ||
        max_torque_.size() != names_.size() || !(frequency_ > 0.) || !(wbc_timeout_ns_ > 0.))
      throw std::invalid_argument("Invalid LLC frequency, WBC timeout, joint names, or gains");

    auto qos = rclcpp::QoS(1).best_effort();
    output_ = create_publisher<std_msgs::msg::Float64MultiArray>(output_topic_, qos);
    wbc_sub_ = create_subscription<JointState>(wbc_topic_, qos,
        [this](JointState::ConstSharedPtr msg) {
          // The callback only replaces the cached reference. The joint-rate loop
          // keeps tracking that latest reference until another one arrives.
          if (active_ && valid_wbc_message(*msg)) {
            wbc_ = *msg;
          }
        });
    fsm_sub_ = create_subscription<std_msgs::msg::String>(fsm_topic_,
        rclcpp::QoS(1).reliable().transient_local(), [this](std_msgs::msg::String::ConstSharedPtr msg) {
          active_ = msg->data == "mpc";
          // Do not carry a reference across FSM modes. MPC must publish a fresh
          // WBC target after each activation before the LLC resumes output.
          wbc_ = JointState{};
          stale_ = false;
          if (active_) {
            report_stamp_ = 0;
            count_ = 0;
            report_ = Steady::now();
          }
        });
    joints_sub_ = create_subscription<JointState>(joint_states_topic_, qos,
        [this](JointState::ConstSharedPtr msg) { on_joints(*msg); });
    RCLCPP_INFO(get_logger(), "LLC ready: target %.1f Hz from joint feedback; tracking WBC command %s",
                frequency_, wbc_topic_.c_str());
  }

 private:
  static int64_t ns(const builtin_interfaces::msg::Time &t) {
    return int64_t(t.sec) * 1000000000LL + t.nanosec;
  }

  bool valid_wbc_message(const JointState &msg) const {
    if (msg.position.size() != msg.name.size() || msg.velocity.size() != msg.name.size() ||
        msg.effort.size() != msg.name.size() ||
        (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0)) return false;
    for (const auto &name : names_) {
      const auto it = std::find(msg.name.begin(), msg.name.end(), name);
      if (it == msg.name.end()) return false;
      const auto i = size_t(std::distance(msg.name.begin(), it));
      if (!std::isfinite(msg.position[i]) || !std::isfinite(msg.velocity[i]) ||
          !std::isfinite(msg.effort[i])) return false;
    }
    return true;
  }

  void on_joints(const JointState &joints) {
    const auto stamp = ns(joints.header.stamp);
    if (!active_ || stamp <= last_stamp_) return;
    if (report_stamp_ == 0) report_stamp_ = stamp;
    const double period_ns = 1e9 / frequency_;
    if (last_stamp_ >= 0 && stamp - last_stamp_ + 1000 < period_ns) return;
    last_stamp_ = stamp;
    if (wbc_.name.empty() ||
        (wbc_.header.stamp.sec == 0 && wbc_.header.stamp.nanosec == 0)) return;
    // WBC stopped or dead: stop publishing instead of tracking a frozen target, so
    // rl_controller's mpc_command_timeout folds the robot.
    const bool stale = stamp - ns(wbc_.header.stamp) > wbc_timeout_ns_;
    if (stale != stale_) {
      stale_ = stale;
      if (stale) RCLCPP_ERROR(get_logger(), "No WBC command for %.2f s: LLC output stopped",
                              (stamp - ns(wbc_.header.stamp)) / 1e9);
      else RCLCPP_WARN(get_logger(), "WBC commands are back: LLC output resumed");
    }
    if (stale_) return;

    std_msgs::msg::Float64MultiArray out;
    out.data.resize(names_.size());
    for (size_t i = 0; i < names_.size(); ++i) {
      auto ji = std::find(joints.name.begin(), joints.name.end(), names_[i]);
      auto wi = std::find(wbc_.name.begin(), wbc_.name.end(), names_[i]);
      if (ji == joints.name.end() || wi == wbc_.name.end()) return;
      const auto j = size_t(std::distance(joints.name.begin(), ji));
      const auto w = size_t(std::distance(wbc_.name.begin(), wi));
      if (j >= joints.position.size() || j >= joints.velocity.size() ||
          w >= wbc_.position.size() || w >= wbc_.velocity.size() || w >= wbc_.effort.size()) return;
      const double tau = wbc_.effort[w] + kp_[i] * (wbc_.position[w] - joints.position[j]) +
                         kd_[i] * (wbc_.velocity[w] - joints.velocity[j]);
      if (!std::isfinite(tau)) return;
      out.data[i] = std::clamp(tau, -max_torque_[i], max_torque_[i]);
    }
    output_->publish(out);
    ++count_;
    if (Steady::now() - report_ >= std::chrono::seconds(5)) {
      const double elapsed_sim = (stamp - report_stamp_) * 1e-9;
      RCLCPP_INFO(get_logger(), "LLC: %.1f Hz sim, WBC command age %.1f ms",
                  elapsed_sim > 0. ? count_ / elapsed_sim : 0.,
                  (stamp - ns(wbc_.header.stamp)) / 1e6);
      count_ = 0;
      report_stamp_ = stamp;
      report_ = Steady::now();
    }
  }

  std::string joint_states_topic_, wbc_topic_, fsm_topic_, output_topic_;
  std::vector<std::string> names_;
  std::vector<double> kp_, kd_, max_torque_;
  double frequency_{0.}, wbc_timeout_ns_{0.};
  bool active_{false}, stale_{false};
  JointState wbc_;
  Steady::time_point report_{Steady::now()};
  int64_t last_stamp_{-1}, report_stamp_{0};
  size_t count_{0};
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr output_;
  rclcpp::Subscription<JointState>::SharedPtr wbc_sub_, joints_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsm_sub_;
};

int main(int argc, char **argv) {
  mpx::DefaultArguments args(argc, argv);
  rclcpp::init(args.argc(), args.argv());
  auto options = rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<LowLevelController>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
