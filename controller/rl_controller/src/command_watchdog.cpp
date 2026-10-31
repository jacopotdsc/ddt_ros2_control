// Safety watchdog. Runs on the robot next to rl_controller and requests an FSM mode on its
// own topic, command/watchdog_mode. check() is a list of cases, one function each:
//
//   yaw_rate_too_high()         |wz| of the body IMU > max_yaw_rate             -> idle (key 6)
//   roll_rate_too_high()        |wx| of the body IMU > max_roll_rate            -> idle
//   pitch_rate_too_high()       |wy| of the body IMU > max_pitch_rate           -> idle
//   roll_too_high()             |roll| from the IMU orientation > max_roll_deg   -> idle
//   tilt_too_high()             base tilt from the IMU orientation > max_tilt_deg -> idle
//   linear_speed_too_high()     |vx| of the filter (base_link), mean over speed_window s
//                               > max_linear_speed                             -> idle
//   torque_too_high()           a joint |effort| > max_torque[i] for torque_hold s in a row
//                                                                              -> idle
//   imu_lost()                  no IMU data for `imu_timeout` s (armed by the first one)
//                                                                              -> idle
//   command_lost()              no twist for `timeout` s (WiFi link to the keyboard PC
//                               lost; armed by the first twist)               -> transform_down
//
// When a case becomes true the mode (e.g. "transform_down") is published once on
// command/watchdog_mode; the reason is printed only in this node's log. rl_controller
// applies it like a key, MPX starts the handoff from it (also without the keyboard) and the
// keyboard takes it as its own mode, then keeps control. The topic is latched, so a keyboard
// reached late (WiFi) still gets it. The mode is forced once: afterwards the keys act again.
// A case still true does not ask at every check. Once the FSM is in idle (both modes end
// there) "" is published: the request is done, and nobody starting later takes an old one.
// The cases are checked only in the FSM states of active_states (watchdog.yaml: mpc and the
// RL policies), never in idle, transform_up (stand-up), transform_down or joint_pd. Entering
// an active state restarts the torque timers and the speed mean, so the stand-up does not
// count. When the FSM leaves idle (e.g. key 7) the watchdog is armed again: a case still true
// (e.g. IMU still lost) asks again once an active state is reached.
// transform_down, like key 8: from RL the FSM folds at once, from MPC after MPX has lowered
// the COM to h_rl, then idle.
// The rate peaks are reset at every check, so without IMU data they stay 0; roll and tilt keep
// their last value, so roll_too_high() and tilt_too_high() need fresh data (else an old value
// would request idle). Speed and torque have no freshness check: lost filter or joint data
// already raise errors elsewhere.
// Joint arrays follow joint_names (watchdog.yaml), matched by name in joint_states.
// Parameters: config/watchdog.yaml only (no defaults here: a missing one stops the node).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ros_utils/topic_names.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class CommandWatchdog : public rclcpp::Node
{
public:
  CommandWatchdog() : Node("command_watchdog")
  {
    timeout_ = std::chrono::duration<double>(declare_parameter<double>("timeout"));
    max_yaw_rate_ = declare_parameter<double>("max_yaw_rate");                // rad/s
    max_roll_rate_ = declare_parameter<double>("max_roll_rate");              // rad/s
    max_pitch_rate_ = declare_parameter<double>("max_pitch_rate");            // rad/s
    max_roll_ = declare_parameter<double>("max_roll_deg") * M_PI / 180.0;    // rad
    max_tilt_ = declare_parameter<double>("max_tilt_deg") * M_PI / 180.0;    // rad
    imu_timeout_ = std::chrono::duration<double>(declare_parameter<double>("imu_timeout"));
    max_linear_speed_ = declare_parameter<double>("max_linear_speed");  // m/s
    speed_window_ = std::chrono::duration<double>(declare_parameter<double>("speed_window"));
    torque_hold_ = std::chrono::duration<double>(declare_parameter<double>("torque_hold"));
    joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");
    max_torque_ = joint_parameter("max_torque");  // Nm
    active_states_ = declare_parameter<std::vector<std::string>>("active_states");
    const size_t n = joint_names_.size();
    torque_above_since_.assign(n, std::nullopt);
    twist_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      ros_topic::manager_twist_command, rclcpp::SystemDefaultsQoS(),
      [this](geometry_msgs::msg::Twist::SharedPtr) { last_command_ = Clock::now(); });
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      ros_topic::body_imu, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Imu::SharedPtr m) {
        // Peaks since the last check: the 100 ms timer does not miss a short spike.
        const auto & w = m->angular_velocity;
        const auto & q = m->orientation;
        yaw_rate_peak_ = std::max(yaw_rate_peak_, std::abs(w.z));
        roll_rate_peak_ = std::max(roll_rate_peak_, std::abs(w.x));
        pitch_rate_peak_ = std::max(pitch_rate_peak_, std::abs(w.y));
        roll_ = std::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));
        tilt_ = std::acos(std::clamp(1.0 - 2.0 * (q.x * q.x + q.y * q.y), -1.0, 1.0));
        last_imu_ = Clock::now();
      });
    joint_state_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      ros_topic::joint_states, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::JointState::SharedPtr m) { on_joint_state(*m); });
    filtered_state_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      ros_topic::filtered_state, rclcpp::SensorDataQoS(),
      [this](nav_msgs::msg::Odometry::SharedPtr m) {
        // twist in base_link: linear.x is the forward speed.
        const auto now = Clock::now();
        speed_samples_.emplace_back(now, m->twist.twist.linear.x);
        while (now - speed_samples_.front().first > speed_window_) speed_samples_.pop_front();
      });
    fsm_state_subscription_ = create_subscription<std_msgs::msg::String>(
      ros_topic::fsm_state, rclcpp::QoS(1).reliable().transient_local(),
      [this](std_msgs::msg::String::SharedPtr m) {
        if (fsm_state_ == "idle" && m->data != "idle") requested_ = false;  // out of idle: armed
        const bool was_active = active_;
        fsm_state_ = m->data;
        active_ = std::find(active_states_.begin(), active_states_.end(), fsm_state_) !=
                  active_states_.end();
        if (active_ && !was_active) {  // e.g. end of the stand-up: start counting from here
          torque_above_since_.assign(torque_above_since_.size(), std::nullopt);
          speed_samples_.clear();
        }
      });
    watchdog_mode_publisher_ = create_publisher<std_msgs::msg::String>(
      ros_topic::watchdog_mode, rclcpp::QoS(1).reliable().transient_local());
    timer_ = create_wall_timer(100ms, [this]() { check(); });
    RCLCPP_INFO(
      get_logger(),
      "idle if |wz| > %.2f, |wx| > %.2f, |wy| > %.2f rad/s, |roll| > %.1f or tilt > %.1f deg "
      "(%s); transform_down after %.2f s without commands on %s (armed by the first one).",
      max_yaw_rate_, max_roll_rate_, max_pitch_rate_, max_roll_ * 180.0 / M_PI,
      max_tilt_ * 180.0 / M_PI,
      imu_subscription_->get_topic_name(), timeout_.count(),
      twist_subscription_->get_topic_name());
    RCLCPP_INFO(
      get_logger(),
      "idle if |vx| > %.2f m/s (%.2f s mean, %s), a joint |effort| > max_torque for %.2f s "
      "(%s, %zu joints).",
      max_linear_speed_, speed_window_.count(), filtered_state_subscription_->get_topic_name(),
      torque_hold_.count(), joint_state_subscription_->get_topic_name(), n);
  }

private:
  std::vector<double> joint_parameter(const std::string & name)
  {
    auto values = declare_parameter<std::vector<double>>(name);
    if (values.size() != joint_names_.size()) {
      throw std::runtime_error(name + ": one value per joint_names entry");
    }
    return values;
  }

  void on_joint_state(const sensor_msgs::msg::JointState & m)
  {
    if (joint_index_.size() != joint_names_.size() || m.name.size() != joint_state_size_) {
      joint_index_.clear();
      for (const auto & name : joint_names_) {
        const auto it = std::find(m.name.begin(), m.name.end(), name);
        if (it == m.name.end()) return;  // joint not in this message: wait for a full one
        joint_index_.push_back(it - m.name.begin());
      }
      joint_state_size_ = m.name.size();
    }
    const auto now = Clock::now();
    for (size_t i = 0; i < joint_index_.size(); ++i) {
      const size_t k = joint_index_[i];
      if (k >= m.effort.size()) continue;
      if (std::abs(m.effort[k]) <= max_torque_[i]) {
        torque_above_since_[i].reset();
      } else if (!torque_above_since_[i]) {
        torque_above_since_[i] = now;
      }
    }
  }

  void check()
  {
    // Most severe first.
    if (!active_) {
      // Not in mpc or RL: no case is checked (stand-up, fold, idle, joint_pd).
    } else if (yaw_rate_too_high()) {
      request("idle", "|wz| %.2f rad/s", yaw_rate_peak_);
    } else if (roll_rate_too_high()) {
      request("idle", "|wx| (roll rate) %.2f rad/s", roll_rate_peak_);
    } else if (pitch_rate_too_high()) {
      request("idle", "|wy| (pitch rate) %.2f rad/s", pitch_rate_peak_);
    } else if (roll_too_high()) {
      request("idle", "roll %.1f deg", std::abs(roll_) * 180.0 / M_PI);
    } else if (tilt_too_high()) {
      request("idle", "tilt %.1f deg", tilt_ * 180.0 / M_PI);
    } else if (linear_speed_too_high()) {
      request("idle", "|vx| %.2f m/s (" + fmt("%.2f", speed_window_.count()) + " s mean)",
              std::abs(mean_speed()));
    } else if (torque_too_high()) {
      const int i = torque_joint();
      request("idle", joint_names_[i] + " |effort| above " + fmt("%.1f", max_torque_[i]) +
              " Nm for %.2f s", std::chrono::duration<double>(Clock::now() - *torque_above_since_[i]).count());
    } else if (imu_lost()) {
      request("idle", "no IMU data for %.2f s", imu_timeout_.count());
    } else if (command_lost()) {
      request("transform_down", "no command for %.2f s (connection lost?)", timeout_.count());
    } else {
      requested_ = false;  // no case true: armed again
    }
    yaw_rate_peak_ = roll_rate_peak_ = pitch_rate_peak_ = 0.0;
    if (pending_ && fsm_state_ == "idle") {
      pending_ = false;
      publish("");  // request done
    }
  }

  // Cases: true while the condition holds.
  bool imu_fresh() const { return last_imu_ && Clock::now() - *last_imu_ < imu_timeout_; }
  bool yaw_rate_too_high() const { return yaw_rate_peak_ > max_yaw_rate_; }
  bool roll_rate_too_high() const { return roll_rate_peak_ > max_roll_rate_; }
  bool pitch_rate_too_high() const { return pitch_rate_peak_ > max_pitch_rate_; }
  bool roll_too_high() const { return imu_fresh() && std::abs(roll_) > max_roll_; }
  bool tilt_too_high() const { return imu_fresh() && std::abs(tilt_) > max_tilt_; }
  double mean_speed() const
  {
    double sum = 0.0;
    for (const auto & sample : speed_samples_) sum += sample.second;
    return speed_samples_.empty() ? 0.0 : sum / speed_samples_.size();
  }
  bool linear_speed_too_high() const
  {
    return std::abs(mean_speed()) > max_linear_speed_;
  }
  // Index of the first joint above max_torque for torque_hold, -1 if none.
  int torque_joint() const
  {
    const auto now = Clock::now();
    for (size_t i = 0; i < torque_above_since_.size(); ++i) {
      if (torque_above_since_[i] && now - *torque_above_since_[i] >= torque_hold_) return i;
    }
    return -1;
  }
  bool torque_too_high() const { return torque_joint() >= 0; }
  bool imu_lost() const { return last_imu_ && !imu_fresh(); }
  bool command_lost() const { return last_command_ && Clock::now() - *last_command_ >= timeout_; }

  static std::string fmt(const char * format, double value)
  {
    char text[32];
    std::snprintf(text, sizeof(text), format, value);
    return text;
  }

  // Action: request mode once, while a case stays true. reason holds one %-format for value.
  void request(const std::string & mode, const std::string & reason, double value)
  {
    if (requested_) return;
    requested_ = true;
    char text[160];
    std::snprintf(text, sizeof(text), reason.c_str(), value);
    RCLCPP_ERROR(get_logger(), "%s: requesting %s.", text, mode.c_str());
    pending_ = true;
    publish(mode);
  }

  void publish(const std::string & request)
  {
    std_msgs::msg::String message;
    message.data = request;
    watchdog_mode_publisher_->publish(message);
  }

  std::chrono::duration<double> timeout_, imu_timeout_, speed_window_, torque_hold_;
  double max_yaw_rate_, max_roll_rate_, max_pitch_rate_, max_roll_, max_tilt_;
  std::optional<Clock::time_point> last_command_, last_imu_;
  double yaw_rate_peak_ = 0.0, roll_rate_peak_ = 0.0, pitch_rate_peak_ = 0.0;
  double roll_ = 0.0, tilt_ = 0.0;
  double max_linear_speed_;
  std::deque<std::pair<Clock::time_point, double>> speed_samples_;  // filter vx in the window
  std::vector<std::string> joint_names_;
  std::vector<double> max_torque_;
  std::vector<size_t> joint_index_;  // joint_names_ -> index in joint_states
  size_t joint_state_size_ = 0;
  std::vector<std::optional<Clock::time_point>> torque_above_since_;
  bool requested_ = false;
  bool pending_ = false;  // request published, FSM not yet in idle
  std::string fsm_state_;
  std::vector<std::string> active_states_;  // FSM states where the cases are checked
  bool active_ = false;  // fsm_state_ is in active_states_
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr filtered_state_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsm_state_subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr watchdog_mode_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CommandWatchdog>());
  rclcpp::shutdown();
  return 0;
}
