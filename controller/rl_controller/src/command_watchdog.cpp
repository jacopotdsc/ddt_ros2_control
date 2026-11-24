// Safety watchdog. Runs on the robot next to rl_controller and requests an FSM mode on its
// own topic, command/watchdog_mode. Each FSM state listed in `states` (watchdog.yaml) has its own
// block: `<state>.checks` says which cases are checked in that state, and the block holds the
// thresholds of those cases. States without a block (idle, joint_pd, ...) check nothing.
//
//   case (name in checks)  condition                                                  request
//   yaw_rate               |wz| of the body IMU > max_yaw_rate                        idle (key 6)
//   roll_rate              |wx| of the body IMU > max_roll_rate                       idle
//   pitch_rate             |wy| of the body IMU > max_pitch_rate                      idle
//   roll                   |roll| from the IMU orientation > max_roll_deg             idle
//   tilt                   base tilt from the IMU orientation > max_tilt_deg          idle
//   linear_speed           |vx| of the filter (base_link), mean over speed_window
//                          > max_linear_speed                                         idle
//   torque                 a joint |effort| > max_torque[i] for torque_hold s in a row idle
//   imu_lost               no IMU data for imu_timeout s (armed by the first one)     idle
//   command_lost           no twist for `timeout` s (WiFi link to the keyboard PC
//                          lost; armed by the first twist)                            transform_down
//
// Parameters of a case are read only for the states that list it, as <state>.<parameter>
// (e.g. transform_up.max_torque). Shared by all states: joint_names, imu_timeout (also the age
// above which roll and tilt are not trusted), speed_window.
// When a case becomes true the mode is published once on command/watchdog_mode; the reason is
// printed only in this node's log. rl_controller applies it like a key (idle and joint_pd are
// immediate from every state), MPX starts the handoff from it (also without the keyboard) and
// the keyboard takes it as its own mode, then keeps control. The topic is latched, so a keyboard
// reached late (WiFi) still gets it. The mode is forced once: afterwards the keys act again.
// A case still true does not ask at every check. Once the FSM is in idle (both modes end
// there) "" is published: the request is done, and nobody starting later takes an old one.
// Every FSM state change restarts the torque timers and the speed mean, so a state is judged
// only on its own data (e.g. the stand-up torques do not count in mpc). When the FSM leaves idle
// (e.g. key 7) the watchdog is armed again.
// The rate peaks are reset at every check, so without IMU data they stay 0; roll and tilt keep
// their last value, so the roll and tilt cases need fresh data (else an old value would request
// idle). Speed and torque have no freshness check: lost filter or joint data already raise
// errors elsewhere.
// Joint arrays follow joint_names, matched by name in joint_states.
// Parameters: config/watchdog.yaml only (no defaults here: a missing one stops the node).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <set>
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

enum class Check {
  yaw_rate, roll_rate, pitch_rate, roll, tilt, linear_speed, torque, imu_lost, command_lost
};

// Cases and thresholds of one FSM state (block <state> of watchdog.yaml).
struct StateConfig
{
  std::set<Check> checks;
  double max_yaw_rate = 0.0, max_roll_rate = 0.0, max_pitch_rate = 0.0;  // rad/s
  double max_roll = 0.0, max_tilt = 0.0;                                  // rad
  double max_linear_speed = 0.0;                                          // m/s
  std::vector<double> max_torque;                                         // Nm, per joint
  std::chrono::duration<double> torque_hold{0.0}, timeout{0.0};
  bool has(Check c) const { return checks.count(c) > 0; }
};

class CommandWatchdog : public rclcpp::Node
{
public:
  CommandWatchdog() : Node("command_watchdog")
  {
    joint_names_ = declare_parameter<std::vector<std::string>>("joint_names");
    imu_timeout_ = std::chrono::duration<double>(declare_parameter<double>("imu_timeout"));
    speed_window_ = std::chrono::duration<double>(declare_parameter<double>("speed_window"));
    for (const auto & state : declare_parameter<std::vector<std::string>>("states")) {
      configs_[state] = load_state(state);
    }
    torque_above_since_.assign(joint_names_.size(), std::nullopt);
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
        if (m->data != fsm_state_) {  // a state is judged on its own data only
          torque_above_since_.assign(torque_above_since_.size(), std::nullopt);
          speed_samples_.clear();
        }
        fsm_state_ = m->data;
        const auto it = configs_.find(fsm_state_);
        current_ = it == configs_.end() ? nullptr : &it->second;
      });
    watchdog_mode_publisher_ = create_publisher<std_msgs::msg::String>(
      ros_topic::watchdog_mode, rclcpp::QoS(1).reliable().transient_local());
    timer_ = create_wall_timer(100ms, [this]() { check(); });
  }

private:
  static const std::map<std::string, Check> & check_names()
  {
    static const std::map<std::string, Check> names = {
      {"yaw_rate", Check::yaw_rate}, {"roll_rate", Check::roll_rate},
      {"pitch_rate", Check::pitch_rate}, {"roll", Check::roll}, {"tilt", Check::tilt},
      {"linear_speed", Check::linear_speed}, {"torque", Check::torque},
      {"imu_lost", Check::imu_lost}, {"command_lost", Check::command_lost}};
    return names;
  }

  // Block <state>: its checks and only the parameters those checks need.
  StateConfig load_state(const std::string & state)
  {
    StateConfig c;
    const auto p = [&state](const char * name) { return state + "." + name; };
    for (const auto & name : declare_parameter<std::vector<std::string>>(p("checks"))) {
      const auto it = check_names().find(name);
      if (it == check_names().end()) {
        throw std::runtime_error(p("checks") + ": unknown check '" + name + "'");
      }
      c.checks.insert(it->second);
    }
    if (c.has(Check::yaw_rate)) c.max_yaw_rate = declare_parameter<double>(p("max_yaw_rate"));
    if (c.has(Check::roll_rate)) c.max_roll_rate = declare_parameter<double>(p("max_roll_rate"));
    if (c.has(Check::pitch_rate)) {
      c.max_pitch_rate = declare_parameter<double>(p("max_pitch_rate"));
    }
    if (c.has(Check::roll)) c.max_roll = declare_parameter<double>(p("max_roll_deg")) * M_PI / 180.0;
    if (c.has(Check::tilt)) c.max_tilt = declare_parameter<double>(p("max_tilt_deg")) * M_PI / 180.0;
    if (c.has(Check::linear_speed)) {
      c.max_linear_speed = declare_parameter<double>(p("max_linear_speed"));
    }
    if (c.has(Check::torque)) {
      c.max_torque = declare_parameter<std::vector<double>>(p("max_torque"));
      if (c.max_torque.size() != joint_names_.size()) {
        throw std::runtime_error(p("max_torque") + ": one value per joint_names entry");
      }
      c.torque_hold = std::chrono::duration<double>(declare_parameter<double>(p("torque_hold")));
    }
    if (c.has(Check::command_lost)) {
      c.timeout = std::chrono::duration<double>(declare_parameter<double>(p("timeout")));
    }
    std::string list;
    for (const auto & [name, check] : check_names()) {
      if (c.has(check)) list += (list.empty() ? "" : ", ") + name;
    }
    RCLCPP_INFO(get_logger(), "%s: %s", state.c_str(), list.empty() ? "no checks" : list.c_str());
    if (c.has(Check::torque)) {
      std::string limits;
      for (double v : c.max_torque) limits += fmt(" %.0f", v);
      RCLCPP_INFO(get_logger(), "%s: idle if a joint |effort| >%s Nm for %.2f s", state.c_str(),
                  limits.c_str(), c.torque_hold.count());
    }
    return c;
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
    if (!current_ || !current_->has(Check::torque)) {  // not checked in this state
      torque_above_since_.assign(torque_above_since_.size(), std::nullopt);
      return;
    }
    const auto now = Clock::now();
    for (size_t i = 0; i < joint_index_.size(); ++i) {
      const size_t k = joint_index_[i];
      if (k >= m.effort.size()) continue;
      if (std::abs(m.effort[k]) <= current_->max_torque[i]) {
        torque_above_since_[i].reset();
      } else if (!torque_above_since_[i]) {
        torque_above_since_[i] = now;
      }
    }
  }

  void check()
  {
    // Most severe first; only the cases of the current state's block.
    const StateConfig * c = current_;
    const auto on = [c](Check k) { return c && c->has(k); };
    if (!c) {
      // No block for this state (idle, joint_pd, ...): nothing is checked.
    } else if (on(Check::yaw_rate) && yaw_rate_peak_ > c->max_yaw_rate) {
      request("idle", "|wz| %.2f rad/s", yaw_rate_peak_);
    } else if (on(Check::roll_rate) && roll_rate_peak_ > c->max_roll_rate) {
      request("idle", "|wx| (roll rate) %.2f rad/s", roll_rate_peak_);
    } else if (on(Check::pitch_rate) && pitch_rate_peak_ > c->max_pitch_rate) {
      request("idle", "|wy| (pitch rate) %.2f rad/s", pitch_rate_peak_);
    } else if (on(Check::roll) && imu_fresh() && std::abs(roll_) > c->max_roll) {
      request("idle", "roll %.1f deg", std::abs(roll_) * 180.0 / M_PI);
    } else if (on(Check::tilt) && imu_fresh() && std::abs(tilt_) > c->max_tilt) {
      request("idle", "tilt %.1f deg", tilt_ * 180.0 / M_PI);
    } else if (on(Check::linear_speed) && std::abs(mean_speed()) > c->max_linear_speed) {
      request("idle", "|vx| %.2f m/s (" + fmt("%.2f", speed_window_.count()) + " s mean)",
              std::abs(mean_speed()));
    } else if (on(Check::torque) && torque_joint(*c) >= 0) {
      const int i = torque_joint(*c);
      request("idle", joint_names_[i] + " |effort| above " + fmt("%.1f", c->max_torque[i]) +
              " Nm for %.2f s",
              std::chrono::duration<double>(Clock::now() - *torque_above_since_[i]).count());
    } else if (on(Check::imu_lost) && last_imu_ && !imu_fresh()) {
      request("idle", "no IMU data for %.2f s", imu_timeout_.count());
    } else if (on(Check::command_lost) && last_command_ &&
               Clock::now() - *last_command_ >= c->timeout) {
      request("transform_down", "no command for %.2f s (connection lost?)", c->timeout.count());
    } else {
      requested_ = false;  // no case true: armed again
    }
    yaw_rate_peak_ = roll_rate_peak_ = pitch_rate_peak_ = 0.0;
    if (pending_ && fsm_state_ == "idle") {
      pending_ = false;
      publish("");  // request done
    }
  }

  bool imu_fresh() const { return last_imu_ && Clock::now() - *last_imu_ < imu_timeout_; }
  double mean_speed() const
  {
    double sum = 0.0;
    for (const auto & sample : speed_samples_) sum += sample.second;
    return speed_samples_.empty() ? 0.0 : sum / speed_samples_.size();
  }
  // Index of the first joint above max_torque for torque_hold, -1 if none.
  int torque_joint(const StateConfig & c) const
  {
    const auto now = Clock::now();
    for (size_t i = 0; i < torque_above_since_.size(); ++i) {
      if (torque_above_since_[i] && now - *torque_above_since_[i] >= c.torque_hold) return i;
    }
    return -1;
  }

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
    RCLCPP_ERROR(get_logger(), "%s (FSM %s): requesting %s.", text, fsm_state_.c_str(),
                 mode.c_str());
    pending_ = true;
    publish(mode);
  }

  void publish(const std::string & request)
  {
    std_msgs::msg::String message;
    message.data = request;
    watchdog_mode_publisher_->publish(message);
  }

  std::chrono::duration<double> imu_timeout_, speed_window_;
  std::map<std::string, StateConfig> configs_;  // block of each state in `states`
  const StateConfig * current_ = nullptr;       // block of fsm_state_, nullptr = no checks
  std::optional<Clock::time_point> last_command_, last_imu_;
  double yaw_rate_peak_ = 0.0, roll_rate_peak_ = 0.0, pitch_rate_peak_ = 0.0;
  double roll_ = 0.0, tilt_ = 0.0;
  std::deque<std::pair<Clock::time_point, double>> speed_samples_;  // filter vx in the window
  std::vector<std::string> joint_names_;
  std::vector<size_t> joint_index_;  // joint_names_ -> index in joint_states
  size_t joint_state_size_ = 0;
  std::vector<std::optional<Clock::time_point>> torque_above_since_;
  bool requested_ = false;
  bool pending_ = false;  // request published, FSM not yet in idle
  std::string fsm_state_;
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
