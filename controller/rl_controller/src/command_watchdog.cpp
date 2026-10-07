// Folds the robot when every command source goes silent (e.g. the WiFi link to the
// keyboard PC is lost). Runs on the robot next to rl_controller.
//
// The keyboard and the remote publish the twist continuously. When no twist arrives for
// `timeout` seconds, this node requests "transform_down" on the mode topic, like key 8:
// from RL the FSM folds at once, from MPC MPX first lowers the COM and hands off.
// It is armed by the first twist, and stops requesting once commands come back.

#include <chrono>
#include <memory>
#include <optional>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ros_utils/topic_names.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

class CommandWatchdog : public rclcpp::Node
{
public:
  CommandWatchdog() : Node("command_watchdog")
  {
    timeout_ = std::chrono::duration<double>(declare_parameter<double>("timeout", 1.0));
    twist_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      ros_topic::manager_twist_command, rclcpp::SystemDefaultsQoS(),
      [this](geometry_msgs::msg::Twist::SharedPtr) {
        last_command_ = Clock::now();
        if (tripped_) {
          tripped_ = false;
          RCLCPP_WARN(get_logger(), "Commands are back: transform_down no longer requested.");
        }
      });
    // Same QoS as the keyboard mode publisher.
    rclcpp::QoS qos(rclcpp::KeepLast(10));
    qos.reliable().transient_local();
    mode_publisher_ = create_publisher<std_msgs::msg::String>(ros_topic::manager_key_command, qos);
    timer_ = create_wall_timer(100ms, [this]() { check(); });
    RCLCPP_INFO(
      get_logger(), "Armed on the first message on %s; transform_down after %.2f s without commands.",
      twist_subscription_->get_topic_name(), timeout_.count());
  }

private:
  void check()
  {
    if (!last_command_ || Clock::now() - *last_command_ < timeout_) return;
    if (!tripped_) {
      tripped_ = true;
      RCLCPP_ERROR(
        get_logger(), "No command on %s for %.2f s (connection lost?): requesting transform_down.",
        twist_subscription_->get_topic_name(), timeout_.count());
    }
    std_msgs::msg::String mode;
    mode.data = "transform_down";
    mode_publisher_->publish(mode);
  }

  std::chrono::duration<double> timeout_;
  std::optional<Clock::time_point> last_command_;
  bool tripped_ = false;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CommandWatchdog>());
  rclcpp::shutdown();
  return 0;
}
