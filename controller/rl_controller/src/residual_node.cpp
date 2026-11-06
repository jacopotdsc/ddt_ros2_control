// Residual network node. Runs next to MPX: while the FSM of rl_controller is in "residual"
// (fsm_state), every time_interval (wall timer, asynchronous from the WBC) it runs the policy
// and publishes a torque offset per joint on residual/joint_command (sensor_msgs/JointState,
// effort; names = `joints`). rl_controller adds the latest offset to the latest WBC tau_ff
// in the low level (FSMState_MPC::run()). Outside "residual" nothing runs or is published.
//
// Dummy step:
//   - observations all 0 (TODO: q, dq, IMU, command, as FSMState_RL::update_observations);
//   - published offset fixed to 0 (TODO: output[i] * action_scales[i]).
// The network still runs at every step, to measure its rate: every 5 s the log reports the
// inferences per second and the inference time.
// Model inputs: 1 = history (ppo, as FSMState_RLPPO); 2 = observation + history (np3o, as
// FSMState_RL, e.g. flat.onnx).
// Parameters: config/tita/residual.yaml only (no defaults here: a missing one stops the node).

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rl_controller/inferrer/onnx_inferrer.hpp"
#include "ros_utils/topic_names.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"

using Clock = std::chrono::steady_clock;

class ResidualNode : public rclcpp::Node
{
public:
  ResidualNode() : Node("residual_node")
  {
    policy_path_ = declare_parameter<std::string>("policy_path");
    if (!policy_path_.empty() && policy_path_.front() != '/') {
      policy_path_ =
        ament_index_cpp::get_package_share_directory("rl_controller") + "/" + policy_path_;
    }
    const auto output_name = declare_parameter<std::string>("output_name");
    num_obs_ = declare_parameter<int>("num_obs");
    num_actions_ = declare_parameter<int>("num_actions");
    history_len_ = declare_parameter<int>("history_len");
    const double time_interval = declare_parameter<double>("time_interval");
    action_scales_ = declare_parameter<std::vector<double>>("action_scales");
    joints_ = declare_parameter<std::vector<std::string>>("joints");
    if (num_obs_ <= 0 || num_actions_ <= 0 || history_len_ <= 0 || time_interval <= 0.0) {
      throw std::runtime_error("residual_node: num_obs, num_actions, history_len, time_interval > 0");
    }
    if (joints_.size() != static_cast<size_t>(num_actions_)) {
      throw std::runtime_error("residual_node: joints must have num_actions names");
    }
    if (action_scales_.size() != static_cast<size_t>(num_actions_)) {
      throw std::runtime_error("residual_node: action_scales must have num_actions values");
    }

    inferrer_.loadModel(policy_path_);
    inferrer_.setOutput(output_name, num_actions_);
    num_inputs_ = inferrer_.getInputNames().size();
    if (num_inputs_ != 1 && num_inputs_ != 2) {
      throw std::runtime_error("residual_node: the model must have 1 or 2 inputs");
    }
    obs_.assign(num_obs_, 0.0f);
    history_.assign(static_cast<size_t>(num_obs_) * history_len_, 0.0f);
    msg_.name = joints_;
    msg_.effort.assign(num_actions_, 0.0);

    publisher_ = create_publisher<sensor_msgs::msg::JointState>(
      ros_topic::residual_joint_command, rclcpp::QoS(1).best_effort());
    fsm_state_subscription_ = create_subscription<std_msgs::msg::String>(
      ros_topic::fsm_state, rclcpp::QoS(1).reliable().transient_local(),
      [this](const std_msgs::msg::String::SharedPtr msg) { fsm_state_cb(msg->data); });
    timer_ = create_wall_timer(
      std::chrono::duration<double>(time_interval), [this]() { step(); });
    RCLCPP_INFO(get_logger(),
      "residual_node: %s (%zu input(s)), %d obs x %d history -> %d offsets, %.1f Hz while the FSM "
      "is in residual. Dummy: observations 0, offset 0.",
      policy_path_.c_str(), num_inputs_, num_obs_, history_len_, num_actions_, 1.0 / time_interval);
  }

private:
  void fsm_state_cb(const std::string & state)
  {
    const bool active = state == "residual";
    if (active == active_) return;
    active_ = active;
    // Each activation starts from an empty history and a new statistics window.
    std::fill(history_.begin(), history_.end(), 0.0f);
    infer_ms_.clear();
    stats_start_ = Clock::now();
    if (active) {
      RCLCPP_INFO(get_logger(), "FSM in residual: network running, offsets published");
    } else {
      RCLCPP_INFO(get_logger(), "FSM left residual (now %s): network stopped", state.c_str());
    }
  }

  void step()
  {
    if (!active_) return;
    // Dummy observations (TODO: real ones).
    std::fill(obs_.begin(), obs_.end(), 0.0f);
    std::vector<std::vector<tensor_element_t>> inputs;
    if (num_inputs_ == 1) {
      push_history();  // ppo: the history already contains the current observation
      inputs.push_back(history_);
    } else {
      inputs.push_back(obs_);  // np3o: current observation + previous ones
      inputs.push_back(history_);
    }
    const auto t0 = Clock::now();
    const std::vector<tensor_element_t> output = inferrer_.computeActions(inputs);
    infer_ms_.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    if (num_inputs_ == 2) push_history();
    if (output.size() != static_cast<size_t>(num_actions_)) {
      throw std::runtime_error("residual_node: network output size != num_actions");
    }
    // Offset [N m] = network output * action_scales (residual.yaml), per joint.
    for (size_t i = 0; i < msg_.effort.size(); ++i) msg_.effort[i] = output[i] * action_scales_[i];
    msg_.header.stamp = now();
    publisher_->publish(msg_);
    log_stats();
  }

  void push_history()
  {
    std::copy(history_.begin() + num_obs_, history_.end(), history_.begin());
    std::copy(obs_.begin(), obs_.end(), history_.end() - num_obs_);
  }

  void log_stats()
  {
    const double elapsed = std::chrono::duration<double>(Clock::now() - stats_start_).count();
    if (elapsed < 5.0) return;
    std::vector<double> v = infer_ms_;
    std::sort(v.begin(), v.end());
    RCLCPP_INFO(get_logger(),
      "network: %zu inferences in %.1f s (%.1f Hz), inference ms median=%.3f max=%.3f",
      v.size(), elapsed, v.size() / elapsed, v[v.size() / 2], v.back());
    infer_ms_.clear();
    stats_start_ = Clock::now();
  }

  std::string policy_path_;
  int num_obs_ = 0, num_actions_ = 0, history_len_ = 0;
  size_t num_inputs_ = 0;
  std::vector<double> action_scales_;
  std::vector<std::string> joints_;
  ONNXInferrer inferrer_;
  std::vector<tensor_element_t> obs_, history_;
  sensor_msgs::msg::JointState msg_;
  bool active_ = false;
  std::vector<double> infer_ms_;
  Clock::time_point stats_start_{};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr fsm_state_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ResidualNode>());
  rclcpp::shutdown();
  return 0;
}
