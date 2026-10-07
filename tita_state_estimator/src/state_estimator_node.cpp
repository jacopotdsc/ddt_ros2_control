// Floating-base state estimator for TITA: IMU + joint encoders -> base pose/twist.
// The Kalman filter (StateFilter_no_bias.hpp) and the input/output conventions come from
// TITA-dynamic-obstacle-avoidance/tita_controller/src/controller_node.cpp; this node keeps
// only the estimation part (no controller, no CSV logs).

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "ros_utils/topic_names.hpp"
#include "tita_state_estimator/StateFilter_no_bias.hpp"

class StateEstimatorNode : public rclcpp::Node
{
  static constexpr int N_JOINTS = 8;

public:
  StateEstimatorNode()
  : Node("state_estimator")
  {
    // The filter steps on every joint_states message, timed by the message stamps
    // (not by now(): in Gazebo /clock is published at only 10 Hz). It re-initializes
    // if IMU and joint stamps differ, or two joint messages are apart, by more than this.
    max_input_age_ = declare_parameter<double>("max_input_age", 0.1);
    // Constant offset added to the IMU angular velocity (manual gyro bias calibration).
    // Robot specific: the original node used {-0.0017, 0.0036, 0.000230} on its TITA.
    const auto gyro_offset =
      declare_parameter<std::vector<double>>("gyro_offset", {0.0, 0.0, 0.0});
    if (gyro_offset.size() != 3) throw std::invalid_argument("gyro_offset must have 3 elements");
    gyro_offset_ = Eigen::Vector3d(gyro_offset[0], gyro_offset[1], gyro_offset[2]);

    description_sub_ = create_subscription<std_msgs::msg::String>(
      ros_topic::robot_description, rclcpp::QoS(1).transient_local(),
      [this](const std_msgs::msg::String::SharedPtr msg) { build_model(msg->data); });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      ros_topic::body_imu, rclcpp::SensorDataQoS(),
      std::bind(&StateEstimatorNode::imu_callback, this, std::placeholders::_1));
    joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      ros_topic::joint_states, rclcpp::SensorDataQoS(),
      std::bind(&StateEstimatorNode::joint_states_callback, this, std::placeholders::_1));

    filtered_state_pub_ = create_publisher<nav_msgs::msg::Odometry>(ros_topic::filtered_state, 10);
    reset_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/reset",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        pause("reset requested", 0.0);
        res->success = true;
        res->message = "Filter will be re-initialized from the current joint/IMU state.";
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
      });

    RCLCPP_INFO(
      get_logger(), "Waiting for robot description on %s", description_sub_->get_topic_name());
  }

private:
  void build_model(const std::string & urdf)
  {
    if (state_filter_ptr_) return;
    pinocchio::urdf::buildModelFromXML(urdf, pinocchio::JointModelFreeFlyer(), robot_model_);
    if (robot_model_.nq != 7 + N_JOINTS) {
      RCLCPP_FATAL(get_logger(), "Expected %d actuated joints, URDF has nq=%d", N_JOINTS, robot_model_.nq);
      throw std::runtime_error("unexpected robot model");
    }
    for (const char * frame : {"left_leg_4", "right_leg_4"}) {
      if (!robot_model_.existFrame(frame)) throw std::runtime_error(std::string("missing frame ") + frame);
    }
    state_filter_ptr_ = std::make_shared<labrob::KF>(robot_model_);
    RCLCPP_INFO(get_logger(), "Pinocchio model built (%d joints); filter ready", N_JOINTS);
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    imu_orientation_ = Eigen::Quaterniond(
      msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
    imu_orientation_.normalize();
    imu_angular_velocity_ = Eigen::Vector3d(
      msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    imu_linear_acceleration_ = Eigen::Vector3d(
      msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    last_imu_stamp_ = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);
    received_imu_ = true;
  }

  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      auto & joint = joints_[msg->name[i]];
      joint.pos = (i < msg->position.size()) ? msg->position[i] : 0.0;
      joint.vel = (i < msg->velocity.size()) ? msg->velocity[i] : 0.0;
    }
    publish_filtered_state(rclcpp::Time(msg->header.stamp, RCL_ROS_TIME));
  }

  void publish_filtered_state(const rclcpp::Time & t_now)
  {
    if (!state_filter_ptr_ || !received_imu_) return;

    const double imu_age = (t_now - last_imu_stamp_).seconds();
    if (std::abs(imu_age) > max_input_age_) {
      // No IMU in step with the joints (e.g. IMU messages lost): restart from the current
      // state once data resumes.
      pause("IMU and joint stamps out of sync", imu_age, t_now);
      return;
    }

    // ---------- fill filter parameters ----------
    // params: [base quaternion (x,y,z,w), joint positions]; input: [acc, gyro, joint velocities]
    Eigen::Vector<double, 12> filter_params;
    filter_params.segment<4>(0) = imu_orientation_.coeffs();

    Eigen::Vector<double, 14> filter_input;
    filter_input.segment<3>(0) = imu_linear_acceleration_;
    filter_input.segment<3>(3) = imu_angular_velocity_ + gyro_offset_;

    for (pinocchio::JointIndex joint_id = 2; static_cast<int>(joint_id) < robot_model_.njoints; ++joint_id) {
      const std::string & name = robot_model_.names[joint_id];
      auto it = joints_.find(name);
      if (it == joints_.end()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Joint %s missing in joint states", name.c_str());
        return;
      }
      filter_params(4 + joint_id - 2) = it->second.pos;
      filter_input(6 + joint_id - 2) = it->second.vel;
    }

    if (!initialized_filter_) {
      initialized_filter_ = true;
      t_prev_ = t_now;
      state_filter_ptr_->set_initial_condition(filter_params.segment<N_JOINTS>(4), imu_orientation_);
      if (reinit_count_++ == 0) {
        RCLCPP_INFO(get_logger(), "Filter initialized (wheels assumed on the ground)");
      } else {
        RCLCPP_INFO(
          get_logger(), "Filter re-initialized #%d after %.3f s paused (%s); x, y and velocity restart from 0",
          reinit_count_ - 1, pause_start_.nanoseconds() > 0 ? (t_now - pause_start_).seconds() : 0.0,
          pause_reason_.c_str());
      }
      pause_start_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    }

    const double dt = (t_now - t_prev_).seconds();
    if (dt < 0.0 || dt > max_input_age_) {
      pause(dt < 0.0 ? "time jumped back (simulation reset?)" : "gap in the joint states", dt, t_now);
      return;
    }
    t_prev_ = t_now;

    // --------- Run filter ---------
    const bool in_contact = true;
    const Eigen::Vector<double, 12> filtered_state =
      state_filter_ptr_->compute_KF_estimate(filter_input, filter_params, dt, in_contact);

    // Same convention as the original node: pose in the odom (world) frame,
    // twist (linear and angular) in the base frame.
    const Eigen::Vector3d linear_velocity_base =
      imu_orientation_.toRotationMatrix().transpose() * filtered_state.segment<3>(3);
    const Eigen::Vector3d angular_velocity_base = filter_input.segment<3>(3);

    nav_msgs::msg::Odometry msg;
    msg.header.stamp = t_now;
    msg.header.frame_id = "odom";
    msg.child_frame_id = "base_link";
    msg.pose.pose.position.x = filtered_state(0);
    msg.pose.pose.position.y = filtered_state(1);
    msg.pose.pose.position.z = filtered_state(2);
    msg.pose.pose.orientation.x = imu_orientation_.x();
    msg.pose.pose.orientation.y = imu_orientation_.y();
    msg.pose.pose.orientation.z = imu_orientation_.z();
    msg.pose.pose.orientation.w = imu_orientation_.w();
    msg.twist.twist.linear.x = linear_velocity_base.x();
    msg.twist.twist.linear.y = linear_velocity_base.y();
    msg.twist.twist.linear.z = linear_velocity_base.z();
    msg.twist.twist.angular.x = angular_velocity_base.x();
    msg.twist.twist.angular.y = angular_velocity_base.y();
    msg.twist.twist.angular.z = angular_velocity_base.z();
    filtered_state_pub_->publish(msg);
  }

  // Stops the estimate until the next synchronized sample, which re-initializes the filter.
  // Logs once per pause (not per sample) with the reason.
  void pause(const std::string & reason, double seconds, const rclcpp::Time & t_now = rclcpp::Time(0, 0, RCL_ROS_TIME))
  {
    if (!initialized_filter_) return;  // already paused or never started
    initialized_filter_ = false;
    pause_reason_ = reason;
    pause_start_ = t_now;
    RCLCPP_WARN(get_logger(), "%s (%.3f s): estimate paused", reason.c_str(), seconds);
  }

  struct JointSample
  {
    double pos = 0.0;
    double vel = 0.0;
  };

  pinocchio::Model robot_model_;
  std::shared_ptr<labrob::KF> state_filter_ptr_;

  Eigen::Quaterniond imu_orientation_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d imu_angular_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_linear_acceleration_ = Eigen::Vector3d(0, 0, 9.81);
  Eigen::Vector3d gyro_offset_ = Eigen::Vector3d::Zero();
  std::unordered_map<std::string, JointSample> joints_;

  bool received_imu_ = false;
  bool initialized_filter_ = false;
  int reinit_count_ = 0;  // initializations so far (the first is the start-up one)
  std::string pause_reason_;
  rclcpp::Time pause_start_{0, 0, RCL_ROS_TIME};
  double max_input_age_ = 0.1;
  rclcpp::Time t_prev_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr description_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr filtered_state_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StateEstimatorNode>());
  rclcpp::shutdown();
  return 0;
}
