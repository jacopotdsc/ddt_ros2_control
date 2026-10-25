// Shared C++ pieces of the MPX nodes (mpx_node_cpp = MPC, mpx_wbc_node = WBC):
// frames, timing helpers, feedback acquisition and the MuJoCo generalized state.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <gazebo_msgs/msg/model_states.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <mpx/default_params.hpp>
#include <pybind11/embed.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mpx {
using Clock = std::chrono::steady_clock;
using Pose = geometry_msgs::msg::Pose;
using Twist = geometry_msgs::msg::Twist;
using Joints = sensor_msgs::msg::JointState;
using Odom = nav_msgs::msg::Odometry;
using Lock = std::lock_guard<std::mutex>;
using Rotation = std::array<std::array<double, 3>, 3>;

inline Rotation rotation(const geometry_msgs::msg::Quaternion &q) {
  const auto x = q.x, y = q.y, z = q.z, w = q.w;
  return {
      {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
       {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
       {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}
inline geometry_msgs::msg::Vector3 rotate(const Rotation &r,
                                          const geometry_msgs::msg::Vector3 &v,
                                          bool transpose = false) {
  geometry_msgs::msg::Vector3 out;
  const double a[] = {v.x, v.y, v.z};
  double b[3] = {};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      b[i] += (transpose ? r[j][i] : r[i][j]) * a[j];
  out.x = b[0];
  out.y = b[1];
  out.z = b[2];
  return out;
}
inline double seconds(Clock::duration dt) {
  return std::chrono::duration<double>(dt).count();
}
inline int64_t stamp_ns(const builtin_interfaces::msg::Time &s) {
  return int64_t(s.sec) * 1000000000LL + s.nanosec;
}
// median, p95, max
inline std::array<double, 3> stats(std::vector<double> v) {
  if (v.empty())
    return {0., 0., 0.};
  std::sort(v.begin(), v.end());
  auto percentile = [&](double p) {
    double i = (v.size() - 1) * p;
    auto a = size_t(i);
    return v[a] + (v[std::min(a + 1, v.size() - 1)] - v[a]) * (i - a);
  };
  return {percentile(.5), percentile(.95), v.back()};
}

struct Base {
  Pose pose;
  Twist twist;
  Clock::time_point received{};
  bool valid = false;
};

// Latest base and joint feedback, stored under the owner's mutex. Callbacks
// never touch Python. Every new sample notifies the owner's condition variable.
class Feedback {
public:
  // on_time_reset: called (mutex held) when the joint stamps go backwards.
  Feedback(rclcpp::Node &node, std::mutex &mutex, std::condition_variable &cv,
           std::function<void()> on_time_reset = {})
      : mutex_(mutex), cv_(cv), on_time_reset_(std::move(on_time_reset)) {
    auto str = [&](const char *key) { return node.get_parameter(key).as_string(); };
    model_name_ = str("inputs.model_name");
    source_ = str("inputs.base_state_source");
    if (source_ != "ground_truth" && source_ != "filter" && source_ != "odometry")
      throw std::invalid_argument(
          "inputs.base_state_source must be ground_truth/filter/odometry");
    auto qos = rclcpp::QoS(1).best_effort();
    if (source_ == "ground_truth") {
      subs_.push_back(node.create_subscription<gazebo_msgs::msg::ModelStates>(
          str("inputs.model_states_topic"), qos,
          [this](gazebo_msgs::msg::ModelStates::ConstSharedPtr m) {
            auto it = std::find(m->name.begin(), m->name.end(), model_name_);
            if (it == m->name.end())
              return;
            auto i = std::distance(m->name.begin(), it);
            if (size_t(i) >= m->pose.size() || size_t(i) >= m->twist.size())
              return;
            Lock lock(mutex_);
            base = {m->pose[i], m->twist[i], Clock::now(), true};
            cv_.notify_all();
          }));
    } else {
      if (source_ == "odometry")
        subs_.push_back(node.create_subscription<Odom>(
            str("inputs.odometry_topic"), qos, [this](Odom::ConstSharedPtr m) {
              Lock lock(mutex_);
              odom_ = *m;
              odom_time_ = Clock::now();
            }));
      subs_.push_back(node.create_subscription<Odom>(
          str("inputs.filtered_state_topic"), qos, [this](Odom::ConstSharedPtr m) {
            Lock lock(mutex_);
            auto pose = m->pose.pose;
            auto r = rotation(pose.orientation);
            Twist twist;
            twist.linear = rotate(r, m->twist.twist.linear);
            twist.angular = rotate(r, m->twist.twist.angular);
            auto received = Clock::now();
            if (source_ == "odometry") {
              if (!odom_)
                return;
              auto yaw = std::atan2(r[1][0], r[0][0]);
              auto c = std::cos(yaw), s = std::sin(yaw);
              auto v = odom_->twist.twist.linear;
              twist.linear.x = c * v.x - s * v.y;
              twist.linear.y = s * v.x + c * v.y;
              pose.position.x = odom_->pose.pose.position.x;
              pose.position.y = odom_->pose.pose.position.y;
              received = std::min(received, odom_time_);
            }
            base = {pose, twist, received, true};
            cv_.notify_all();
          }));
    }
    subs_.push_back(node.create_subscription<Joints>(
        str("inputs.joint_states_topic"), qos, [this](Joints::ConstSharedPtr m) {
          Lock lock(mutex_);
          auto stamp = stamp_ns(m->header.stamp);
          if (have_joint && stamp < joint_stamp && on_time_reset_)
            on_time_reset_();
          joints = *m;
          joint_stamp = stamp;
          joint_time = Clock::now();
          have_joint = true;
          cv_.notify_all();
        }));
  }
  const std::string &source() const { return source_; }
  // Both base and joints younger than 0.1 s (owner's mutex held).
  bool fresh(Clock::time_point now) const {
    return base.valid && have_joint && seconds(now - base.received) < .1 &&
           seconds(now - joint_time) < .1;
  }

  // Data, read under the owner's mutex.
  Base base;
  Joints joints;
  int64_t joint_stamp = 0;
  Clock::time_point joint_time{};
  bool have_joint = false;

private:
  std::mutex &mutex_;
  std::condition_variable &cv_;
  std::function<void()> on_time_reset_;
  std::string model_name_, source_;
  std::optional<Odom> odom_;
  Clock::time_point odom_time_{};
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
};

// Joint order and MuJoCo addresses of the solver; fills qpos/qvel (no Python).
struct StateLayout {
  std::vector<std::string> names;
  std::vector<std::pair<int, int>> addresses;  // (qpos, dof)
  size_t nq = 0, nv = 0;

  void fill(const Base &base, const Joints &joints, std::vector<double> &qpos,
            std::vector<double> &qvel) const {
    qpos[0] = base.pose.position.x;
    qpos[1] = base.pose.position.y;
    qpos[2] = base.pose.position.z;
    qpos[3] = base.pose.orientation.w;
    qpos[4] = base.pose.orientation.x;
    qpos[5] = base.pose.orientation.y;
    qpos[6] = base.pose.orientation.z;
    qvel[0] = base.twist.linear.x;
    qvel[1] = base.twist.linear.y;
    qvel[2] = base.twist.linear.z;
    // World angular velocity -> body frame (MuJoCo free joint).
    auto angular = rotate(rotation(base.pose.orientation), base.twist.angular, true);
    qvel[3] = angular.x;
    qvel[4] = angular.y;
    qvel[5] = angular.z;
    for (size_t i = 0; i < names.size(); ++i) {
      auto it = std::find(joints.name.begin(), joints.name.end(), names[i]);
      auto j = std::distance(joints.name.begin(), it);
      if (it == joints.name.end() || size_t(j) >= joints.position.size() ||
          size_t(j) >= joints.velocity.size())
        throw std::runtime_error("Missing joint feedback: " + names[i]);
      qpos.at(addresses[i].first) = joints.position[j];
      qvel.at(addresses[i].second) = joints.velocity[j];
    }
  }
};

// mpx/trajectory (std_msgs/Float64MultiArray), MPC node -> WBC node:
//   data[0..1]  stamp of the feedback the MPC solved from (sec, nanosec; robot clock)
//   data[2]     activation generation (changes on every activation / reset)
//   data[3]     COM height reference (ramped) for the WBC
//   data[4]     1 = active, 0 = standby (the WBC stops publishing at once)
//   data[5]     weight phase: index in config_tita.weight_phases (W_mpc / W_wbc entry)
//   data[6..]   U, horizon x nu row-major; WBC currently holds the first row U[0]
// layout.dim: {"header", 6}, {"horizon", N, N*nu}, {"input", nu, nu}; data_offset 0.
constexpr size_t kTrajectoryHeader = 6;

// Embedded interpreter setup shared by the executables: JAX on CPU, one BLAS thread,
// and mpx_cpp_solver.py found next to the executable.
inline void python_environment() {
  setenv("JAX_PLATFORMS", "cpu", 1);
  setenv("OPENBLAS_NUM_THREADS", "1", 0);
  setenv("OMP_NUM_THREADS", "1", 0);
  setenv("MKL_NUM_THREADS", "1", 0);
  setenv("XLA_FLAGS",
         "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1", 0);
}
inline void python_path(const char *argv0) {
  // With --symlink-install the executable resolves into build/: search the
  // invoked path (install/lib/mpx) first, then the resolved one.
  auto path = pybind11::module_::import("sys").attr("path");
  path.attr("insert")(
      0, std::filesystem::canonical("/proc/self/exe").parent_path().string());
  path.attr("insert")(0, std::filesystem::absolute(argv0).parent_path().string());
}
}  // namespace mpx
