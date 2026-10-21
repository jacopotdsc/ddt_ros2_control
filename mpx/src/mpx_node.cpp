// MPX MPC node (rclcpp front end; the MPC is the JAX function of MPX). It owns the
// feedback, the modes, the height ramp and the handoff, solves the MPC on the
// joint_states timestamps and publishes the whole input trajectory on mpx/trajectory.
// mpx_wbc_node and mpx_llc_node (separate processes) consume the plan's first input
// before rl_controller writes the final torque to hardware.
#include "mpx/node_common.hpp"
#include "mpx/python_buffers.hpp"
#include <atomic>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <iostream>
#include <pybind11/stl.h>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>

namespace py = pybind11;
using namespace mpx;
using String = std_msgs::msg::String;
using Array = std_msgs::msg::Float64MultiArray;

// Hidden like the pybind11 types it holds (py::object member).
class __attribute__((visibility("hidden"))) MPXNode : public rclcpp::Node {
public:
  explicit MPXNode(const rclcpp::NodeOptions &options)
      : Node("controller", options) {
    controlled_ = !str("inputs.mode_command_topic").empty();
    enabled_ = !controlled_;
    h_mpc_ = num("mpc_rl_handoff.h_mpc");
    h_rl_ = num("mpc_rl_handoff.h_rl");
    command_[3] = h_mpc_;
    height_tol_ = num("mpc_rl_handoff.height_tol");
    vz_tol_ = num("mpc_rl_handoff.vz_tol");
    settle_ns_ = num("mpc_rl_handoff.settle_time") * 1e9;
    timeout_ns_ = num("mpc_rl_handoff.timeout") * 1e9;
    rl_modes_ = get_parameter("mpc_rl_handoff.rl_modes").as_string_array();
    pending_compile_ = get_parameter("behavior.precompile").as_bool();
    auto feedback = rclcpp::QoS(1).best_effort();
    auto latched = rclcpp::QoS(1).transient_local();
    trajectory_pub_ =
        create_publisher<Array>(str("outputs.trajectory_topic"), feedback);
    lock_pub_ = create_publisher<std_msgs::msg::Bool>(
        str("outputs.keys_locked_topic"), latched);
    handoff_pub_ = create_publisher<String>(str("outputs.handoff_topic"), 1);
    publish_lock(true);
    subscribe<String>(str("inputs.robot_description_topic"), latched,
                      [this](const String &m) {
                        Lock lock(mutex_);
                        if (description_.empty())
                          description_ = m.data;
                        cv_.notify_all();
                      });
    feedback_ = std::make_unique<Feedback>(*this, mutex_, cv_, [this] {
      // Simulation reset: restart the activation.
      ++generation_;
      handoff_start_.reset();
      handoff_settled_.reset();
      publish_standby();
    });
    subscribe<Twist>(
        str("inputs.cmd_vel_topic"), rclcpp::QoS(1), [this](const Twist &m) {
          Lock lock(mutex_);
          if (!blocked())
            command_ = {m.linear.x, m.linear.z, m.angular.z, command_[3]};
        });
    subscribe<std_msgs::msg::Float64>(str("inputs.com_height_topic"), latched,
                                      [this](const std_msgs::msg::Float64 &m) {
                                        Lock lock(mutex_);
                                        set_height(m.data);
                                      });
    if (controlled_)
      subscribe<String>(str("inputs.mode_command_topic"), latched,
                        [this](const String &m) {
                          Lock lock(mutex_);
                          bool changed = m.data != last_mode_;
                          last_mode_ = m.data;
                          apply_mode(m.data);
                          publish_lock(changed);
                        });
    if (!str("inputs.fsm_state_topic").empty())
      subscribe<String>(
          str("inputs.fsm_state_topic"), latched, [this](const String &m) {
            Lock lock(mutex_);
            fsm_ = m.data;
            if (enabled_ && took_over_ && fsm_blocks()) {
              enabled_ = took_over_ = activating_ = false;
              handoff_.clear();
              ++generation_;
              publish_standby();
              publish_lock();
              cv_.notify_all();
              RCLCPP_WARN(get_logger(), "RL FSM switched to %s: MPX standby",
                          fsm_.c_str());
            }
          });
    if (!str("inputs.pose_command_topic").empty())
      subscribe<geometry_msgs::msg::PoseStamped>(
          str("inputs.pose_command_topic"), rclcpp::QoS(1),
          [this](const geometry_msgs::msg::PoseStamped &m) {
            Lock lock(mutex_);
            auto z = m.pose.position.z;
            if (!last_pose_z_ || z != *last_pose_z_) {
              last_pose_z_ = z;
              set_height(z);
            }
          });
    if (get_parameter("behavior.publish_default_commands").as_bool()) {
      default_pub_ = create_publisher<Twist>(str("inputs.cmd_vel_topic"), 1);
      default_timer_ =
          create_wall_timer(std::chrono::milliseconds(100), [this] {
            for (const auto &e :
                 get_publishers_info_by_topic(str("inputs.cmd_vel_topic"))) {
              if (e.node_name() != get_name() ||
                  e.node_namespace() != get_namespace()) {
                default_timer_->cancel();
                default_pub_.reset();
                return;
              }
            }
            default_pub_->publish(Twist{});
          });
    }
    estimates_ = get_parameter("behavior.publish_estimate").as_bool();
    if (estimates_) {
      auto prefix = str("outputs.estimate_topic_prefix");
      for (auto suffix : {"/com", "/left_wheel", "/right_wheel"})
        point_pubs_.push_back(
            create_publisher<geometry_msgs::msg::PointStamped>(prefix + suffix,
                                                               feedback));
      for (auto suffix : {"/com_vel", "/base_rpy"})
        vector_pubs_.push_back(
            create_publisher<geometry_msgs::msg::Vector3Stamped>(
                prefix + suffix, feedback));
    }
    RCLCPP_INFO(get_logger(),
                "MPX MPC node (C++ ROS, JAX MPC); the WBC runs in mpx_wbc_node");
    RCLCPP_INFO(get_logger(), "Base state source: %s",
                feedback_->source().c_str());
  }
  ~MPXNode() override { stop(); }
  void start() { worker_ = std::thread(&MPXNode::run, this); }
  void stop() {
    {
      Lock lock(mutex_);
      stop_ = true;
      cv_.notify_all();
    }
    if (worker_.joinable())
      worker_.join();
  }
  bool failed() const { return failed_.load(); }

private:
  template <class M, class Callback>
  void subscribe(const std::string &topic, const rclcpp::QoS &qos,
                 Callback cb) {
    subs_.push_back(create_subscription<M>(
        topic, qos, [cb](const typename M::ConstSharedPtr m) { cb(*m); }));
  }
  std::string str(const char *key) const {
    return get_parameter(key).as_string();
  }
  // Accept integer YAML values too (e.g. "250" for a frequency).
  double num(const char *key) const {
    auto p = get_parameter(key);
    return p.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER
               ? double(p.as_int())
               : p.as_double();
  }
  bool fsm_blocks() const { return fsm_ == "idle" || fsm_ == "transform_down"; }
  bool blocked() const {
    return (compiling_ && enabled_) || activating_ || !handoff_.empty();
  }
  void publish_lock(bool force = false) {
    bool value = blocked();
    if (force || !last_lock_ || value != *last_lock_) {
      std_msgs::msg::Bool msg;
      msg.data = value;
      lock_pub_->publish(msg);
      last_lock_ = value;
    }
  }
  void set_height(double z) {
    if (!std::isfinite(z)) {
      RCLCPP_WARN(get_logger(), "Ignoring non-finite COM height command");
      return;
    }
    if (!blocked())
      command_[3] = z;
  }
  void apply_mode(const std::string &mode) {
    bool rl =
        std::find(rl_modes_.begin(), rl_modes_.end(), mode) != rl_modes_.end();
    bool enable = mode == "mpc";
    if (!enable && !rl && mode != "idle" && mode != "joint_pd" &&
        mode != "transform_down" && mode != "transform_up")
      return;
    if (enable && !enabled_ && fsm_blocks())
      return;
    bool in_control = enabled_ && took_over_ && !fsm_blocks();
    if (mode == "transform_up" && in_control)
      return;
    // transform_down is the safety command: no handoff, MPX goes to standby
    // while the FSM folds right away.
    if (rl && in_control) {
      activating_ = false;
      double h = h_rl_;
      if (handoff_.empty() || h != command_[3]) {
        command_ = {0., 0., 0., h};
        handoff_start_.reset();
        handoff_settled_.reset();
        RCLCPP_INFO(
            get_logger(),
            "%s requested: MPX moves the COM to %.3f m before handing off",
            mode.c_str(), h);
      }
      handoff_ = mode;
      return;
    }
    bool cancelled = enable && !handoff_.empty();
    handoff_.clear();
    if (cancelled) {
      command_ = {0., 0., 0., h_mpc_};
      activating_ = true;
      activation_start_.reset();
    } else if (enabled_ != enable) {
      enabled_ = enable;
      took_over_ = false;
      activating_ = enable;
      activation_start_.reset();
      if (enable)
        command_ = {0., 0., 0., h_mpc_};
      ++generation_;
      if (!enable)
        publish_standby();
      cv_.notify_all();
      RCLCPP_INFO(get_logger(), "MPC computation %s",
                  enable ? "enabled" : "standby");
    }
  }
  void check_transitions(int64_t generation, int64_t stamp, double z, double vz,
                         double target, double ref) {
    Lock lock(mutex_);
    if (generation != generation_)
      return;
    if (!handoff_.empty()) {
      if (!handoff_start_)
        handoff_start_ = stamp;
      bool settled =
          std::abs(z - command_[3]) < height_tol_ && std::abs(vz) < vz_tol_;
      if (!settled)
        handoff_settled_.reset();
      else if (!handoff_settled_)
        handoff_settled_ = stamp;
      bool timeout = stamp - *handoff_start_ >= timeout_ns_;
      if (timeout ||
          (handoff_settled_ && stamp - *handoff_settled_ >= settle_ns_)) {
        String msg;
        msg.data = handoff_;
        handoff_.clear();
        enabled_ = took_over_ = false;
        ++generation_;
        publish_standby();
        handoff_pub_->publish(msg);
        publish_lock();
        cv_.notify_all();
        RCLCPP_INFO(get_logger(), "COM at %.3f m: handing off to %s%s",
                    command_[3], msg.data.c_str(), timeout ? " (timeout)" : "");
        RCLCPP_INFO(get_logger(), "MPC computation standby");
      }
    }
    if (activating_ && took_over_ && generation == generation_) {
      if (!activation_start_)
        activation_start_ = stamp;
      bool settled = ref == target && std::abs(z - target) < height_tol_ &&
                     std::abs(vz) < vz_tol_;
      bool timeout = stamp - *activation_start_ >= timeout_ns_;
      if (settled || timeout) {
        activating_ = false;
        publish_lock();
        RCLCPP_INFO(get_logger(),
                    "COM at h_mpc=%.3f m: MPC commands unlocked%s", target,
                    settled ? "" : " anyway (timeout)");
      }
    }
  }
  // mpx/trajectory, layout in mpx/node_common.hpp (mutex_ held).
  void publish_trajectory(const builtin_interfaces::msg::Time &stamp,
                          int64_t generation, double height,
                          const std::vector<double> &U, bool active) {
    Array msg;
    msg.layout.dim.resize(3);
    msg.layout.dim[0].label = "header";
    msg.layout.dim[0].size = kTrajectoryHeader;
    msg.layout.dim[0].stride = kTrajectoryHeader;
    msg.layout.dim[1].label = "horizon";
    msg.layout.dim[1].size = horizon_;
    msg.layout.dim[1].stride = horizon_ * nu_;
    msg.layout.dim[2].label = "input";
    msg.layout.dim[2].size = nu_;
    msg.layout.dim[2].stride = nu_;
    msg.data.reserve(kTrajectoryHeader + U.size());
    msg.data = {double(stamp.sec), double(stamp.nanosec), double(generation),
                height, active ? 1. : 0.};
    msg.data.insert(msg.data.end(), U.begin(), U.end());
    trajectory_pub_->publish(msg);
  }
  // Tells the WBC node to stop publishing at once (mutex_ held).
  void publish_standby() {
    publish_trajectory(builtin_interfaces::msg::Time(), generation_, 0., {},
                       false);
  }
  void publish_estimates(const std::vector<double> &d,
                         const builtin_interfaces::msg::Time &stamp) {
    if (!estimates_)
      return;
    geometry_msgs::msg::PointStamped p;
    p.header.stamp = stamp;
    p.header.frame_id = "world";
    for (size_t i = 0; i < 3; ++i) {
      size_t off = i == 0 ? 0 : 6 + (i - 1) * 3;
      p.point.x = d[off];
      p.point.y = d[off + 1];
      p.point.z = d[off + 2];
      point_pubs_[i]->publish(p);
    }
    geometry_msgs::msg::Vector3Stamped v;
    v.header = p.header;
    v.vector.x = d[3];
    v.vector.y = d[4];
    v.vector.z = d[5];
    vector_pubs_[0]->publish(v);
    double w = d[12], x = d[13], y = d[14], z = d[15];
    v.vector.x = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    v.vector.y = std::asin(std::clamp(2 * (w * y - z * x), -1., 1.));
    v.vector.z = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    vector_pubs_[1]->publish(v);
  }
  static std::vector<double> values(const py::object &object) {
    // Do not use the NumPy C API from Humble's pybind11 2.9 with NumPy 2.
    // The Python buffer protocol (inputs) and tolist (small outputs) are
    // ABI-safe.
    return mpx::array_values(object);
  }

  // Worker thread: owns the solver, plans on the joint_states timestamps.
  void run() {
    try {
      std::string description;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        auto timeout = std::chrono::duration<double>(
            num("inputs.robot_description_timeout"));
        if (!cv_.wait_for(lock, timeout,
                          [this] { return stop_ || !description_.empty(); }))
          throw std::runtime_error("Timed out waiting for robot description");
        if (stop_)
          return;
        description = description_;
      }
      py::gil_scoped_acquire gil;
      // Drop the solver with the GIL held, before the interpreter ends.
      struct DropSolver {
        py::object &solver;
        ~DropSolver() { solver = py::object(); }
      } drop_solver{solver_};
      solver_ = py::module_::import("mpx_cpp_solver")
                    .attr("Solver")(description, controlled_, h_mpc_,
                                    py::cpp_function([this](const std::string &s) {
                                      RCLCPP_INFO(get_logger(), "%s", s.c_str());
                                    }),
                                    num("timing.mpc_frequency"));
      layout_.names =
          solver_.attr("joint_names").cast<std::vector<std::string>>();
      layout_.addresses = solver_.attr("joint_addresses")
                              .cast<std::vector<std::pair<int, int>>>();
      layout_.nq = solver_.attr("nq").cast<size_t>();
      layout_.nv = solver_.attr("nv").cast<size_t>();
      auto period = solver_.attr("period_ns").cast<int64_t>();
      {
        Lock lock(mutex_);
        horizon_ = solver_.attr("horizon").cast<size_t>();
        nu_ = solver_.attr("nu").cast<size_t>();
      }
      if (period <= 0 || layout_.names.size() != layout_.addresses.size())
        throw std::runtime_error("Invalid solver configuration");
      std::vector<double> qpos(layout_.nq), qvel(layout_.nv);
      if (pending_compile_ && !controlled_) {
        solver_.attr("compile_plan")();
        pending_compile_ = false;
      }
      RCLCPP_INFO(get_logger(),
                  "MPX %s: MPC %.1f Hz on the joint_states timestamps; "
                  "trajectory for mpx_wbc_node.",
                  controlled_ ? "standby" : "ready on CPU", 1e9 / period);
      int64_t generation = -1, next_due = 0, previous_stamp = 0,
              window_start = 0;
      bool first = true;
      int warmup = 0;
      double ref = 0.;
      std::vector<double> durations;
      int64_t missed = 0, invalid = 0;
      auto report = Clock::now(), last_wait = Clock::now(),
           last_error = Clock::now() - std::chrono::seconds(2);
      while (true) {
        Base base;
        Joints joints;
        Clock::time_point received;
        int64_t stamp;
        std::array<double, 4> command;
        {
          // Release the GIL while idle and during C++ message/state handling.
          py::gil_scoped_release release;
          std::unique_lock<std::mutex> lock(mutex_);
          while (!stop_) {
            auto now = Clock::now();
            if (enabled_ && feedback_->fresh(now) &&
                (generation != generation_ || next_due == 0 ||
                 feedback_->joint_stamp >= next_due))
              break;
            if (enabled_ && !feedback_->fresh(now) &&
                seconds(now - last_wait) >= 2.) {
              RCLCPP_WARN(get_logger(),
                          "MPX waiting for fresh base/joint feedback");
              last_wait = now;
            }
            cv_.wait_for(lock, std::chrono::milliseconds(100));
          }
          if (stop_)
            return;
          base = feedback_->base;
          joints = feedback_->joints;
          received = feedback_->joint_time;
          stamp = feedback_->joint_stamp;
          command = command_;
          if (generation != generation_) {
            generation = generation_;
            next_due = 0;
            first = true;
            previous_stamp = 0;
            durations.clear();
            missed = invalid = 0;
            window_start = stamp;
            report = Clock::now();
          }
          if (next_due == 0)
            next_due = stamp;
          auto skipped = (stamp - next_due) / period;
          missed += skipped;
          next_due += (skipped + 1) * period;
        }
        auto start = Clock::now();
        std::vector<double> diagnostic;
        try {
          if (pending_compile_) {
            {
              Lock lock(mutex_);
              compiling_ = true;
              publish_lock();
            }
            RCLCPP_WARN(
                get_logger(),
                "Compiling the MPC after key 4; previous control remains active.");
            try {
              solver_.attr("compile_plan")();
            } catch (...) {
              Lock lock(mutex_);
              compiling_ = false;
              publish_lock();
              throw;
            }
            {
              Lock lock(mutex_);
              compiling_ = false;
              publish_lock();
            }
            pending_compile_ = false;
            generation = -1;
            next_due = 0;
            RCLCPP_INFO(get_logger(), "MPX MPC ready on CPU: compilation "
                                      "complete; preparing the trajectory.");
            continue;
          }
          layout_.fill(base, joints, qpos, qvel);
          diagnostic = values(solver_.attr("prepare")(
              mpx::readonly_buffer(qpos), mpx::readonly_buffer(qvel), first));
          if (diagnostic.size() != 17)
            throw std::runtime_error("Invalid diagnostic size");
          if (first) {
            ref = diagnostic[2];
            warmup = controlled_ ? 10 : 0;
            first = false;
            RCLCPP_INFO(get_logger(),
                        "First feedback received; starting simulation-time "
                        "control (%d warm-up MPC solves before the trajectory).",
                        warmup);
          }
          auto dt = previous_stamp == 0 ? period / 1e9
                                        : (stamp - previous_stamp) / 1e9;
          previous_stamp = stamp;
          ref += std::clamp(command[3] - ref, -.1 * dt, .1 * dt);
          auto ramped = command;
          ramped[3] = ref;
          auto trajectory = values(solver_.attr("plan")(ramped));
          if (trajectory.size() != horizon_ * nu_)
            throw std::runtime_error("Invalid MPC trajectory size");
          if (warmup > 0) {
            if (--warmup == 0) {
              durations.clear();
              missed = invalid = 0;
              window_start = stamp;
              report = Clock::now();
            }
            continue;
          }
          {
            // mpx_wbc_node publishes the joint commands from this trajectory.
            Lock lock(mutex_);
            if (enabled_ && generation == generation_ &&
                seconds(Clock::now() - received) < .15) {
              publish_trajectory(joints.header.stamp, generation, ref,
                                 trajectory, true);
              took_over_ = true;
            }
          }
        } catch (const std::exception &e) {
          ++invalid;
          if (seconds(Clock::now() - last_error) >= 1.) {
            RCLCPP_ERROR(get_logger(), "MPX MPC step failed: %s", e.what());
            last_error = Clock::now();
          }
        }
        if (diagnostic.size() == 17)
          check_transitions(generation, stamp, diagnostic[2], diagnostic[5],
                            command[3], ref);
        durations.push_back(seconds(Clock::now() - start) * 1000.);
        if (diagnostic.size() == 17)
          publish_estimates(diagnostic, joints.header.stamp);
        if (seconds(Clock::now() - report) >= 5. && !durations.empty()) {
          auto s = stats(durations);
          double elapsed = (stamp - window_start) / 1e9;
          RCLCPP_INFO(get_logger(),
                      "MPC: %.1f Hz sim, plan ms median=%.2f p95=%.2f "
                      "max=%.2f, missed periods=%ld, invalid=%ld.",
                      elapsed > 0 ? durations.size() / elapsed : 0., s[0], s[1],
                      s[2], long(missed), long(invalid));
          durations.clear();
          missed = invalid = 0;
          window_start = stamp;
          report = Clock::now();
        }
      }
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "MPX worker failed: %s", e.what());
      failed_ = true;
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  py::object solver_;  // touched only with the GIL held
  std::unique_ptr<Feedback> feedback_;
  StateLayout layout_;
  size_t horizon_ = 0, nu_ = 0;  // mutex_ (read by publish_standby)
  std::atomic<bool> failed_{false};
  bool stop_ = false, controlled_ = false, enabled_ = false, compiling_ = false,
       activating_ = false, took_over_ = false;
  bool pending_compile_ = false, estimates_ = false;
  double h_mpc_ = 0., h_rl_ = 0., height_tol_ = 0., vz_tol_ = 0.;
  int64_t settle_ns_ = 0, timeout_ns_ = 0, generation_ = 0;
  std::optional<int64_t> handoff_start_, handoff_settled_, activation_start_;
  std::optional<double> last_pose_z_;
  std::optional<bool> last_lock_;
  std::array<double, 4> command_{};
  std::vector<std::string> rl_modes_;
  std::string description_, fsm_, last_mode_, handoff_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  rclcpp::Publisher<Array>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr lock_pub_;
  rclcpp::Publisher<String>::SharedPtr handoff_pub_;
  rclcpp::Publisher<Twist>::SharedPtr default_pub_;
  rclcpp::TimerBase::SharedPtr default_timer_;
  std::vector<rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr>
      point_pubs_;
  std::vector<rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr>
      vector_pubs_;
};

int main(int argc, char **argv) {
  python_environment();  // before the interpreter imports JAX
  mpx::DefaultArguments args(argc, argv);
  rclcpp::init(args.argc(), args.argv());
  int result = 0;
  try {
    py::scoped_interpreter interpreter{};
    python_path(argv[0]);
    py::gil_scoped_release release;
    auto options =
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(
            true);
    auto node = std::make_shared<MPXNode>(options);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    node->start();
    while (rclcpp::ok() && !node->failed())
      executor.spin_once(std::chrono::milliseconds(100));
    node->stop();
    result = node->failed() ? 1 : 0;
    executor.remove_node(node);
    node.reset();
  } catch (const std::exception &e) {
    std::cerr << "MPX C++: " << e.what() << std::endl;
    result = 1;
  }
  if (rclcpp::ok())
    rclcpp::shutdown();
  return result;
}
