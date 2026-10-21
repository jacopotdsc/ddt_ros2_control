// MPX WBC node (rclcpp front end; the WBC is the JAX function of MPX), separate
// process from the MPC node (mpx_node_cpp), so the two never share a GIL.
// On the joint_states timestamps, at timing.wbc_frequency, it uses U[0] from the
// latest mpx/trajectory, runs the WBC on the measured state and publishes mpx/joint_command:
// effort = tau_ff, position = q_des, velocity = qd_des, for the separate LLC node.
#include "mpx/node_common.hpp"
#include "mpx/python_buffers.hpp"
#include <atomic>
#include <iostream>
#include <pybind11/stl.h>
#include <std_msgs/msg/string.hpp>
#include <thread>

namespace py = pybind11;
using namespace mpx;
using String = std_msgs::msg::String;
using Array = std_msgs::msg::Float64MultiArray;

// Hidden like the pybind11 types it holds (py::object member).
class __attribute__((visibility("hidden"))) WBCNode : public rclcpp::Node {
public:
  explicit WBCNode(const rclcpp::NodeOptions &options)
      : Node("wbc", options) {
    wbc_frequency_ = num("timing.wbc_frequency");
    trajectory_timeout_ = num("timing.trajectory_timeout");
    if (!(wbc_frequency_ > 0.) || !(trajectory_timeout_ > 0.))
      throw std::invalid_argument(
          "timing.wbc_frequency and timing.trajectory_timeout must be positive");
    joint_command_pub_ = create_publisher<Joints>(
        str("outputs.joint_command_topic"), rclcpp::QoS(1).best_effort());
    subs_.push_back(create_subscription<String>(
        str("inputs.robot_description_topic"), rclcpp::QoS(1).transient_local(),
        [this](String::ConstSharedPtr m) {
          Lock lock(mutex_);
          if (description_.empty())
            description_ = m->data;
          cv_.notify_all();
        }));
    subs_.push_back(create_subscription<Array>(
        str("outputs.trajectory_topic"), rclcpp::QoS(1).best_effort(),
        [this](Array::ConstSharedPtr m) { on_trajectory(*m); }));
    feedback_ = std::make_unique<Feedback>(*this, mutex_, cv_);
    RCLCPP_INFO(get_logger(),
                "MPX WBC node (C++ ROS, JAX WBC): %.1f Hz on the joint_states "
                "timestamps; base state source: %s",
                wbc_frequency_, feedback_->source().c_str());
  }
  ~WBCNode() override { stop(); }
  void start() { worker_ = std::thread(&WBCNode::run, this); }
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
  struct Trajectory {
    bool active = false;
    int64_t generation = -1, stamp = 0;
    double height = 0.;
    std::vector<double> U;
    Clock::time_point received{};
  };
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

  void on_trajectory(const Array &m) {
    Lock lock(mutex_);
    if (m.data.size() < kTrajectoryHeader)
      return;
    trajectory_.active = m.data[4] != 0.;
    trajectory_.generation = int64_t(m.data[2]);
    trajectory_.received = Clock::now();
    if (!trajectory_.active) {
      trajectory_.U.clear();  // standby: stop publishing at once
      return;
    }
    // Size checked by the worker against the solver horizon x nu.
    trajectory_.stamp = int64_t(m.data[0]) * 1000000000LL + int64_t(m.data[1]);
    trajectory_.height = m.data[3];
    trajectory_.U.assign(m.data.begin() + kTrajectoryHeader, m.data.end());
    cv_.notify_all();
  }
  // Active, current and not older than the timeout (mutex_ held).
  bool usable(Clock::time_point now) const {
    return ready_ && trajectory_.active &&
           trajectory_.U.size() == horizon_ * nu_ &&
           seconds(now - trajectory_.received) < trajectory_timeout_;
  }

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
      struct DropSolver {
        py::object &solver;
        ~DropSolver() { solver = py::object(); }
      } drop_solver{solver_};
      solver_ = py::module_::import("mpx_cpp_solver")
                    .attr("Solver")(description, true,
                                    num("mpc_rl_handoff.h_mpc"),
                                    py::cpp_function([this](const std::string &s) {
                                      RCLCPP_INFO(get_logger(), "%s", s.c_str());
                                    }),
                                    num("timing.mpc_frequency"));
      StateLayout layout;
      layout.names = solver_.attr("joint_names").cast<std::vector<std::string>>();
      layout.addresses = solver_.attr("joint_addresses")
                             .cast<std::vector<std::pair<int, int>>>();
      layout.nq = solver_.attr("nq").cast<size_t>();
      layout.nv = solver_.attr("nv").cast<size_t>();
      auto limits = solver_.attr("effort_limits").cast<std::vector<double>>();
      const auto nj = layout.names.size();
      if (layout.addresses.size() != nj || limits.size() != nj)
        throw std::runtime_error("Invalid solver configuration");
      // Compile at startup: the robot is still under the previous controller.
      RCLCPP_INFO(get_logger(), "Compiling the WBC...");
      auto compile_start = Clock::now();
      solver_.attr("compile_wbc")();
      {
        Lock lock(mutex_);
        horizon_ = solver_.attr("horizon").cast<size_t>();
        nu_ = solver_.attr("nu").cast<size_t>();
        ready_ = true;
      }
      RCLCPP_INFO(get_logger(), "MPX WBC ready on CPU (compiled in %.1f s).",
                  seconds(Clock::now() - compile_start));

      const int64_t period = int64_t(std::llround(1e9 / wbc_frequency_));
      const double dt = period / 1e9;
      std::vector<double> qpos(layout.nq), qvel(layout.nv), u(nu_);
      int64_t generation = -1, next_due = 0, last_stamp = -1, window_start = 0;
      int64_t missed = 0, invalid = 0;
      std::vector<double> durations, ages;
      auto report = Clock::now(),
           last_error = Clock::now() - std::chrono::seconds(2);
      Joints msg;
      msg.name = layout.names;
      msg.position.resize(nj);
      msg.velocity.resize(nj);
      msg.effort.resize(nj);
      while (true) {
        Base base;
        Joints joints;
        Clock::time_point received;
        int64_t stamp, plan_stamp, plan_generation;
        double height;
        bool reset = false;
        {
          py::gil_scoped_release release;
          std::unique_lock<std::mutex> lock(mutex_);
          while (!stop_) {
            auto now = Clock::now();
            if (usable(now) && feedback_->fresh(now) &&
                feedback_->joint_stamp != last_stamp &&
                (trajectory_.generation != generation || next_due == 0 ||
                 feedback_->joint_stamp >= next_due))
              break;
            cv_.wait_for(lock, std::chrono::milliseconds(20));
          }
          if (stop_)
            return;
          base = feedback_->base;
          joints = feedback_->joints;
          received = feedback_->joint_time;
          stamp = last_stamp = feedback_->joint_stamp;
          plan_stamp = trajectory_.stamp;
          plan_generation = trajectory_.generation;
          // Match mjx_tita.py: use the first MPC input and hold it until a new
          // plan arrives. The WBC still recomputes on this cycle's measured state.
          std::copy_n(trajectory_.U.begin(), nu_, u.begin());
          height = trajectory_.height;
          if (plan_generation != generation) {
            generation = plan_generation;
            next_due = 0;
            reset = true;
            durations.clear();
            ages.clear();
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
        try {
          layout.fill(base, joints, qpos, qvel);
          solver_.attr("prepare_wbc")(mpx::readonly_buffer(qpos),
                                      mpx::readonly_buffer(qvel), reset);
          auto out =
              mpx::array_values(solver_.attr("wbc")(mpx::readonly_buffer(u), height));
          if (out.size() != nj + layout.nv)
            throw std::runtime_error("Invalid WBC output size");
          // Joint targets one WBC period ahead, for the low-level PD.
          for (size_t i = 0; i < nj; ++i) {
            double q = qpos[layout.addresses[i].first];
            double qd = qvel[layout.addresses[i].second];
            double qdd = out[nj + layout.addresses[i].second];
            if (!std::isfinite(out[i]))
              throw std::runtime_error("Invalid torque");
            msg.effort[i] = std::clamp(out[i], -limits[i], limits[i]);
            msg.position[i] = q + qd * dt + .5 * qdd * dt * dt;
            msg.velocity[i] = qd + qdd * dt;
          }
          msg.header.stamp = joints.header.stamp;
          Lock lock(mutex_);
          // Never publish for an old activation, after standby or on stale
          // feedback.
          if (usable(Clock::now()) && trajectory_.generation == generation &&
              seconds(Clock::now() - received) < .15)
            joint_command_pub_->publish(msg);
        } catch (const std::exception &e) {
          ++invalid;
          if (seconds(Clock::now() - last_error) >= 1.) {
            RCLCPP_ERROR(get_logger(), "MPX WBC step failed: %s", e.what());
            last_error = Clock::now();
          }
        }
        durations.push_back(seconds(Clock::now() - start) * 1000.);
        ages.push_back((stamp - plan_stamp) / 1e6);
        if (seconds(Clock::now() - report) >= 5. && !durations.empty()) {
          auto s = stats(durations), a = stats(ages);
          double elapsed = (stamp - window_start) / 1e9;
          RCLCPP_INFO(get_logger(),
                      "WBC: %.1f Hz sim, ms median=%.2f p95=%.2f max=%.2f, "
                      "MPC trajectory age ms median=%.1f max=%.1f, missed "
                      "periods=%ld, invalid=%ld.",
                      elapsed > 0 ? durations.size() / elapsed : 0., s[0], s[1],
                      s[2], a[0], a[2], long(missed), long(invalid));
          durations.clear();
          ages.clear();
          missed = invalid = 0;
          window_start = stamp;
          report = Clock::now();
        }
      }
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "MPX WBC worker failed: %s", e.what());
      failed_ = true;
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  py::object solver_;  // touched only with the GIL held
  std::unique_ptr<Feedback> feedback_;
  Trajectory trajectory_;
  size_t horizon_ = 0, nu_ = 0;
  bool ready_ = false, stop_ = false;
  std::atomic<bool> failed_{false};
  double wbc_frequency_ = 0., trajectory_timeout_ = 0.;
  std::string description_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  rclcpp::Publisher<Joints>::SharedPtr joint_command_pub_;
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
    auto node = std::make_shared<WBCNode>(options);
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
    std::cerr << "MPX WBC C++: " << e.what() << std::endl;
    result = 1;
  }
  if (rclcpp::ok())
    rclcpp::shutdown();
  return result;
}
