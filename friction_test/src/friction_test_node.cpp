#include "friction_test/friction_test_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace
{
// Key t: seconds holding the default pose between two tests of the sequence.
constexpr double SEQUENCE_PAUSE = 3.0;
constexpr double DT = JointTrajectory::DT;
const std::vector<std::string> SEGMENTS = {
  "start", "pause", "approach", "sweep+", "sweep-", "speed+", "speed-", "return"};
enum Segment : uint8_t { START, PAUSE, APPROACH, SWEEP_POS, SWEEP_NEG, SPEED_POS, SPEED_NEG, RETURN };

double smoothstep(double x)
{
  x = std::clamp(x, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

size_t samples(double duration) { return static_cast<size_t>(std::ceil(duration / DT - 1e-9)); }

// Rest-to-rest quintic from q0 to q1 (zero velocity and acceleration at both ends).
std::vector<double> quintic(double q0, double q1, double duration)
{
  std::vector<double> q(samples(duration));
  for (size_t i = 0; i < q.size(); ++i) {
    const double s = i * DT / duration;
    q[i] = q0 + (q1 - q0) * (10 * std::pow(s, 3) - 15 * std::pow(s, 4) + 6 * std::pow(s, 5));
  }
  return q;
}

// Velocity: smooth ramp 0 -> v, cruise_time at v, ramp v -> 0 (peak acceleration acc_max).
std::vector<double> speed_profile(double v, double acc_max, double cruise_time)
{
  const double ramp_time = 1.5 * std::abs(v) / acc_max;  // smoothstep ramp: peak acceleration 1.5 v / T
  std::vector<double> up(samples(ramp_time));
  for (size_t i = 0; i < up.size(); ++i) up[i] = v * smoothstep(i * DT / ramp_time);
  std::vector<double> vel(up);
  vel.insert(vel.end(), static_cast<size_t>(std::round(cruise_time / DT)), v);
  vel.insert(vel.end(), up.rbegin(), up.rend());
  return vel;
}

// Duration of a smooth stroke of length D with peak acceleration a_peak (smooth_stroke).
double stroke_time(double D, double a_peak) { return 2.0 * std::sqrt(2.0 * D / a_peak); }

// Smooth rest-to-rest stroke from qa to qb. Acceleration: a triangle in each half of the stroke,
// 0 -> a_peak (at a quarter of the time) -> 0 at mid-stroke, then the same negative down to 0 at qb.
// Velocity and acceleration are 0 at both ends, the velocity is highest at mid-stroke:
// half duration tau = sqrt(2 D / a_peak), v_mid = a_peak tau / 2 = sqrt(a_peak D / 2).
std::vector<double> smooth_stroke(double qa, double qb, double a_peak)
{
  const double D = std::abs(qb - qa), sign = qb > qa ? 1.0 : -1.0;
  const double T = stroke_time(D, a_peak), tau = 0.5 * T, a = a_peak;
  auto half = [a, tau](double t) {  // displacement in the first half, 0 <= t <= tau
    if (t <= 0.5 * tau) return a / (3 * tau) * t * t * t;
    const double u = t - 0.5 * tau;
    return a * tau * tau / 24 + a * tau / 4 * u + 0.5 * a * u * u - a / (3 * tau) * u * u * u;
  };
  const size_t n = samples(T);
  std::vector<double> q(n);
  for (size_t i = 1; i <= n; ++i) {
    const double t = std::min(i * DT, T);
    const double disp = t <= tau ? half(t) : D - half(T - t);
    q[i - 1] = qa + sign * disp;
  }
  q.back() = qb;
  return q;
}

std::vector<double> gradient(const std::vector<double> & x)
{
  std::vector<double> d(x.size(), 0.0);
  if (x.size() < 2) return d;
  d.front() = (x[1] - x[0]) / DT;
  d.back() = (x[x.size() - 1] - x[x.size() - 2]) / DT;
  for (size_t i = 1; i + 1 < x.size(); ++i) d[i] = (x[i + 1] - x[i - 1]) / (2 * DT);
  return d;
}

// Concatenates trajectory pieces with their speed level and segment label.
struct Builder
{
  JointTrajectory t;
  explicit Builder(double q0)
  {
    t.q = {q0};
    t.level = {-1};
    t.segment = {START};
  }
  double last() const { return t.q.back(); }
  void add(const std::vector<double> & q, int level, uint8_t segment)
  {
    t.q.insert(t.q.end(), q.begin(), q.end());
    t.level.insert(t.level.end(), q.size(), level);
    t.segment.insert(t.segment.end(), q.size(), segment);
  }
  // Positions integrated from a velocity profile, starting at the last position.
  void add_velocity(const std::vector<double> & vel, int level, uint8_t segment)
  {
    std::vector<double> q(vel.size());
    double x = last();
    for (size_t i = 0; i < vel.size(); ++i) q[i] = (x += vel[i] * DT);
    add(q, level, segment);
  }
  void pause(double duration)
  {
    add(std::vector<double>(static_cast<size_t>(std::round(duration / DT)), last()), -1, PAUSE);
  }
  JointTrajectory build(const std::string & joint)
  {
    t.joint = joint;
    t.dq = gradient(t.q);
    t.ddq = gradient(t.dq);
    return std::move(t);
  }
};

}  // namespace

size_t JointTrajectory::index(double t) const
{
  const double k = std::round(t / DT);
  return static_cast<size_t>(std::clamp(k, 0.0, static_cast<double>(q.size() - 1)));
}

std::string JointTrajectory::write_csv(
  const std::string & dir, const std::string & suffix, const std::vector<std::string> & joints) const
{
  std::filesystem::create_directories(dir);
  const std::string path = dir + "/" + joint + suffix + ".csv";
  std::ofstream f;
  f.exceptions(std::ios::failbit | std::ios::badbit);
  f.open(path);
  f << "t_sim,q_des,dq_des,ddq_des,q,dq,tau_applied,kp,kd,speed_level,segment";
  // then the state of every joint, as their data files hold all joints (q, qd, tau_obs)
  for (const char * name : {"q_", "dq_", "tau_"}) {
    for (const auto & j : joints) f << "," << name << j;
  }
  f << "\n" << std::setprecision(10);
  for (const auto & r : rows) {
    f << std::fixed << std::setprecision(4) << r.t << std::defaultfloat << std::setprecision(10) << ","
      << r.q_des << "," << r.dq_des << "," << r.ddq_des << "," << r.q << "," << r.dq << "," << r.tau
      << "," << r.kp << "," << r.kd << "," << r.level << "," << SEGMENTS[r.segment];
    for (const auto * v : {&r.all_q, &r.all_dq, &r.all_tau}) {
      for (const double x : *v) f << "," << x;
    }
    f << "\n";
  }
  f.close();  // report buffered write errors before announcing success
  return path;
}

// {key, FSM state requested on command/cmd_key, name}; "test_*" keys start a test.
const std::vector<std::tuple<char, std::string, std::string>> FrictionTestNode::key_mapping = {
  {'0', "idle", "idle"}, {'1', "joint_cmd", "joint_pd"},
  {'2', "joint_cmd", "test_left_wheel"}, {'3', "joint_cmd", "test_right_wheel"},
  {'4', "joint_cmd", "test_left_knee"}, {'5', "joint_cmd", "test_right_knee"},
  {'6', "joint_cmd", "test_left_hip_pitch"}, {'7', "joint_cmd", "test_right_hip_pitch"},
  {'8', "joint_cmd", "test_left_abduction"}, {'9', "joint_cmd", "test_right_abduction"}};

FrictionTestNode::FrictionTestNode(const rclcpp::NodeOptions & options)
: Node("friction_test", options)
{
  joints_ = declare_parameter<std::vector<std::string>>("joints");
  kp_ = declare_parameter<std::vector<double>>("joint_kp");
  kd_ = declare_parameter<std::vector<double>>("joint_kd");
  default_pose_ = declare_parameter<std::vector<double>>("default_pose");
  command_rate_ = declare_parameter<double>("command_rate");
  home_ramp_time_ = declare_parameter<double>("home_ramp_time");
  csv_dir_ = declare_parameter<std::string>("csv_dir");
  speed_fractions_ = declare_parameter<std::vector<double>>("trajectory.speed_fractions");
  sweeps_per_level_ = declare_parameter<int>("trajectory.sweeps_per_level");
  acc_fractions_ = declare_parameter<std::vector<double>>("trajectory.acc_fractions");
  lower_bound_ = declare_parameter<std::vector<double>>("lower_bound");
  upper_bound_ = declare_parameter<std::vector<double>>("upper_bound");
  wheel_hold_time_ = declare_parameter<double>("trajectory.wheel_hold_time");
  pause_time_ = declare_parameter<double>("trajectory.pause_time");
  for (const auto & [key, state, action] : key_mapping) {
    if (action.rfind("test_", 0) == 0) {
      tests_[key] = declare_parameter<std::vector<std::string>>(std::string("tests.key_") + key);
    }
  }
  // acc_max.<joint>, the only acceleration of the trajectories: legs, peak of the strokes; wheels, peak of the
  // ramps between the speed steps (the wheel speeds come from lower_bound/upper_bound)
  for (const auto & j : joints_) {
    acc_max_[j] = declare_parameter<double>("acc_max." + j);
    if (acc_max_[j] <= 0.) throw std::invalid_argument("acc_max." + j + " must be > 0");
  }
  if (lower_bound_.size() != joints_.size() || upper_bound_.size() != joints_.size()) {
    throw std::invalid_argument("lower_bound/upper_bound need one value per joint");
  }
  for (size_t i = 0; i < joints_.size(); ++i) {
    const bool wheel = joints_[i].back() == '4';
    if (wheel ? !(lower_bound_[i] < 0. && upper_bound_[i] > 0.) : lower_bound_[i] >= upper_bound_[i]) {
      throw std::invalid_argument(
        wheel ? "wheel " + joints_[i] + ": lower_bound < 0 < upper_bound (speeds, rad/s)"
              : "lower_bound must be < upper_bound for " + joints_[i]);
    }
  }
  if (kp_.size() != joints_.size() || kd_.size() != joints_.size() || default_pose_.size() != 4) {
    throw std::invalid_argument("joint_kp/joint_kd need one gain per joint, default_pose 4 values per leg");
  }
  if (csv_dir_.rfind("~/", 0) == 0) csv_dir_ = std::string(std::getenv("HOME")) + csv_dir_.substr(1);

  const size_t n = joints_.size();
  q_.assign(n, 0.);
  dq_.assign(n, 0.);
  effort_.assign(n, 0.);
  hold_.assign(n, 0.);
  cmd_.assign(n, {0., 0., 0., 0., 0.});
  cmd_info_.assign(n, {0., -1, PAUSE});

  // Same QoS as keyboard_controller for command/cmd_key (reliable, transient local).
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST).keep_last(10);
  fsm_goal_publisher_ = create_publisher<std_msgs::msg::String>(ros_topic::manager_key_command, qos);
  joint_command_publisher_ =
    create_publisher<std_msgs::msg::Float64MultiArray>(ros_topic::joint_command, 10);
  joint_states_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
    ros_topic::joint_states, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::JointState::SharedPtr msg) { on_joint_states(*msg); });
  fsm_state_subscription_ = create_subscription<std_msgs::msg::String>(
    ros_topic::fsm_state, rclcpp::QoS(1).reliable().transient_local(),
    [this](const std_msgs::msg::String::SharedPtr msg) { on_fsm_state(msg->data); });

  // Joint limits from the URDF (robot_state_publisher, latched).
  robot_description_subscription_ = create_subscription<std_msgs::msg::String>(
    ros_topic::robot_description, rclcpp::QoS(1).reliable().transient_local(),
    [this](const std_msgs::msg::String::SharedPtr msg) { on_robot_description(msg->data); });

  fsm_goal_timer_ = create_wall_timer(10ms, std::bind(&FrictionTestNode::PubFsmGoalCallBack, this));
  control_timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Duration::from_seconds(1.0 / command_rate_),
    std::bind(&FrictionTestNode::ControlCallBack, this));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    publish_zero();
    print_interface();
  }
  std::thread(&FrictionTestNode::ReadKeyThread, this).detach();
}

// ── interface (as keyboard_controller) ────────────────────────────────────
void FrictionTestNode::print_interface()
{
  // Keep the control callback independent of spawning a shell/clear process.
  std::cout << "\033[2J\033[H";
  std::cout << R"(
  -------------------------------------------------------
                      State Machines)"
            << std::endl;
  int count = 0;
  for (const auto & [key, state, name] : key_mapping) {
    std::cout << "  " << key << ": " << name;
    count++;
    if (count % 3 == 0) {
      std::cout << std::endl;
    } else {
      std::cout << "  ";
    }
  }
  if (count % 3 != 0) {
    std::cout << std::endl;
  }
  std::cout << R"(
  -------------------------------------------------------
  )" << std::endl;
  std::cout << std::fixed << std::setprecision(0);
  std::cout << std::setw(36) << "state machine now: " << std::setw(6) << GREEN << fsm_goal_ << RESET;
  if (!fsm_state_.empty() && fsm_state_ != fsm_goal_) {
    std::cout << "  (FSM: " << fsm_state_ << ")";
  }
  std::cout << std::endl;
  std::cout << std::setw(36) << "friction test now: " << std::setw(6);
  switch (mode_) {
    case Mode::ZERO: std::cout << RED << "zero torque"; break;
    case Mode::WAIT_JOINT_CMD: std::cout << YELLOW << "joint_pd (waiting for joint_cmd)"; break;
    case Mode::RAMP: std::cout << GREEN << "joint_pd (ramp to the default pose)"; break;
    case Mode::HOLD: std::cout << GREEN << "joint_pd (default pose)"; break;
    case Mode::TEST: {
      double duration = 0.;
      for (const auto & [i, t] : test_) duration = std::max(duration, t.duration());
      const auto name = std::find_if(key_mapping.begin(), key_mapping.end(),
        [this](const auto & m) { return std::get<0>(m) == test_key_; });
      std::cout << GREEN << std::get<2>(*name) << "  " << std::min(now_s() - test_t0_, duration) << " / "
                << duration << " s";
      break;
    }
  }
  std::cout << RESET << std::endl;
  if (sequence_) std::cout << RED << "  " << sequence_status() << RESET << std::endl;
  if (!message_.empty()) {
    std::cout << (message_warn_ ? RED : "") << "  " << message_ << RESET << std::endl;
  }
  std::cout << R"(
  -------------------------------------------------------
       "0": idle (always), "1": joint_pd (always)
       "2"-"9": tests (one joint each) from joint_pd, "t": all in sequence, Ctrl+C: idle and quit
  )" << std::endl;
  last_print_ = now_s();
}

void FrictionTestNode::note(const std::string & text, bool warn)
{
  message_ = text;
  message_warn_ = warn;
  print_interface();
}

int FrictionTestNode::get_key()
{
  int ch;
  struct termios oldt;
  struct termios newt;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;

  newt.c_lflag &= ~(ICANON | ECHO);
  newt.c_iflag |= IGNBRK;
  newt.c_iflag &= ~(INLCR | ICRNL | IXON | IXOFF);
  newt.c_lflag &= ~(ICANON | ECHO | ECHOK | ECHOE | ECHONL | ISIG | IEXTEN);
  newt.c_cc[VMIN] = 1;
  newt.c_cc[VTIME] = 0;
  tcsetattr(fileno(stdin), TCSANOW, &newt);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return ch;
}

void FrictionTestNode::ReadKeyThread()
{
  while (rclcpp::ok()) {
    const int key = get_key();
    std::lock_guard<std::mutex> lock(mutex_);
    if (key == '\x03' || key == EOF) {  // Ctrl+C: idle and quit
      fsm_goal_ = "idle";
      fsm_goal_publisher_->publish(std_msgs::msg::String().set__data(fsm_goal_));
      zero_torque("Ctrl+C");
      rclcpp::shutdown();
      return;
    }
    if (key == 't' || key == 'T') {  // every test in sequence
      if (mode_ != Mode::HOLD || fsm_state_ != "joint_cmd") {
        note("t: the sequence starts only while holding the default pose (press 1)", true);
      } else if (start_test(tests_.begin()->first)) {
        sequence_ = true;
        sequence_index_ = 0;
        note("t: sequence started, test 1/" + std::to_string(tests_.size()), true);
      }
      continue;
    }
    const auto mapping = std::find_if(key_mapping.begin(), key_mapping.end(),
      [key](const auto & m) { return std::get<0>(m) == key; });
    if (mapping == key_mapping.end()) continue;
    const auto & [k, state, action] = *mapping;
    if (action == "idle") {
      fsm_goal_ = state;
      if (mode_ != Mode::ZERO) zero_torque("idle");
      else print_interface();
    } else if (action == "joint_pd") {
      fsm_goal_ = state;
      sequence_ = false;
      if (mode_ == Mode::TEST) abort_test();
      if (fsm_state_ == "joint_cmd" && have_state_) {
        start_joint_pd(std::string(1, k));
      } else {
        mode_ = Mode::WAIT_JOINT_CMD;
        note(std::string(1, k) + ": joint_pd, requesting FSM joint_cmd");
      }
    } else if (mode_ != Mode::HOLD || fsm_state_ != "joint_cmd") {
      note(std::string(1, k) + ": tests start only while holding the default pose (press 1)", true);
    } else {
      start_test(k);
    }
  }
}

bool FrictionTestNode::start_test(char k)
{
  if (std::any_of(tests_.at(k).begin(), tests_.at(k).end(),
        [this](const auto & j) { return !joint_limits_.count(j); })) {
    note(std::string(1, k) + ": no joint limits yet (robot_description)", true);
    return false;
  }
  // The test range (lower_bound, upper_bound of the yaml) must lie inside the URDF limits: positions for the
  // legs, speeds for the wheels (continuous joints, URDF velocity limit).
  for (const auto & j : tests_.at(k)) {
    const size_t i = std::find(joints_.begin(), joints_.end(), j) - joints_.begin();
    const bool wheel = j.back() == '4';
    const auto & l = joint_limits_.at(j);
    const double lo = wheel ? -joint_velocity_limits_.at(j) : l[0], hi = wheel ? joint_velocity_limits_.at(j) : l[1];
    if (lower_bound_[i] < lo || upper_bound_[i] > hi) {
      std::ostringstream e;
      e << k << ": " << j << (wheel ? " speeds [" : " range [") << lower_bound_[i] << ", " << upper_bound_[i]
        << "] outside the URDF limits [" << lo << ", " << hi << "], test refused";
      note(e.str(), true);
      return false;
    }
  }
  test_.clear();
  double duration = 0.;
  for (const auto & j : tests_.at(k)) {
    const size_t i = std::find(joints_.begin(), joints_.end(), j) - joints_.begin();
    test_[i] = generate_joint_trajectory(j, lower_bound_[i], upper_bound_[i]);
    duration = std::max(duration, test_[i].duration());
  }
  test_key_ = k;
  test_t0_ = now_s();
  mode_ = Mode::TEST;
  std::ostringstream s;
  s << k << ": test started (" << std::fixed << std::setprecision(0) << duration << " s)";
  if (!trajectory_info_.empty()) s << "  " << trajectory_info_;
  note(s.str());
  RCLCPP_INFO(get_logger(), "%s", s.str().c_str());
  return true;
}

std::string FrictionTestNode::sequence_status()
{
  const size_t total = tests_.size();
  auto name = [this](size_t index) {
    const char key = std::next(tests_.begin(), index)->first;
    return std::get<2>(*std::find_if(key_mapping.begin(), key_mapping.end(),
      [key](const auto & m) { return std::get<0>(m) == key; }));
  };
  std::ostringstream s;
  s << "sequence t, test " << sequence_index_ + 1 << "/" << total << ": ";
  if (mode_ == Mode::TEST) {
    s << name(sequence_index_) << " running";
  } else if (mode_ == Mode::RAMP) {
    s << name(sequence_index_) << " done, joint_pd: ramp to the default pose";
  } else if (mode_ == Mode::HOLD && sequence_index_ + 1 < total) {
    s << name(sequence_index_) << " done, joint_pd: holding, next " << name(sequence_index_ + 1) << " in "
      << std::fixed << std::setprecision(0) << std::max(0.0, sequence_next_ - now_s()) << " s";
  } else {
    s << "joint_pd";
  }
  return s.str();
}

// ── ROS callbacks ─────────────────────────────────────────────────────────
void FrictionTestNode::PubFsmGoalCallBack()
{
  std::lock_guard<std::mutex> lock(mutex_);
  fsm_goal_publisher_->publish(std_msgs::msg::String().set__data(fsm_goal_));
}

void FrictionTestNode::on_fsm_state(const std::string & state)
{
  std::lock_guard<std::mutex> lock(mutex_);
  fsm_state_ = state;
  // As keyboard_controller: the requested state follows the FSM, whoever changed it (watchdog, ...).
  if ((state == "idle" || state == "joint_cmd") && state != fsm_goal_ && mode_ != Mode::WAIT_JOINT_CMD) {
    fsm_goal_ = state;
  }
  if (state != "joint_cmd" && mode_ != Mode::ZERO && mode_ != Mode::WAIT_JOINT_CMD) {
    zero_torque("FSM in " + state);
  } else if (state == "joint_cmd" && mode_ == Mode::WAIT_JOINT_CMD && have_state_) {
    start_joint_pd("FSM in joint_cmd");
  } else {
    note("FSM state: " + state);
  }
}

void FrictionTestNode::on_robot_description(const std::string & xml)
{
  urdf::Model model;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model.initString(xml)) {
    note("robot_description: URDF not parsed, no joint limits", true);
    return;
  }
  for (const auto & j : joints_) {
    const auto joint = model.getJoint(j);
    if (joint && joint->limits) {
      joint_limits_[j] = {joint->limits->lower, joint->limits->upper};
      joint_velocity_limits_[j] = joint->limits->velocity;
    }
  }
  note("joint limits from robot_description (" + std::to_string(joint_limits_.size()) + " joints)");
}

void FrictionTestNode::on_joint_states(const sensor_msgs::msg::JointState & msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  size_t found = 0;
  for (size_t k = 0; k < msg.name.size(); ++k) {
    const auto it = std::find(joints_.begin(), joints_.end(), msg.name[k]);
    if (it == joints_.end()) continue;
    const size_t i = it - joints_.begin();
    q_[i] = msg.position[k];
    dq_[i] = msg.velocity[k];
    effort_[i] = k < msg.effort.size() ? msg.effort[k] : NAN;
    ++found;
  }
  have_state_ = have_state_ || found == joints_.size();
  if (mode_ != Mode::TEST) return;
  const double t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
  for (auto & [i, traj] : test_) {
    const auto & [ddq, level, segment] = cmd_info_[i];
    traj.rows.push_back({t, cmd_[i][0], cmd_[i][1], ddq, q_[i], dq_[i], effort_[i], kp_[i], kd_[i], level,
                         segment, q_, dq_, effort_});
  }
}

void FrictionTestNode::ControlCallBack()
{
  std::lock_guard<std::mutex> lock(mutex_);
  while (!csv_jobs_.empty() && csv_jobs_.front().wait_for(0s) == std::future_status::ready) {
    try {
      for (const auto & message : csv_jobs_.front().get()) note(message);
    } catch (const std::exception & e) {
      sequence_ = false;
      note(std::string("CSV save failed: ") + e.what(), true);
      RCLCPP_ERROR(get_logger(), "CSV save failed: %s", e.what());
    }
    csv_jobs_.pop_front();
  }
  const double now = now_s();
  const size_t n = joints_.size();
  switch (mode_) {
    case Mode::ZERO:
    case Mode::WAIT_JOINT_CMD:
      publish_zero();
      return;
    case Mode::RAMP: {
      const double s = std::min((now - ramp_t0_) / home_ramp_time_, 1.0);
      const double blend = 10 * std::pow(s, 3) - 15 * std::pow(s, 4) + 6 * std::pow(s, 5);
      const double dblend =
        s < 1.0 ? (30 * s * s - 60 * std::pow(s, 3) + 30 * std::pow(s, 4)) / home_ramp_time_ : 0.0;
      for (size_t i = 0; i < n; ++i) {
        const double d = ramp_target_[i] - ramp_start_[i];
        set_command(i, ramp_start_[i] + d * blend, d * dblend, 0., -1, PAUSE);
      }
      if (s >= 1.0) {
        hold_ = ramp_target_;
        mode_ = Mode::HOLD;
        if (sequence_) {
          sequence_next_ = now + SEQUENCE_PAUSE;
          print_interface();
        } else {
          note("holding the default pose: tests available");
        }
      }
      break;
    }
    case Mode::HOLD:
      for (size_t i = 0; i < n; ++i) set_command(i, hold_[i], 0., 0., -1, PAUSE);
      if (sequence_ && now >= sequence_next_) {  // next test of the sequence
        if (++sequence_index_ < tests_.size()) {
          if (!start_test(std::next(tests_.begin(), sequence_index_)->first)) sequence_ = false;
        } else {
          sequence_ = false;
          note("t: sequence finished, holding the default pose", true);
        }
      }
      break;
    case Mode::TEST: {
      const double t = now - test_t0_;
      double duration = 0.;
      for (size_t i = 0; i < n; ++i) {
        const auto it = test_.find(i);
        if (it == test_.end()) {
          set_command(i, hold_[i], 0., 0., -1, PAUSE);
          continue;
        }
        const auto & traj = it->second;
        const size_t k = traj.index(t);
        set_command(i, traj.q[k], traj.dq[k], traj.ddq[k], traj.level[k], traj.segment[k]);
        duration = std::max(duration, traj.duration());
      }
      if (t > duration + 0.5) {
        csv_jobs_.push_back(std::async(std::launch::async,
          [completed = std::move(test_), dir = csv_dir_, joints = joints_, key = test_key_]() {
            std::vector<std::string> messages;
            for (const auto & [i, traj] : completed) {
              const auto path = traj.write_csv(dir, "", joints);
              messages.push_back(std::string(1, key) + ": " + traj.joint + " done, " +
                std::to_string(traj.rows.size()) + " rows -> " + path);
            }
            return messages;
          }));
        test_.clear();
        start_joint_pd(std::string(1, test_key_) + " done");  // as key 1
      }
      break;
    }
  }
  std_msgs::msg::Float64MultiArray msg;
  msg.data.reserve(5 * n);
  for (const auto & c : cmd_) msg.data.insert(msg.data.end(), c.begin(), c.end());
  joint_command_publisher_->publish(msg);
  if ((mode_ == Mode::RAMP || mode_ == Mode::TEST || sequence_) && now - last_print_ > 1.0) print_interface();
}

// ── modes and commands (mutex_ held) ───────────────────────────────────────
void FrictionTestNode::set_command(
  size_t i, double q_des, double dq_des, double ddq_des, int level, uint8_t segment)
{
  cmd_[i] = {q_des, dq_des, 0.0, kp_[i], kd_[i]};
  cmd_info_[i] = {ddq_des, level, segment};
}

void FrictionTestNode::publish_zero()
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data.assign(5 * joints_.size(), 0.0);
  joint_command_publisher_->publish(msg);
}

void FrictionTestNode::zero_torque(const std::string & reason)
{
  sequence_ = false;
  if (mode_ == Mode::TEST) abort_test();
  mode_ = Mode::ZERO;
  publish_zero();
  note(reason + ": zero torque", true);
}

void FrictionTestNode::abort_test()
{
  if (!test_.empty()) note(std::string(1, test_key_) + ": test aborted, data discarded", true);
  test_.clear();
}

void FrictionTestNode::start_joint_pd(const std::string & reason)
{
  ramp_start_ = q_;
  ramp_target_.resize(joints_.size());
  for (size_t i = 0; i < joints_.size(); ++i) {
    const char leg = joints_[i].back();  // joint_<side>_leg_<1..4>
    // Legs to the default pose; wheels (continuous) stay where they are.
    ramp_target_[i] = leg == '4' ? q_[i] : default_pose_[leg - '1'];
  }
  ramp_t0_ = now_s();
  mode_ = Mode::RAMP;
  note(reason + ": joint_pd, ramp to the default pose");
}

// ── trajectories ──────────────────────────────────────────────────────────
// Legs: from the hold pose to q_lim_low, then for each level of acc_fractions sweeps_per_level times
// q_lim_low -> q_lim_high -> q_lim_low with smooth strokes (smooth_stroke: velocity 0 at both ends and highest at
// mid-range, acceleration a triangle per half up to a_peak = fraction * acc_max.<joint>), pause_time at the ends,
// then back to the hold pose. q_lim_low/high = lower_bound/upper_bound of the yaml (checked inside the URDF limits
// by the caller). The
// strokes pass the same positions in both directions with opposite velocities, so gravity cancels in
// tau(q, +v) - tau(q, -v). No sinusoids.
// Wheels (continuous): constant speed steps +v / -v, v = speed_fractions of upper_bound / |lower_bound| (rad/s), of
// wheel_hold_time each, ramps with peak acceleration acc_max.<wheel>. Rows go to <csv_dir>/<joint_name>.csv when
// the test ends.
JointTrajectory FrictionTestNode::generate_joint_trajectory(
  const std::string & joint_name, double q_lim_low, double q_lim_high)
{
  const size_t i = std::find(joints_.begin(), joints_.end(), joint_name) - joints_.begin();
  const double q0 = hold_[i];
  Builder b(q0);
  std::ostringstream info;
  info << std::fixed << std::setprecision(2);
  if (joint_name.back() == '4') {
    for (size_t level = 0; level < speed_fractions_.size(); ++level) {
      for (const double v : {q_lim_high, q_lim_low}) {  // wheels: the bounds are the highest speeds
        b.add_velocity(speed_profile(speed_fractions_[level] * v, acc_max_.at(joint_name), wheel_hold_time_),
                       static_cast<int>(level), v > 0 ? SPEED_POS : SPEED_NEG);
        b.pause(pause_time_);
      }
    }
    info << joint_name << ": speed steps up to " << q_lim_low << " / " << q_lim_high << " rad/s";
    trajectory_info_ = info.str();
    return b.build(joint_name);
  }
  const double lo = q_lim_low, hi = q_lim_high, D = hi - lo, acc_max = acc_max_.at(joint_name);
  // From the hold pose to lower_bound and back: the same quintic ramp as joint_pd (key 1, home_ramp_time),
  // tracked by the PD.
  auto approach = [this](double qa, double qb) { return quintic(qa, qb, home_ramp_time_); };
  b.add(approach(q0, lo), -1, APPROACH);
  b.pause(pause_time_);
  info << joint_name << " [" << lo << ", " << hi << "] rad:";
  for (size_t level = 0; level < acc_fractions_.size(); ++level) {
    const double a_peak = acc_fractions_[level] * acc_max;
    for (int k = 0; k < sweeps_per_level_; ++k) {
      b.add(smooth_stroke(lo, hi, a_peak), static_cast<int>(level), SWEEP_POS);
      b.pause(pause_time_);
      b.add(smooth_stroke(hi, lo, a_peak), static_cast<int>(level), SWEEP_NEG);
      b.pause(pause_time_);
    }
    info << " a " << a_peak << " v " << std::sqrt(a_peak * D / 2) << " T " << stroke_time(D, a_peak) << ";";
  }
  b.add(approach(lo, q0), -1, RETURN);
  b.pause(pause_time_);
  trajectory_info_ = info.str();
  return b.build(joint_name);
}
