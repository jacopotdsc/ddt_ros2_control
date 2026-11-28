// Keyboard radio for the simulation: CRSF frames on a pty, read by the real teleop_command node
// running in this process. See include/keyboard_controller/radio_keyboard.hpp.
#include "keyboard_controller/radio_keyboard.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rcl/logging.h"
#include "rcl/logging_rosout.h"
#include "rcutils/logging.h"
#include "ros_utils/topic_names.hpp"
#include "teleop_command/teleop_command_node.hpp"

namespace keyboard_controller
{
namespace
{
// Channel indices and switch values as teleop_command reads them (joy_axis_t, CRSF limits), so
// they follow teleop_command_node.hpp / elrs_emwave_device.hpp.
using teleop_command::ADJUST_HEIGHT;
using teleop_command::FORWARD;
using teleop_command::JUMP;
using teleop_command::ROLL;
using teleop_command::SPEED_REGULATION;
using teleop_command::STAND_AND_CROUCH;
using teleop_command::YAW;
constexpr uint16_t kLow = CRSF_CHANNEL_MIN, kMid = CRSF_CHANNEL_MID, kHigh = CRSF_CHANNEL_MAX;
constexpr uint16_t kButtonNone = 988;  // CH9: no mode button (teleop_command: (ch - 988) / 4)
constexpr int kButtonChannel = 8;
constexpr double kStickStep = 0.1;
constexpr auto kRightStickPush = std::chrono::milliseconds(500);
// JUMP is spring-loaded: pulled while Space is held. The terminal repeats a held key only after
// ~0.5 s, so each Space keeps it pulled for 0.6 s (a tap = 0.6 s pulled).
constexpr auto kJumpHold = std::chrono::milliseconds(600);
constexpr auto kFramePeriod = std::chrono::milliseconds(10);  // 100 Hz
constexpr auto kDrawPeriod = std::chrono::milliseconds(50);   // screen refresh, when changed
const char * const kRed = "\033[1;31m";
const char * const kGreen = "\033[1;32m";
const char * const kReset = "\033[0m";

const char * const kHelp[] = {
  "  7 / 8        CH5  STAND / CROUCH",
  "  1 / 2 / 3    CH8  SPEED low (none) / medium / high",
  "  PgUp / PgDn  CH6  HEIGHT up / down (low, medium, high)",
  "  Space        CH7  JUMP pulled while held, a tap = 0.6 s (safety -> idle)",
  "  q            right stick pushed for 0.5 s (safety -> idle)",
  "  w / s        left stick forward / back (vx), steps of 10 %",
  "  a / d        left stick left / right (wz), steps of 10 %",
  "  x            left stick to the centre",
  "  Ctrl+C       quit",
};
// Same keys on 3 lines, for terminals at least kWideHelp columns wide.
const char * const kHelpWide[] = {
  "  7/8 CH5 STAND/CROUCH   1/2/3 CH8 SPEED low(none)/medium/high   PgUp/PgDn CH6 HEIGHT",
  "  Space CH7 JUMP (held, tap = 0.6 s)   q right stick 0.5 s   w/s vx, a/d wz (10 %)   x centre",
  "  Ctrl+C quit.  Arm: JUMP released, sticks centred, SPEED low, HEIGHT low, CROUCH",
};
constexpr size_t kWideHelp = 100;

struct termios g_terminal;
bool g_terminal_saved = false;

// ROS log output of this process -> the screen (last lines) and /rosout, not the console.
std::atomic<RadioKeyboard *> g_radio{nullptr};
rcutils_logging_output_handler_t g_previous_handler = nullptr;

void log_handler(
  const rcutils_log_location_t * location, int severity, const char * name,
  rcutils_time_point_value_t timestamp, const char * format, va_list * args)
{
  RadioKeyboard * radio = g_radio.load();
  if (radio == nullptr) {
    if (g_previous_handler) g_previous_handler(location, severity, name, timestamp, format, args);
    return;
  }
  if (rcl_logging_rosout_enabled()) {
    va_list copy;
    va_copy(copy, *args);
    rcl_logging_rosout_output_handler(location, severity, name, timestamp, format, &copy);
    va_end(copy);
  }
  char message[1024];
  va_list copy;
  va_copy(copy, *args);
  vsnprintf(message, sizeof(message), format, copy);
  va_end(copy);
  const char * level = severity >= RCUTILS_LOG_SEVERITY_FATAL   ? "FATAL"
                       : severity >= RCUTILS_LOG_SEVERITY_ERROR ? "ERROR"
                       : severity >= RCUTILS_LOG_SEVERITY_WARN  ? "WARN"
                       : severity >= RCUTILS_LOG_SEVERITY_INFO  ? "INFO"
                                                                : "DEBUG";
  time_t seconds = static_cast<time_t>(timestamp / 1000000000LL);
  struct tm local;
  localtime_r(&seconds, &local);
  char clock[16];
  strftime(clock, sizeof(clock), "%H:%M:%S", &local);
  radio->add_log(clock, std::string(level) + " [" + (name ? name : "") + "] " + message);
}

uint8_t crc8(const uint8_t * data, size_t length)
{
  uint8_t crc = 0;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x80) ? static_cast<uint8_t>((crc << 1) ^ 0xD5) : static_cast<uint8_t>(crc << 1);
    }
  }
  return crc;
}

// RC channels frame as elrs_emwave_device.cpp expects: address, length 24, type 0x16,
// 16 channels x 11 bits (22 bytes), CRC8 over type + payload.
std::array<uint8_t, 26> frame(const std::array<uint16_t, 16> & channels)
{
  std::array<uint8_t, 26> f{};
  f[0] = 0xC8;
  f[1] = 24;
  f[2] = 0x16;
  uint32_t buffer = 0;
  int bits = 0;
  size_t out = 3;
  for (uint16_t ch : channels) {
    buffer |= static_cast<uint32_t>(ch & 0x07FF) << bits;
    bits += 11;
    while (bits >= 8) {
      f[out++] = static_cast<uint8_t>(buffer & 0xFF);
      buffer >>= 8;
      bits -= 8;
    }
  }
  f[25] = crc8(&f[2], 23);
  return f;
}

const char * switch3(uint16_t value, const char * low, const char * mid, const char * high)
{
  return value == kLow ? low : (value == kHigh ? high : mid);
}
}  // namespace

RadioKeyboard::RadioKeyboard(const rclcpp::NodeOptions & options) : Node("radio_keyboard", options)
{
  channels_.fill(kMid);
  channels_[STAND_AND_CROUCH] = kLow;  // CROUCH
  channels_[ADJUST_HEIGHT] = kLow;
  channels_[JUMP] = kLow;              // released
  channels_[SPEED_REGULATION] = kLow;
  channels_[kButtonChannel] = kButtonNone;
  channels_[FORWARD] = stick(0.);
  channels_[YAW] = stick(0.);

  char name[128];
  if (openpty(&master_, &slave_, name, nullptr, nullptr) != 0) {
    throw std::runtime_error(std::string("radio_keyboard: openpty failed: ") + strerror(errno));
  }
  device_ = name;
  struct termios raw;
  tcgetattr(slave_, &raw);
  cfmakeraw(&raw);
  tcsetattr(slave_, TCSANOW, &raw);
  fcntl(master_, F_SETFL, fcntl(master_, F_GETFL) | O_NONBLOCK);

  fsm_state_subscription_ = create_subscription<std_msgs::msg::String>(
    ros_topic::fsm_state, rclcpp::QoS(1).reliable().transient_local(),
    [this](const std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      fsm_state_ = msg->data;
      dirty_ = true;
    });
  // joy.axes as teleop_command reads the channels: arming check of teleop_command on them.
  joy_subscription_ = create_subscription<sensor_msgs::msg::Joy>(
    ros_topic::joy, rclcpp::QoS(10), [this](const sensor_msgs::msg::Joy::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (msg->axes != axes_) dirty_ = true;
      axes_ = msg->axes;
    });
  // COM height: command (teleop_command cmd_pose), MPX estimate, ground truth.
  pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
    ros_topic::manager_pose_command, rclcpp::QoS(10),
    [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (msg->pose.position.z != height_command_) dirty_ = true;
      height_command_ = msg->pose.position.z;
    });
  com_subscription_ = create_subscription<geometry_msgs::msg::PointStamped>(
    ros_topic::mpx_estimate_com, rclcpp::SensorDataQoS(),
    [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      com_height_ = msg->point.z;
      com_received_ = std::chrono::steady_clock::now();
    });
  description_subscription_ = create_subscription<std_msgs::msg::String>(
    ros_topic::robot_description, rclcpp::QoS(1).reliable().transient_local(),
    [this](const std_msgs::msg::String::SharedPtr msg) {
      auto model = std::make_shared<urdf::Model>();
      if (!model->initString(msg->data)) {
        RCLCPP_WARN(get_logger(), "robot_description: invalid URDF, no ground-truth COM");
        return;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      urdf_ = model;
    });
  joint_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
    ros_topic::joint_states, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
        joint_positions_[msg->name[i]] = msg->position[i];
      }
      joints_received_ = std::chrono::steady_clock::now();
    });
  gazebo_model_ = declare_parameter<std::string>("gazebo_model", "tita");
  model_subscription_ = create_subscription<gazebo_msgs::msg::ModelStates>(
    ros_topic::gazebo_model_states, rclcpp::SensorDataQoS(),
    [this](const gazebo_msgs::msg::ModelStates::SharedPtr msg) {
      for (size_t i = 0; i < msg->name.size() && i < msg->pose.size(); ++i) {
        if (msg->name[i] != gazebo_model_) continue;
        const auto & p = msg->pose[i];
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
        pose.linear() =
          Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
            .normalized()
            .toRotationMatrix();
        std::lock_guard<std::mutex> lock(mutex_);
        base_pose_ = pose;
        base_received_ = std::chrono::steady_clock::now();
        return;
      }
    });
  // COM values change continuously: refresh the screen 4 times a second.
  com_timer_ = create_wall_timer(std::chrono::milliseconds(250), [this] { dirty_ = true; });
}

double RadioKeyboard::ground_truth_com_z() const
{
  const auto now = std::chrono::steady_clock::now();
  const auto fresh = std::chrono::milliseconds(500);
  if (!urdf_ || now - base_received_ > fresh || now - joints_received_ > fresh) {
    return std::nan("");
  }
  auto to_eigen = [](const urdf::Pose & p) {
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
    double x, y, z, w;
    p.rotation.getQuaternion(x, y, z, w);
    t.linear() = Eigen::Quaterniond(w, x, y, z).toRotationMatrix();
    return t;
  };
  double mass = 0.;
  Eigen::Vector3d moment = Eigen::Vector3d::Zero();
  // Forward kinematics from the root link (base_link, the Gazebo model pose).
  std::vector<std::pair<urdf::LinkConstSharedPtr, Eigen::Isometry3d>> stack{
    {urdf_->getRoot(), base_pose_}};
  while (!stack.empty()) {
    auto [link, pose] = stack.back();
    stack.pop_back();
    if (link->inertial) {
      mass += link->inertial->mass;
      moment += link->inertial->mass * (pose * to_eigen(link->inertial->origin)).translation();
    }
    for (const auto & joint : link->child_joints) {
      Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
      auto q = joint_positions_.find(joint->name);
      const double value = q == joint_positions_.end() ? 0. : q->second;
      const Eigen::Vector3d axis(joint->axis.x, joint->axis.y, joint->axis.z);
      if (joint->type == urdf::Joint::REVOLUTE || joint->type == urdf::Joint::CONTINUOUS) {
        motion.linear() = Eigen::AngleAxisd(value, axis.normalized()).toRotationMatrix();
      } else if (joint->type == urdf::Joint::PRISMATIC) {
        motion.translation() = value * axis;
      }
      auto child = urdf_->getLink(joint->child_link_name);
      if (child) {
        stack.push_back({child, pose * to_eigen(joint->parent_to_joint_origin_transform) * motion});
      }
    }
  }
  return mass > 0. ? moment.z() / mass : std::nan("");
}

RadioKeyboard::~RadioKeyboard()
{
  stop();
  if (master_ >= 0) close(master_);
  if (slave_ >= 0) close(slave_);
}

void RadioKeyboard::stop()
{
  stop_ = true;
  if (writer_.joinable()) writer_.join();
  if (reader_.joinable()) reader_.join();
  if (drawer_.joinable()) drawer_.join();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    teleop_ = nullptr;
  }
  if (g_terminal_saved) {
    tcsetattr(STDIN_FILENO, TCSANOW, &g_terminal);
    g_terminal_saved = false;
    std::printf("\033[?25h\n");  // cursor back
    std::fflush(stdout);
  }
}

uint16_t RadioKeyboard::stick(double position)
{
  // teleop_command: normalized = ((ch - 172) / 1638) * 2 - 1.
  position = std::clamp(position, -1.0, 1.0);
  return static_cast<uint16_t>(std::lround(172. + (position + 1.) / 2. * 1638.));
}

void RadioKeyboard::add_log(const std::string & time, const std::string & text)
{
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (!logs_.empty() && logs_.back().text == text) {
      logs_.back().time = time;
      ++logs_.back().count;
    } else {
      logs_.push_back({time, text, 1});
      while (logs_.size() > kLogLines) logs_.pop_front();
    }
  }
  dirty_ = true;
}

void RadioKeyboard::start()
{
  writer_ = std::thread(&RadioKeyboard::write_loop, this);
  reader_ = std::thread(&RadioKeyboard::key_loop, this);
  drawer_ = std::thread(&RadioKeyboard::draw_loop, this);
}

void RadioKeyboard::write_loop()
{
  auto next = std::chrono::steady_clock::now();
  while (!stop_ && rclcpp::ok()) {
    std::array<uint16_t, 16> channels;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = std::chrono::steady_clock::now();
      channels_[ROLL] = now < right_stick_until_ ? kHigh : kMid;
      channels_[JUMP] = now < jump_until_ ? kHigh : kLow;
      channels = channels_;
    }
    const auto f = frame(channels);
    if (write(master_, f.data(), f.size()) < 0 && errno == EAGAIN) {
      // Input queue full (teleop_command not reading yet): drop the stale frames, otherwise no
      // new data -> no SIGIO -> teleop_command never reads.
      tcflush(slave_, TCIFLUSH);
    }
    next += kFramePeriod;
    std::this_thread::sleep_until(next);
  }
}

void RadioKeyboard::key_loop()
{
  if (tcgetattr(STDIN_FILENO, &g_terminal) == 0) {
    g_terminal_saved = true;
    struct termios t = g_terminal;
    t.c_lflag &= ~(ICANON | ECHO);  // ISIG kept: Ctrl+C stops ROS
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
  }
  // teleop_command's SIGIO handler has no SA_RESTART: poll / read may return EINTR.
  auto read_byte = [](int timeout_ms) -> int {
    struct pollfd p{STDIN_FILENO, POLLIN, 0};
    if (poll(&p, 1, timeout_ms) <= 0) return -1;
    unsigned char c;
    return read(STDIN_FILENO, &c, 1) == 1 ? c : -1;
  };
  while (!stop_ && rclcpp::ok()) {
    int c = read_byte(100);
    if (c < 0) continue;
    if (c == 27) {  // escape sequence: PgUp "\e[5~", PgDn "\e[6~"
      int c1 = read_byte(50), c2 = read_byte(50), c3 = read_byte(50);
      if (c1 == '[' && c3 == '~' && (c2 == '5' || c2 == '6')) c = c2 == '5' ? 1000 : 1001;
      else continue;
    }
    if (!handle_key(c)) break;
    dirty_ = true;
  }
  if (g_terminal_saved) tcsetattr(STDIN_FILENO, TCSANOW, &g_terminal);
}

void RadioKeyboard::draw_loop()
{
  std::printf("\033[?25l");  // hide the cursor
  while (!stop_ && rclcpp::ok()) {
    bool pushed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = std::chrono::steady_clock::now();
      pushed = now < right_stick_until_ + kDrawPeriod * 2 || now < jump_until_ + kDrawPeriod * 2;
    }
    if (pushed) dirty_ = true;  // show the right stick coming back to the centre
    if (dirty_.exchange(false)) draw();
    std::this_thread::sleep_for(kDrawPeriod);
  }
}

bool RadioKeyboard::handle_key(int key)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto & height = channels_[ADJUST_HEIGHT];
  switch (key) {
    case '7': channels_[STAND_AND_CROUCH] = kHigh; break;
    case '8': channels_[STAND_AND_CROUCH] = kLow; break;
    case '1': channels_[SPEED_REGULATION] = kLow; break;
    case '2': channels_[SPEED_REGULATION] = kMid; break;
    case '3': channels_[SPEED_REGULATION] = kHigh; break;
    case 1000: height = height == kLow ? kMid : kHigh; break;  // PgUp
    case 1001: height = height == kHigh ? kMid : kLow; break;  // PgDn
    case ' ': jump_until_ = std::chrono::steady_clock::now() + kJumpHold; break;
    case 'q': right_stick_until_ = std::chrono::steady_clock::now() + kRightStickPush; break;
    case 'w': forward_ = std::min(1., forward_ + kStickStep); break;
    case 's': forward_ = std::max(-1., forward_ - kStickStep); break;
    case 'a': turn_ = std::min(1., turn_ + kStickStep); break;
    case 'd': turn_ = std::max(-1., turn_ - kStickStep); break;
    case 'x': forward_ = turn_ = 0.; break;
    default: return true;
  }
  forward_ = std::round(forward_ * 10.) / 10.;
  turn_ = std::round(turn_ * 10.) / 10.;
  // teleop_command: vx = +normalized(CH3) * max, wz = -normalized(CH4) * max.
  channels_[FORWARD] = stick(forward_);
  channels_[YAW] = stick(-turn_);
  return true;
}

void RadioKeyboard::draw()
{
  struct winsize w;
  const bool size_known = ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0;
  const size_t width = size_known && w.ws_col > 20 ? w.ws_col : 120;
  const size_t rows = size_known && w.ws_row > 5 ? w.ws_row : 40;
  std::vector<std::pair<std::string, const char *>> lines;  // text, ANSI colour (nullptr = none)
  lines.push_back({"radio_keyboard: simulated radio on " + device_ +
                     ", read by teleop_command (same process)",
                   nullptr});
  lines.push_back({"", nullptr});
  lines.push_back({" Keys", nullptr});
  if (width >= kWideHelp) {
    for (const char * h : kHelpWide) lines.push_back({h, nullptr});
  } else {
    for (const char * h : kHelp) lines.push_back({h, nullptr});
    lines.push_back(
      {"  Arm: JUMP released, sticks centred, SPEED low, HEIGHT low, CROUCH", nullptr});
  }
  lines.push_back({"", nullptr});
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Not armed: the sticks / switches teleop_command wants elsewhere, in red.
    const bool armed = teleop_ != nullptr && teleop_->armed();
    std::vector<int> blockers;
    if (!armed && !axes_.empty()) blockers = teleop_command::TeleopCmdNode::arming_blockers(axes_);
    auto red = [&](std::initializer_list<int> indices) {
      for (int i : indices) {
        if (std::find(blockers.begin(), blockers.end(), i) != blockers.end()) return true;
      }
      return false;
    };
    auto metres = [](double v) {
      char b[32];
      if (std::isfinite(v)) std::snprintf(b, sizeof(b), "%.3f m", v);
      else std::snprintf(b, sizeof(b), "-");
      return std::string(b);
    };
    const auto now = std::chrono::steady_clock::now();
    const double estimate =
      now - com_received_ < std::chrono::milliseconds(500) ? com_height_ : std::nan("");
    lines.push_back({armed ? " Radio: ARMED" : " Radio: NOT ARMED (switch the red ones)",
                     armed ? kGreen : kRed});
    lines.push_back({" Channels", nullptr});
    lines.push_back({std::string("  CH5 STAND/CROUCH  ") +
                       (channels_[STAND_AND_CROUCH] == kHigh ? "STAND" : "CROUCH"),
                     red({STAND_AND_CROUCH}) ? kRed : nullptr});
    lines.push_back({std::string("  CH6 HEIGHT        ") +
                       switch3(channels_[ADJUST_HEIGHT], "low", "medium", "high") + " (" +
                       metres(height_command_) + "), mpx est " + metres(estimate) +
                       ", gt: " + metres(ground_truth_com_z()),
                     red({ADJUST_HEIGHT}) ? kRed : nullptr});
    lines.push_back({std::string("  CH7 JUMP          ") +
                       (now < jump_until_ ? "PULLED" : "released"),
                     red({JUMP}) ? kRed : nullptr});
    lines.push_back({std::string("  CH8 SPEED         ") +
                       switch3(channels_[SPEED_REGULATION], "low", "medium", "high"),
                     red({SPEED_REGULATION}) ? kRed : nullptr});
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "  left stick        vx %+.1f   wz %+.1f", forward_, turn_);
    lines.push_back({buffer, red({FORWARD, YAW}) ? kRed : nullptr});
    lines.push_back({std::string("  right stick       ") +
                       (now < right_stick_until_ ? "PUSHED" : "centre"),
                     red({ROLL, teleop_command::PITCH}) ? kRed : nullptr});
    lines.push_back({"", nullptr});
    lines.push_back({" FSM state: " + (fsm_state_.empty() ? std::string("- (no fsm_state yet)")
                                                          : fsm_state_),
                     nullptr});
  }
  lines.push_back({"", nullptr});
  lines.push_back({" Last logs", nullptr});
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    for (const auto & l : logs_) {
      lines.push_back({"  " + l.time + " " + l.text +
                         (l.count > 1 ? " (x" + std::to_string(l.count) + ")" : ""),
                       nullptr});
    }
    for (size_t i = logs_.size(); i < kLogLines; ++i) lines.push_back({"", nullptr});
  }
  // Never more lines than the terminal has (otherwise every redraw scrolls): drop the empty
  // lines first, then the oldest log lines.
  for (size_t k = lines.size(); lines.size() > rows && k-- > 0;) {
    if (lines[k].first.empty()) lines.erase(lines.begin() + k);
  }
  auto logs_header = std::find_if(lines.begin(), lines.end(),
                                  [](const auto & l) { return l.first == " Last logs"; });
  while (lines.size() > rows && logs_header != lines.end() && logs_header + 1 != lines.end()) {
    lines.erase(logs_header + 1);  // oldest log line (the header iterator stays valid)
  }
  if (lines.size() > rows) lines.resize(rows);
  // Redraw in place: home, each line cleared to its end, no newline after the last one.
  std::string screen = "\033[H";
  for (size_t k = 0; k < lines.size(); ++k) {
    auto & [text, colour] = lines[k];
    if (text.size() > width - 1) text = text.substr(0, width - 1);  // cut before the colour codes
    screen += colour ? std::string(colour) + text + kReset : text;
    screen += "\033[K";
    if (k + 1 < lines.size()) screen += "\n";
  }
  screen += "\033[J";  // clear what is below
  std::fwrite(screen.data(), 1, screen.size(), stdout);
  std::fflush(stdout);
}
}  // namespace keyboard_controller

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto radio = std::make_shared<keyboard_controller::RadioKeyboard>();
    // ROS logs of this process (teleop_command included) go to the screen, not the console.
    keyboard_controller::g_previous_handler = rcutils_logging_get_output_handler();
    keyboard_controller::g_radio = radio.get();
    rcutils_logging_set_output_handler(keyboard_controller::log_handler);
    // The real teleop_command node, with its own parameters, reading the simulated radio.
    const std::string params =
      ament_index_cpp::get_package_share_directory("teleop_command") + "/config/param.yaml";
    auto options = rclcpp::NodeOptions()
                     .automatically_declare_parameters_from_overrides(true)
                     .arguments({"--ros-args", "--params-file", params, "-p",
                                 "uart_interface:=" + radio->device()});
    auto teleop = std::make_shared<teleop_command::TeleopCmdNode>(options);
    teleop->init();
    radio->set_teleop(teleop.get());
    radio->start();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(radio);
    executor.add_node(teleop);
    executor.spin();
    radio->stop();  // threads stop before the teleop_command node is destroyed
    keyboard_controller::g_radio = nullptr;
    rcutils_logging_set_output_handler(keyboard_controller::g_previous_handler);
    rclcpp::shutdown();
    // teleop_command's CRSF reader threads are detached and use their object after it is
    // destroyed: leave without destructors (the terminal is already restored).
    std::fflush(stdout);
    std::_Exit(0);
  } catch (const std::exception & e) {
    keyboard_controller::g_radio = nullptr;
    rcutils_logging_set_output_handler(keyboard_controller::g_previous_handler);
    std::fprintf(stderr, "radio_keyboard: %s\n", e.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
