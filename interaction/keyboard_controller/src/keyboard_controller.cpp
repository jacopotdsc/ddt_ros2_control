#include "keyboard_controller/keyboard_controller.hpp"
#include <algorithm>

const std::vector<std::pair<char, std::string>> KeyboardControllerNode::fsm_state_mapping = {
{'0', "rl_0"}, {'1', "rl_1"}, {'2', "rl_2"},         {'3', "rl_3"},           {'4', "mpc"},
  {'5', "residual"}, {'6', "idle"}, {'7', "transform_up"}, {'8', "transform_down"}, {'9', "joint_pd"}};

KeyboardControllerNode::KeyboardControllerNode(const rclcpp::NodeOptions & options)
: Node("keyboard_controller", options)
{
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST).keep_last(10);
  this->cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
    ros_topic::manager_twist_command, rclcpp::SystemDefaultsQoS());
  this->posestamped_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
    ros_topic::manager_pose_command, rclcpp::SystemDefaultsQoS());
  this->fsm_goal_publisher_ =
    this->create_publisher<std_msgs::msg::String>(ros_topic::manager_key_command, qos);
  realtime_cmd_vel_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::Twist>>(
      cmd_vel_publisher_);
  realtime_posestamped_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>>(
      posestamped_publisher_);
  realtime_fsm_goal_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::String>>(fsm_goal_publisher_);
  mpx_keys_locked_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
    ros_topic::mpx_keys_locked, qos, [this](const std_msgs::msg::Bool::SharedPtr msg) {
      if (mpx_keys_locked_.exchange(msg->data) != msg->data) print_interface();
    });
  fsm_state_subscription_ = this->create_subscription<std_msgs::msg::String>(
    ros_topic::fsm_state, rclcpp::QoS(1).reliable().transient_local(),
    [this](const std_msgs::msg::String::SharedPtr msg) {
      {
        std::lock_guard<std::mutex> lock(fsm_state_mutex_);
        if (fsm_state_ == msg->data) return;
        fsm_state_ = msg->data;
      }
      // The keyboard mode always follows the FSM state it sees, whoever caused the change
      // (MPX stopped -> transform_down, end of the fold -> idle, command_watchdog, another
      // command source). RL states carry the policy name (e.g. rl_flat): rl_N is kept.
      const auto & state = msg->data;
      const bool known = std::any_of(
        fsm_state_mapping.begin(), fsm_state_mapping.end(),
        [&state](const auto & pair) { return pair.second == state; });
      const bool follow = known && state != fsm_goal_.data;
      if (follow) {
        fsm_goal_.data = state;
        twist_ = geometry_msgs::msg::Twist();
        mpx_keys_locked_ = false;
      }
      print_interface();
      if (follow) {
        std::cout << RED << "  FSM in " << state << " (not from this keyboard): mode set to "
                  << state << RESET << std::endl;
      }
    });

  this->timer_ =
    this->create_wall_timer(10ms, std::bind(&KeyboardControllerNode::PubCmdVelCallBack, this));
  std::shared_ptr<std::thread> read_key_thread =
    std::make_shared<std::thread>(&KeyboardControllerNode::ReadKeyThread, this);
  (*read_key_thread).detach();
  pose_.pose.position.z = MIN_HEIGHT;
  for (size_t i = 0; i < 3; i++) {
    rpy_[i] = 0.0;
  }
  fsm_goal_.data = "idle";
  print_interface();
}

void KeyboardControllerNode::print_interface()
{
  system("clear");
  std::cout << R"(
  -------------------------------------------------------
                      State Machines)"
            << std::endl;
  int count = 0;
  for (const auto & pair : fsm_state_mapping) {
    std::cout << "  " << pair.first << ": " << pair.second;
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
  // std::cout << R"(
  // -------------------------------------------------------
  // Moving Around    Postion Control    Orientation Control
  //       w                 ↑                u i o
  //     a s d             ← ↓ →              j k l
  // -------------------------------------------------------
  // )";
  std::cout << R"(
  -------------------------------------------------------
  )" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::setw(36) << "state machine now: " << std::setw(6) << GREEN << fsm_goal_.data
            << RESET;
  const std::string fsm_state = FsmState();
  if (!fsm_state.empty() && fsm_state != fsm_goal_.data) {
    std::cout << "  (FSM: " << fsm_state << ")";
  }
  std::cout << std::endl;
  if (fsm_goal_.data == "mpc" && !MpxRunning()) {
    std::cout << RED << "  MPX NOT RUNNING: MPC unavailable, the current state stays active"
              << RESET << std::endl;
  }
  if (mpx_keys_locked_) {
    std::cout << RED << "  MPX transition (compile / height ramp / RL handoff): motion keys locked"
              << RESET << std::endl;
  }
  std::cout << R"(
  -------------------------------------------------------
       "r": reset velocity, "y" : reset orientation
  )" << std::endl;
  // std::cout << std::setw(20) << "  speed scale:" << std::setw(6) << RED << speed_scale_ << RESET
  //           << "  pose scale:" << std::setw(6) << RED << pose_scale_ << RESET << std::endl
  //           << std::endl;
  std::cout << "  x_vel(ws):" << std::setw(6) << MAGENTA << twist_.linear.x
            << RESET
            // << "  y_vel(ad):" << std::setw(6) << MAGENTA << twist_.linear.y << RESET
            << " wz_vel(ad):" << std::setw(6) << MAGENTA << twist_.angular.z << RESET << std::endl;
  // std::cout << "  z_vel(zc):" << std::setw(6) << MAGENTA << twist_.linear.x << RESET
  std::cout << "  y_vel(←→):" << std::setw(6) << YELLOW << twist_.linear.y << RESET
            << "  z_vel(↑↓):" << std::setw(6) << YELLOW << twist_.linear.z << RESET << std::endl;

  std::cout << "  height(PgUp/PgDn):" << std::setw(6) << YELLOW << pose_.pose.position.z
            << RESET << " m" << std::endl;

  std::cout << "  pitch(jl):" << std::setw(6) << CYAN << rpy_[0] << RESET
            << "   roll(ik):" << std::setw(6) << CYAN << rpy_[1] << RESET
            << "   yaw(uo): " << std::setw(6) << CYAN << rpy_[2] << RESET << std::endl;
  std::cout << "  Space: all velocities and rpy to 0, height kept" << std::endl;
}

int KeyboardControllerNode::get_key()
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
  // std::cout << "ch : " << ch << std::endl;
  return ch;
}

void KeyboardControllerNode::ReadKeyThread()
{
  while (rclcpp::ok()) {
    int key = get_key();
    // Nobody releases the lock if MPX is not running (e.g. never started on the robot).
    const bool mpx_running = MpxRunning();
    if (!mpx_running) mpx_keys_locked_ = false;
    const auto mode = std::find_if(
      fsm_state_mapping.begin(), fsm_state_mapping.end(),
      [key](const auto & pair) { return pair.first == key; });
    if (mode != fsm_state_mapping.end()) {
      const std::string & goal = mode->second;
      // The FSM may still be in mpc after another key (e.g. an unknown rl_N).
      const bool in_mpc = FsmState() == "mpc";
      const bool from_mpc = fsm_goal_.data == "mpc" || in_mpc;
      // The FSM never leaves mpc for these modes: keep showing mpc.
      if (in_mpc && (goal == "transform_up" || goal == "residual")) continue;
      if (goal == "mpc" && !from_mpc) {
        // MPX starts at rest and ramps to h_mpc: mirror it and lock the motion keys
        // until MPX releases them.
        twist_ = geometry_msgs::msg::Twist();
        pose_.pose.position.z = MPC_START_HEIGHT;
        mpx_keys_locked_ = mpx_running;
      } else if ((goal.rfind("rl_", 0) == 0 || goal == "transform_down") && from_mpc) {
        // MPX stops the robot and lowers it before handing off (keys stay locked meanwhile).
        twist_ = geometry_msgs::msg::Twist();
      } else if (goal != "mpc") {
        mpx_keys_locked_ = false;
      }
      fsm_goal_.data = goal;
      print_interface();
      continue;
    }
    // While MPX is in a transition only Space and Ctrl+C act: the other keys are read
    // (so escape sequences are consumed) and then undone.
    const bool locked = mpx_keys_locked_ && key != ' ';
    const auto twist = twist_;
    const auto pose = pose_;
    const std::array<double, 3> rpy{rpy_[0], rpy_[1], rpy_[2]};
    switch (key) {
      case 'w':
        twist_.linear.x += STEP_ACCL_X * speed_scale_;
        break;
      case 's':
        twist_.linear.x -= STEP_ACCL_X * speed_scale_;
        break;
      // case 'a':
      //   twist_.linear.y += STEP_ACCL_X * speed_scale_;
      //   break;
      // case 'd':
      //   twist_.linear.y -= STEP_ACCL_X * speed_scale_;
      //   break;
      case 'a':
        twist_.angular.z += STEP_ACCL_W * speed_scale_;
        break;
      case 'd':
        twist_.angular.z -= STEP_ACCL_W * speed_scale_;
        break;
      case 'r':
        twist_.linear.x = 0.0;
        twist_.linear.y = 0.0;
        twist_.linear.z = 0.0;
        twist_.angular.x = 0.0;
        twist_.angular.y = 0.0;
        twist_.angular.z = 0.0;
        break;
      // case '+':
      //   speed_scale_ += 0.1;
      //   break;
      // case '-':
      //   speed_scale_ -= 0.1;
      //   break;
      // case '*':
      //   pose_scale_ += 0.1;
      //   break;
      // case '/':
      //   pose_scale_ -= 0.1;
      //   break;
      case '\033':
        if (get_key() == '[') {
          switch (get_key()) {
            case 'A':  // Up
              twist_.linear.z += STEP_ACCL_X * speed_scale_ * 0.1;
              break;
            case 'B':  // Down
              twist_.linear.z -= STEP_ACCL_X * speed_scale_ * 0.1;
              break;
            case 'C':  // Right
              twist_.linear.y -= STEP_ACCL_X * speed_scale_;
              break;
            case 'D':  // Left
              twist_.linear.y += STEP_ACCL_X * speed_scale_;
              break;
            case '5':  // Page Up: ESC [ 5 ~
              if (get_key() == '~') {
                pose_.pose.position.z += STEP_HEIGHT * pose_scale_;
              }
              break;
            case '6':  // Page Down: ESC [ 6 ~
              if (get_key() == '~') {
                pose_.pose.position.z -= STEP_HEIGHT * pose_scale_;
              }
              break;
            default:
              break;
          }
        }
        break;
      case 'i':
        rpy_[1] += STEP_ORIENTATION * pose_scale_;
        break;
      case 'k':
        rpy_[1] -= STEP_ORIENTATION * pose_scale_;
        break;
      case 'j':
        rpy_[0] -= STEP_ORIENTATION * pose_scale_;
        break;
      case 'l':
        rpy_[0] += STEP_ORIENTATION * pose_scale_;
        break;
      case 'u':
        rpy_[2] -= STEP_ORIENTATION * pose_scale_;
        break;
      case 'o':
        rpy_[2] += STEP_ORIENTATION * pose_scale_;
        break;
      case 'y':
        rpy_[0] = rpy_[1] = rpy_[2] = 0;
        break;
      case ' ':  // Space: stop everything, keep the commanded height
        twist_ = geometry_msgs::msg::Twist();
        rpy_[0] = rpy_[1] = rpy_[2] = 0;
        pose_.pose.position.y = 0.0;
        break;
      case '\x03':  // Ctrl+C: quit
        rclcpp::shutdown();
        return;
      default:
        break;
    }
    if (locked) {
      twist_ = twist;
      pose_ = pose;
      std::copy(rpy.begin(), rpy.end(), rpy_);
    }
    speed_scale_ = clamp(speed_scale_, 0.1, 4.0);
    pose_scale_ = clamp(pose_scale_, 0.1, 4.0);
    twist_.linear.x = clamp(twist_.linear.x, -MAX_VEL_X, MAX_VEL_X);
    twist_.linear.y = clamp(twist_.linear.y, -MAX_VEL_X, MAX_VEL_X);
    twist_.linear.z = clamp(twist_.linear.z, -MAX_VEL_X, MAX_VEL_X);
    twist_.angular.z = clamp(twist_.angular.z, -MAX_VEL_W, MAX_VEL_W);
    for (size_t i = 0; i < 3; i++) {
      rpy_[i] = clamp(rpy_[i], -MAX_ORIENTATION, MAX_ORIENTATION);
    }
    tf2::Quaternion q;
    q.setRPY(rpy_[0], rpy_[1], rpy_[2]);
    pose_.pose.orientation.x = q.x();
    pose_.pose.orientation.y = q.y();
    pose_.pose.orientation.z = q.z();
    pose_.pose.orientation.w = q.w();
    pose_.pose.position.y = clamp(pose_.pose.position.y, -MAX_POSITION, MAX_POSITION);
    pose_.pose.position.z = clamp(pose_.pose.position.z, MIN_HEIGHT, MAX_HEIGHT);
    print_interface();
  }
}

std::string KeyboardControllerNode::FsmState()
{
  std::lock_guard<std::mutex> lock(fsm_state_mutex_);
  return fsm_state_;
}

bool KeyboardControllerNode::MpxRunning()
{
  return this->count_publishers(mpx_keys_locked_subscription_->get_topic_name()) > 0;
}

void KeyboardControllerNode::CheckRlControllerAlive()
{
  bool alive = false;
  for (const auto & info : this->get_subscriptions_info_by_topic(fsm_goal_publisher_->get_topic_name())) {
    const auto & name = info.node_name();
    const std::string suffix = "_rl_controller";
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      alive = true;
    }
  }
  if (rl_controller_seen_ && !alive && fsm_goal_.data != "idle") {
    // A restarted controller must not receive the old mode (e.g. rl_0 or mpc).
    fsm_goal_.data = "idle";
    mpx_keys_locked_ = false;
    print_interface();
    std::cout << RED << "  rl_controller gone: state set to idle" << RESET << std::endl;
  }
  rl_controller_seen_ = alive;
}

void KeyboardControllerNode::PubCmdVelCallBack()
{
  if (++alive_check_count_ >= 50) {  // every 0.5 s
    alive_check_count_ = 0;
    CheckRlControllerAlive();
  }
  if (realtime_cmd_vel_publisher_ && realtime_cmd_vel_publisher_->trylock()) {
    realtime_cmd_vel_publisher_->msg_ = twist_;
    realtime_cmd_vel_publisher_->unlockAndPublish();
  }
  if (realtime_posestamped_publisher_ && realtime_posestamped_publisher_->trylock()) {
    realtime_posestamped_publisher_->msg_.header.stamp = this->now();
    realtime_posestamped_publisher_->msg_ = pose_;
    realtime_posestamped_publisher_->unlockAndPublish();
  }
  if (realtime_fsm_goal_publisher_ && realtime_fsm_goal_publisher_->trylock()) {
    realtime_fsm_goal_publisher_->msg_ = fsm_goal_;
    realtime_fsm_goal_publisher_->unlockAndPublish();
  }
}
