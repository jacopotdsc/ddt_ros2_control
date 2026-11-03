#include "ament_index_cpp/get_package_share_directory.hpp"
#include "friction_test/friction_test_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Load defaults first, then user arguments so hardware can override use_sim_time,
  // csv_dir, etc. Local NodeOptions arguments take precedence over rclcpp::init's
  // global arguments; loading only the YAML here would silently discard CLI parameters.
  rclcpp::NodeOptions options;
  std::vector<std::string> arguments{
    "--ros-args", "--params-file",
    ament_index_cpp::get_package_share_directory("friction_test") + "/config/friction_test.yaml"};
  // End the defaults' ROS argument block before appending the original CLI.
  arguments.emplace_back("--");
  arguments.insert(arguments.end(), argv + 1, argv + argc);
  options.arguments(arguments);
  auto node = std::make_shared<FrictionTestNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
