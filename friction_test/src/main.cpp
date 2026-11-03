#include "ament_index_cpp/get_package_share_directory.hpp"
#include "friction_test/friction_test_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Parameters from the installed config/friction_test.yaml (edit it in src/ and rebuild).
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "--params-file",
                     ament_index_cpp::get_package_share_directory("friction_test") +
                       "/config/friction_test.yaml"});
  auto node = std::make_shared<FrictionTestNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
