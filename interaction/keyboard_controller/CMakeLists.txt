cmake_minimum_required(VERSION 3.5)
project(keyboard_controller)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

include_directories()

ament_auto_add_executable(keyboard_controller_node src/main.cpp src/keyboard_controller.cpp)

# Keyboard radio: CRSF frames on a pty read by the real teleop_command node (same process).
# Plain add_executable so keyboard_controller_node does not link teleop_command.
add_executable(radio_keyboard src/radio_keyboard.cpp)
target_include_directories(radio_keyboard PRIVATE include)
set_target_properties(radio_keyboard PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
find_package(Eigen3 REQUIRED)
target_include_directories(radio_keyboard PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(radio_keyboard rclcpp std_msgs sensor_msgs geometry_msgs gazebo_msgs urdf
  ros_utils teleop_command ament_index_cpp)
target_link_libraries(radio_keyboard util)  # openpty
install(TARGETS radio_keyboard DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
