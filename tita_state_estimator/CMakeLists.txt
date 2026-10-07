cmake_minimum_required(VERSION 3.8)
project(tita_state_estimator)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ros_utils REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

add_executable(state_estimator_node src/state_estimator_node.cpp)
target_include_directories(state_estimator_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(state_estimator_node pinocchio::pinocchio Eigen3::Eigen)
ament_target_dependencies(state_estimator_node rclcpp nav_msgs sensor_msgs std_msgs std_srvs ros_utils)

install(TARGETS state_estimator_node DESTINATION lib/${PROJECT_NAME})

ament_package()
