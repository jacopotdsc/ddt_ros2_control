cmake_minimum_required(VERSION 3.5)
project(gazebo_bridge)

# Simulation only: on the robot (aarch64) Gazebo, Webots and MuJoCo are not installed, so the
# package is registered empty instead of failing the whole workspace build.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "${PROJECT_NAME}: skipped on ${CMAKE_SYSTEM_PROCESSOR} (simulation only)")
  find_package(ament_cmake REQUIRED)
  ament_package()
  return()
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -O3")

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

include_directories()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/gazebo_bridge.cpp
)

pluginlib_export_plugin_description_file(gazebo_ros2_control gazebo_bridge_plugin.xml)

# gzclient plugin limiting the GUI render rate (loaded by sim_gazebo.launch.py).
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
add_library(tita_gui_render_rate SHARED src/gui_render_rate.cpp)
target_include_directories(tita_gui_render_rate PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(tita_gui_render_rate ${GAZEBO_LIBRARIES} Qt5::Core Qt5::Widgets)
install(TARGETS tita_gui_render_rate LIBRARY DESTINATION lib)

## Install
install(DIRECTORY config launch worlds DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
