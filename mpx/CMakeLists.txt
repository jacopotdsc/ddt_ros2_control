cmake_minimum_required(VERSION 3.8)
project(mpx)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(mpx_node_cpp src/mpx_node.cpp)
target_include_directories(mpx_node_cpp PRIVATE include)
target_compile_features(mpx_node_cpp PRIVATE cxx_std_17)
target_link_libraries(mpx_node_cpp pybind11::embed)
ament_target_dependencies(mpx_node_cpp rclcpp ament_index_cpp gazebo_msgs nav_msgs sensor_msgs geometry_msgs std_msgs)
install(TARGETS mpx_node_cpp DESTINATION lib/${PROJECT_NAME})

add_executable(mpx_wbc_node src/mpx_wbc_node.cpp)
target_include_directories(mpx_wbc_node PRIVATE include)
target_compile_features(mpx_wbc_node PRIVATE cxx_std_17)
target_link_libraries(mpx_wbc_node pybind11::embed)
ament_target_dependencies(mpx_wbc_node rclcpp ament_index_cpp gazebo_msgs nav_msgs sensor_msgs geometry_msgs std_msgs)
install(TARGETS mpx_wbc_node DESTINATION lib/${PROJECT_NAME})

add_executable(mpx_llc_node src/mpx_llc_node.cpp)
target_include_directories(mpx_llc_node PRIVATE include)
target_compile_features(mpx_llc_node PRIVATE cxx_std_17)
ament_target_dependencies(mpx_llc_node rclcpp ament_index_cpp sensor_msgs std_msgs)
install(TARGETS mpx_llc_node DESTINATION lib/${PROJECT_NAME})

add_executable(benchmark_mpx_cpp src/benchmark_mpx_cpp.cpp)
target_include_directories(benchmark_mpx_cpp PRIVATE include)
target_compile_features(benchmark_mpx_cpp PRIVATE cxx_std_17)
target_link_libraries(benchmark_mpx_cpp pybind11::embed)
install(TARGETS benchmark_mpx_cpp DESTINATION lib/${PROJECT_NAME})

install(FILES scripts/mpx_cpp_solver.py scripts/mpx_benchmark.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  add_executable(test_cpp_buffers tests/test_cpp_buffers.cpp)
  target_include_directories(test_cpp_buffers PRIVATE include)
  target_link_libraries(test_cpp_buffers pybind11::embed)
  add_test(NAME cpp_numpy_buffers COMMAND test_cpp_buffers)
  add_test(NAME benchmark_metrics COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_metrics.py)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(PROGRAMS
  scripts/mpx_node_python.py
  scripts/benchmark_mpx.py
  scripts/test_mpx_ros.py
  scripts/benchmark_ros_feedback.py
  scripts/bench_mpx_cpp.sh
  scripts/check_mpx_rates.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  mpx_tita/mpx
  DESTINATION lib/${PROJECT_NAME}/python
)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
