cmake_minimum_required(VERSION 3.8)
project(friction_test)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_executable(friction_test_node src/main.cpp src/friction_test_node.cpp)

install(DIRECTORY launch urdf config DESTINATION share/${PROJECT_NAME})

ament_auto_package()
