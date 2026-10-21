// Default parameters of the MPX nodes: `ros2 run mpx <node>` works without arguments.
// The installed mpx_node.yaml and rl_controller's tita/mpx.yaml (relative topics, robot
// setup) are loaded first; files and -p given on the command line or by a launch file
// come later and override them.
#pragma once
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace mpx {
class DefaultArguments {
 public:
  DefaultArguments(int argc, char **argv) {
    args_.push_back(argv[0]);
    args_.push_back("--ros-args");
    add("mpx", "config/mpx_node.yaml");
    add("rl_controller", "config/tita/mpx.yaml");
    for (int i = 1; i < argc; ++i) args_.push_back(argv[i]);
    for (auto &a : args_) argv_.push_back(a.data());
  }
  int argc() const { return int(argv_.size()); }
  char **argv() { return argv_.data(); }

 private:
  void add(const std::string &package, const std::string &file) {
    try {
      auto path = ament_index_cpp::get_package_share_directory(package) + "/" + file;
      if (!std::filesystem::exists(path)) return;
      args_.push_back("--params-file");
      args_.push_back(path);
    } catch (const std::exception &) {  // package not installed: nothing to add
    }
  }
  std::vector<std::string> args_;
  std::vector<char *> argv_;
};
}  // namespace mpx
