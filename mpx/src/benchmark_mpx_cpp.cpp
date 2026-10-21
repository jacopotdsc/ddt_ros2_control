// Standalone timing of the production C++ -> Python MPC/WBC path.
#include "mpx/python_buffers.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cerrno>
#include <thread>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
namespace {
struct Stats {
  std::vector<double> ms;
  size_t overruns = 0, invalid = 0;
};
void report(const char *name, Stats &s, double elapsed, double period_ms) {
  if (s.ms.empty()) { std::cout << name << ": no samples\n"; return; }
  double mean = std::accumulate(s.ms.begin(), s.ms.end(), 0.) / s.ms.size();
  auto v = s.ms;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { double x = p * (v.size() - 1); auto lo = size_t(x);
    return v[lo] + (v[std::min(lo + 1, v.size() - 1)] - v[lo]) * (x - lo); };
  std::cout << std::fixed << std::setprecision(2) << name << ": " << v.size() / elapsed
            << " Hz (target " << 1000. / period_ms << "), ms mean " << mean << ", median "
            << pct(.5) << ", p95 " << pct(.95) << ", p99 " << pct(.99) << ", max " << v.back()
            << "; over budget " << s.overruns << "/" << v.size() << ", invalid " << s.invalid
            << '\n';
}

// Allocate/fork before initializing Python or JAX: each worker owns its interpreter.
struct SharedPlan {
  pthread_mutex_t mutex;
  bool mpc_ready, wbc_ready;
  size_t nu;
  double u[256];
  Clock::duration::rep start;
};
SharedPlan *shared = nullptr;
bool mpc_worker = false;
volatile sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }

int benchmark_worker(py::object solver, double seconds, double mpc_hz, double wbc_hz) {
  const char *name = mpc_worker ? "MPC" : "WBC";
  auto qpos = mpx::array_values(solver.attr("data").attr("qpos"));
  std::vector<double> qvel(solver.attr("nv").cast<size_t>(), 0.);
  qpos[2] = solver.attr("config").attr("robot_height").cast<double>();
  qpos[3] = 1.;
  const size_t nu = solver.attr("nu").cast<size_t>();
  if (nu > 256) throw std::runtime_error("MPC input exceeds shared buffer");
  std::vector<double> command{0., 0., 0., 0.4}, u(nu);
  auto t0 = Clock::now();
  solver.attr(mpc_worker ? "compile_plan" : "compile_wbc")();
  solver.attr(mpc_worker ? "prepare" : "prepare_wbc")(
      mpx::readonly_buffer(qpos), mpx::readonly_buffer(qvel), true);
  if (mpc_worker) {
    auto trajectory = mpx::array_values(solver.attr("plan")(command));
    pthread_mutex_lock(&shared->mutex);
    std::copy_n(trajectory.begin(), nu, shared->u);
    shared->nu = nu;
    pthread_mutex_unlock(&shared->mutex);
  }
  std::cout << name << " PID " << getpid() << " compilation: "
            << std::chrono::duration<double>(Clock::now() - t0).count() << " s" << std::endl;
  pthread_mutex_lock(&shared->mutex);
  (mpc_worker ? shared->mpc_ready : shared->wbc_ready) = true;
  pthread_mutex_unlock(&shared->mutex);
  Clock::time_point start;
  for (;;) {
    pthread_mutex_lock(&shared->mutex);
    bool ready = shared->mpc_ready && shared->wbc_ready;
    if (ready && shared->start == 0)
      shared->start = (Clock::now() + std::chrono::milliseconds(50)).time_since_epoch().count();
    start = Clock::time_point(Clock::duration(shared->start));
    pthread_mutex_unlock(&shared->mutex);
    if (ready) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (shared->nu != nu) throw std::runtime_error("MPC/WBC input dimensions differ");
  const double hz = mpc_worker ? mpc_hz : wbc_hz;
  auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1. / hz));
  auto end_time = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
  std::mt19937 rng(mpc_worker ? 0 : 1);
  std::normal_distribution<double> noise(0., .01);
  Stats stats;
  std::this_thread::sleep_until(start);
  for (auto deadline = start; Clock::now() < end_time;) {
    if (!mpc_worker) {
      pthread_mutex_lock(&shared->mutex);
      std::copy_n(shared->u, nu, u.begin());
      pthread_mutex_unlock(&shared->mutex);
    }
    for (auto &v : qvel) v = noise(rng);
    auto a = Clock::now();
    try {
      solver.attr(mpc_worker ? "prepare" : "prepare_wbc")(
          mpx::readonly_buffer(qpos), mpx::readonly_buffer(qvel), false);
      auto out = mpc_worker ? mpx::array_values(solver.attr("plan")(command))
                            : mpx::array_values(solver.attr("wbc")(mpx::readonly_buffer(u), 0.4));
      if (!std::all_of(out.begin(), out.end(), [](double v) { return std::isfinite(v); })) {
        ++stats.invalid;
      } else if (mpc_worker) {
        pthread_mutex_lock(&shared->mutex);
        std::copy_n(out.begin(), nu, shared->u);
        pthread_mutex_unlock(&shared->mutex);
      }
    } catch (const py::error_already_set &e) {
      if (!e.matches(PyExc_ValueError)) throw;
      ++stats.invalid;
    }
    auto b = Clock::now();
    stats.ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    if (b - a > period) ++stats.overruns;
    deadline += period;
    while (deadline < b) deadline += period;
    std::this_thread::sleep_until(std::min(deadline, end_time));
  }
  report(name, stats, std::chrono::duration<double>(Clock::now() - start).count(), 1000. / hz);
  std::cout << std::flush;
  return stats.invalid ? 1 : 0;
}
}  // namespace

int main(int argc, char **argv) {
  double mpc_hz = 40., wbc_hz = 100., seconds = 20.;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help") {
      std::cout << "benchmark_mpx_cpp [--seconds S] [--mpc-hz F] [--wbc-hz F]\n"
                   "Defaults: 20 seconds, MPC 40 Hz, WBC 100 Hz.\n"
                   "MPC and WBC always run in separate processes/interpreters, shared latest U[0].\n"
                   "Synthetic feedback, production backend, no ROS or motor output.\n";
      return 0;
    }
    if ((arg == "--mpc-hz" || arg == "--wbc-hz" || arg == "--seconds") && i + 1 < argc) {
      double v = 0.;
      try { v = std::stod(argv[++i]); } catch (...) {}
      if (!(v > 0.) || !std::isfinite(v)) { std::cerr << arg << " must be positive\n"; return 2; }
      (arg == "--mpc-hz" ? mpc_hz : arg == "--wbc-hz" ? wbc_hz : seconds) = v;
    }
    else { std::cerr << "Unknown/missing argument: " << arg << '\n'; return 2; }
  }
  {
    shared = static_cast<SharedPlan *>(mmap(nullptr, sizeof(SharedPlan),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (shared == MAP_FAILED) { std::cerr << "mmap failed\n"; return 1; }
    *shared = SharedPlan{};
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&shared->mutex, &attr);
    pthread_mutexattr_destroy(&attr);
    struct sigaction action{};
    action.sa_handler = interrupt;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    pid_t workers[2] = {-1, -1};
    bool child = false;
    for (int i = 0; i < 2; ++i) {
      workers[i] = fork();
      if (workers[i] == 0) {
        signal(SIGINT, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        mpc_worker = i == 0; child = true; break;
      }
      if (workers[i] < 0) { interrupted = 1; break; }
    }
    if (!child) {
      int result = 0;
      for (int remaining = (workers[0] > 0) + (workers[1] > 0); remaining > 0;) {
        if (interrupted || result) {
          for (auto pid : workers) if (pid > 0) kill(pid, SIGTERM);
          if (interrupted) result = 130;
        }
        int status = 0;
        auto pid = waitpid(-1, &status, 0);
        if (pid < 0) { if (errno == EINTR) continue; result = 1; break; }
        for (auto &worker : workers) if (worker == pid) worker = -1;
        --remaining;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) result = result ? result : 1;
      }
      pthread_mutex_destroy(&shared->mutex);
      munmap(shared, sizeof(SharedPlan));
      return result;
    }
  }
  setenv("JAX_PLATFORMS", "cpu", 1);
  setenv("OPENBLAS_NUM_THREADS", "1", 0);
  setenv("OMP_NUM_THREADS", "1", 0);
  setenv("MKL_NUM_THREADS", "1", 0);
  setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1", 0);
  py::scoped_interpreter interpreter{};
  try {
    auto directory = std::filesystem::absolute(argv[0]).parent_path();
    py::module_::import("sys").attr("path").attr("insert")(0, directory.string());
    auto share = py::module_::import("ament_index_python.packages")
      .attr("get_package_share_directory")("tita_description").cast<std::string>();
    py::dict mappings; mappings["hw_env"] = "hw";
    auto description = py::module_::import("xacro").attr("process_file")(
      share + "/xacro/robot.xacro", py::arg("mappings") = mappings).attr("toxml")();
    auto solver = py::module_::import("mpx_cpp_solver").attr("Solver")(
      description, true, 0.4, py::cpp_function([](const std::string &s) {
        std::cout << s << std::endl;
      }), mpc_hz);
    return benchmark_worker(solver, seconds, mpc_hz, wbc_hz);
  } catch (const std::exception &e) {
    std::cerr << "Benchmark failed: " << e.what() << '\n'; return 1;
  }
}
