#pragma once
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <vector>

namespace mpx {
// Humble ships pybind11 2.9. Avoid its NumPy C API with NumPy 2: use the
// standard Python buffer protocol. The caller owns storage for the whole call.
inline pybind11::memoryview readonly_buffer(std::vector<double> &data) {
  return pybind11::memoryview::from_buffer(
      data.data(), {pybind11::ssize_t(data.size())},
      {pybind11::ssize_t(sizeof(double))}, true);
}
inline std::vector<double> array_values(const pybind11::object &array) {
  return array.attr("tolist")().cast<std::vector<double>>();
}
} // namespace mpx
