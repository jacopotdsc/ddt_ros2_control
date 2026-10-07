#pragma once

// Wheel-contact helpers used by StateFilter_no_bias.hpp, taken from
// TITA-dynamic-obstacle-avoidance/tita_controller/src/utils.cpp.

#include <Eigen/Dense>

namespace labrob {

// Vector from the wheel centre to the ground contact point (flat ground, world z up).
inline Eigen::Vector3d get_rCP(const Eigen::MatrixXd& wheel_R, const double& wheel_radius){
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Vector3d z_0 = Eigen::Vector3d(0,0,1);
  Eigen::Vector3d n = wheel_R * z_0;
  Eigen::Vector3d a = (I - n*n.transpose()) * z_0;
  Eigen::Vector3d s = a/a.norm(); // normalize
  Eigen::Vector3d rCP = - s * wheel_radius;
  return rCP;
}

// Columns: rolling direction t, wheel axis n, centre-to-top direction s.
inline Eigen::Matrix3d compute_virtual_frame(const Eigen::MatrixXd& wheel_R){
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Vector3d z_0 = Eigen::Vector3d(0,0,1);
  Eigen::Vector3d n = wheel_R * z_0;
  Eigen::Vector3d a = (I - n*n.transpose()) * z_0;
  Eigen::Vector3d s = a / a.norm(); // normalize
  Eigen::Vector3d t = n.cross(s);
  t = t/t.norm(); // normalize
  Eigen::Matrix3d R;
  R.col(0) = t;
  R.col(1) = n;
  R.col(2) = s;
  return R;
}

}  // namespace labrob
