// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cartesian_impedance_control/cartesian_impedance_controller.hpp>
#include <cartesian_impedance_control/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <geometry_msgs/msg/pose.hpp>

// ============================================================================
// POTENTIAL FIELD CONFIGURATION - SET TO true TO ENABLE, false TO DISABLE
// ============================================================================
const bool USE_POTENTIAL_FIELD = true;  
// ============================================================================

namespace {

template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}

namespace cartesian_impedance_control {

void CartesianImpedanceController::update_stiffness_and_references() {
  // Keep your stiffness filter line:
  nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_
                       + (1.0 - filter_params_) * nullspace_stiffness_;

  // NEW: lock while reading the targets written by the subscriber
  std::lock_guard<std::mutex> lock(position_and_orientation_d_target_mutex_);
  position_d_    = filter_params_ * position_d_target_    + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}


void CartesianImpedanceController::arrayToMatrix(const std::array<double,7>& inputArray, Eigen::Matrix<double,7,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 7; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

void CartesianImpedanceController::arrayToMatrix(const std::array<double,6>& inputArray, Eigen::Matrix<double,6,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 6; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceController::saturateTorqueRate(
  const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
  const Eigen::Matrix<double, 7, 1>& tau_J_d_M) {  
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
  double difference = tau_d_calculated[i] - tau_J_d_M[i];
  tau_d_saturated[i] =
         tau_J_d_M[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}


inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
  double lambda_ = damped ? 0.2 : 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);   
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
  Eigen::MatrixXd S_ = M_;  
  S_.setZero();

  for (int i = 0; i < sing_vals_.size(); i++)
     S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

  M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
}

// ============================================================================
// POTENTIAL FIELD HELPER FUNCTIONS - MATCHING RMP STRUCTURE
// ============================================================================


Eigen::Vector3d CartesianImpedanceController::computeRepulsionForceForSingleObstacle(
    const Eigen::Vector3d& d_obs) {
  
  Eigen::Vector3d F_env = Eigen::Vector3d::Zero();
  
  
  double D = d_obs.norm();
  
  
  if (D >= Q_threshold_ || D < 1e-6) {
    return F_env;  
  }
  
 
  Eigen::Vector3d n_hat = d_obs / D;
  
  // Compute repulsion force magnitude using potential field equation
  // F_env = λ * (1/D - 1/Q)^l * (1/D²) * n̂
  double magnitude = lambda_repulsion_ * 
                     std::pow((1.0/D - 1.0/Q_threshold_), l_smoothing_) * 
                     (1.0/(D*D));
  
  F_env = magnitude * n_hat;
  
  
  double max_repulsion_force = 50.0;  // Maximum 50N
  if (F_env.norm() > max_repulsion_force) {
    F_env = (max_repulsion_force / F_env.norm()) * F_env;
  }
  
  return F_env;
}

//like RMP's get_ddq() structure
Eigen::VectorXd CartesianImpedanceController::computeEnvironmentalRepulsionTorques() {
  
  Eigen::VectorXd tau_env_total = Eigen::VectorXd::Zero(7);
  
  if (!USE_POTENTIAL_FIELD) {
    return tau_env_total;  
  }
  
  std::lock_guard<std::mutex> lock(closest_point_mutex_);
  
  if (!closest_point_msg_) {
    return tau_env_total; 
  }
  
  //  like RMP line 449
  int number_obstacles = closest_point_msg_->frame2x.size();
  
  if (number_obstacles == 0) {
    return tau_env_total;
  }
  
  // like RMP lines 451-458
  Eigen::MatrixXd d_obs2(3, number_obstacles);
  Eigen::MatrixXd d_obs3(3, number_obstacles);
  Eigen::MatrixXd d_obs4(3, number_obstacles);
  Eigen::MatrixXd d_obs5(3, number_obstacles);
  Eigen::MatrixXd d_obs6(3, number_obstacles);
  Eigen::MatrixXd d_obs7(3, number_obstacles);
  Eigen::MatrixXd d_obshand(3, number_obstacles);
  Eigen::MatrixXd d_obsEE(3, number_obstacles);
  
  //  like RMP lines 477-484
  for (int i = 0; i < number_obstacles; i++) {
    d_obs2.col(i) << closest_point_msg_->frame2x[i], closest_point_msg_->frame2y[i], closest_point_msg_->frame2z[i];
    d_obs3.col(i) << closest_point_msg_->frame3x[i], closest_point_msg_->frame3y[i], closest_point_msg_->frame3z[i];
    d_obs4.col(i) << closest_point_msg_->frame4x[i], closest_point_msg_->frame4y[i], closest_point_msg_->frame4z[i];
    d_obs5.col(i) << closest_point_msg_->frame5x[i], closest_point_msg_->frame5y[i], closest_point_msg_->frame5z[i];
    d_obs6.col(i) << closest_point_msg_->frame6x[i], closest_point_msg_->frame6y[i], closest_point_msg_->frame6z[i];
    d_obs7.col(i) << closest_point_msg_->frame7x[i], closest_point_msg_->frame7y[i], closest_point_msg_->frame7z[i];
    d_obshand.col(i) << closest_point_msg_->framehandx[i], closest_point_msg_->framehandy[i], closest_point_msg_->framehandz[i];
    d_obsEE.col(i) << closest_point_msg_->frameeex[i], closest_point_msg_->frameeey[i], closest_point_msg_->frameeez[i];
  }
  
  //  like RMP lines 488-495
  Eigen::MatrixXd jacobian2_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobian3_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobian4_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobian5_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobian6_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobian7_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobianhand_obstacle(6, 7 * number_obstacles);
  Eigen::MatrixXd jacobianEE_obstacle(6, 7 * number_obstacles);
  
  //  like RMP lines 498-524
  for (int i = 0; i < number_obstacles; i++) {
    jacobian2_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobian2.data() + 42 * i));
    jacobian3_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobian3.data() + 42 * i));
    jacobian4_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobian4.data() + 42 * i));
    jacobian5_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobian5.data() + 42 * i));
    jacobian6_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobian6.data() + 42 * i));
    jacobian7_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobian7.data() + 42 * i));
    jacobianhand_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobianhand.data() + 42 * i));
    jacobianEE_obstacle.block(0, 7 * i, 6, 7) = 
        Eigen::Map<Eigen::Matrix<double, 6, 7>>(const_cast<double*>(closest_point_msg_->jacobianee.data() + 42 * i));
  }
  
  // ============================================================================
  // COMPUTE TORQUES -  like RMP's get_ddq() lines 374-390
  // ============================================================================
  for (int i = 0; i < number_obstacles; i++) {
    // Frame 2
    Eigen::Vector3d F_env2 = computeRepulsionForceForSingleObstacle(d_obs2.col(i));
    Eigen::Matrix<double, 6, 1> wrench2 = Eigen::Matrix<double, 6, 1>::Zero();
    wrench2.head<3>() = F_env2;
    tau_env_total += jacobian2_obstacle.block(0, 7 * i, 6, 7).transpose() * wrench2;
    
    // Frame 3
    Eigen::Vector3d F_env3 = computeRepulsionForceForSingleObstacle(d_obs3.col(i));
    Eigen::Matrix<double, 6, 1> wrench3 = Eigen::Matrix<double, 6, 1>::Zero();
    wrench3.head<3>() = F_env3;
    tau_env_total += jacobian3_obstacle.block(0, 7 * i, 6, 7).transpose() * wrench3;
    
    // Frame 4
    Eigen::Vector3d F_env4 = computeRepulsionForceForSingleObstacle(d_obs4.col(i));
    Eigen::Matrix<double, 6, 1> wrench4 = Eigen::Matrix<double, 6, 1>::Zero();
    wrench4.head<3>() = F_env4;
    tau_env_total += jacobian4_obstacle.block(0, 7 * i, 6, 7).transpose() * wrench4;
    
    // Frame 5
    Eigen::Vector3d F_env5 = computeRepulsionForceForSingleObstacle(d_obs5.col(i));
    Eigen::Matrix<double, 6, 1> wrench5 = Eigen::Matrix<double, 6, 1>::Zero();
    wrench5.head<3>() = F_env5;
    tau_env_total += jacobian5_obstacle.block(0, 7 * i, 6, 7).transpose() * wrench5;
    
    // Frame 6
    Eigen::Vector3d F_env6 = computeRepulsionForceForSingleObstacle(d_obs6.col(i));
    Eigen::Matrix<double, 6, 1> wrench6 = Eigen::Matrix<double, 6, 1>::Zero();
    wrench6.head<3>() = F_env6;
    tau_env_total += jacobian6_obstacle.block(0, 7 * i, 6, 7).transpose() * wrench6;
    
    // Frame 7
    Eigen::Vector3d F_env7 = computeRepulsionForceForSingleObstacle(d_obs7.col(i));
    Eigen::Matrix<double, 6, 1> wrench7 = Eigen::Matrix<double, 6, 1>::Zero();
    wrench7.head<3>() = F_env7;
    tau_env_total += jacobian7_obstacle.block(0, 7 * i, 6, 7).transpose() * wrench7;
    
    // Hand
    Eigen::Vector3d F_envhand = computeRepulsionForceForSingleObstacle(d_obshand.col(i));
    Eigen::Matrix<double, 6, 1> wrenchhand = Eigen::Matrix<double, 6, 1>::Zero();
    wrenchhand.head<3>() = F_envhand;
    tau_env_total += jacobianhand_obstacle.block(0, 7 * i, 6, 7).transpose() * wrenchhand;
    
    // End-effector
    Eigen::Vector3d F_envEE = computeRepulsionForceForSingleObstacle(d_obsEE.col(i));
    Eigen::Matrix<double, 6, 1> wrenchEE = Eigen::Matrix<double, 6, 1>::Zero();
    wrenchEE.head<3>() = F_envEE;
    tau_env_total += jacobianEE_obstacle.block(0, 7 * i, 6, 7).transpose() * wrenchEE;
  }
  
  // Safety check: limit maximum total torque
  // double max_torque = 10.0;  // Maximum torque per joint in Nm
  // for (int i = 0; i < 7; i++) {
  //   if (std::abs(tau_env_total(i)) > max_torque) {
  //     tau_env_total(i) = std::copysign(max_torque, tau_env_total(i));
  //   }
  // }
  
  return tau_env_total;
}

void CartesianImpedanceController::closestPointCallback(
    const messages_fr3::msg::ClosestPoint::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(closest_point_mutex_);
  closest_point_msg_ = msg;
}
// ============================================================================


controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}


controller_interface::InterfaceConfiguration CartesianImpedanceController::state_interface_configuration()
  const {
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    state_interfaces_config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/position");
    state_interfaces_config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/velocity");
  }

  const std::string full_interface_name = robot_name_ + "/" + state_interface_name_;

  return state_interfaces_config;
}


CallbackReturn CartesianImpedanceController::on_init() {
   // Initialize potential field parameters
   lambda_repulsion_ = 20.0;   // Scaling factor (Strength of repulsion force)
   Q_threshold_ = 0.05;         // Distance threshold in meters (15cm)
   l_smoothing_ = 2.0;          // Smoothing exponent (controls how smoothly force increases)
   closest_point_msg_ = nullptr;
   
   if (USE_POTENTIAL_FIELD) {
     RCLCPP_INFO(get_node()->get_logger(), 
       "Potential Field Repulsion ENABLED: λ=%.1f, Q=%.3fm, l=%.1f", 
       lambda_repulsion_, Q_threshold_, l_smoothing_);
   } else {
     RCLCPP_INFO(get_node()->get_logger(), "Potential Field Repulsion DISABLED");
   }
   
   UserInputServer input_server_obj(&position_d_target_, &rotation_d_target_, &K, &D, &T);
   std::thread input_thread(&UserInputServer::main, input_server_obj, 0, nullptr);
   input_thread.detach();
   RCLCPP_INFO(get_node()->get_logger(), "on_init completed successfully.");
   return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  try {

    RCLCPP_INFO(get_node()->get_logger(), "Starting on_configure...");

    // Retrieve the robot_description parameter
    std::string robot_description;
    auto parameters_client = std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "/robot_state_publisher");
    parameters_client->wait_for_service();
    auto future = parameters_client->get_parameters({"robot_description"});
    auto result = future.get();
    if (!result.empty()) {
      robot_description = result[0].value_to_string();
      RCLCPP_INFO(get_node()->get_logger(), "'robot_description' parameter retrieved successfully.");
      RCLCPP_INFO(get_node()->get_logger(), "URDF content: %s", robot_description.c_str());
    } else {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
      return CallbackReturn::ERROR;
    }
    
    // Parse the URDF using Pinocchio
    pinocchio::urdf::buildModelFromXML(robot_description, model_);
    data_ = pinocchio::Data(model_);
    RCLCPP_INFO(get_node()->get_logger(), "Pinocchio model parsed successfully.");
  
    // Set the end-effector frame ID
    end_effector_frame_id_ = model_.getFrameId("fr3_hand_tcp");

    {
      using geometry_msgs::msg::Pose;
      goal_sub_ = get_node()->create_subscription<Pose>(
          "/cartesian_impedance_controller/target_pose", 10,
          [this](const Pose::SharedPtr msg) {
            std::lock_guard<std::mutex> lk(position_and_orientation_d_target_mutex_);
            position_d_target_ << msg->position.x, msg->position.y, msg->position.z;

            std::cout << "Received target position: " << position_d_target_.transpose() << std::endl;

            Eigen::Quaterniond q(
                msg->orientation.w,
                msg->orientation.x,
                msg->orientation.y,
                msg->orientation.z);
            orientation_d_target_ = q.normalized();
          });
      RCLCPP_INFO(get_node()->get_logger(),
                  "CIC subscribed to /cartesian_impedance_controller/target_pose (geometry_msgs::Pose)");
    }

    // Subscribe to closest point for potential fields (only if enabled)
    if (USE_POTENTIAL_FIELD) {
      closest_point_sub_ = get_node()->create_subscription<messages_fr3::msg::ClosestPoint>(
          "/closest_point", 
          10,
          std::bind(&CartesianImpedanceController::closestPointCallback, this, std::placeholders::_1)
      );
      RCLCPP_INFO(get_node()->get_logger(), "Subscribed to /closest_point for potential field repulsion");
    }

  } 
  catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to load Pinocchio model: %s", e.what());
    return CallbackReturn::ERROR;
  }
  RCLCPP_INFO(get_node()->get_logger(), "on_configure completed successfully.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
  std::cout << "Available frames in the model:" << std::endl;
  for (const auto& frame : model_.frames) {
    std::cout << frame.name << std::endl;
  }
  std::cout << "Number of available velocities:" << model_.nv << std::endl;

  updateJointStates();
  jacobian.resize(6, model_.nv);
  jacobian_transpose_pinv.resize(model_.nv, 6);
  jacobian.setZero();
  jacobian_transpose_pinv.setZero();
  pinocchio::forwardKinematics(model_, data_, q_);
  pinocchio::updateFramePlacements(model_, data_);
  
  Eigen::Affine3d transform;
  transform.linear() = data_.oMf[end_effector_frame_id_].rotation();
  transform.translation() = data_.oMf[end_effector_frame_id_].translation();
  position_d_ = transform.translation();
  orientation_d_ = Eigen::Quaterniond(transform.rotation());
  std::cout << "Completed Activation process" << std::endl;
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn CartesianImpedanceController::on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "Controller deactivated.");
  return CallbackReturn::SUCCESS;
}

std::array<double, 6> CartesianImpedanceController::convertToStdArray(const geometry_msgs::msg::WrenchStamped& wrench) {
    std::array<double, 6> result;
    result[0] = wrench.wrench.force.x;
    result[1] = wrench.wrench.force.y;
    result[2] = wrench.wrench.force.z;
    result[3] = wrench.wrench.torque.x;
    result[4] = wrench.wrench.torque.y;
    result[5] = wrench.wrench.torque.z;
    return result;
}

void CartesianImpedanceController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {  

  Eigen::VectorXd dynamic_torques = pinocchio::rnea(model_, data_, q_,  dq_, Eigen::VectorXd::Zero(model_.nv));
  M = pinocchio::crba(model_, data_, q_);
  pinocchio::forwardKinematics(model_, data_, q_);

  pinocchio::computeJointJacobians(model_, data_, q_);
  pinocchio::getFrameJacobian(model_, data_, end_effector_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  pinocchio::updateFramePlacements(model_, data_);
  Eigen::MatrixXd g = pinocchio::computeGeneralizedGravity(model_, data_, q_);
  coriolis = dynamic_torques - g;
  
  Eigen::Affine3d transform;
  transform.linear() = data_.oMf[end_effector_frame_id_].rotation();
  transform.translation() = data_.oMf[end_effector_frame_id_].translation();
  Eigen::Vector3d position = transform.translation();
  Eigen::Quaterniond orientation(transform.rotation());
  orientation_d_target_ = Eigen::AngleAxisd(rotation_d_target_[0], Eigen::Vector3d::UnitX())
                        * Eigen::AngleAxisd(rotation_d_target_[1], Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(rotation_d_target_[2], Eigen::Vector3d::UnitZ());
  updateJointStates();  
  error.head(3) << position - position_d_;

  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);

  Lambda = ((jacobian * M.inverse() * jacobian.transpose()).inverse()).topLeftCorner(6, 6);
  D =  D_gain* K.cwiseMax(0.0).cwiseSqrt() * Lambda.cwiseMax(0.0).diagonal().cwiseSqrt().asDiagonal();

  F_impedance = -1 * ((D * jacobian * dq_) + K * error);

  // ============================================================================
  // COMPUTE ENVIRONMENTAL REPULSION TORQUES (POTENTIAL FIELD FOR ALL FRAMES)
  // Structure matches RMP's get_ddq() exactly
  // ============================================================================
  Eigen::VectorXd tau_env = computeEnvironmentalRepulsionTorques();
  // ============================================================================

  Eigen::VectorXd tau_nullspace = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd tau_d = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd tau_impedance = Eigen::VectorXd::Zero(7);

  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
  
  tau_impedance = jacobian.topLeftCorner(6,7).transpose() * Sm * F_impedance;
  
  // Add environmental repulsion torques to the control law
  tau_d = tau_impedance + tau_env + tau_nullspace + coriolis.head(7);
  
  tau_d << saturateTorqueRate(tau_d, tau_J_d_M);
  tau_J_d_M = tau_d;

  bool valid = true;
  for (int i = 0; i < tau_d.size(); ++i) {
    if (std::isnan(tau_d(i))) {
      valid = false;
      break;
    }
  }

  if (valid) {
    for (size_t i = 0; i < 7; ++i) {
      command_interfaces_[i].set_value(tau_d(i));
    }
  }
  
  if (outcounter % 10000 == 0){
    for (int i = 0; i < tau_d.size(); ++i) {
      if (std::isnan(tau_d(i))) {
        std::cout << "Error: tau_d contains NaN at index " << i << std::endl;
      }
    }
    std::cout << "jacobian: \n" << jacobian << std::endl;
    std::cout << "F_impedance: " << F_impedance.transpose() << std::endl;
    std::cout << "position: " << position.transpose() << std::endl;
    std::cout << "position_d: " << position_d_.transpose() << std::endl;
    
    // Log environmental repulsion info if active
    if (USE_POTENTIAL_FIELD && tau_env.norm() > 0.01) {
      std::cout << "tau_env (repulsion): " << tau_env.transpose() << " Nm (norm: " << tau_env.norm() << ")" << std::endl;
      if (closest_point_msg_) {
        std::cout << "Number of obstacles: " << closest_point_msg_->frame2x.size() << std::endl;
      }
    }
  }
  outcounter++;
  update_stiffness_and_references();
  return controller_interface::return_type::OK;
}
}

// namespace cartesian_impedance_control
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cartesian_impedance_control::CartesianImpedanceController,
                       controller_interface::ControllerInterface)