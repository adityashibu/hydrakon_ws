#include <hydrakon_can/hydrakon_can.hpp>
#include <string>

using std::placeholders::_1;
using std::placeholders::_2;

HydrakonCanInterface::HydrakonCanInterface() : Node("hydrakon_can") {
  // Declare ROS parameters
  can_debug_ = declare_parameter<int>("can_debug", can_debug_);
  simulate_can_ = declare_parameter<int>("simulate_can", simulate_can_);
  can_interface_ = declare_parameter<std::string>("can_interface", can_interface_);
  loop_rate = declare_parameter<int>("loop_rate", loop_rate);
  max_dec_ = declare_parameter<float>("max_dec", max_dec_);
  engine_threshold_ = declare_parameter<float>("engine_threshold", engine_threshold_);
  rpm_limit_ = declare_parameter<float>("rpm_limit", rpm_limit_);
  cmd_timeout_ = declare_parameter<double>("cmd_timeout", cmd_timeout_);
  if (declare_parameter<bool>("debug_logging", false)) {get_logger().set_level(rclcpp::Logger::Level::Debug);
  }

  // CAN interface setup
  if (can_debug_) RCLCPP_INFO(get_logger(), "Initiating DEBUG MODE");
  if (simulate_can_) RCLCPP_INFO(get_logger(), "Initiating SIMULATION MODE");
  fs_ai_api_init(const_cast<char *>(can_interface_.c_str()), can_debug_, simulate_can_);

  // ROS subscribers
  cmd_sub_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
      "/hydrakon_can/command", 1, std::bind(&HydrakonCanInterface::commandCallback, this, _1));
  flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/hydrakon_can/is_mission_completed", 1, std::bind(&HydrakonCanInterface::flagCallback, this, _1));
  driving_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/hydrakon_can/driving_flag", 1, std::bind(&HydrakonCanInterface::drivingFlagCallback, this, _1));

  // ROS publishers
  state_pub_ = this->create_publisher<hydrakon_can::msg::CanState>("/hydrakon_can/state", 1);
  state_str_pub_ = this->create_publisher<std_msgs::msg::String>("/hydrakon_can/state_str", 1);
  wheel_speed_pub_ = this->create_publisher<hydrakon_can::msg::WheelSpeed>("/hydrakon_can/wheel_speed", 1);
  vehicle_command_pub_ = this->create_publisher<hydrakon_can::msg::VehicleCommand>("/hydrakon_can/vehicle_command", 1);
  twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/hydrakon_can/twist", 1);
  // imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("/hydrakon_can/imu", 1);
  // fix_pub_ = this->create_publisher<sensor_msgs::msg::NavSatFix>("/hydrakon_can/fix", 1);

  // ROS services
  ebs_srv_ = this->create_service<std_srvs::srv::Trigger>("/hydrakon_can/ebs_request", std::bind(&HydrakonCanInterface::requestEBS, this, _1, _2));

  // Setup ROS timer
  std::chrono::duration<float> rate(1 / static_cast<double>(loop_rate));
  timer_ = this->create_wall_timer(rate, std::bind(&HydrakonCanInterface::loop, this));

  // Value of 0.0 means time is uninitialised
  last_cmd_message_time_ = 0.0;
}

void HydrakonCanInterface::loop() {
  // Get fresh data from VCU
  fs_ai_api_vcu2ai_get_data(&vcu2ai_data_);
  // fs_ai_api_gps_get_data(&gps_data_);
  // fs_ai_api_imu_get_data(&imu_data_);

  // Log new data (in one string so log messages don't get separated)
  std::string msg_recv =
      "--- Read CAN data ---\n"
      "AS STATE:    " + std::to_string(vcu2ai_data_.VCU2AI_AS_STATE) + "\n" +
      "AMI STATE:   " + std::to_string(vcu2ai_data_.VCU2AI_AMI_STATE) + "\n" +
      "RR PULSE:    " + std::to_string(vcu2ai_data_.VCU2AI_RR_PULSE_COUNT) + "\n" +
      "RL PULSE:    " + std::to_string(vcu2ai_data_.VCU2AI_RL_PULSE_COUNT) + "\n" +
      "FR PULSE:    " + std::to_string(vcu2ai_data_.VCU2AI_FR_PULSE_COUNT) + "\n" +
      "FL PULSE:    " + std::to_string(vcu2ai_data_.VCU2AI_FL_PULSE_COUNT) + "\n" +
      "RR RPM:      " + std::to_string(vcu2ai_data_.VCU2AI_RR_WHEEL_SPEED_rpm) + "\n" +
      "RL RPM:      " + std::to_string(vcu2ai_data_.VCU2AI_RL_WHEEL_SPEED_rpm) + "\n" +
      "FR RPM:      " + std::to_string(vcu2ai_data_.VCU2AI_FR_WHEEL_SPEED_rpm) + "\n" +
      "FL RPM:      " + std::to_string(vcu2ai_data_.VCU2AI_FL_WHEEL_SPEED_rpm) + "\n" +
      "STEER ANGLE: " + std::to_string(vcu2ai_data_.VCU2AI_STEER_ANGLE_deg) + "\n";
  RCLCPP_DEBUG(get_logger(), "%s", msg_recv.c_str());

  // Update AS state
  as_state_ = vcu2ai_data_.VCU2AI_AS_STATE;
  // as_state_ = AS_DRIVING; //remove this when done
  // as_state_ = fs_ai_api_as_state_e::AS_DRIVING;

  // Reset state variables when in AS_OFF
  if (as_state_ == fs_ai_api_as_state_e::AS_OFF) {
    ebs_state_ = fs_ai_api_estop_request_e::ESTOP_NO;
    driving_flag_ = false;
    mission_complete_ = false;
    steering_ = 0;
    torque_ = 0;
    rpm_request_ = 0.0;
    braking_ = 0;
    last_cmd_message_time_ = 0;
  }

  // publish all received data
  wheel_speed_pub_->publish(HydrakonCanInterface::makeWheelSpeedMessage(vcu2ai_data_));
  twist_pub_->publish(HydrakonCanInterface::makeTwistMessage(vcu2ai_data_));
  vehicle_command_pub_->publish(HydrakonCanInterface::makeVehicleCommandMessage());
  // fix_pub_->publish(HydrakonCanInterface::makeGpsMessage(gps_data_));
  // imu_pub_->publish(HydrakonCanInterface::makeImuMessage(imu_data_));


  // vcu2ai_data_.VCU2AI_AMI_STATE = fs_ai_api_ami_state_e::AMI_TRACK_DRIVE; //remove this when done

  // Read and publish state data
  auto state_msg = HydrakonCanInterface::makeStateMessage(vcu2ai_data_);
  state_pub_->publish(state_msg);
  state_str_pub_->publish(makeStateString(state_msg));

  // Assign data to be sent
  ai2vcu_data_.AI2VCU_ESTOP_REQUEST = ebs_state_;
  ai2vcu_data_.AI2VCU_BRAKE_PRESS_REQUEST_pct = braking_;
  ai2vcu_data_.AI2VCU_AXLE_TORQUE_REQUEST_Nm = torque_;
  ai2vcu_data_.AI2VCU_STEER_ANGLE_REQUEST_deg = steering_;
  ai2vcu_data_.AI2VCU_AXLE_SPEED_REQUEST_rpm = rpm_request_;
  ai2vcu_data_.AI2VCU_HANDSHAKE_SEND_BIT = HydrakonCanInterface::getHandshake(vcu2ai_data_);
  ai2vcu_data_.AI2VCU_DIRECTION_REQUEST = HydrakonCanInterface::getDirectionRequest(vcu2ai_data_);
  ai2vcu_data_.AI2VCU_MISSION_STATUS = HydrakonCanInterface::getMissionStatus(vcu2ai_data_);

  // Log sent data (in one string so log messages don't get separated)
  std::string msg_send =
      "--- Sending CAN data ---\n"
      "EBS:            " + std::to_string(ai2vcu_data_.AI2VCU_ESTOP_REQUEST) + "\n" +
      "Brake pct:      " + std::to_string(ai2vcu_data_.AI2VCU_BRAKE_PRESS_REQUEST_pct) + "\n" +
      "Steering:       " + std::to_string(ai2vcu_data_.AI2VCU_STEER_ANGLE_REQUEST_deg) + "\n" +
      "Torque:         " + std::to_string(ai2vcu_data_.AI2VCU_AXLE_TORQUE_REQUEST_Nm) + "\n" +
      "Axle rpm:       " + std::to_string(ai2vcu_data_.AI2VCU_AXLE_SPEED_REQUEST_rpm) + "\n" +
      "Direction req:  " + std::to_string(ai2vcu_data_.AI2VCU_DIRECTION_REQUEST) + "\n" +
      "Mission status: " + std::to_string(ai2vcu_data_.AI2VCU_MISSION_STATUS) + "\n";
  RCLCPP_DEBUG(get_logger(), "%s", msg_send.c_str());

  // Send data to car
  fs_ai_api_ai2vcu_set_data(&ai2vcu_data_);

  // Only check timeout if driving_flag is true
  if (driving_flag_) {
    checkTimeout();
  }
}

fs_ai_api_handshake_send_bit_e HydrakonCanInterface::getHandshake(const fs_ai_api_vcu2ai_struct data) {
  auto handshake = data.VCU2AI_HANDSHAKE_RECEIVE_BIT;
  if (handshake == fs_ai_api_handshake_receive_bit_e::HANDSHAKE_RECEIVE_BIT_OFF)
    return fs_ai_api_handshake_send_bit_e::HANDSHAKE_SEND_BIT_OFF;
  else
    return fs_ai_api_handshake_send_bit_e::HANDSHAKE_SEND_BIT_ON;
}


fs_ai_api_direction_request_e HydrakonCanInterface::getDirectionRequest(const fs_ai_api_vcu2ai_struct data) {
  if (data.VCU2AI_AS_STATE == fs_ai_api_as_state_e::AS_DRIVING && driving_flag_)
  // if (as_state_ == fs_ai_api_as_state_e::AS_DRIVING && driving_flag_)
    return fs_ai_api_direction_request_e::DIRECTION_FORWARD;
  else
    return fs_ai_api_direction_request_e::DIRECTION_NEUTRAL;
}


fs_ai_api_mission_status_e HydrakonCanInterface::getMissionStatus(const fs_ai_api_vcu2ai_struct data) {

  switch (data.VCU2AI_AS_STATE) {
    case fs_ai_api_as_state_e::AS_OFF:
      // Check whether a mission has been chosen, and acknowledge it if so
      if (data.VCU2AI_AMI_STATE != fs_ai_api_ami_state_e::AMI_NOT_SELECTED)
        return fs_ai_api_mission_status_e::MISSION_SELECTED;
      else
        return fs_ai_api_mission_status_e::MISSION_NOT_SELECTED;
    case fs_ai_api_as_state_e::AS_READY:
      if (driving_flag_) {
        // our stack is ready to start driving
        return fs_ai_api_mission_status_e::MISSION_RUNNING;
      } else {
        // still not ready to start driving
        return fs_ai_api_mission_status_e::MISSION_SELECTED;
      }
    case fs_ai_api_as_state_e::AS_DRIVING:
      if (mission_complete_) {
        // mission has been finished
        return fs_ai_api_mission_status_e::MISSION_FINISHED;
      } else {
        // still doing a mission
        return fs_ai_api_mission_status_e::MISSION_RUNNING;
      }
    case fs_ai_api_as_state_e::AS_FINISHED:
      return fs_ai_api_mission_status_e::MISSION_FINISHED;
    default:
      return fs_ai_api_mission_status_e::MISSION_NOT_SELECTED;
  }
  // return fs_ai_api_mission_status_e::MISSION_RUNNING; //remove this when done
}


void HydrakonCanInterface::commandCallback(ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  if (driving_flag_) {
    const float acceleration = msg->drive.acceleration;

    // Always calculate torque baseline (may be set to 0 later)
    float torque = (TOTAL_MASS_ * WHEEL_RADIUS_ * std::abs(acceleration + 0.5f)) / 2.0f;

    if (acceleration > 0.0f) {
      braking_ = 0.0f;
      torque_ = torque;
      rpm_request_ = rpm_limit_;  // positive torque via RPM
    } 
    else if (acceleration == 0.0f) {
      torque_ = 0.0f;
      braking_ = 0.0f;
      rpm_request_ = 0.0f;
    } 
    else if (acceleration > engine_threshold_) {
      // Engine braking only
      braking_ = 0.0f;
      torque_ = torque;
      rpm_request_ = 0.0f;
    } 
    else {
      // Use mechanical brakes
      torque_ = 0.0f;
      braking_ = (-acceleration / max_dec_) * MAX_BRAKE_;
      rpm_request_ = 0.0f;
    }

    // Convert radians to degrees
    steering_ = static_cast<float>(msg->drive.steering_angle * 180.0 / M_PI);
  } 
  else {
    // Disable movement commands
    steering_ = 0.0f;
    torque_ = 0.0f;
    braking_ = 0.0f;
    rpm_request_ = 0.0f;
  }

  // Clamp outputs to safe ranges
  steering_ = checkAndTrunc(steering_, MAX_STEERING_ANGLE_DEG_, "setting steering", false);
  torque_ = checkAndTrunc(torque_, MAX_TORQUE_, "setting torque");
  braking_ = checkAndTrunc(braking_, MAX_BRAKE_, "setting brake");

  // Record timestamp
  last_cmd_message_time_ = this->now().seconds();
}

void HydrakonCanInterface::flagCallback(std_msgs::msg::Bool::SharedPtr msg) {
  mission_complete_ = msg->data;
}

void HydrakonCanInterface::drivingFlagCallback(std_msgs::msg::Bool::SharedPtr msg) {
  // Driving flag can only be set to true if we're in AS_DRIVING_
  if (msg->data && as_state_ == fs_ai_api_as_state_e::AS_DRIVING) {
    driving_flag_ = true;
  } else if (msg->data) {
    driving_flag_ = false;
    RCLCPP_WARN(get_logger(), "Driving flag is true but as_state is %i", as_state_);
  } else {
    driving_flag_ = msg->data;
  }

  if (driving_flag_ && last_cmd_message_time_ == 0.0) {
    // We start the cmd timeout on the positive edge of `driving_flag_`
    last_cmd_message_time_ = this->now().seconds();
  } else if (!driving_flag_) {
    // Reset last cmd message time back to magic value (0.0) if we stop driving
    last_cmd_message_time_ = 0.0;
  }
}

bool HydrakonCanInterface::requestEBS(std_srvs::srv::Trigger::Request::SharedPtr,
                              std_srvs::srv::Trigger::Response::SharedPtr response) {
  RCLCPP_WARN(this->get_logger(), "Requesting EMERGENCY STOP");
  ebs_state_ = fs_ai_api_estop_request_e::ESTOP_YES;
  response->success = true;
  return response->success;
}

hydrakon_can::msg::VehicleCommand HydrakonCanInterface::makeVehicleCommandMessage() {
  auto msg = hydrakon_can::msg::VehicleCommand();

  // Populate msg
  msg.ebs = ebs_state_;
  msg.braking = braking_;
  msg.torque = torque_;
  msg.steering = steering_;
  msg.rpm = rpm_request_;
  msg.handshake = HydrakonCanInterface::getHandshake(vcu2ai_data_);
  msg.direction = HydrakonCanInterface::getDirectionRequest(vcu2ai_data_);
  msg.mission_status = HydrakonCanInterface::getMissionStatus(vcu2ai_data_);

  return msg;
}

hydrakon_can::msg::WheelSpeed HydrakonCanInterface::makeWheelSpeedMessage(const fs_ai_api_vcu2ai_struct data) {
  auto msg = hydrakon_can::msg::WheelSpeed();

  float steering_feedback = -data.VCU2AI_STEER_ANGLE_deg;  // inverted to match ISO convention
  float fl_speed = data.VCU2AI_FL_WHEEL_SPEED_rpm;
  float fr_speed = data.VCU2AI_FR_WHEEL_SPEED_rpm;
  float rl_speed = data.VCU2AI_RL_WHEEL_SPEED_rpm;
  float rr_speed = data.VCU2AI_RR_WHEEL_SPEED_rpm;

  // check values
  msg.lf_speed = checkAndTrunc(fl_speed, MAX_RPM_, "FL wheelspeed");
  msg.rf_speed = checkAndTrunc(fr_speed, MAX_RPM_, "FR wheelspeed");
  msg.lb_speed = checkAndTrunc(rl_speed, MAX_RPM_, "RL wheelspeed");
  msg.rb_speed = checkAndTrunc(rr_speed, MAX_RPM_, "RR wheelspeed");

  steering_feedback = checkAndTrunc(steering_feedback, MAX_STEERING_ANGLE_DEG_, "steering", false);
  msg.steering = (steering_feedback / 180) * M_PI;  // convert to radians

  return msg;
}

geometry_msgs::msg::TwistWithCovarianceStamped HydrakonCanInterface::makeTwistMessage(
    const fs_ai_api_vcu2ai_struct data) {
  auto msg = geometry_msgs::msg::TwistWithCovarianceStamped();
  msg.header.stamp = get_clock()->now();
  msg.header.frame_id = "base_footprint";

  auto wheel_speed = (checkAndTrunc(data.VCU2AI_RL_WHEEL_SPEED_rpm, MAX_RPM_, "RL ws") +
                      checkAndTrunc(data.VCU2AI_RR_WHEEL_SPEED_rpm, MAX_RPM_, "RR ws")) /
                     2;

  msg.twist.twist.linear.x = wheel_speed * M_PI * WHEEL_RADIUS_ / 30;

  auto steering_angle =
      checkAndTrunc(-data.VCU2AI_STEER_ANGLE_deg, MAX_STEERING_ANGLE_DEG_, "steering", false) /
      180 * M_PI;
  // auto steering_angle = 10.0 / 180.0 * M_PI;

  msg.twist.twist.angular.z = msg.twist.twist.linear.x * sin(steering_angle) / WHEELBASE_;

  msg.twist.covariance = {1e-9, 0, 0, 0, 0,    0,
                             0, 0, 0, 0, 0,    0,
                             0, 0, 0, 0, 0,    0,
                             0, 0, 0, 0, 0,    0,
                             0, 0, 0, 0, 0,    0,
                             0, 0, 0, 0, 0, 1e-9};

  return msg;
}

// sensor_msgs::msg::Imu HydrakonCanInterface::makeImuMessage(const fs_ai_api_imu_struct &data) {
//   // Initialise message
//   sensor_msgs::msg::Imu msg;
//   msg.header.stamp = this->get_clock()->now();
//   msg.header.frame_id = "base_footprint";

//   // Get accelerations
//   const float G_VALUE = 9.80665;
//   msg.linear_acceleration.x = data.IMU_Acceleration_X_mG * 1000 * G_VALUE;
//   msg.linear_acceleration.y = data.IMU_Acceleration_Y_mG * 1000 * G_VALUE;
//   msg.linear_acceleration.z = data.IMU_Acceleration_Z_mG * 1000 * G_VALUE;

//   // Get angular velocity
//   msg.angular_velocity.x = (data.IMU_Rotation_X_degps / 180) * M_PI;
//   msg.angular_velocity.y = (data.IMU_Rotation_Y_degps / 180) * M_PI;
//   msg.angular_velocity.z = (data.IMU_Rotation_Z_degps / 180) * M_PI;

//   return msg;
// }

// sensor_msgs::msg::NavSatFix HydrakonCanInterface::makeGpsMessage(const fs_ai_api_gps_struct &data) {
//   sensor_msgs::msg::NavSatFix msg;
//   msg.header.stamp = this->get_clock()->now();
//   msg.header.frame_id = "base_footprint";

//   // Double check these with real values
//   msg.altitude = data.GPS_Altitude;
//   msg.latitude = data.GPS_Latitude_Degree + data.GPS_Latitude_Minutes / 60;
//   msg.longitude = data.GPS_Longitude_Degree + data.GPS_Longitude_Minutes / 60;

//   return msg;
// }

hydrakon_can::msg::CanState HydrakonCanInterface::makeStateMessage(const fs_ai_api_vcu2ai_struct &data) {
  static const std::unordered_map<uint8_t, uint8_t> as_state_map = {
    {fs_ai_api_as_state_e::AS_OFF, hydrakon_can::msg::CanState::AS_OFF},
    {fs_ai_api_as_state_e::AS_READY, hydrakon_can::msg::CanState::AS_READY},
    {fs_ai_api_as_state_e::AS_DRIVING, hydrakon_can::msg::CanState::AS_DRIVING},
    {fs_ai_api_as_state_e::AS_EMERGENCY_BRAKE, hydrakon_can::msg::CanState::AS_EMERGENCY_BRAKE},
    {fs_ai_api_as_state_e::AS_FINISHED, hydrakon_can::msg::CanState::AS_FINISHED}
  };

  static const std::unordered_map<uint8_t, uint8_t> ami_state_map = {
    {fs_ai_api_ami_state_e::AMI_NOT_SELECTED, hydrakon_can::msg::CanState::AMI_NOT_SELECTED},
    {fs_ai_api_ami_state_e::AMI_ACCELERATION, hydrakon_can::msg::CanState::AMI_ACCELERATION},
    {fs_ai_api_ami_state_e::AMI_SKIDPAD, hydrakon_can::msg::CanState::AMI_SKIDPAD},
    {fs_ai_api_ami_state_e::AMI_AUTOCROSS, hydrakon_can::msg::CanState::AMI_AUTOCROSS},
    {fs_ai_api_ami_state_e::AMI_TRACK_DRIVE, hydrakon_can::msg::CanState::AMI_TRACK_DRIVE},
    {fs_ai_api_ami_state_e::AMI_STATIC_INSPECTION_A, hydrakon_can::msg::CanState::AMI_DDT_INSPECTION_A},
    {fs_ai_api_ami_state_e::AMI_STATIC_INSPECTION_B, hydrakon_can::msg::CanState::AMI_DDT_INSPECTION_B},
    {fs_ai_api_ami_state_e::AMI_AUTONOMOUS_DEMO, hydrakon_can::msg::CanState::AMI_AUTONOMOUS_DEMO}
  };

  hydrakon_can::msg::CanState msg;

  auto as_it = as_state_map.find(data.VCU2AI_AS_STATE);
  if (as_it != as_state_map.end()) {
    msg.as_state = as_it->second;
  } else {
    msg.as_state = hydrakon_can::msg::CanState::AS_OFF;
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "Invalid AS state from vehicle.");
  }

  auto ami_it = ami_state_map.find(data.VCU2AI_AMI_STATE);
  if (ami_it != ami_state_map.end()) {
    msg.ami_state = ami_it->second;
  } else {
    msg.ami_state = hydrakon_can::msg::CanState::AMI_NOT_SELECTED;
    RCLCPP_WARN(this->get_logger(), "Invalid AMI state from vehicle.");
  }

  return msg;
}

std_msgs::msg::String HydrakonCanInterface::makeStateString(hydrakon_can::msg::CanState &state) {
  static const std::unordered_map<uint8_t, std::string> as_state_map = {
    {hydrakon_can::msg::CanState::AS_OFF, "AS:OFF"},
    {hydrakon_can::msg::CanState::AS_READY, "AS:READY"},
    {hydrakon_can::msg::CanState::AS_DRIVING, "AS:DRIVING"},
    {hydrakon_can::msg::CanState::AS_FINISHED, "AS:FINISHED"},
    {hydrakon_can::msg::CanState::AS_EMERGENCY_BRAKE, "AS:EMERGENCY"}
  };

  static const std::unordered_map<uint8_t, std::string> ami_state_map = {
    {hydrakon_can::msg::CanState::AMI_NOT_SELECTED, "AMI:NOT_SELECTED"},
    {hydrakon_can::msg::CanState::AMI_ACCELERATION, "AMI:ACCELERATION"},
    {hydrakon_can::msg::CanState::AMI_SKIDPAD, "AMI:SKIDPAD"},
    {hydrakon_can::msg::CanState::AMI_AUTOCROSS, "AMI:AUTOCROSS"},
    {hydrakon_can::msg::CanState::AMI_TRACK_DRIVE, "AMI:TRACKDRIVE"},
    {hydrakon_can::msg::CanState::AMI_DDT_INSPECTION_A, "AMI:INSPECTION_A"},
    {hydrakon_can::msg::CanState::AMI_DDT_INSPECTION_B, "AMI:INSPECTION_B"},
    {hydrakon_can::msg::CanState::AMI_AUTONOMOUS_DEMO, "AMI:AUTONOMOUS_DEMO"},
    {hydrakon_can::msg::CanState::AMI_ADS_INSPECTION, "AMI:ADS_INSPECTION"},
    {hydrakon_can::msg::CanState::AMI_ADS_EBS, "AMI:ADS_EBS"},
    {hydrakon_can::msg::CanState::AMI_JOYSTICK, "AMI:JOYSTICK"},
    {hydrakon_can::msg::CanState::AMI_MANUAL, "AMI:MANUAL"}
  };

  std::string str1 = as_state_map.count(state.as_state) ? as_state_map.at(state.as_state) : "NO_SUCH_MESSAGE";
  std::string str2 = ami_state_map.count(state.ami_state) ? ami_state_map.at(state.ami_state) : "NO_SUCH_MESSAGE";
  std::string str3 = driving_flag_ ? "DRIVING:TRUE" : "DRIVING:FALSE";

  std_msgs::msg::String msg;
  msg.data = str1 + " " + str2 + " " + str3;
  return msg;
}

// Checks if value exceeds max allowed value, if it does truncate it and warn
float HydrakonCanInterface::checkAndTrunc(const float val, const float max_val, const std::string type,
                                  bool trunc_at_zero) {
  float min_val = trunc_at_zero ? 0 : -max_val;
  if (val > max_val) {
    RCLCPP_DEBUG(get_logger(), "%s was %f but max is %f", type.c_str(), val, max_val);
    return max_val;
  } else {
    if (val < min_val) {
      RCLCPP_DEBUG(get_logger(), "%s was %f but min is %f", type.c_str(), val, min_val);
      return min_val;
    }
  }
  return val;
}

int HydrakonCanInterface::checkAndTrunc(const int val, const int max_val, std::string type,
                                bool trunc_at_zero) {
  // Replicated because casting to float from int could lose information
  int min_val = trunc_at_zero ? 0 : -max_val;
  if (val > max_val) {
    RCLCPP_DEBUG(get_logger(), "%s was %i but max is %i", type.c_str(), val, max_val);
    return max_val;
  } else {
    if (val < min_val) {
      RCLCPP_DEBUG(get_logger(), "%s was %i but min is %i", type.c_str(), val, min_val);
      return min_val;
    }
  }
  return val;
}

void HydrakonCanInterface::checkTimeout() {
  // Engage EBS if the duration between last message time and now exceeds threshold
  if (this->now().seconds() - last_cmd_message_time_ > cmd_timeout_) {
    RCLCPP_ERROR(get_logger(), "/hydrakon_can/cmd sent nothing for %f seconds, requesting EMERGENCY STOP",
                 cmd_timeout_);
    ebs_state_ = fs_ai_api_estop_request_e::ESTOP_YES;
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HydrakonCanInterface>());
  rclcpp::shutdown();
  return 0;
}