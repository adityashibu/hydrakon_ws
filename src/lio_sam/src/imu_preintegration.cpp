#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// GTSAM includes
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <deque>
#include <mutex>

using std::placeholders::_1;
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class IMUPreintegrationNode : public rclcpp::Node
{
public:
    IMUPreintegrationNode() : Node("lio_sam_imu_preintegration")
    {
        // Initialize parameters
        initializeParameters();
        
        // Initialize GTSAM components
        initializeGTSAM();
        
        // Create callback groups for multi-threading
        callback_group_imu_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callback_group_odom_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        // Setup subscription options
        auto imu_options = rclcpp::SubscriptionOptions();
        imu_options.callback_group = callback_group_imu_;
        auto odom_options = rclcpp::SubscriptionOptions();
        odom_options.callback_group = callback_group_odom_;

        // Create subscribers
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/carla/imu_sensor", 
            rclcpp::QoS(2000),
            std::bind(&IMUPreintegrationNode::imuCallback, this, _1),
            imu_options);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", 
            rclcpp::QoS(2000),
            std::bind(&IMUPreintegrationNode::odometryCallback, this, _1),
            odom_options);

        // Create publishers
        imu_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "odometry/imu_incremental", 
            rclcpp::QoS(2000));

        // Transform broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "IMU Preintegration node initialized with GTSAM backend.");
    }

private:
    void initializeParameters()
    {
        // IMU noise parameters (you may want to make these ROS parameters)
        imu_acc_noise_ = 3.9939570888238808e-03;
        imu_gyr_noise_ = 1.5636343949698187e-03;
        imu_acc_bias_n_ = 6.4356659353532566e-05;
        imu_gyr_bias_n_ = 3.5640318696367613e-05;
        imu_gravity_ = 9.80511;
        
        // Frame IDs
        odometry_frame_ = "odom";
        baselink_frame_ = "base_link";
        
        // System state
        system_initialized_ = false;
        done_first_opt_ = false;
        last_imu_t_imu_ = -1.0;
        last_imu_t_opt_ = -1.0;
        key_ = 1;
        delta_t_ = 0.0;
    }

    void initializeGTSAM()
    {
        // Setup preintegration parameters
        auto p = gtsam::PreintegrationParams::MakeSharedU(imu_gravity_);
        p->accelerometerCovariance = gtsam::Matrix33::Identity(3,3) * pow(imu_acc_noise_, 2);
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3,3) * pow(imu_gyr_noise_, 2);
        p->integrationCovariance = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2);
        
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
        
        // Initialize integrators
        imu_integrator_opt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
        imu_integrator_imu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
        
        // Setup noise models
        prior_pose_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());
        prior_vel_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);
        prior_bias_noise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        correction_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());
        correction_noise2_ = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());
        noise_model_between_bias_ = (gtsam::Vector(6) << 
            imu_acc_bias_n_, imu_acc_bias_n_, imu_acc_bias_n_, 
            imu_gyr_bias_n_, imu_gyr_bias_n_, imu_gyr_bias_n_).finished();
        
        // Initialize optimizer
        resetOptimization();
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params opt_parameters;
        opt_parameters.relinearizeThreshold = 0.1;
        opt_parameters.relinearizeSkip = 1;
        optimizer_ = gtsam::ISAM2(opt_parameters);
        
        gtsam::NonlinearFactorGraph new_graph_factors;
        graph_factors_ = new_graph_factors;
        
        gtsam::Values new_graph_values;
        graph_values_ = new_graph_values;
    }

    void resetParams()
    {
        last_imu_t_imu_ = -1.0;
        done_first_opt_ = false;
        system_initialized_ = false;
    }

    double stamp2Sec(const builtin_interfaces::msg::Time& stamp)
    {
        return stamp.sec + stamp.nanosec * 1e-9;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // Store IMU data
        imu_queue_opt_.push_back(*msg);
        imu_queue_imu_.push_back(*msg);
        
        // Manage queue size
        if (imu_queue_opt_.size() > 2000) {
            imu_queue_opt_.pop_front();
        }
        if (imu_queue_imu_.size() > 2000) {
            imu_queue_imu_.pop_front();
        }

        if (!done_first_opt_) {
            return;
        }

        // Integrate IMU measurements for odometry prediction
        double imu_time = stamp2Sec(msg->header.stamp);
        double dt = (last_imu_t_imu_ < 0) ? (1.0 / 500.0) : (imu_time - last_imu_t_imu_);
        last_imu_t_imu_ = imu_time;

        // Integrate this single IMU message
        imu_integrator_imu_->integrateMeasurement(
            gtsam::Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
            gtsam::Vector3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z), 
            dt);

        // Predict odometry
        gtsam::NavState current_state = imu_integrator_imu_->predict(prev_state_odom_, prev_bias_odom_);

        // Publish odometry
        publishOdometry(current_state, *msg);

        RCLCPP_DEBUG(this->get_logger(), "Processed IMU at time: %.4f", imu_time);
    }

    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        double current_correction_time = stamp2Sec(odom_msg->header.stamp);

        // Make sure we have IMU data to integrate
        if (imu_queue_opt_.empty()) {
            return;
        }

        // Extract pose from odometry
        float p_x = odom_msg->pose.pose.position.x;
        float p_y = odom_msg->pose.pose.position.y;
        float p_z = odom_msg->pose.pose.position.z;
        float r_x = odom_msg->pose.pose.orientation.x;
        float r_y = odom_msg->pose.pose.orientation.y;
        float r_z = odom_msg->pose.pose.orientation.z;
        float r_w = odom_msg->pose.pose.orientation.w;
        
        bool degenerate = (int)odom_msg->pose.covariance[0] == 1;
        gtsam::Pose3 lidar_pose = gtsam::Pose3(
            gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), 
            gtsam::Point3(p_x, p_y, p_z));

        // Initialize system
        if (!system_initialized_) {
            initializeSystem(lidar_pose, current_correction_time);
            return;
        }

        // Optimize with new measurement
        optimizeWithMeasurement(lidar_pose, current_correction_time, degenerate);

        RCLCPP_DEBUG(this->get_logger(), "Processed odometry correction at time: %.4f", current_correction_time);
    }

    void initializeSystem(const gtsam::Pose3& lidar_pose, double current_time)
    {
        resetOptimization();

        // Pop old IMU messages
        while (!imu_queue_opt_.empty()) {
            if (stamp2Sec(imu_queue_opt_.front().header.stamp) < current_time - delta_t_) {
                last_imu_t_opt_ = stamp2Sec(imu_queue_opt_.front().header.stamp);
                imu_queue_opt_.pop_front();
            } else {
                break;
            }
        }

        // Initial pose (assuming lidar and IMU are co-located for simplicity)
        prev_pose_ = lidar_pose;
        gtsam::PriorFactor<gtsam::Pose3> prior_pose(X(0), prev_pose_, prior_pose_noise_);
        graph_factors_.add(prior_pose);

        // Initial velocity
        prev_vel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> prior_vel(V(0), prev_vel_, prior_vel_noise_);
        graph_factors_.add(prior_vel);

        // Initial bias
        prev_bias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> prior_bias(B(0), prev_bias_, prior_bias_noise_);
        graph_factors_.add(prior_bias);

        // Add values
        graph_values_.insert(X(0), prev_pose_);
        graph_values_.insert(V(0), prev_vel_);
        graph_values_.insert(B(0), prev_bias_);

        // Optimize once
        optimizer_.update(graph_factors_, graph_values_);
        graph_factors_.resize(0);
        graph_values_.clear();

        // Reset integrators
        imu_integrator_imu_->resetIntegrationAndSetBias(prev_bias_);
        imu_integrator_opt_->resetIntegrationAndSetBias(prev_bias_);

        key_ = 1;
        system_initialized_ = true;
        done_first_opt_ = true;

        // Initialize odometry prediction state
        prev_state_odom_ = gtsam::NavState(prev_pose_, prev_vel_);
        prev_bias_odom_ = prev_bias_;

        RCLCPP_INFO(this->get_logger(), "System initialized successfully");
    }

    void optimizeWithMeasurement(const gtsam::Pose3& lidar_pose, double current_time, bool degenerate)
    {
        // Integrate IMU data between optimizations
        while (!imu_queue_opt_.empty()) {
            sensor_msgs::msg::Imu* this_imu = &imu_queue_opt_.front();
            double imu_time = stamp2Sec(this_imu->header.stamp);
            
            if (imu_time < current_time - delta_t_) {
                double dt = (last_imu_t_opt_ < 0) ? (1.0 / 500.0) : (imu_time - last_imu_t_opt_);
                imu_integrator_opt_->integrateMeasurement(
                    gtsam::Vector3(this_imu->linear_acceleration.x, this_imu->linear_acceleration.y, this_imu->linear_acceleration.z),
                    gtsam::Vector3(this_imu->angular_velocity.x, this_imu->angular_velocity.y, this_imu->angular_velocity.z), 
                    dt);
                
                last_imu_t_opt_ = imu_time;
                imu_queue_opt_.pop_front();
            } else {
                break;
            }
        }

        // Add IMU factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = 
            dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imu_integrator_opt_);
        gtsam::ImuFactor imu_factor(X(key_ - 1), V(key_ - 1), X(key_), V(key_), B(key_ - 1), preint_imu);
        graph_factors_.add(imu_factor);

        // Add IMU bias between factor
        graph_factors_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            B(key_ - 1), B(key_), gtsam::imuBias::ConstantBias(),
            gtsam::noiseModel::Diagonal::Sigmas(sqrt(imu_integrator_opt_->deltaTij()) * noise_model_between_bias_)));

        // Add pose factor
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key_), lidar_pose, 
            degenerate ? correction_noise2_ : correction_noise_);
        graph_factors_.add(pose_factor);

        // Insert predicted values
        gtsam::NavState prop_state = imu_integrator_opt_->predict(prev_state_, prev_bias_);
        graph_values_.insert(X(key_), prop_state.pose());
        graph_values_.insert(V(key_), prop_state.v());
        graph_values_.insert(B(key_), prev_bias_);

        // Optimize
        optimizer_.update(graph_factors_, graph_values_);
        optimizer_.update();
        graph_factors_.resize(0);
        graph_values_.clear();

        // Update state
        gtsam::Values result = optimizer_.calculateEstimate();
        prev_pose_ = result.at<gtsam::Pose3>(X(key_));
        prev_vel_ = result.at<gtsam::Vector3>(V(key_));
        prev_state_ = gtsam::NavState(prev_pose_, prev_vel_);
        prev_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(key_));

        // Reset the optimization preintegration object
        imu_integrator_opt_->resetIntegrationAndSetBias(prev_bias_);

        // Check for failure
        if (failureDetection(prev_vel_, prev_bias_)) {
            resetParams();
            return;
        }

        // Update odometry prediction state
        updateOdometryPrediction(current_time);

        ++key_;
    }

    void updateOdometryPrediction(double current_time)
    {
        prev_state_odom_ = prev_state_;
        prev_bias_odom_ = prev_bias_;

        // Pop old IMU messages from the IMU queue
        double last_imu_qt = -1;
        while (!imu_queue_imu_.empty() && 
               stamp2Sec(imu_queue_imu_.front().header.stamp) < current_time - delta_t_) {
            last_imu_qt = stamp2Sec(imu_queue_imu_.front().header.stamp);
            imu_queue_imu_.pop_front();
        }

        // Re-propagate
        if (!imu_queue_imu_.empty()) {
            // Reset bias with newly optimized bias
            imu_integrator_imu_->resetIntegrationAndSetBias(prev_bias_odom_);
            
            // Integrate IMU messages from the beginning of this optimization
            for (size_t i = 0; i < imu_queue_imu_.size(); ++i) {
                sensor_msgs::msg::Imu* this_imu = &imu_queue_imu_[i];
                double imu_time = stamp2Sec(this_imu->header.stamp);
                double dt = (last_imu_qt < 0) ? (1.0 / 500.0) : (imu_time - last_imu_qt);

                imu_integrator_imu_->integrateMeasurement(
                    gtsam::Vector3(this_imu->linear_acceleration.x, this_imu->linear_acceleration.y, this_imu->linear_acceleration.z),
                    gtsam::Vector3(this_imu->angular_velocity.x, this_imu->angular_velocity.y, this_imu->angular_velocity.z), 
                    dt);
                last_imu_qt = imu_time;
            }
        }
    }

    bool failureDetection(const gtsam::Vector3& vel_cur, const gtsam::imuBias::ConstantBias& bias_cur)
    {
        Eigen::Vector3f vel(vel_cur.x(), vel_cur.y(), vel_cur.z());
        if (vel.norm() > 30) {
            RCLCPP_WARN(this->get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(bias_cur.accelerometer().x(), bias_cur.accelerometer().y(), bias_cur.accelerometer().z());
        Eigen::Vector3f bg(bias_cur.gyroscope().x(), bias_cur.gyroscope().y(), bias_cur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0) {
            RCLCPP_WARN(this->get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    void publishOdometry(const gtsam::NavState& current_state, const sensor_msgs::msg::Imu& imu_msg)
    {
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = imu_msg.header.stamp;
        odometry.header.frame_id = odometry_frame_;
        odometry.child_frame_id = "odom_imu";

        // Get pose from NavState
        gtsam::Pose3 imu_pose = gtsam::Pose3(current_state.quaternion(), current_state.position());

        odometry.pose.pose.position.x = imu_pose.translation().x();
        odometry.pose.pose.position.y = imu_pose.translation().y();
        odometry.pose.pose.position.z = imu_pose.translation().z();
        odometry.pose.pose.orientation.x = imu_pose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = imu_pose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = imu_pose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = imu_pose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = current_state.velocity().x();
        odometry.twist.twist.linear.y = current_state.velocity().y();
        odometry.twist.twist.linear.z = current_state.velocity().z();
        odometry.twist.twist.angular.x = imu_msg.angular_velocity.x + prev_bias_odom_.gyroscope().x();
        odometry.twist.twist.angular.y = imu_msg.angular_velocity.y + prev_bias_odom_.gyroscope().y();
        odometry.twist.twist.angular.z = imu_msg.angular_velocity.z + prev_bias_odom_.gyroscope().z();

        imu_odom_pub_->publish(odometry);
    }

    // Member variables
    std::mutex mtx_;
    
    // ROS components
    rclcpp::CallbackGroup::SharedPtr callback_group_imu_;
    rclcpp::CallbackGroup::SharedPtr callback_group_odom_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr imu_odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // IMU data queues
    std::deque<sensor_msgs::msg::Imu> imu_queue_opt_;
    std::deque<sensor_msgs::msg::Imu> imu_queue_imu_;
    
    // GTSAM components
    gtsam::ISAM2 optimizer_;
    gtsam::NonlinearFactorGraph graph_factors_;
    gtsam::Values graph_values_;
    gtsam::PreintegratedImuMeasurements* imu_integrator_opt_;
    gtsam::PreintegratedImuMeasurements* imu_integrator_imu_;
    
    // Noise models
    gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr prior_vel_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr prior_bias_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr correction_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr correction_noise2_;
    gtsam::Vector noise_model_between_bias_;
    
    // State variables
    gtsam::Pose3 prev_pose_;
    gtsam::Vector3 prev_vel_;
    gtsam::NavState prev_state_;
    gtsam::imuBias::ConstantBias prev_bias_;
    gtsam::NavState prev_state_odom_;
    gtsam::imuBias::ConstantBias prev_bias_odom_;
    
    // Parameters
    double imu_acc_noise_;
    double imu_gyr_noise_;
    double imu_acc_bias_n_;
    double imu_gyr_bias_n_;
    double imu_gravity_;
    std::string odometry_frame_;
    std::string baselink_frame_;
    
    // System state
    bool system_initialized_;
    bool done_first_opt_;
    double last_imu_t_imu_;
    double last_imu_t_opt_;
    int key_;
    double delta_t_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor executor;
    
    auto node = std::make_shared<IMUPreintegrationNode>();
    executor.add_node(node);
    
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}