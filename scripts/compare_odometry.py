#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
import numpy as np
from tf_transformations import euler_from_quaternion
import math

class OdometryComparer(Node):
    def __init__(self):
        super().__init__('odometry_comparer')
        
        # Subscribe to filtered odometry
        self.filtered_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.filtered_callback,
            10
        )
        
        # Subscribe to GPS-derived odometry
        self.gps_odom_sub = self.create_subscription(
            Odometry,
            '/odometry/gps',
            self.gps_odom_callback,
            10
        )
        
        # Subscribe to raw IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/carla/imu_sensor',
            self.imu_callback,
            10
        )
        
        # Store the latest messages
        self.latest_filtered = None
        self.latest_gps_odom = None
        self.latest_imu = None
        
        # Create a timer to compare odometry periodically
        self.timer = self.create_timer(1.0, self.compare_odometry)
    
    def filtered_callback(self, msg):
        self.latest_filtered = msg
        
    def gps_odom_callback(self, msg):
        self.latest_gps_odom = msg
    
    def imu_callback(self, msg):
        self.latest_imu = msg
    
    def compare_odometry(self):
        if not self.latest_filtered or not self.latest_gps_odom or not self.latest_imu:
            self.get_logger().info("Waiting for all messages...")
            return
        
        # Compare filtered odometry with GPS-derived odometry
        self.compare_with_gps_odom()
        
        # Compare filtered orientation and velocity with IMU data
        self.compare_with_imu()
    
    def compare_with_gps_odom(self):
        # Extract position data
        gps_pos = self.latest_gps_odom.pose.pose.position
        filtered_pos = self.latest_filtered.pose.pose.position
        
        # Extract orientation (convert quaternions to euler angles)
        gps_quat = self.latest_gps_odom.pose.pose.orientation
        filtered_quat = self.latest_filtered.pose.pose.orientation
        
        gps_euler = euler_from_quaternion([gps_quat.x, gps_quat.y, gps_quat.z, gps_quat.w])
        filtered_euler = euler_from_quaternion([filtered_quat.x, filtered_quat.y, filtered_quat.z, filtered_quat.w])
        
        # Extract linear velocity
        gps_vel = self.latest_gps_odom.twist.twist.linear
        filtered_vel = self.latest_filtered.twist.twist.linear
        
        # Calculate differences
        pos_diff = np.sqrt((gps_pos.x - filtered_pos.x)**2 + 
                          (gps_pos.y - filtered_pos.y)**2 + 
                          (gps_pos.z - filtered_pos.z)**2)
        
        yaw_diff = abs(gps_euler[2] - filtered_euler[2])
        # Normalize angle difference to [-pi, pi]
        if yaw_diff > np.pi:
            yaw_diff = 2 * np.pi - yaw_diff
        
        vel_diff = np.sqrt((gps_vel.x - filtered_vel.x)**2 + 
                          (gps_vel.y - filtered_vel.y)**2 + 
                          (gps_vel.z - filtered_vel.z)**2)
        
        # Print comparison with GPS-derived odometry
        self.get_logger().info("----- Comparison with GPS Odometry -----")
        self.get_logger().info(f"Position difference: {pos_diff:.4f} m")
        self.get_logger().info(f"Yaw difference: {np.degrees(yaw_diff):.4f} degrees")
        self.get_logger().info(f"Velocity difference: {vel_diff:.4f} m/s")
        self.get_logger().info(f"GPS   - Pos: ({gps_pos.x:.2f}, {gps_pos.y:.2f}, {gps_pos.z:.2f})")
        self.get_logger().info(f"EKF   - Pos: ({filtered_pos.x:.2f}, {filtered_pos.y:.2f}, {filtered_pos.z:.2f})")
        self.get_logger().info(f"GPS   - Yaw: {np.degrees(gps_euler[2]):.2f} deg")
        self.get_logger().info(f"EKF   - Yaw: {np.degrees(filtered_euler[2]):.2f} deg")
        self.get_logger().info("---------------------------------------")
    
    def compare_with_imu(self):
        # Extract orientation data
        imu_quat = self.latest_imu.orientation
        filtered_quat = self.latest_filtered.pose.pose.orientation
        
        imu_euler = euler_from_quaternion([imu_quat.x, imu_quat.y, imu_quat.z, imu_quat.w])
        filtered_euler = euler_from_quaternion([filtered_quat.x, filtered_quat.y, filtered_quat.z, filtered_quat.w])
        
        # Extract angular velocity
        imu_ang_vel = self.latest_imu.angular_velocity
        filtered_ang_vel = self.latest_filtered.twist.twist.angular
        
        # Extract linear acceleration
        imu_accel = self.latest_imu.linear_acceleration
        
        # Calculate differences
        roll_diff = abs(imu_euler[0] - filtered_euler[0])
        pitch_diff = abs(imu_euler[1] - filtered_euler[1])
        yaw_diff = abs(imu_euler[2] - filtered_euler[2])
        
        # Normalize angle differences to [-pi, pi]
        if roll_diff > np.pi: roll_diff = 2 * np.pi - roll_diff
        if pitch_diff > np.pi: pitch_diff = 2 * np.pi - pitch_diff
        if yaw_diff > np.pi: yaw_diff = 2 * np.pi - yaw_diff
        
        ang_vel_diff = np.sqrt((imu_ang_vel.x - filtered_ang_vel.x)**2 + 
                              (imu_ang_vel.y - filtered_ang_vel.y)**2 + 
                              (imu_ang_vel.z - filtered_ang_vel.z)**2)
        
        # Print comparison with IMU data
        self.get_logger().info("----- Comparison with IMU Data -----")
        self.get_logger().info(f"Roll difference: {np.degrees(roll_diff):.4f} degrees")
        self.get_logger().info(f"Pitch difference: {np.degrees(pitch_diff):.4f} degrees")
        self.get_logger().info(f"Yaw difference: {np.degrees(yaw_diff):.4f} degrees")
        self.get_logger().info(f"Angular velocity difference: {ang_vel_diff:.4f} rad/s")
        self.get_logger().info(f"IMU   - Roll/Pitch/Yaw: ({np.degrees(imu_euler[0]):.2f}, {np.degrees(imu_euler[1]):.2f}, {np.degrees(imu_euler[2]):.2f}) deg")
        self.get_logger().info(f"EKF   - Roll/Pitch/Yaw: ({np.degrees(filtered_euler[0]):.2f}, {np.degrees(filtered_euler[1]):.2f}, {np.degrees(filtered_euler[2]):.2f}) deg")
        self.get_logger().info(f"IMU   - Ang Vel: ({imu_ang_vel.x:.4f}, {imu_ang_vel.y:.4f}, {imu_ang_vel.z:.4f}) rad/s")
        self.get_logger().info(f"EKF   - Ang Vel: ({filtered_ang_vel.x:.4f}, {filtered_ang_vel.y:.4f}, {filtered_ang_vel.z:.4f}) rad/s")
        self.get_logger().info(f"IMU   - Lin Accel: ({imu_accel.x:.4f}, {imu_accel.y:.4f}, {imu_accel.z:.4f}) m/s²")
        self.get_logger().info("-----------------------------------")


def main(args=None):
    rclpy.init(args=args)
    node = OdometryComparer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()