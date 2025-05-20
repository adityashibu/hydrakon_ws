#!/usr/bin/env python3
"""
Sensor Fusion Node for Formula Student Driverless
This module handles sensor fusion between LiDAR, IMU, and GNSS data in the CARLA simulator.
Optimized filtering for CARLA's specific coordinate system.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import time
import threading
import math
from typing import Optional, List, Dict, Tuple
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as R

# Message types
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, Point
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray, Marker


class SensorData:
    """Class to store the latest sensor data with timestamps."""
    
    def __init__(self):
        # Raw sensor data
        self.imu_data: Optional[Imu] = None
        self.gnss_data: Optional[NavSatFix] = None
        self.lidar_data: Optional[PointCloud2] = None
        self.odometry_data: Optional[Odometry] = None
        
        # Processed data
        self.filtered_pose: Optional[PoseStamped] = None
        self.vehicle_speed: float = 0.0
        self.vehicle_yaw: float = 0.0
        
        # Timestamps
        self.imu_timestamp: Optional[float] = None
        self.gnss_timestamp: Optional[float] = None
        self.lidar_timestamp: Optional[float] = None
        
        # Thread safety
        self.lock = threading.RLock()


class FusionTimerNode(Node):
    def __init__(self):
        super().__init__('fusion_timer_node')
        
        # Declare parameters
        self.declare_parameter('fusion_rate', 5.0)            # Hz
        self.declare_parameter('cone_detection_rate', 10.0)   # Hz for more real-time detection
        self.declare_parameter('gnss_weight', 0.3)            # Weight for GNSS in the fusion
        self.declare_parameter('imu_weight', 0.7)             # Weight for IMU in the fusion
        self.declare_parameter('cone_cluster_distance', 1.0)  # Max distance between points
        self.declare_parameter('min_points_per_cone', 1)      # Minimum points per cone
        self.declare_parameter('use_ekf', True)               # Whether to use EKF for fusion
        self.declare_parameter('debug_mode', True)            # Enable more detailed logging
        self.declare_parameter('filter_method', 'height_range')  # Filtering method: 'none', 'ground_plane', 'height_range'
        self.declare_parameter('max_cone_distance', 40.0)     # Maximum distance to show cones (meters)
        self.declare_parameter('show_cone_labels', True)      # Whether to show text labels on cones
        self.declare_parameter('cone_color_by_height', True)  # Color cones by height
        
        # Height range filtering parameters (adjusted for CARLA's negative Z values)
        self.declare_parameter('min_height', -3.0)  # Minimum height for cone detection (meters)
        self.declare_parameter('max_height', -1.0)  # Maximum height for cone detection (meters)
        
        # Ground plane filtering parameters
        self.declare_parameter('ground_z', -2.5)        # Approximate ground plane Z coordinate
        self.declare_parameter('ground_thickness', 0.3) # Thickness of ground plane to filter
        
        # Get parameters
        self.fusion_rate = self.get_parameter('fusion_rate').value
        self.cone_detection_rate = self.get_parameter('cone_detection_rate').value
        self.gnss_weight = self.get_parameter('gnss_weight').value
        self.imu_weight = self.get_parameter('imu_weight').value
        self.cone_cluster_distance = self.get_parameter('cone_cluster_distance').value
        self.min_points_per_cone = self.get_parameter('min_points_per_cone').value
        self.use_ekf = self.get_parameter('use_ekf').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.filter_method = self.get_parameter('filter_method').value
        self.max_cone_distance = self.get_parameter('max_cone_distance').value
        self.show_cone_labels = self.get_parameter('show_cone_labels').value
        self.cone_color_by_height = self.get_parameter('cone_color_by_height').value
        
        # Height filtering params
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        
        # Ground plane params
        self.ground_z = self.get_parameter('ground_z').value
        self.ground_thickness = self.get_parameter('ground_thickness').value
        
        # Create callback groups to prevent deadlocks
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.subscription_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Initialize sensor data
        self.sensor_data = SensorData()
        
        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            '/carla/imu_sensor',
            self.imu_callback,
            sensor_qos,
            callback_group=self.subscription_callback_group
        )
        
        self.gnss_sub = self.create_subscription(
            NavSatFix,
            '/carla/gnss',
            self.gnss_callback,
            sensor_qos,
            callback_group=self.subscription_callback_group
        )
        
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/carla/lidar',
            self.lidar_callback,
            sensor_qos,
            callback_group=self.subscription_callback_group
        )
        
        self.filtered_odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.filtered_odom_callback,
            10,
            callback_group=self.subscription_callback_group
        )
        
        # Create publishers
        self.fused_pose_pub = self.create_publisher(
            PoseStamped,
            '/fusion/pose',
            10
        )
        
        self.fused_velocity_pub = self.create_publisher(
            TwistStamped,
            '/fusion/velocity',
            10
        )
        
        self.cone_markers_pub = self.create_publisher(
            MarkerArray,
            '/fusion/cone_markers',
            10
        )
        
        self.cone_positions_pub = self.create_publisher(
            PointCloud2,
            '/fusion/cone_positions',
            10
        )
        
        # Raw point cloud for debugging
        self.raw_cloud_pub = self.create_publisher(
            PointCloud2,
            '/fusion/raw_cloud',
            10
        )
        
        # Filtered point cloud for debugging
        self.filtered_cloud_pub = self.create_publisher(
            PointCloud2,
            '/fusion/filtered_cloud',
            10
        )
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State variables
        self.last_fusion_time = self.get_clock().now()
        self.last_lidar_process_time = self.get_clock().now()
        self.detected_cones = []  # List of tuples (cone_position, point_count)
        self.lidar_debug_count = 0  # Counter for debugging
        self.marker_id_counter = 0  # Counter for marker IDs
        self.active_marker_ids = set()  # Track active marker IDs
        
        # Z-value statistics for adaptive filtering
        self.z_min = None
        self.z_max = None
        self.z_mean = None
        self.z_std = None
        self.z_stats_initialized = False
        
        # Set up fusion timer
        self.fusion_timer = self.create_timer(
            1.0 / self.fusion_rate, 
            self.fusion_timer_callback,
            callback_group=self.timer_callback_group
        )
        
        # Set up cone detection timer (runs at higher rate for more responsive visualization)
        self.cone_timer = self.create_timer(
            1.0 / self.cone_detection_rate,  # Higher rate for real-time visualization
            self.cone_detection_callback,
            callback_group=self.timer_callback_group
        )
        
        # EKF state
        if self.use_ekf:
            self.ekf_state = np.zeros(6)  # [x, y, z, vx, vy, vz]
            self.ekf_covariance = np.eye(6) * 0.1
        
        self.get_logger().info(f"Fusion timer node initialized at {self.fusion_rate}Hz")
        self.get_logger().info(f"Cone detection rate: {self.cone_detection_rate}Hz")
        self.get_logger().info(f"Filter method: {self.filter_method}")
        self.get_logger().info(f"Height range: {self.min_height}m to {self.max_height}m")
        self.get_logger().info(f"Ground plane: z={self.ground_z}m, thickness={self.ground_thickness}m")
        self.get_logger().info(f"Cone cluster distance: {self.cone_cluster_distance}m")
        self.get_logger().info(f"Min points per cone: {self.min_points_per_cone}")
        self.get_logger().info(f"Max cone distance: {self.max_cone_distance}m")
        
    def imu_callback(self, msg: Imu):
        """Callback for IMU sensor data."""
        with self.sensor_data.lock:
            self.sensor_data.imu_data = msg
            self.sensor_data.imu_timestamp = self.get_clock().now().nanoseconds / 1e9
            
    def gnss_callback(self, msg: NavSatFix):
        """Callback for GNSS (GPS) sensor data."""
        with self.sensor_data.lock:
            self.sensor_data.gnss_data = msg
            self.sensor_data.gnss_timestamp = self.get_clock().now().nanoseconds / 1e9
    
    def lidar_callback(self, msg: PointCloud2):
        """Callback for LiDAR point cloud data."""
        with self.sensor_data.lock:
            self.sensor_data.lidar_data = msg
            self.sensor_data.lidar_timestamp = self.get_clock().now().nanoseconds / 1e9
            
            # Check LiDAR message structure for debugging
            if self.debug_mode and (self.lidar_debug_count < 3):
                self.debug_lidar_structure(msg)
                self.lidar_debug_count += 1
    
    def debug_lidar_structure(self, msg: PointCloud2):
        """Print debug information about the LiDAR message structure."""
        try:
            self.get_logger().info(f"LiDAR message fields: {[f.name for f in msg.fields]}")
            self.get_logger().info(f"LiDAR message height: {msg.height}, width: {msg.width}")
            self.get_logger().info(f"LiDAR message point_step: {msg.point_step}, row_step: {msg.row_step}")
            self.get_logger().info(f"LiDAR message is_bigendian: {msg.is_bigendian}, is_dense: {msg.is_dense}")
            
            # Try to read the first few points to understand the structure
            try:
                points = list(pc2.read_points(msg, field_names=None, skip_nans=True))
                if points:
                    self.get_logger().info(f"First point: {points[0]}")
                    self.get_logger().info(f"Point type: {type(points[0])}")
                    
                    # Sample the first 5 points to see the range of values
                    if len(points) >= 5:
                        for i in range(min(5, len(points))):
                            if isinstance(points[i], tuple) and len(points[i]) >= 3:
                                self.get_logger().info(f"Sample point {i}: x={points[i][0]:.2f}, y={points[i][1]:.2f}, z={points[i][2]:.2f}")
                            else:
                                self.get_logger().info(f"Sample point {i}: {points[i]}")
            except Exception as e:
                self.get_logger().error(f"Error reading points: {str(e)}")
                
        except Exception as e:
            self.get_logger().error(f"Error in debug_lidar_structure: {str(e)}")
    
    def filtered_odom_callback(self, msg: Odometry):
        """Callback for filtered odometry (from robot_localization EKF)."""
        with self.sensor_data.lock:
            self.sensor_data.odometry_data = msg
            
            # Extract speed
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            self.sensor_data.vehicle_speed = math.sqrt(vx**2 + vy**2 + vz**2)
            
            # Extract orientation (yaw)
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            
            # Convert quaternion to Euler angles
            r = R.from_quat([qx, qy, qz, qw])
            euler = r.as_euler('xyz', degrees=False)
            self.sensor_data.vehicle_yaw = euler[2]  # Yaw is the rotation around Z
            
            # Create and store pose
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = msg.pose.pose
            self.sensor_data.filtered_pose = pose
            
    def fusion_timer_callback(self):
        """Main sensor fusion processing function."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_fusion_time).nanoseconds / 1e9
        self.last_fusion_time = current_time
        
        with self.sensor_data.lock:
            # Check if we have received all required sensor data
            if (self.sensor_data.imu_data is None or 
                self.sensor_data.gnss_data is None):
                self.get_logger().warn("Missing sensor data for fusion")
                return
                
            # Get latest sensor data
            imu_data = self.sensor_data.imu_data
            gnss_data = self.sensor_data.gnss_data
            
            # If we're using the EKF and have received odometry, use that
            if self.use_ekf and self.sensor_data.odometry_data is not None:
                self.update_ekf(dt)
                
                # Publish fused pose using EKF state
                self.publish_fused_state()
            else:
                # Simple fusion: Use weighted average of IMU and GNSS
                self.simple_fusion()
        
    def update_ekf(self, dt: float):
        """Update the EKF state with the latest sensor measurements."""
        # This is a simplified EKF implementation
        # A full implementation would be more complex
        
        # State transition model (constant velocity)
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Process noise (assumed constant)
        Q = np.eye(6) * 0.01
        
        # Predict step
        x_pred = F @ self.ekf_state
        P_pred = F @ self.ekf_covariance @ F.T + Q
        
        if self.sensor_data.odometry_data is not None:
            # Measurement from odometry
            odom = self.sensor_data.odometry_data
            z = np.array([
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.linear.z
            ])
            
            # Measurement model (direct observation of state)
            H = np.eye(6)
            
            # Measurement noise
            R = np.eye(6) * 0.1
            
            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            
            # Update step
            self.ekf_state = x_pred + K @ (z - H @ x_pred)
            self.ekf_covariance = (np.eye(6) - K @ H) @ P_pred
        else:
            # No measurement, just use prediction
            self.ekf_state = x_pred
            self.ekf_covariance = P_pred
    
    def simple_fusion(self):
        """Perform simple weighted fusion of IMU and GNSS data."""
        if self.sensor_data.filtered_pose is not None:
            # We already have filtered pose from robot_localization
            # Just publish it as our fused pose
            self.fused_pose_pub.publish(self.sensor_data.filtered_pose)
            
            # Create and publish velocity
            if self.sensor_data.odometry_data is not None:
                twist = TwistStamped()
                twist.header = self.sensor_data.odometry_data.header
                twist.twist = self.sensor_data.odometry_data.twist.twist
                self.fused_velocity_pub.publish(twist)
        else:
            # Create a simple pose from GNSS data
            # In a real implementation, you would convert lat/lon to map coordinates
            # This is a placeholder that assumes you've already handled this conversion
            if self.sensor_data.gnss_data is not None:
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = "map"
                
                # Convert GNSS to a local coordinate system (simplified)
                # In a real world scenario, you would need to implement UTM conversion
                gnss = self.sensor_data.gnss_data
                
                # Placeholder: use relative position from a reference point
                # This is not correct for real-world use but serves as example
                pose.pose.position.x = 0.0  # Should convert from lat/lon
                pose.pose.position.y = 0.0  # Should convert from lat/lon
                pose.pose.position.z = 0.0
                
                # Use orientation from IMU if available
                if self.sensor_data.imu_data is not None:
                    pose.pose.orientation = self.sensor_data.imu_data.orientation
                
                self.fused_pose_pub.publish(pose)
            
                # Create a simple velocity estimate from IMU
                if self.sensor_data.imu_data is not None:
                    twist = TwistStamped()
                    twist.header.stamp = self.get_clock().now().to_msg()
                    twist.header.frame_id = "base_link"
                    
                    # Linear acceleration integrated to velocity
                    # This is simplified and would need proper integration in real use
                    accel = self.sensor_data.imu_data.linear_acceleration
                    twist.twist.linear.x = accel.x * 0.1  # Simple integration
                    twist.twist.linear.y = accel.y * 0.1
                    twist.twist.linear.z = accel.z * 0.1
                    
                    # Angular velocity directly from IMU
                    twist.twist.angular = self.sensor_data.imu_data.angular_velocity
                    
                    self.fused_velocity_pub.publish(twist)
                    
    def publish_fused_state(self):
        """Publish the fused state estimate from the EKF."""
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = self.ekf_state[0]
        pose_msg.pose.position.y = self.ekf_state[1]
        pose_msg.pose.position.z = self.ekf_state[2]
        
        # For orientation, use the orientation from filtered odometry if available
        if self.sensor_data.odometry_data is not None:
            pose_msg.pose.orientation = self.sensor_data.odometry_data.pose.pose.orientation
        else:
            # Default orientation (identity quaternion)
            pose_msg.pose.orientation.w = 1.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            
        self.fused_pose_pub.publish(pose_msg)
        
        # Publish velocity
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        
        twist_msg.twist.linear.x = self.ekf_state[3]
        twist_msg.twist.linear.y = self.ekf_state[4]
        twist_msg.twist.linear.z = self.ekf_state[5]
        
        # For angular velocity, use the angular velocity from IMU if available
        if self.sensor_data.imu_data is not None:
            twist_msg.twist.angular = self.sensor_data.imu_data.angular_velocity
            
        self.fused_velocity_pub.publish(twist_msg)
            
    def cone_detection_callback(self):
        """Process LiDAR data to detect cones with improved filtering."""
        with self.sensor_data.lock:
            self.detected_cones = []

            if self.sensor_data.lidar_data is None:
                return
                
            # Get the LiDAR data
            lidar_data = self.sensor_data.lidar_data
            
            try:
                # Extract the xyz coordinates from the point cloud
                points_xyz = []
                
                # Get field names from message
                field_names = [field.name for field in lidar_data.fields]
                
                if 'x' in field_names and 'y' in field_names and 'z' in field_names:
                    # Extract points using field names for xyz
                    for point in pc2.read_points(lidar_data, field_names=("x", "y", "z"), skip_nans=True):
                        points_xyz.append([point[0], point[1], point[2]])
                    
                    if self.debug_mode:
                        self.get_logger().info(f"Extracted {len(points_xyz)} points using x,y,z fields")
                else:
                    # Try to extract using index positions (assuming xyz are the first 3 values)
                    for point in pc2.read_points(lidar_data, field_names=None, skip_nans=True):
                        if isinstance(point, tuple) and len(point) >= 3:
                            points_xyz.append([point[0], point[1], point[2]])
                    
                    if self.debug_mode:
                        self.get_logger().info(f"Extracted {len(points_xyz)} points using index positions")
                
                # Check if we have points
                if not points_xyz:
                    self.get_logger().warn("No points extracted from LiDAR data")
                    return
                
                # Convert to numpy array
                points_array = np.array(points_xyz, dtype=np.float32)
                
                # Publish raw point cloud for visualization
                self.publish_raw_cloud(points_array)
                
                # Calculate Z statistics for adaptive filtering if not initialized
                if not self.z_stats_initialized and len(points_array) > 10:
                    self.z_min = np.min(points_array[:, 2])
                    self.z_max = np.max(points_array[:, 2])
                    self.z_mean = np.mean(points_array[:, 2])
                    self.z_std = np.std(points_array[:, 2])
                    self.z_stats_initialized = True
                    
                    # Update filtering parameters if using adaptive filtering
                    if self.filter_method == 'adaptive':
                        # Set ground plane level at the mean Z minus some margin
                        self.ground_z = self.z_mean - 0.3 * self.z_std
                        # Set height range to be mean +/- 2 standard deviations
                        self.min_height = self.z_mean - 2.0 * self.z_std
                        self.max_height = self.z_mean + 2.0 * self.z_std
                        
                        self.get_logger().info(f"Adaptive filtering parameters updated:")
                        self.get_logger().info(f"  Z stats: min={self.z_min:.2f}, max={self.z_max:.2f}, mean={self.z_mean:.2f}, std={self.z_std:.2f}")
                        self.get_logger().info(f"  Ground plane: z={self.ground_z:.2f}")
                        self.get_logger().info(f"  Height range: min={self.min_height:.2f}, max={self.max_height:.2f}")
                
                if self.debug_mode:
                    self.get_logger().info(f"Points array shape: {points_array.shape}")
                    
                    # Print range of z values for debugging
                    if len(points_array) > 0:
                        z_min = np.min(points_array[:, 2])
                        z_max = np.max(points_array[:, 2])
                        self.get_logger().info(f"Z-range: min={z_min:.2f}m, max={z_max:.2f}m")
                
                # Initial distance filtering (only keep points within max_cone_distance)
                distances = np.sqrt(np.sum(points_array[:, :2]**2, axis=1))  # XY distance
                distance_mask = distances < self.max_cone_distance  # Only use points within range
                filtered_by_distance = points_array[distance_mask]
                
                if self.debug_mode:
                    self.get_logger().info(f"After distance filtering ({self.max_cone_distance}m): {len(filtered_by_distance)} of {len(points_array)} points")
                
                # Apply filtering based on the selected method
                if self.filter_method == 'none':
                    # No additional filtering
                    filtered_points = filtered_by_distance
                    if self.debug_mode:
                        self.get_logger().info(f"No additional filtering applied")
                
                elif self.filter_method == 'height_range':
                    # Filter by height range (specific to CARLA's coordinate system)
                    height_mask = (filtered_by_distance[:, 2] >= self.min_height) & (filtered_by_distance[:, 2] <= self.max_height)
                    filtered_points = filtered_by_distance[height_mask]
                    
                    if self.debug_mode:
                        self.get_logger().info(f"After height range filtering ({self.min_height}m to {self.max_height}m): {len(filtered_points)} points")
                
                elif self.filter_method == 'ground_plane':
                    # Filter out ground plane points
                    # Keep points that are NOT near the ground_z value
                    non_ground_mask = np.abs(filtered_by_distance[:, 2] - self.ground_z) > self.ground_thickness
                    filtered_points = filtered_by_distance[non_ground_mask]
                    
                    if self.debug_mode:
                        self.get_logger().info(f"After ground plane filtering (z={self.ground_z}±{self.ground_thickness}m): {len(filtered_points)} points")
                
                else:  # Default to no filtering
                    filtered_points = filtered_by_distance
                
                # Publish filtered cloud for visualization
                self.publish_filtered_cloud(filtered_points)
                
                # Skip processing if we don't have enough points after filtering
                if len(filtered_points) < self.min_points_per_cone:
                    if self.debug_mode:
                        self.get_logger().info("Not enough points for cone detection")
                    # Clear existing markers if we don't have any cones
                    self.clear_all_markers()
                    return
                
                # Simplified DBSCAN-like clustering for cones
                cones = self.simple_cluster_points(filtered_points)
                
                if self.debug_mode:
                    self.get_logger().info(f"Detected {len(cones)} potential cones")
                
                # Update detected cones
                self.detected_cones = cones
                
                # Publish cone markers for visualization
                self.publish_cone_markers()
                
                # Publish cone positions as point cloud
                self.publish_cone_positions()
                
            except Exception as e:
                self.get_logger().error(f"Error processing LiDAR data: {str(e)}")
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def publish_raw_cloud(self, points_array: np.ndarray):
        """Publish raw point cloud for visualization."""
        # Create point cloud message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"  # Adjust frame if needed
        
        # Create fields for PointCloud2
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        
        # Create structured array for points
        cloud_points = np.zeros(len(points_array), 
                              dtype=[
                                  ('x', np.float32),
                                  ('y', np.float32),
                                  ('z', np.float32)
                              ])
        
        # Fill structured array
        cloud_points['x'] = points_array[:, 0]
        cloud_points['y'] = points_array[:, 1]
        cloud_points['z'] = points_array[:, 2]
        
        # Create PointCloud2 message
        pc_msg = pc2.create_cloud(header, fields, cloud_points)
        self.raw_cloud_pub.publish(pc_msg)
    
    def publish_filtered_cloud(self, points_array: np.ndarray):
        """Publish filtered point cloud for visualization."""
        if len(points_array) == 0:
            return
            
        # Create point cloud message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"  # Adjust frame if needed
        
        # Create fields for PointCloud2
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        
        # Create structured array for points
        cloud_points = np.zeros(len(points_array), 
                              dtype=[
                                  ('x', np.float32),
                                  ('y', np.float32),
                                  ('z', np.float32)
                              ])
        
        # Fill structured array
        cloud_points['x'] = points_array[:, 0]
        cloud_points['y'] = points_array[:, 1]
        cloud_points['z'] = points_array[:, 2]
        
        # Create PointCloud2 message
        pc_msg = pc2.create_cloud(header, fields, cloud_points)
        self.filtered_cloud_pub.publish(pc_msg)
                
    def simple_cluster_points(self, points: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Simple clustering algorithm to group points into cones.
        
        Parameters:
        -----------
        points : np.ndarray
            Nx3 array of points (x, y, z)
            
        Returns:
        --------
        List[Tuple[np.ndarray, int]]
            List of tuples containing (cone_position, point_count)
        """
        if len(points) == 0:
            return []
            
        # List to store clusters
        clusters = []
        
        # Process each point
        remaining_points = points.copy()
        
        while len(remaining_points) > 0:
            # Start a new cluster with the first point
            current_cluster = [remaining_points[0]]
            
            # If we only have one point left, handle it specially
            if len(remaining_points) == 1:
                remaining_points = np.empty((0, 3))
            else:
                remaining_points = remaining_points[1:]
            
            # Flag to track if we added any points in this iteration
            added_points = True
            
            # Keep adding points to the current cluster until no more are added
            while added_points and len(remaining_points) > 0:
                added_points = False
                
                # List to track points to remove from remaining_points
                points_to_remove = []
                
                # Check each remaining point
                for i, point in enumerate(remaining_points):
                    # Check distance to any point in the current cluster
                    for cluster_point in current_cluster:
                        # Use only XY distance for clustering (ignore height)
                        distance = np.linalg.norm(point[:2] - cluster_point[:2])
                        
                        if distance < self.cone_cluster_distance:
                            # Add point to cluster
                            current_cluster.append(point)
                            points_to_remove.append(i)
                            added_points = True
                            break
                
                # Remove points added to cluster (safely)
                if len(points_to_remove) > 0:
                    remaining_points = np.delete(remaining_points, points_to_remove, axis=0)
                    if len(remaining_points) == 0:
                        # Reshape to keep dimensions consistent
                        remaining_points = np.empty((0, 3))
            
            # If cluster has enough points, consider it a cone
            if len(current_cluster) >= self.min_points_per_cone:
                # Calculate the centroid of the cluster
                centroid = np.mean(current_cluster, axis=0)
                
                # Only include cones within the maximum distance
                distance_to_origin = np.linalg.norm(centroid[:2])
                if distance_to_origin <= self.max_cone_distance:
                    # Store centroid and point count
                    clusters.append((centroid, len(current_cluster)))
                
                if self.debug_mode and len(clusters) == 1:
                    self.get_logger().info(f"First cone cluster centroid: {centroid}")
                    self.get_logger().info(f"First cone cluster has {len(current_cluster)} points")
        
        return clusters
    
    def clear_all_markers(self):
        """Clear all existing markers."""
        if not self.active_marker_ids:
            return
            
        marker_array = MarkerArray()
        
        # Create a deletion marker for each active marker ID
        for marker_id in self.active_marker_ids:
            # Delete cylinder marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cone_markers"
            marker.id = marker_id
            marker.action = Marker.DELETE
            
            marker_array.markers.append(marker)
            
            # Delete text marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "cone_labels"
            text_marker.id = marker_id
            text_marker.action = Marker.DELETE
            
            marker_array.markers.append(text_marker)
        
        # Publish the deletion markers
        self.cone_markers_pub.publish(marker_array)
        
        # Clear the active marker IDs
        self.active_marker_ids.clear()
        
    def publish_cone_markers(self):
        """Publish visualization markers for detected cones."""
        self.clear_all_markers()

        marker_array = MarkerArray()
        
        # Track new marker IDs
        new_marker_ids = set()
        
        for i, (cone_pos, point_count) in enumerate(self.detected_cones):
            # Use continuously increasing IDs
            marker_id = i
            new_marker_ids.add(marker_id)
            
            # Calculate distance for coloring and size
            distance = np.linalg.norm(cone_pos[:2])
            distance_factor = float(max(0.0, 1.0 - (distance / self.max_cone_distance)))
            
            # Cylinder marker for cone
            marker = Marker()
            marker.header.frame_id = "map"  # Adjust if needed
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cone_markers"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(cone_pos[0])
            marker.pose.position.y = float(cone_pos[1])
            marker.pose.position.z = float(cone_pos[2])
            
            marker.pose.orientation.w = 1.0
            
            # Size based on point count and distance (bigger for closer cones)
            point_scale = float(min(1.0, 0.1 + (point_count / 10.0)))
            marker.scale.x = float(0.3 * point_scale * (0.5 + 0.5 * distance_factor))  # Base diameter
            marker.scale.y = float(0.3 * point_scale * (0.5 + 0.5 * distance_factor))  # Base diameter
            marker.scale.z = float(0.5 * point_scale * (0.5 + 0.5 * distance_factor))  # Height
            
            # Color based on different schemes
            if self.cone_color_by_height:
                # Color by relative height 
                # Normalize based on the observed z range
                if self.z_stats_initialized:
                    height_normalized = float((cone_pos[2] - self.z_min) / (self.z_max - self.z_min)) if self.z_max > self.z_min else 0.5
                else:
                    # Default normalization if stats not available
                    height_normalized = float(max(0.0, min(1.0, (cone_pos[2] - self.min_height) / (self.max_height - self.min_height))))
                
                marker.color.r = float(1.0 - height_normalized)
                marker.color.g = 0.5
                marker.color.b = float(height_normalized)
            else:
                # Color by distance (red for close, green for far)
                marker.color.r = float(distance_factor)
                marker.color.g = float(1.0 - distance_factor)
                marker.color.b = 0.0
            
            # Alpha depends on distance (more transparent as distance increases)
            marker.color.a = float(0.8 * distance_factor + 0.2)  # Never completely transparent
            
            marker_array.markers.append(marker)
            
            # Add text marker with point count and distance if enabled
            if self.show_cone_labels:
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "cone_labels"
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = float(cone_pos[0])
                text_marker.pose.position.y = float(cone_pos[1])
                text_marker.pose.position.z = float(cone_pos[2]) + 0.5  # Above the cone
                
                text_marker.text = f"{point_count}pts {distance:.1f}m z:{cone_pos[2]:.1f}"
                
                # Size depends on distance (smaller for further cones)
                text_marker.scale.z = float(0.2 * (0.5 + 0.5 * distance_factor))  # Text height
                
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = float(distance_factor)  # More transparent with distance
                
                marker_array.markers.append(text_marker)
        
        # Keep track of active markers
        self.active_marker_ids = new_marker_ids
        
        # Publish the marker array
        self.cone_markers_pub.publish(marker_array)
        
    def publish_cone_positions(self):
        """Publish cone positions as a point cloud."""
        if not self.detected_cones:
            return
            
        # Create point cloud from cone positions
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        
        # Create fields for PointCloud2
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        ]
        
        # Create structured array for points
        cone_points = np.zeros(len(self.detected_cones), 
                             dtype=[
                                 ('x', np.float32),
                                 ('y', np.float32),
                                 ('z', np.float32)
                             ])
        
        # Fill structured array
        for i, (cone, _) in enumerate(self.detected_cones):
            cone_points[i]['x'] = float(cone[0])
            cone_points[i]['y'] = float(cone[1])
            cone_points[i]['z'] = float(cone[2])
        
        # Create PointCloud2 message
        pc_msg = pc2.create_cloud(header, fields, cone_points)
        self.cone_positions_pub.publish(pc_msg)
        
    def get_transform(self, target_frame: str, source_frame: str):
        """Get transform between two frames."""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                now,
                rclpy.duration.Duration(seconds=1.0)
            )
            return trans
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform {source_frame} to {target_frame}: {ex}")
            return None

def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    node = FusionTimerNode()
    
    # Use a multithreaded executor to prevent callback deadlocks
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        # Spin the node
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in main: {str(e)}")
        import traceback
        node.get_logger().error(traceback.format_exc())
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()