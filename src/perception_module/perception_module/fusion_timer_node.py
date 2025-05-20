#!/usr/bin/env python3
"""
Sensor Fusion Node for Formula Student Driverless
This module handles sensor fusion between LiDAR, IMU, and GNSS data in the CARLA simulator.
No filtering approach to ensure cone detection works.
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
from std_msgs.msg import Header
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
        self.declare_parameter('fusion_rate', 5.0)           # Hz
        self.declare_parameter('gnss_weight', 0.3)            # Weight for GNSS in the fusion
        self.declare_parameter('imu_weight', 0.7)             # Weight for IMU in the fusion
        self.declare_parameter('cone_cluster_distance', 1.0)  # Max distance between points (increased)
        self.declare_parameter('min_points_per_cone', 1)      # Minimum points (reduced to 1)
        self.declare_parameter('use_ekf', True)               # Whether to use EKF for fusion
        self.declare_parameter('debug_mode', True)            # Enable more detailed logging
        self.declare_parameter('disable_filtering', True)     # Completely disable filtering
        
        # Get parameters
        self.fusion_rate = self.get_parameter('fusion_rate').value
        self.gnss_weight = self.get_parameter('gnss_weight').value
        self.imu_weight = self.get_parameter('imu_weight').value
        self.cone_cluster_distance = self.get_parameter('cone_cluster_distance').value
        self.min_points_per_cone = self.get_parameter('min_points_per_cone').value
        self.use_ekf = self.get_parameter('use_ekf').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.disable_filtering = self.get_parameter('disable_filtering').value
        
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
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State variables
        self.last_fusion_time = self.get_clock().now()
        self.last_lidar_process_time = self.get_clock().now()
        self.detected_cones = []  # List of tuples (cone_position, point_count)
        self.lidar_debug_count = 0  # Counter for debugging
        
        # Set up fusion timer
        self.fusion_timer = self.create_timer(
            1.0 / self.fusion_rate, 
            self.fusion_timer_callback,
            callback_group=self.timer_callback_group
        )
        
        # Set up cone detection timer (runs at 5Hz to save computational resources)
        self.cone_timer = self.create_timer(
            0.2,  # 5Hz 
            self.cone_detection_callback,
            callback_group=self.timer_callback_group
        )
        
        # EKF state
        if self.use_ekf:
            self.ekf_state = np.zeros(6)  # [x, y, z, vx, vy, vz]
            self.ekf_covariance = np.eye(6) * 0.1
        
        self.get_logger().info(f"Fusion timer node initialized at {self.fusion_rate}Hz")
        self.get_logger().info(f"Debug mode: {'Enabled' if self.debug_mode else 'Disabled'}")
        self.get_logger().info(f"Filtering: {'DISABLED' if self.disable_filtering else 'Enabled'}")
        self.get_logger().info(f"Cone cluster distance: {self.cone_cluster_distance}m")
        self.get_logger().info(f"Min points per cone: {self.min_points_per_cone}")
        
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
        """Process LiDAR data to detect cones. No filtering approach."""
        with self.sensor_data.lock:
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
                
                if self.debug_mode:
                    self.get_logger().info(f"Points array shape: {points_array.shape}")
                    self.get_logger().info(f"Points array dtype: {points_array.dtype}")
                    
                    # Print range of z values for debugging
                    if len(points_array) > 0:
                        z_min = np.min(points_array[:, 2])
                        z_max = np.max(points_array[:, 2])
                        self.get_logger().info(f"Z-range: min={z_min:.2f}m, max={z_max:.2f}m")
                
                # If filtering is completely disabled, use all points
                if self.disable_filtering:
                    # Skip all filtering
                    filtered_points = points_array
                    if self.debug_mode:
                        self.get_logger().info(f"Using ALL {len(filtered_points)} points (filtering disabled)")
                else:
                    # Apply filtering (not used by default now but kept for future use)
                    # Distance filtering
                    distances = np.sqrt(np.sum(points_array[:, :2]**2, axis=1))  # XY distance
                    distance_mask = distances < 50.0  # 50 meter max range
                    filtered_points = points_array[distance_mask]
                    
                    if self.debug_mode:
                        self.get_logger().info(f"After filtering: {len(filtered_points)} points remain")
                
                # Skip processing if we don't have enough points after filtering
                if len(filtered_points) < self.min_points_per_cone:
                    if self.debug_mode:
                        self.get_logger().info("Not enough points for cone detection")
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
                # Store centroid and point count
                clusters.append((centroid, len(current_cluster)))
                
                if self.debug_mode and len(clusters) == 1:
                    self.get_logger().info(f"First cone cluster centroid: {centroid}")
                    self.get_logger().info(f"First cone cluster has {len(current_cluster)} points")
        
        return clusters
        
    def publish_cone_markers(self):
        """Publish visualization markers for detected cones."""
        marker_array = MarkerArray()
        
        for i, (cone_pos, point_count) in enumerate(self.detected_cones):
            marker = Marker()
            marker.header.frame_id = "map"  # Adjust if needed
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cone_markers"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(cone_pos[0])
            marker.pose.position.y = float(cone_pos[1])
            marker.pose.position.z = float(cone_pos[2])
            
            marker.pose.orientation.w = 1.0
            
            # Size based on point count (bigger for more points)
            scale_factor = min(1.0, 0.1 + (point_count / 10.0))
            marker.scale.x = 0.3 * scale_factor  # Base diameter
            marker.scale.y = 0.3 * scale_factor  # Base diameter
            marker.scale.z = 0.5 * scale_factor  # Height
            
            # Color based on height (red for low, blue for high)
            height_normalized = max(0.0, min(1.0, (cone_pos[2] + 0.5) / 2.0))
            marker.color.r = 1.0 - height_normalized
            marker.color.g = 0.5
            marker.color.b = height_normalized
            marker.color.a = 0.8
            
            # Add text marker with point count
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "cone_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(cone_pos[0])
            text_marker.pose.position.y = float(cone_pos[1])
            text_marker.pose.position.z = float(cone_pos[2]) + 0.5  # Above the cone
            
            text_marker.text = f"{point_count}pts (z:{cone_pos[2]:.2f})"
            
            text_marker.scale.z = 0.2  # Text height
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            marker_array.markers.append(text_marker)
        
        # Add deletion markers if needed
        current_marker_count = len(self.detected_cones) * 2  # Each cone has a cylinder and text
        last_marker_id = len(marker_array.markers)
        
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