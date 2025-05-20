#!/usr/bin/env python3
"""
Diagnostic version of the fusion node for Formula Student
This version focuses on analyzing the LiDAR data characteristics
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

# Message types
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray, Marker


class DiagnosticFusionNode(Node):
    def __init__(self):
        super().__init__('diagnostic_fusion_node')
        
        # Create callback groups to prevent deadlocks
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.subscription_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/carla/lidar',
            self.lidar_callback,
            sensor_qos,
            callback_group=self.subscription_callback_group
        )
        
        # Create publishers for visualization
        self.filtered_cloud_pub = self.create_publisher(
            PointCloud2,
            '/diagnostic/filtered_cloud',
            10
        )
        
        self.cone_markers_pub = self.create_publisher(
            MarkerArray,
            '/diagnostic/cone_markers',
            10
        )
        
        # State variables
        self.lidar_data = None
        self.lock = threading.RLock()
        self.full_analysis_done = False
        self.lidar_debug_count = 0
        
        # Set up analysis timer
        self.analysis_timer = self.create_timer(
            1.0, 
            self.analyze_lidar_callback,
            callback_group=self.timer_callback_group
        )
        
        self.get_logger().info("Diagnostic Fusion Node initialized")
        
    def lidar_callback(self, msg: PointCloud2):
        """Callback for LiDAR point cloud data."""
        with self.lock:
            self.lidar_data = msg
            
            # Check LiDAR message structure for debugging
            if self.lidar_debug_count < 3:
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
            
    def analyze_lidar_callback(self):
        """Analyze LiDAR data to understand its characteristics."""
        with self.lock:
            if self.lidar_data is None:
                self.get_logger().info("Waiting for LiDAR data...")
                return
                
            # Skip if we've already done a full analysis
            if self.full_analysis_done:
                return
                
            self.get_logger().info("=== Starting LiDAR Data Analysis ===")
            
            try:
                # Extract points
                points_xyz = []
                
                # Get field names from message
                field_names = [field.name for field in self.lidar_data.fields]
                
                if 'x' in field_names and 'y' in field_names and 'z' in field_names:
                    # Extract points using field names for xyz
                    for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
                        points_xyz.append([point[0], point[1], point[2]])
                    
                    self.get_logger().info(f"Extracted {len(points_xyz)} points using x,y,z fields")
                else:
                    # Try to extract using index positions
                    for point in pc2.read_points(self.lidar_data, field_names=None, skip_nans=True):
                        if isinstance(point, tuple) and len(point) >= 3:
                            points_xyz.append([point[0], point[1], point[2]])
                    
                    self.get_logger().info(f"Extracted {len(points_xyz)} points using index positions")
                
                # Check if we have points
                if not points_xyz:
                    self.get_logger().warn("No points extracted from LiDAR data")
                    return
                
                # Convert to numpy array
                points_array = np.array(points_xyz, dtype=np.float32)
                
                self.get_logger().info(f"Points array shape: {points_array.shape}")
                self.get_logger().info(f"Points array dtype: {points_array.dtype}")
                
                # Analyze point distribution
                # X distribution
                x_min, x_max = np.min(points_array[:, 0]), np.max(points_array[:, 0])
                x_mean, x_std = np.mean(points_array[:, 0]), np.std(points_array[:, 0])
                self.get_logger().info(f"X-range: min={x_min:.2f}, max={x_max:.2f}, mean={x_mean:.2f}, std={x_std:.2f}")
                
                # Y distribution
                y_min, y_max = np.min(points_array[:, 1]), np.max(points_array[:, 1])
                y_mean, y_std = np.mean(points_array[:, 1]), np.std(points_array[:, 1])
                self.get_logger().info(f"Y-range: min={y_min:.2f}, max={y_max:.2f}, mean={y_mean:.2f}, std={y_std:.2f}")
                
                # Z distribution
                z_min, z_max = np.min(points_array[:, 2]), np.max(points_array[:, 2])
                z_mean, z_std = np.mean(points_array[:, 2]), np.std(points_array[:, 2])
                self.get_logger().info(f"Z-range: min={z_min:.2f}, max={z_max:.2f}, mean={z_mean:.2f}, std={z_std:.2f}")
                
                # Histogram of Z values to understand height distribution
                z_values = points_array[:, 2]
                hist, bin_edges = np.histogram(z_values, bins=10)
                
                self.get_logger().info("Z-value histogram:")
                for i in range(len(hist)):
                    self.get_logger().info(f"  {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {hist[i]} points")
                
                # Distance analysis
                distances = np.sqrt(np.sum(points_array[:, :2]**2, axis=1))
                dist_min, dist_max = np.min(distances), np.max(distances)
                dist_mean, dist_std = np.mean(distances), np.std(distances)
                self.get_logger().info(f"Distance range: min={dist_min:.2f}, max={dist_max:.2f}, mean={dist_mean:.2f}, std={dist_std:.2f}")
                
                # Histogram of distances
                hist, bin_edges = np.histogram(distances, bins=10)
                
                self.get_logger().info("Distance histogram:")
                for i in range(len(hist)):
                    self.get_logger().info(f"  {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {hist[i]} points")
                
                # Analyze possible cone clusters
                self.get_logger().info("Analyzing potential cone clusters with various distance thresholds:")
                
                for distance_threshold in [0.3, 0.5, 1.0, 2.0]:
                    # Count number of clusters at different thresholds
                    clusters = self.find_clusters(points_array, distance_threshold, min_points=2)
                    self.get_logger().info(f"  Distance threshold {distance_threshold}m, min_points=2: {len(clusters)} clusters")
                    
                    # For the first threshold, publish markers
                    if distance_threshold == 0.5:
                        self.publish_potential_cones(clusters)
                
                # Mark the analysis as completed
                self.full_analysis_done = True
                self.get_logger().info("=== LiDAR Data Analysis Complete ===")
                
            except Exception as e:
                self.get_logger().error(f"Error analyzing LiDAR data: {str(e)}")
                import traceback
                self.get_logger().error(traceback.format_exc())
    
    def find_clusters(self, points: np.ndarray, distance_threshold: float, min_points: int) -> List[np.ndarray]:
        """
        Find clusters in the point cloud using a simple clustering algorithm.
        
        Parameters:
        -----------
        points : np.ndarray
            Nx3 array of points (x, y, z)
        distance_threshold : float
            Maximum distance between points to be considered part of the same cluster
        min_points : int
            Minimum number of points to constitute a valid cluster
            
        Returns:
        --------
        List[np.ndarray]
            List of cluster centroids (x, y, z)
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
                        # Use only XY distance for clustering
                        distance = np.linalg.norm(point[:2] - cluster_point[:2])
                        
                        if distance < distance_threshold:
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
            
            # If cluster has enough points, consider it valid
            if len(current_cluster) >= min_points:
                # Calculate the centroid of the cluster
                centroid = np.mean(current_cluster, axis=0)
                clusters.append((centroid, len(current_cluster)))
        
        return clusters
    
    def publish_potential_cones(self, clusters):
        """
        Publish markers for potential cones.
        
        Parameters:
        -----------
        clusters : List[Tuple[np.ndarray, int]]
            List of tuples containing cluster centroids and point counts
        """
        if not clusters:
            return
            
        marker_array = MarkerArray()
        
        for i, (cone_pos, point_count) in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cone_candidates"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(cone_pos[0])
            marker.pose.position.y = float(cone_pos[1])
            marker.pose.position.z = float(cone_pos[2])
            
            marker.pose.orientation.w = 1.0
            
            # Size of the cone
            marker.scale.x = 0.2  # Base diameter
            marker.scale.y = 0.2  # Base diameter
            marker.scale.z = 0.3  # Height
            
            # Color (scale from red to green based on point count)
            intensity = min(1.0, point_count / 10.0)  # Normalize
            marker.color.r = 1.0 - intensity
            marker.color.g = intensity
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            # Add label with point count
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "cone_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(cone_pos[0])
            text_marker.pose.position.y = float(cone_pos[1])
            text_marker.pose.position.z = float(cone_pos[2]) + 0.3
            
            text_marker.text = f"{point_count} pts"
            
            text_marker.scale.z = 0.2  # Text height
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            marker_array.markers.append(text_marker)
        
        self.cone_markers_pub.publish(marker_array)
        
        # Also publish the original point cloud
        self.publish_point_cloud()
        
    def publish_point_cloud(self):
        """Publish the original point cloud for visualization."""
        if not self.lidar_data:
            return
            
        # Create a copy of the point cloud
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        
        # Extract points
        points_xyz = []
        
        # Get field names from message
        field_names = [field.name for field in self.lidar_data.fields]
        
        if 'x' in field_names and 'y' in field_names and 'z' in field_names:
            # Extract points using field names for xyz
            for point in pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True):
                points_xyz.append([point[0], point[1], point[2]])
        else:
            # Try to extract using index positions
            for point in pc2.read_points(self.lidar_data, field_names=None, skip_nans=True):
                if isinstance(point, tuple) and len(point) >= 3:
                    points_xyz.append([point[0], point[1], point[2]])
        
        if not points_xyz:
            return
        
        # Convert to numpy array
        points_array = np.array(points_xyz, dtype=np.float32)
        
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


def main(args=None):
    rclpy.init(args=args)
    
    # Create node
    node = DiagnosticFusionNode()
    
    # Use a multithreaded executor
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
