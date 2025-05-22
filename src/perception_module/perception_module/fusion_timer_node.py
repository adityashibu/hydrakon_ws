#!/usr/bin/env python3
"""
Sensor Fusion Node for Formula Student Driverless
This module handles sensor fusion between LiDAR, IMU, GNSS, and Camera data in the CARLA simulator.
Optimized for cone detection and tracking.
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
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from sklearn.cluster import DBSCAN
import os

# Message types
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry, Path
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
        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        
        # Processed data
        self.filtered_pose: Optional[PoseStamped] = None
        self.vehicle_speed: float = 0.0
        self.vehicle_yaw: float = 0.0
        
        # Cone detections
        self.camera_cones = []  # From YOLO
        self.lidar_cones = []   # From LiDAR clustering
        self.fused_cones = []   # Combined detections
        
        # Timestamps
        self.imu_timestamp: Optional[float] = None
        self.gnss_timestamp: Optional[float] = None
        self.lidar_timestamp: Optional[float] = None
        self.rgb_timestamp: Optional[float] = None
        self.depth_timestamp: Optional[float] = None
        
        # Thread safety
        self.lock = threading.RLock()
        self.rgb_lock = threading.RLock()
        self.depth_lock = threading.RLock()
        self.lidar_lock = threading.RLock()
        self.cone_lock = threading.RLock()


class FusionTimerNode(Node):
    def __init__(self):
        super().__init__('fusion_timer_node')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_parameter('fusion_rate', 20.0)            # Hz
        self.declare_parameter('cone_detection_rate', 10.0)    # Hz for cone detection
        self.declare_parameter('model_path', '')               # Path to YOLO model
        self.declare_parameter('use_carla', True)              # Whether to use CARLA simulator
        self.declare_parameter('carla.host', 'localhost')      # CARLA host
        self.declare_parameter('carla.port', 2000)             # CARLA port
        self.declare_parameter('output_dir', './output')       # Output directory
        self.declare_parameter('show_opencv_windows', True)    # Show OpenCV windows
        self.declare_parameter('lidar_point_size', 0.4)        # Point size for visualization
        self.declare_parameter('accumulate_lidar_frames', 3)   # Number of frames to accumulate
        
        # Cone detection parameters
        self.declare_parameter('max_cone_distance', 40.0)      # Maximum distance to detect cones
        self.declare_parameter('cone_cluster_distance', 0.5)   # Distance threshold for clustering
        self.declare_parameter('min_points_per_cone', 5)       # Minimum points per cone
        self.declare_parameter('show_cone_labels', True)       # Show labels on cones
        self.declare_parameter('cone_color_by_height', True)   # Color cones by height
        
        # Filtering parameters
        self.declare_parameter('filter_method', 'height_range') # Filtering method
        self.declare_parameter('min_height', -3.0)             # Min height for cone detection
        self.declare_parameter('max_height', -1.0)             # Max height for cone detection
        self.declare_parameter('ground_z', -2.5)               # Ground plane z coordinate
        self.declare_parameter('ground_thickness', 0.3)        # Ground plane thickness
        
        # YOLO parameters
        self.declare_parameter('yolo.confidence', 0.5)         # Confidence threshold
        self.declare_parameter('yolo.nms_threshold', 0.45)     # NMS threshold
        self.declare_parameter('yolo.img_size', 640)           # Image size for YOLO
        
        # Camera parameters
        self.declare_parameter('camera_fx', 700.0)             # Focal length x
        self.declare_parameter('camera_fy', 700.0)             # Focal length y
        self.declare_parameter('camera_cx', 640.0)             # Principal point x
        self.declare_parameter('camera_cy', 360.0)             # Principal point y
        
        # Get parameters
        self.fusion_rate = self.get_parameter('fusion_rate').value
        self.cone_detection_rate = self.get_parameter('cone_detection_rate').value
        self.model_path = self.get_parameter('model_path').value
        self.use_carla = self.get_parameter('use_carla').value
        self.carla_host = self.get_parameter('carla.host').value
        self.carla_port = self.get_parameter('carla.port').value
        self.output_dir = self.get_parameter('output_dir').value
        self.show_opencv_windows = self.get_parameter('show_opencv_windows').value
        self.lidar_point_size = self.get_parameter('lidar_point_size').value
        self.accumulate_frames = self.get_parameter('accumulate_lidar_frames').value
        
        # Cone detection parameters
        self.max_cone_distance = self.get_parameter('max_cone_distance').value
        self.cone_cluster_distance = self.get_parameter('cone_cluster_distance').value
        self.min_points_per_cone = self.get_parameter('min_points_per_cone').value
        self.show_cone_labels = self.get_parameter('show_cone_labels').value
        self.cone_color_by_height = self.get_parameter('cone_color_by_height').value
        
        # Filtering parameters
        self.filter_method = self.get_parameter('filter_method').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.ground_z = self.get_parameter('ground_z').value
        self.ground_thickness = self.get_parameter('ground_thickness').value
        
        # YOLO parameters
        self.yolo_confidence = self.get_parameter('yolo.confidence').value
        self.yolo_nms_threshold = self.get_parameter('yolo.nms_threshold').value
        self.yolo_img_size = self.get_parameter('yolo.img_size').value
        
        # Camera parameters
        self.camera_fx = self.get_parameter('camera_fx').value
        self.camera_fy = self.get_parameter('camera_fy').value
        self.camera_cx = self.get_parameter('camera_cx').value
        self.camera_cy = self.get_parameter('camera_cy').value
        
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
        
        # Load YOLO model if path provided
        self.yolo_model = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
                self.yolo_model.conf = self.yolo_confidence  # Confidence threshold
                self.yolo_model.iou = self.yolo_nms_threshold  # NMS threshold
                self.get_logger().info(f"YOLO model loaded from {self.model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {str(e)}")
        else:
            self.get_logger().warn(f"YOLO model path not provided or does not exist: {self.model_path}")
        
        # Create subscribers
        # LiDAR subscriber
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/carla/lidar',
            self.lidar_callback,
            sensor_qos,
            callback_group=self.subscription_callback_group
        )
        
        # Camera subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/zed2i/rgb/image',
            self.rgb_callback,
            10,
            callback_group=self.subscription_callback_group
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/zed2i/depth/image',
            self.depth_callback,
            10,
            callback_group=self.subscription_callback_group
        )
        
        # Other sensor subscribers
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
        
        self.filtered_odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.filtered_odom_callback,
            10,
            callback_group=self.subscription_callback_group
        )
        
        # Create publishers
        # Vision and LiDAR publishers
        self.rgb_pub = self.create_publisher(Image, '/carla/rgb_image', 10)
        self.depth_pub = self.create_publisher(Image, '/carla/depth_image', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/carla/lidar_points', 10)
        self.fused_pub = self.create_publisher(PointCloud2, '/carla/fused_points', 10)
        
        # Cone publishers
        self.cone_marker_pub = self.create_publisher(MarkerArray, '/carla/cone_markers', 10)
        self.lidar_cone_pub = self.create_publisher(MarkerArray, '/carla/lidar_cones', 10)
        self.camera_cone_pub = self.create_publisher(MarkerArray, '/carla/camera_cones', 10)
        self.fused_cone_pub = self.create_publisher(MarkerArray, '/carla/fused_cones', 10)
        
        # Path visualization
        self.path_pub = self.create_publisher(Path, '/carla/path', 10)
        self.path_img_pub = self.create_publisher(Image, '/carla/path_image', 10)
        
        # Debug visualization
        self.debug_img_pub = self.create_publisher(Image, '/carla/debug_image', 10)
        
        # Transform broadcaster for tf tree
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State variables
        self.last_fusion_time = self.get_clock().now()
        self.lidar_data = None
        self.lidar_lock = threading.Lock()
        self.lidar_debug_count = 0
        
        # LiDAR history for point accumulation
        self.lidar_history = []
        self.lidar_history_lock = threading.Lock()
        
        # Marker tracking
        self.active_marker_ids = set()
        self.lidar_marker_ids = set()
        self.camera_marker_ids = set()
        
        # Vehicle tracking
        self.vehicle_poses = []
        self.cone_map = []
        
        # Z-value statistics for adaptive filtering
        self.z_min = None
        self.z_max = None
        self.z_mean = None
        self.z_std = None
        self.z_stats_initialized = False
        
        # Set up timers
        self.fusion_timer = self.create_timer(
            1.0 / self.fusion_rate, 
            self.fusion_timer_callback,
            callback_group=self.timer_callback_group
        )
        
        self.cone_detection_timer = self.create_timer(
            1.0 / self.cone_detection_rate,
            self.cone_detection_callback,
            callback_group=self.timer_callback_group
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Start visualization thread if enabled
        self.vis_thread = None
        if self.show_opencv_windows:
            self.vis_thread = threading.Thread(target=self.visualization_thread)
            self.vis_thread.daemon = True
            self.vis_thread.start()
            self.get_logger().info("OpenCV visualization enabled")
        
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
        with self.sensor_data.lidar_lock:
            self.sensor_data.lidar_data = msg
            self.sensor_data.lidar_timestamp = self.get_clock().now().nanoseconds / 1e9
            
            # Check LiDAR message structure for debugging
            if hasattr(self, 'lidar_debug_count') and self.lidar_debug_count < 3:
                self.debug_lidar_structure(msg)
                self.lidar_debug_count += 1
    
    def rgb_callback(self, msg: Image):
        """Callback for RGB camera data."""
        try:
            with self.sensor_data.rgb_lock:
                self.sensor_data.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.sensor_data.rgb_timestamp = self.get_clock().now().nanoseconds / 1e9
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting RGB image: {str(e)}")
    
    def depth_callback(self, msg: Image):
        """Callback for depth camera data."""
        try:
            with self.sensor_data.depth_lock:
                self.sensor_data.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                self.sensor_data.depth_timestamp = self.get_clock().now().nanoseconds / 1e9
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting depth image: {str(e)}")
    
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
    
    def fusion_timer_callback(self):
        """Main sensor fusion processing function."""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_fusion_time).nanoseconds / 1e9
        self.last_fusion_time = current_time
        
        try:
            # Process LiDAR data
            self.process_lidar_data()
            
            # Process camera data
            self.process_camera_data()
            
            # Publish data for visualization
            self.publish_data()
            
            # Broadcast transforms
            self.broadcast_tf()
            
        except Exception as e:
            self.get_logger().error(f"Error in fusion timer callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def cone_detection_callback(self):
        """Process LiDAR and camera data to detect cones."""
        try:
            # Detect cones from LiDAR
            if hasattr(self, 'latest_filtered_points') and self.latest_filtered_points is not None:
                self.detect_cones_from_lidar()
            
            # Detect cones from camera
            if self.yolo_model is not None and hasattr(self, 'latest_rgb_image') and self.latest_rgb_image is not None:
                self.detect_cones_from_camera()
            
            # Fuse cone detections
            self.fuse_cone_detections()
            
            # Publish cone markers for visualization
            self.publish_cone_markers()
            
        except Exception as e:
            self.get_logger().error(f"Error in cone detection callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def process_lidar_data(self):
        """Process and analyze LiDAR data."""
        with self.sensor_data.lidar_lock:
            if self.sensor_data.lidar_data is None:
                return
            
            # Get LiDAR data
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
            else:
                # Try to extract using index positions (assuming xyz are the first 3 values)
                for point in pc2.read_points(lidar_data, field_names=None, skip_nans=True):
                    if isinstance(point, tuple) and len(point) >= 3:
                        points_xyz.append([point[0], point[1], point[2]])
            
            # Convert to numpy array
            points_array = np.array(points_xyz, dtype=np.float32)
            
            if len(points_array) == 0:
                self.get_logger().warn("No points extracted from LiDAR data")
                return
            
            # Filter out points that are too far away
            distances = np.sqrt(np.sum(points_array[:, :2]**2, axis=1))  # XY distance
            distance_mask = distances < self.max_cone_distance  # Only use points within range
            filtered_by_distance = points_array[distance_mask]
            
            # Apply filtering based on the selected method
            if self.filter_method == 'none':
                # No additional filtering
                filtered_points = filtered_by_distance
            
            elif self.filter_method == 'height_range':
                # Filter by height range (specific to CARLA's coordinate system)
                height_mask = (filtered_by_distance[:, 2] >= self.min_height) & (filtered_by_distance[:, 2] <= self.max_height)
                filtered_points = filtered_by_distance[height_mask]
            
            elif self.filter_method == 'ground_plane':
                # Filter out ground plane points
                # Keep points that are NOT near the ground_z value
                non_ground_mask = np.abs(filtered_by_distance[:, 2] - self.ground_z) > self.ground_thickness
                filtered_points = filtered_by_distance[non_ground_mask]
            
            else:  # Default to no filtering
                filtered_points = filtered_by_distance
            
            # Store filtered points for cone detection
            self.latest_filtered_points = filtered_points
            
            # Accumulate points across multiple frames
            with self.lidar_history_lock:
                self.lidar_history.append(filtered_points)
                if len(self.lidar_history) > self.accumulate_frames:
                    self.lidar_history.pop(0)  # Remove oldest frame
                
                # Combine points from all frames in history
                all_points = np.vstack(self.lidar_history) if self.lidar_history else filtered_points
            
            # Calculate Z statistics for adaptive filtering if not initialized
            if not self.z_stats_initialized and len(filtered_points) > 10:
                self.z_min = np.min(filtered_points[:, 2])
                self.z_max = np.max(filtered_points[:, 2])
                self.z_mean = np.mean(filtered_points[:, 2])
                self.z_std = np.std(filtered_points[:, 2])
                self.z_stats_initialized = True
            
            # Create PointCloud2 header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "map"
            
            # Create fields for PointCloud2
            fields = [
                pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
            ]
            
            # Create structured array for world points
            structured_points = np.zeros(len(all_points), 
                                        dtype=[
                                            ('x', np.float32),
                                            ('y', np.float32),
                                            ('z', np.float32),
                                            ('intensity', np.float32)
                                        ])
            
            # Fill structured array
            structured_points['x'] = all_points[:, 0]
            structured_points['y'] = all_points[:, 1]
            structured_points['z'] = all_points[:, 2]
            
            # Color points based on height (z value)
            min_z = np.min(all_points[:, 2])
            max_z = np.max(all_points[:, 2])
            z_range = max_z - min_z
            if z_range > 0:
                intensity = (all_points[:, 2] - min_z) / z_range
            else:
                intensity = np.ones(len(all_points))
            
            structured_points['intensity'] = intensity
            
            # Create and publish the point cloud
            pc_msg = pc2.create_cloud(header, fields, structured_points)
            self.lidar_pub.publish(pc_msg)
            
            self.get_logger().info(f"Processed {len(points_array)} LiDAR points, filtered to {len(filtered_points)}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def process_camera_data(self):
        """Process camera data for detection and visualization."""
        with self.sensor_data.rgb_lock:
            if self.sensor_data.rgb_image is None:
                return
            
            # Make a copy of the RGB image
            rgb_image = self.sensor_data.rgb_image.copy()
            # Store for YOLO processing
            self.latest_rgb_image = rgb_image
        
        try:
            # Publish RGB image
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = "camera_link"
            self.rgb_pub.publish(rgb_msg)
            
            # Process depth image if available
            with self.sensor_data.depth_lock:
                if self.sensor_data.depth_image is not None:
                    depth_image = self.sensor_data.depth_image.copy()
                    
                    # Create a visualization of the depth image
                    # Normalize for visualization
                    depth_min = np.nanmin(depth_image)
                    depth_max = np.nanmax(depth_image)
                    if depth_max > depth_min:
                        normalized_depth = (depth_image - depth_min) / (depth_max - depth_min)
                        depth_viz = (normalized_depth * 255).astype(np.uint8)
                        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
                        
                        # Publish depth visualization
                        try:
                            depth_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding="bgr8")
                            depth_msg.header.stamp = self.get_clock().now().to_msg()
                            depth_msg.header.frame_id = "camera_link"
                            self.depth_pub.publish(depth_msg)
                        except CvBridgeError as e:
                            self.get_logger().error(f"Error converting depth image: {str(e)}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing camera data: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def detect_cones_from_lidar(self):
        """Detect cones from LiDAR point cloud using DBSCAN clustering."""
        if not hasattr(self, 'latest_filtered_points') or self.latest_filtered_points is None:
            return
            
        try:
            # Get a copy of the filtered points
            points = self.latest_filtered_points.copy()
            
            if len(points) < self.min_points_per_cone:
                with self.sensor_data.cone_lock:
                    self.sensor_data.lidar_cones = []
                return
            
            # Run DBSCAN clustering to find cone-like objects
            clustering = DBSCAN(eps=self.cone_cluster_distance, min_samples=self.min_points_per_cone).fit(points)
            labels = clustering.labels_
            
            # Process clusters
            lidar_cones = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                # Get all points in this cluster
                cluster_points = points[labels == label]
                
                # Calculate centroid
                centroid = np.mean(cluster_points, axis=0)
                
                # Calculate distance from origin
                distance = np.linalg.norm(centroid[:2])
                
                # Skip cones that are too far away
                if distance > self.max_cone_distance:
                    continue
                
                # Calculate confidence based on point count
                confidence = min(1.0, len(cluster_points) / 50.0)  # Normalize by expected point count
                
                # Add to cone list
                lidar_cones.append({
                    'position': centroid,
                    'confidence': confidence,
                    'num_points': len(cluster_points),
                    'type': 'lidar'
                })
            
            # Update cones in sensor data
            with self.sensor_data.cone_lock:
                self.sensor_data.lidar_cones = lidar_cones
            
            self.get_logger().info(f"Detected {len(lidar_cones)} cones from LiDAR")
            
        except Exception as e:
            self.get_logger().error(f"Error detecting cones from LiDAR: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            with self.sensor_data.cone_lock:
                self.sensor_data.lidar_cones = []
    
    def detect_cones_from_camera(self):
        """Detect cones using YOLO model and convert to 3D positions using depth."""
        if self.yolo_model is None or not hasattr(self, 'latest_rgb_image') or self.latest_rgb_image is None:
            return
            
        # Get copies of RGB and depth images
        with self.sensor_data.rgb_lock:
            if self.sensor_data.rgb_image is None:
                return
            rgb_image = self.sensor_data.rgb_image.copy()
        
        with self.sensor_data.depth_lock:
            if self.sensor_data.depth_image is None:
                return
            depth_image = self.sensor_data.depth_image.copy()
        
        try:
            # Run YOLO detection
            results = self.yolo_model(rgb_image)
            
            # Get detections
            detections = results.pandas().xyxy[0]  # Get detections in pandas DataFrame format
            
            # Create debug image with detections
            debug_img = rgb_image.copy()
            
            # Process detections
            camera_cones = []
            
            for _, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                cls = det['class']
                
                # Get class name if available
                if 'name' in det:
                    class_name = det['name']
                else:
                    class_name = f"Class {cls}"
                
                # Draw on debug image
                color = (0, 255, 0)  # Default green
                if class_name.lower() == 'yellow':
                    color = (0, 255, 255)  # Yellow
                    cls = 0  # Map to yellow class
                elif class_name.lower() == 'blue':
                    color = (255, 0, 0)  # Blue (BGR)
                    cls = 1  # Map to blue class
                
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Calculate center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get depth at center point
                if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                    # Get average depth in a small region around the center
                    roi_size = 5  # 5x5 pixel ROI
                    y_start = max(0, center_y - roi_size)
                    y_end = min(depth_image.shape[0], center_y + roi_size)
                    x_start = max(0, center_x - roi_size)
                    x_end = min(depth_image.shape[1], center_x + roi_size)
                    
                    depth_roi = depth_image[y_start:y_end, x_start:x_end]
                    
                    # Filter out invalid depths (NaN, inf, zeros)
                    valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
                    
                    if len(valid_depths) == 0:
                        continue  # Skip this detection if no valid depth
                    
                    # Use median depth to reduce noise
                    depth = np.median(valid_depths)
                    
                    # Skip if beyond max distance
                    if depth > self.max_cone_distance:
                        continue
                    
                    # Convert to 3D coordinates
                    # Formula: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
                    x_3d = (center_x - self.camera_cx) * depth / self.camera_fx
                    y_3d = (center_y - self.camera_cy) * depth / self.camera_fy
                    z_3d = depth  # Z is forward in camera coordinates
                    
                    # Get vehicle transform
                    vehicle_position = np.zeros(3)
                    vehicle_orientation = np.eye(3)
                    
                    if self.sensor_data.filtered_pose is not None:
                        pose = self.sensor_data.filtered_pose
                        vehicle_position = np.array([
                            pose.pose.position.x,
                            pose.pose.position.y,
                            pose.pose.position.z
                        ])
                        
                        # Get rotation matrix from quaternion
                        qx = pose.pose.orientation.x
                        qy = pose.pose.orientation.y
                        qz = pose.pose.orientation.z
                        qw = pose.pose.orientation.w
                        r = R.from_quat([qx, qy, qz, qw])
                        vehicle_orientation = r.as_matrix()
                    
                    # Apply vehicle transform (assuming camera is at vehicle position)
                    # Note: Adjust for camera mounting position if needed
                    # Assuming camera forward is vehicle forward
                    camera_relative_pos = np.array([z_3d, -x_3d, -y_3d])  # Convert to vehicle frame
                    world_pos = vehicle_position + vehicle_orientation @ camera_relative_pos
                    
                    # Add to cone list
                    camera_cones.append({
                        'position': world_pos,
                        'confidence': conf,
                        'class': cls,
                        'type': 'camera',
                        'depth': depth,
                        'box': [x1, y1, x2, y2],
                        'class_name': class_name
                    })
            
            # Update cones in sensor data
            with self.sensor_data.cone_lock:
                self.sensor_data.camera_cones = camera_cones
            
            # Publish debug image
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.header.frame_id = "camera_link"
                self.debug_img_pub.publish(debug_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting debug image: {str(e)}")
            
            self.get_logger().info(f"Detected {len(camera_cones)} cones from camera")
            
        except Exception as e:
            self.get_logger().error(f"Error detecting cones from camera: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            with self.sensor_data.cone_lock:
                self.sensor_data.camera_cones = []
    
    def fuse_cone_detections(self):
        """Fuse LiDAR and camera cone detections."""
        with self.sensor_data.cone_lock:
            lidar_cones = self.sensor_data.lidar_cones.copy()
            camera_cones = self.sensor_data.camera_cones.copy()
            
            # If no cones from either source, just use the other source
            if not lidar_cones and not camera_cones:
                self.sensor_data.fused_cones = []
                return
                
            if not lidar_cones:
                self.sensor_data.fused_cones = camera_cones
                return
                
            if not camera_cones:
                self.sensor_data.fused_cones = lidar_cones
                return
            
            # Initialize fused cones with all LiDAR cones
            fused_cones = lidar_cones.copy()
            
            # Track which camera cones have been matched
            matched_camera_cones = [False] * len(camera_cones)
            
            # For each LiDAR cone, find the closest camera cone
            for i, lidar_cone in enumerate(lidar_cones):
                lidar_pos = lidar_cone['position']
                
                # Find closest camera cone
                best_dist = float('inf')
                best_match = -1
                
                for j, camera_cone in enumerate(camera_cones):
                    camera_pos = camera_cone['position']
                    
                    # Calculate distance between cones
                    dist = np.linalg.norm(lidar_pos - camera_pos)
                    
                    # If close enough, consider a match
                    if dist < 1.0 and dist < best_dist:  # 1.0 meter threshold
                        best_dist = dist
                        best_match = j
                
                # If found a match, update the LiDAR cone with camera info
                if best_match >= 0:
                    matched_camera_cones[best_match] = True
                    camera_cone = camera_cones[best_match]
                    
                    # Merge data - weighted position based on confidence
                    lidar_conf = lidar_cone['confidence']
                    camera_conf = camera_cone['confidence']
                    total_conf = lidar_conf + camera_conf
                    
                    merged_position = (
                        (lidar_pos * lidar_conf + camera_cone['position'] * camera_conf) / total_conf
                    )
                    
                    # Update fused cone
                    fused_cones[i] = {
                        'position': merged_position,
                        'confidence': max(lidar_conf, camera_conf),  # Take max confidence
                        'num_points': lidar_cone.get('num_points', 0),
                        'class': camera_cone.get('class', 0),  # Use camera class if available
                        'class_name': camera_cone.get('class_name', 'Unknown'),
                        'type': 'fused',
                        'sources': ['lidar', 'camera']
                    }
            
            # Add camera cones that weren't matched
            for j, matched in enumerate(matched_camera_cones):
                if not matched:
                    fused_cones.append(camera_cones[j])
            
            # Update fused cones
            self.sensor_data.fused_cones = fused_cones
            
            self.get_logger().info(f"Fused {len(lidar_cones)} LiDAR cones and {len(camera_cones)} camera cones into {len(fused_cones)} cones")
    
    def publish_cone_markers(self):
        """Publish visualization markers for all cone detections."""
        try:
            # Get cones from sensor data
            with self.sensor_data.cone_lock:
                lidar_cones = self.sensor_data.lidar_cones.copy()
                camera_cones = self.sensor_data.camera_cones.copy()
                fused_cones = self.sensor_data.fused_cones.copy()
            
            # Publish fused cone markers (main visualization)
            self.publish_fused_cone_markers(fused_cones)
            
            # Optionally publish separate markers for LiDAR and camera cones
            self.publish_lidar_cone_markers(lidar_cones)
            self.publish_camera_cone_markers(camera_cones)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing cone markers: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_lidar_cone_markers(self, cones):
        """Publish visualization markers for LiDAR-detected cones."""
        marker_array = MarkerArray()
        
        # Clear existing markers if no cones
        if not cones:
            self.clear_lidar_markers()
            return
        
        # Track new marker IDs
        new_marker_ids = set()
        
        for i, cone in enumerate(cones):
            # Use index as marker ID
            marker_id = i
            new_marker_ids.add(marker_id)
            
            # Get cone data
            position = cone['position']
            confidence = cone['confidence']
            num_points = cone.get('num_points', 0)
            
            # Calculate distance for sizing
            distance = np.linalg.norm(position[:2])
            distance_factor = float(max(0.0, 1.0 - (distance / self.max_cone_distance)))
            
            # Create marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lidar_cones"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            
            # Set orientation (upright)
            marker.pose.orientation.w = 1.0
            
            # Set scale
            size_factor = float(min(1.0, 0.5 + 0.5 * confidence))
            marker.scale.x = float(0.3 * size_factor * (0.5 + 0.5 * distance_factor))
            marker.scale.y = float(0.3 * size_factor * (0.5 + 0.5 * distance_factor))
            marker.scale.z = float(0.5 * size_factor * (0.5 + 0.5 * distance_factor))
            
            # Set color (red for LiDAR)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = float(0.6 + 0.4 * confidence)
            
            marker_array.markers.append(marker)
            
            # Add text label
            if self.show_cone_labels:
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "lidar_cone_labels"
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = float(position[0])
                text_marker.pose.position.y = float(position[1])
                text_marker.pose.position.z = float(position[2] + 0.6)  # Above cone
                
                text_marker.text = f"L {num_points}pts {distance:.1f}m"
                
                text_marker.scale.z = float(0.2 * (0.5 + 0.5 * distance_factor))
                
                text_marker.color.r = 1.0
                text_marker.color.g = 0.0
                text_marker.color.b = 0.0
                text_marker.color.a = float(0.7 * distance_factor + 0.3)
                
                marker_array.markers.append(text_marker)
        
        # Keep track of active markers
        self.lidar_marker_ids = new_marker_ids
        
        # Publish the marker array
        self.lidar_cone_pub.publish(marker_array)
    
    def publish_camera_cone_markers(self, cones):
        """Publish visualization markers for camera-detected cones."""
        marker_array = MarkerArray()
        
        # Clear existing markers if no cones
        if not cones:
            self.clear_camera_markers()
            return
        
        # Track new marker IDs
        new_marker_ids = set()
        
        for i, cone in enumerate(cones):
            # Use index as marker ID
            marker_id = i
            new_marker_ids.add(marker_id)
            
            # Get cone data
            position = cone['position']
            confidence = cone['confidence']
            cone_class = cone.get('class', 0)
            class_name = cone.get('class_name', 'Unknown')
            depth = cone.get('depth', 0.0)
            
            # Calculate distance for sizing
            distance = np.linalg.norm(position[:2])
            distance_factor = float(max(0.0, 1.0 - (distance / self.max_cone_distance)))
            
            # Create marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "camera_cones"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            
            # Set orientation (upright)
            marker.pose.orientation.w = 1.0
            
            # Set scale
            size_factor = float(min(1.0, 0.5 + 0.5 * confidence))
            marker.scale.x = float(0.3 * size_factor * (0.5 + 0.5 * distance_factor))
            marker.scale.y = float(0.3 * size_factor * (0.5 + 0.5 * distance_factor))
            marker.scale.z = float(0.5 * size_factor * (0.5 + 0.5 * distance_factor))
            
            # Set color based on class
            if cone_class == 0 or class_name.lower() == 'yellow':  # Yellow cone
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif cone_class == 1 or class_name.lower() == 'blue':  # Blue cone
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:  # Default green for camera
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            
            marker.color.a = float(0.6 + 0.4 * confidence)
            
            marker_array.markers.append(marker)
            
            # Add text label
            if self.show_cone_labels:
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "camera_cone_labels"
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = float(position[0])
                text_marker.pose.position.y = float(position[1])
                text_marker.pose.position.z = float(position[2] + 0.6)  # Above cone
                
                text_marker.text = f"C {class_name} {confidence:.2f} {depth:.1f}m"
                
                text_marker.scale.z = float(0.2 * (0.5 + 0.5 * distance_factor))
                
                text_marker.color.r = marker.color.r
                text_marker.color.g = marker.color.g
                text_marker.color.b = marker.color.b
                text_marker.color.a = float(0.7 * distance_factor + 0.3)
                
                marker_array.markers.append(text_marker)
        
        # Keep track of active markers
        self.camera_marker_ids = new_marker_ids
        
        # Publish the marker array
        self.camera_cone_pub.publish(marker_array)
    
    def publish_fused_cone_markers(self, cones):
        """Publish visualization markers for fused cone detections."""
        marker_array = MarkerArray()
        
        # Clear existing markers if no cones
        if not cones:
            self.clear_all_markers()
            return
        
        # Track new marker IDs
        new_marker_ids = set()
        
        for i, cone in enumerate(cones):
            # Use index as marker ID
            marker_id = i
            new_marker_ids.add(marker_id)
            
            # Get cone data
            position = cone['position']
            confidence = cone['confidence']
            cone_type = cone.get('type', 'unknown')
            cone_class = cone.get('class', 0)
            class_name = cone.get('class_name', 'Unknown')
            
            # Calculate distance for sizing
            distance = np.linalg.norm(position[:2])
            distance_factor = float(max(0.0, 1.0 - (distance / self.max_cone_distance)))
            
            # Create marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fused_cones"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = float(position[2])
            
            # Set orientation (upright)
            marker.pose.orientation.w = 1.0
            
            # Set scale - bigger for fused cones
            size_factor = float(min(1.0, 0.5 + 0.5 * confidence))
            marker.scale.x = float(0.4 * size_factor * (0.5 + 0.5 * distance_factor))
            marker.scale.y = float(0.4 * size_factor * (0.5 + 0.5 * distance_factor))
            marker.scale.z = float(0.6 * size_factor * (0.5 + 0.5 * distance_factor))
            
            # Set color based on type and class
            if cone_type == 'fused':
                # Magenta for fused cones as base color
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                
                # But use class color if available
                if cone_class == 0 or class_name.lower() == 'yellow':  # Yellow
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif cone_class == 1 or class_name.lower() == 'blue':  # Blue
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
            elif cone_type == 'camera':
                # Use class color for camera cones
                if cone_class == 0 or class_name.lower() == 'yellow':  # Yellow
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif cone_class == 1 or class_name.lower() == 'blue':  # Blue
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                else:  # Default green for camera
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
            elif cone_type == 'lidar':
                # Red for LiDAR
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                # White for unknown
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
            
            marker.color.a = float(0.7 + 0.3 * confidence)
            
            marker_array.markers.append(marker)
            
            # Add text label
            if self.show_cone_labels:
                text_marker = Marker()
                text_marker.header = marker.header
                text_marker.ns = "fused_cone_labels"
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = float(position[0])
                text_marker.pose.position.y = float(position[1])
                text_marker.pose.position.z = float(position[2] + 0.6)  # Above cone
                
                # Format label based on type
                if cone_type == 'fused':
                    sources = ','.join(cone.get('sources', ['?']))
                    text_marker.text = f"F {class_name} {confidence:.2f} {distance:.1f}m"
                elif cone_type == 'camera':
                    depth = cone.get('depth', 0.0)
                    text_marker.text = f"C {class_name} {confidence:.2f} {depth:.1f}m"
                elif cone_type == 'lidar':
                    num_points = cone.get('num_points', 0)
                    text_marker.text = f"L {num_points}pts {distance:.1f}m"
                else:
                    text_marker.text = f"{cone_type} {confidence:.2f} {distance:.1f}m"
                
                text_marker.scale.z = float(0.2 * (0.5 + 0.5 * distance_factor))
                
                text_marker.color.r = marker.color.r
                text_marker.color.g = marker.color.g
                text_marker.color.b = marker.color.b
                text_marker.color.a = float(0.8 * distance_factor + 0.2)
                
                marker_array.markers.append(text_marker)
        
        # Keep track of active markers
        self.active_marker_ids = new_marker_ids
        
        # Publish the marker array
        self.cone_marker_pub.publish(marker_array)
        
        # Also publish to the fused cone topic
        self.fused_cone_pub.publish(marker_array)
    
    def clear_all_markers(self):
        """Clear all cone markers."""
        if not self.active_marker_ids:
            return
            
        marker_array = MarkerArray()
        
        for marker_id in self.active_marker_ids:
            # Delete cylinder marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fused_cones"
            marker.id = marker_id
            marker.action = Marker.DELETE
            
            marker_array.markers.append(marker)
            
            # Delete text marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "fused_cone_labels"
            text_marker.id = marker_id
            text_marker.action = Marker.DELETE
            
            marker_array.markers.append(text_marker)
        
        # Publish the deletion markers
        self.cone_marker_pub.publish(marker_array)
        self.fused_cone_pub.publish(marker_array)
        
        # Clear the marker IDs
        self.active_marker_ids.clear()
    
    def clear_lidar_markers(self):
        """Clear all LiDAR cone markers."""
        if not self.lidar_marker_ids:
            return
            
        marker_array = MarkerArray()
        
        for marker_id in self.lidar_marker_ids:
            # Delete cylinder marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "lidar_cones"
            marker.id = marker_id
            marker.action = Marker.DELETE
            
            marker_array.markers.append(marker)
            
            # Delete text marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "lidar_cone_labels"
            text_marker.id = marker_id
            text_marker.action = Marker.DELETE
            
            marker_array.markers.append(text_marker)
        
        # Publish the deletion markers
        # Publish the deletion markers
        self.lidar_cone_pub.publish(marker_array)
        
        # Clear the marker IDs
        self.lidar_marker_ids.clear()
    
    def clear_camera_markers(self):
        """Clear all camera cone markers."""
        if not self.camera_marker_ids:
            return
            
        marker_array = MarkerArray()
        
        for marker_id in self.camera_marker_ids:
            # Delete cylinder marker
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "camera_cones"
            marker.id = marker_id
            marker.action = Marker.DELETE
            
            marker_array.markers.append(marker)
            
            # Delete text marker
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "camera_cone_labels"
            text_marker.id = marker_id
            text_marker.action = Marker.DELETE
            
            marker_array.markers.append(text_marker)
        
        # Publish the deletion markers
        self.camera_cone_pub.publish(marker_array)
        
        # Clear the marker IDs
        self.camera_marker_ids.clear()
    
    def publish_data(self):
        """Publish camera, LiDAR, and fused data for visualization."""
        # Publish RGB and depth images
        with self.sensor_data.rgb_lock:
            if self.sensor_data.rgb_image is not None:
                try:
                    rgb_msg = self.bridge.cv2_to_imgmsg(self.sensor_data.rgb_image, encoding="bgr8")
                    rgb_msg.header.stamp = self.get_clock().now().to_msg()
                    rgb_msg.header.frame_id = "camera_link"
                    self.rgb_pub.publish(rgb_msg)
                except CvBridgeError as e:
                    self.get_logger().error(f"Error converting RGB image: {str(e)}")
        
        # Publish cone point cloud for tracking
        with self.sensor_data.cone_lock:
            fused_cones = self.sensor_data.fused_cones.copy()
            
            if fused_cones:
                # Create point cloud header
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"
                
                # Create fields for PointCloud2
                fields = [
                    pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                    pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                    pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                    pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
                ]
                
                # Create structured array for cone points
                cone_points = np.zeros(len(fused_cones), 
                                     dtype=[
                                         ('x', np.float32),
                                         ('y', np.float32),
                                         ('z', np.float32),
                                         ('intensity', np.float32)
                                     ])
                
                # Fill structured array
                for i, cone in enumerate(fused_cones):
                    position = cone['position']
                    cone_points[i]['x'] = float(position[0])
                    cone_points[i]['y'] = float(position[1])
                    cone_points[i]['z'] = float(position[2])
                    
                    # Use color based on type for intensity
                    # 0.0 = LiDAR, 0.5 = camera, 1.0 = fused
                    if cone['type'] == 'lidar':
                        cone_points[i]['intensity'] = 0.0
                    elif cone['type'] == 'camera':
                        cone_points[i]['intensity'] = 0.5
                    else:  # fused
                        cone_points[i]['intensity'] = 1.0
                
                # Create PointCloud2 message
                pc_msg = pc2.create_cloud(header, fields, cone_points)
                self.fused_pub.publish(pc_msg)
    
    def broadcast_tf(self):
        """Broadcast coordinate transforms for the vehicle and sensors."""
        try:
            if not hasattr(self, 'sensor_data') or self.sensor_data.filtered_pose is None:
                return
                
            # Get vehicle transform
            pose = self.sensor_data.filtered_pose
            
            # Broadcast map -> base_link transform
            t = TransformStamped()
            t.header = pose.header
            t.child_frame_id = "base_link"
            
            # Set translation
            t.transform.translation.x = pose.pose.position.x
            t.transform.translation.y = pose.pose.position.y
            t.transform.translation.z = pose.pose.position.z
            
            # Set rotation
            t.transform.rotation = pose.pose.orientation
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(t)
            
            # Store vehicle pose for mapping
            new_pose = (
                pose.pose.position.x,
                pose.pose.position.y,
                self.sensor_data.vehicle_yaw
            )
            self.vehicle_poses.append(new_pose)
            
            # Limit stored poses to last 1000
            if len(self.vehicle_poses) > 1000:
                self.vehicle_poses = self.vehicle_poses[-1000:]
            
            # Base link -> camera_link transform
            camera_t = TransformStamped()
            camera_t.header.stamp = self.get_clock().now().to_msg()
            camera_t.header.frame_id = "base_link"
            camera_t.child_frame_id = "camera_link"
            
            # Set camera position relative to base_link (forward and elevated)
            camera_t.transform.translation.x = 2.0  # 2m forward
            camera_t.transform.translation.y = 0.0  # Centered
            camera_t.transform.translation.z = 1.2  # 1.2m above base_link
            
            # Set camera orientation (pointing forward)
            camera_t.transform.rotation.w = 1.0
            camera_t.transform.rotation.x = 0.0
            camera_t.transform.rotation.y = 0.0
            camera_t.transform.rotation.z = 0.0
            
            # Broadcast the camera transform
            self.tf_broadcaster.sendTransform(camera_t)
            
            # Base link -> lidar_link transform
            lidar_t = TransformStamped()
            lidar_t.header.stamp = self.get_clock().now().to_msg()
            lidar_t.header.frame_id = "base_link"
            lidar_t.child_frame_id = "lidar_link"
            
            # Set LiDAR position relative to base_link (on roof)
            lidar_t.transform.translation.x = 0.0  # Centered
            lidar_t.transform.translation.y = 0.0  # Centered
            lidar_t.transform.translation.z = 1.8  # 1.8m above base_link (on roof)
            
            # Set LiDAR orientation (upright)
            lidar_t.transform.rotation.w = 1.0
            lidar_t.transform.rotation.x = 0.0
            lidar_t.transform.rotation.y = 0.0
            lidar_t.transform.rotation.z = 0.0
            
            # Broadcast the LiDAR transform
            self.tf_broadcaster.sendTransform(lidar_t)
            
        except Exception as e:
            self.get_logger().error(f"Error broadcasting transforms: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def visualization_thread(self):
        """Thread for OpenCV visualization."""
        while self.show_opencv_windows:
            try:
                # Get images with thread safety
                rgb_img = None
                depth_img = None
                cones_info = None
                
                with self.sensor_data.rgb_lock:
                    if self.sensor_data.rgb_image is not None:
                        rgb_img = self.sensor_data.rgb_image.copy()
                
                with self.sensor_data.depth_lock:
                    if self.sensor_data.depth_image is not None:
                        # Create visualization of depth image
                        depth_data = self.sensor_data.depth_image.copy()
                        depth_min = np.nanmin(depth_data)
                        depth_max = np.nanmax(depth_data)
                        if depth_max > depth_min:
                            normalized = (depth_data - depth_min) / (depth_max - depth_min)
                            depth_img = (normalized * 255).astype(np.uint8)
                            depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
                
                with self.sensor_data.cone_lock:
                    if self.sensor_data.fused_cones:
                        cones_info = self.sensor_data.fused_cones.copy()
                
                # Show RGB image if available
                if rgb_img is not None:
                    # Draw detections on RGB image if available
                    if cones_info:
                        # Make a copy for drawing
                        display_img = rgb_img.copy()
                        
                        # Draw cone info
                        with self.sensor_data.cone_lock:
                            camera_cones = self.sensor_data.camera_cones.copy()
                            
                            for cone in camera_cones:
                                if 'box' in cone:
                                    # Draw bounding box
                                    x1, y1, x2, y2 = cone['box']
                                    
                                    # Color based on class
                                    cls = cone.get('class', 0)
                                    class_name = cone.get('class_name', 'Unknown')
                                    
                                    if cls == 0 or class_name.lower() == 'yellow':
                                        color = (0, 255, 255)  # Yellow in BGR
                                    elif cls == 1 or class_name.lower() == 'blue':
                                        color = (255, 0, 0)  # Blue in BGR
                                    else:
                                        color = (0, 255, 0)  # Green in BGR
                                    
                                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Draw label
                                    conf = cone.get('confidence', 0.0)
                                    depth = cone.get('depth', 0.0)
                                    label = f"{class_name} {conf:.2f} {depth:.1f}m"
                                    cv2.putText(display_img, label, (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Show image with detections
                        cv2.imshow('RGB with Detections', display_img)
                    else:
                        # Show plain RGB image
                        cv2.imshow('RGB Camera', rgb_img)
                
                # Show depth image if available
                if depth_img is not None:
                    cv2.imshow('Depth Image', depth_img)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit if 'q' is pressed
                    self.show_opencv_windows = False
                    break
                
                # Sleep to avoid high CPU usage
                time.sleep(0.05)  # 20 Hz
                
            except Exception as e:
                self.get_logger().error(f"Error in visualization thread: {str(e)}")
                time.sleep(0.1)  # Sleep longer on error
        
        # Cleanup
        cv2.destroyAllWindows()
    
    def destroy_node(self):
        """Clean up resources on node shutdown."""
        self.get_logger().info("Shutting down fusion timer node...")
        
        # Disable OpenCV windows
        self.show_opencv_windows = False
        if hasattr(self, 'vis_thread') and self.vis_thread and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=1.0)
        
        # Clean up YOLO model
        if hasattr(self, 'yolo_model'):
            del self.yolo_model
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = FusionTimerNode()
        
        # Use multithreaded executor for better performance
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            node.destroy_node()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()