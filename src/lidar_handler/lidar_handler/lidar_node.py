#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import carla
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import sys
import threading


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_handler')
        
        # Parameters
        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)
        self.declare_parameter("publish_enhanced", True)  # For LIO-SAM compatibility
        
        self.host = self.get_parameter("carla_host").get_parameter_value().string_value
        self.port = self.get_parameter("carla_port").get_parameter_value().integer_value
        self.publish_enhanced = self.get_parameter("publish_enhanced").get_parameter_value().bool_value
        
        # Publishers
        self.publisher_ = self.create_publisher(PointCloud2, '/carla/lidar', 10)
        self.raw_publisher_ = self.create_publisher(PointCloud2, '/carla/lidar_raw', 10)
        
        # State variables
        self.vehicle = None
        self.world = None
        self.lidar = None
        self.lidar_data = None
        self.data_lock = threading.Lock()
        
        self.get_logger().info(f"Using Python interpreter: {sys.executable}")
        self.get_logger().info(f"Enhanced LiDAR mode: {self.publish_enhanced}")
        
        # Setup timer for processing (20 Hz to match your rotation_frequency * 2)
        self.timer = self.create_timer(0.05, self.process_and_publish)
        
        self.setup()
    
    def setup(self):
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(5.0)
            self.world = client.get_world()
            
            # Find existing vehicle
            actors = self.world.get_actors().filter('vehicle.*')
            if not actors:
                self.get_logger().error("No vehicle found.")
                return
                
            self.vehicle = actors[0]
            self.get_logger().info(f"Found vehicle: {self.vehicle.type_id}")
            self.setup_lidar()
            
        except Exception as e:
            self.get_logger().error(f"Error setting up CARLA: {str(e)}")
    
    def setup_lidar(self):
        try:
            blueprint_library = self.world.get_blueprint_library()
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            
            # Configure LiDAR to match your settings
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('points_per_second', '500000')
            lidar_bp.set_attribute('rotation_frequency', '10')  # 10 Hz
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('upper_fov', '15')
            lidar_bp.set_attribute('lower_fov', '-25')
            
            # Position LiDAR (matching your setup)
            lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.2))
            self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
            
            # Listen to LiDAR data
            self.lidar.listen(lambda data: self.lidar_callback(data))
            self.get_logger().info("LiDAR sensor attached and streaming.")
            
        except Exception as e:
            self.get_logger().error(f"Error setting up LiDAR: {str(e)}")
    
    def lidar_callback(self, data):
        """Store LiDAR data for processing in main thread."""
        try:
            # Convert raw data to numpy array
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            
            with self.data_lock:
                self.lidar_data = {
                    'points': points.copy(),
                    'timestamp': data.timestamp,
                    'frame': data.frame
                }
                
        except Exception as e:
            self.get_logger().error(f"Error in LiDAR callback: {str(e)}")
    
    def process_and_publish(self):
        """Process and publish LiDAR data with LIO-SAM enhancements."""
        with self.data_lock:
            if self.lidar_data is None:
                return
            data = self.lidar_data.copy()
        
        try:
            points = data['points']
            if len(points) == 0:
                return
            
            # Create header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "lidar_link"
            
            if self.publish_enhanced:
                # Enhanced point cloud with ring and time for LIO-SAM
                enhanced_cloud = self.create_enhanced_pointcloud2(points, header)
                self.publisher_.publish(enhanced_cloud)
                
                # Also publish simple version for existing pipeline
                simple_cloud = self.create_simple_pointcloud2(points[:, :3], header)
                self.raw_publisher_.publish(simple_cloud)
            else:
                # Simple point cloud only
                simple_cloud = self.create_simple_pointcloud2(points[:, :3], header)
                self.publisher_.publish(simple_cloud)
                
        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {str(e)}")
    
    def create_enhanced_pointcloud2(self, points, header):
        """Create enhanced PointCloud2 with ring and time fields for LIO-SAM."""
        try:
            # Extract XYZ and intensity
            xyz_points = points[:, :3]
            intensity = points[:, 3]
            
            # Calculate ring numbers based on vertical angle
            # This is crucial for LIO-SAM's feature extraction
            vertical_angles = np.arctan2(xyz_points[:, 2], 
                                       np.sqrt(xyz_points[:, 0]**2 + xyz_points[:, 1]**2))
            
            # Map vertical angles to ring numbers (0-63 for 64 channels)
            # CARLA LiDAR: upper_fov=15°, lower_fov=-25°, total=40°
            ring_numbers = np.round((vertical_angles + np.radians(25)) / np.radians(40) * 63).astype(np.uint16)
            ring_numbers = np.clip(ring_numbers, 0, 63)
            
            # Calculate time stamps within the scan (motion compensation)
            # Assume 10 Hz rotation, so 0.1s per full rotation
            horizontal_angles = np.arctan2(xyz_points[:, 1], xyz_points[:, 0])
            # Normalize to [0, 2π]
            horizontal_angles = (horizontal_angles + 2*np.pi) % (2*np.pi)
            # Convert to time within scan [0, 0.1s]
            times = (horizontal_angles / (2*np.pi) * 0.1).astype(np.float32)
            
            # Create enhanced fields
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
                PointField(name='time', offset=18, datatype=PointField.FLOAT32, count=1)
            ]
            
            # Create structured array
            enhanced_points = np.zeros(len(xyz_points), 
                                     dtype=[
                                         ('x', np.float32),
                                         ('y', np.float32),
                                         ('z', np.float32),
                                         ('intensity', np.float32),
                                         ('ring', np.uint16),
                                         ('time', np.float32)
                                     ])
            
            enhanced_points['x'] = xyz_points[:, 0]
            enhanced_points['y'] = xyz_points[:, 1]
            enhanced_points['z'] = xyz_points[:, 2]
            enhanced_points['intensity'] = intensity
            enhanced_points['ring'] = ring_numbers
            enhanced_points['time'] = times
            
            # Create PointCloud2 message
            cloud_msg = pc2.create_cloud(header, fields, enhanced_points)
            
            self.get_logger().debug(f"Published enhanced LiDAR with {len(xyz_points)} points")
            return cloud_msg
            
        except Exception as e:
            self.get_logger().error(f"Error creating enhanced point cloud: {str(e)}")
            # Fallback to simple version
            return self.create_simple_pointcloud2(points[:, :3], header)
    
    def create_simple_pointcloud2(self, points, header):
        """Create simple PointCloud2 (your original version)."""
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * len(points)
        msg.data = np.asarray(points, dtype=np.float32).tobytes()
        
        return msg
    
    def destroy_node(self):
        """Clean up resources."""
        self.get_logger().info("Shutting down LiDAR node...")
        
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.destroy()
                self.get_logger().info("LiDAR sensor destroyed")
            except Exception as e:
                self.get_logger().error(f"Error destroying LiDAR: {str(e)}")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = LidarNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()