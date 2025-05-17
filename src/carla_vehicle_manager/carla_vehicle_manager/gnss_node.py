"""
CARLA GNSS (GPS) Sensor Node
This node publishes GPS data from a CARLA vehicle with configurable noise parameters.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import NavSatStatus
from std_msgs.msg import Header
import carla
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import threading


class CarlaGNSSNode(Node):
    def __init__(self):
        super().__init__('carla_gnss_node')
        
        self.declare_parameter('carla.host', 'localhost')
        self.declare_parameter('carla.port', 2000)
        self.declare_parameter('carla.timeout', 10.0)
        self.declare_parameter('update_frequency', 10.0)
        
        self.declare_parameter('gnss.noise_lat_bias', 0.0)
        self.declare_parameter('gnss.noise_lat_stddev', 0.0000001)
        self.declare_parameter('gnss.noise_lon_bias', 0.0)
        self.declare_parameter('gnss.noise_lon_stddev', 0.0000001)
        self.declare_parameter('gnss.noise_alt_bias', 0.0)
        self.declare_parameter('gnss.noise_alt_stddev', 0.01)
        self.declare_parameter('gnss.sensor_tick', 0.1)
        self.declare_parameter('gnss.noise_seed', 0)
        
        self.declare_parameter('frame_id.gnss', 'gnss_link')
        self.declare_parameter('frame_id.vehicle', 'base_link')
        self.declare_parameter('topic_name', '/carla/gnss')
        
        self.host = self.get_parameter('carla.host').value
        self.port = self.get_parameter('carla.port').value
        self.timeout = self.get_parameter('carla.timeout').value
        self.update_freq = self.get_parameter('update_frequency').value
        
        self.lat_bias = self.get_parameter('gnss.noise_lat_bias').value
        self.lat_stddev = self.get_parameter('gnss.noise_lat_stddev').value
        self.lon_bias = self.get_parameter('gnss.noise_lon_bias').value
        self.lon_stddev = self.get_parameter('gnss.noise_lon_stddev').value
        self.alt_bias = self.get_parameter('gnss.noise_alt_bias').value
        self.alt_stddev = self.get_parameter('gnss.noise_alt_stddev').value
        self.sensor_tick = self.get_parameter('gnss.sensor_tick').value
        self.noise_seed = self.get_parameter('gnss.noise_seed').value
        
        self.gnss_frame = self.get_parameter('frame_id.gnss').value
        self.vehicle_frame = self.get_parameter('frame_id.vehicle').value
        self.topic_name = self.get_parameter('topic_name').value
        
        if self.noise_seed > 0:
            np.random.seed(self.noise_seed)
        
        self.gnss_pub = self.create_publisher(NavSatFix, self.topic_name, 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.client = None
        self.world = None
        self.vehicle = None
        self.gnss_sensor = None
        
        self.latest_gnss_data = None
        self.data_lock = threading.Lock()
        
        self.gnss_count = 0
        self.setup_carla()
        self.gnss_timer = self.create_timer(1.0/self.update_freq, self.gnss_publish_callback)
        self.status_timer = self.create_timer(5.0, self.status_callback)
    
    def setup_carla(self):
        """Connect to CARLA and setup vehicle and sensors."""
        try:
            self.get_logger().info(f"Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            
            self.world = self.client.get_world()
            self.get_logger().info("Connected to CARLA world")
            self.find_vehicle()
            if self.vehicle:
                self.setup_gnss_sensor()
            else:
                self.get_logger().error("No vehicle found to attach GNSS sensor to")
                
        except Exception as e:
            self.get_logger().error(f"Error connecting to CARLA: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def find_vehicle(self):
        """Find an existing vehicle in the world."""
        try:
            all_actors = self.world.get_actors()
            vehicles = all_actors.filter('vehicle.*')
            
            if vehicles:
                self.vehicle = vehicles[0]
                self.get_logger().info(f"Found vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
            else:
                self.get_logger().warn("No vehicles found in the world")
                
        except Exception as e:
            self.get_logger().error(f"Error finding vehicle: {str(e)}")
    
    def setup_gnss_sensor(self):
        """Setup GNSS sensor on the vehicle."""
        if not self.vehicle:
            return
            
        try:
            blueprint_library = self.world.get_blueprint_library()
            gnss_bp = blueprint_library.find('sensor.other.gnss')
            if not gnss_bp:
                self.get_logger().error("GNSS blueprint not found")
                return
            gnss_bp.set_attribute('sensor_tick', str(self.sensor_tick))
            gnss_bp.set_attribute('noise_lat_bias', str(self.lat_bias))
            gnss_bp.set_attribute('noise_lat_stddev', str(self.lat_stddev))
            gnss_bp.set_attribute('noise_lon_bias', str(self.lon_bias))
            gnss_bp.set_attribute('noise_lon_stddev', str(self.lon_stddev))
            gnss_bp.set_attribute('noise_alt_bias', str(self.alt_bias))
            gnss_bp.set_attribute('noise_alt_stddev', str(self.alt_stddev))
            
            if self.noise_seed > 0:
                gnss_bp.set_attribute('noise_seed', str(self.noise_seed))
            gnss_transform = carla.Transform(carla.Location(z=0.5))
            self.gnss_sensor = self.world.spawn_actor(
                gnss_bp,
                gnss_transform,
                attach_to=self.vehicle
            )
            self.gnss_sensor.listen(self.gnss_callback)
            self.get_logger().info("GNSS sensor created and attached to vehicle")
            
        except Exception as e:
            self.get_logger().error(f"Error setting up GNSS sensor: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def gnss_callback(self, data):
        """Process GNSS data from CARLA."""
        try:
            with self.data_lock:
                self.latest_gnss_data = {
                    'timestamp': data.timestamp,
                    'frame': data.frame,
                    'latitude': data.latitude,
                    'longitude': data.longitude,
                    'altitude': data.altitude,
                    'transform': data.transform
                }
            self.gnss_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in GNSS callback: {str(e)}")
    
    def gnss_publish_callback(self):
        """Publish GNSS data at the specified rate."""
        if not self.latest_gnss_data:
            return
            
        try:
            with self.data_lock:
                gnss_data = self.latest_gnss_data.copy() if self.latest_gnss_data else None
            
            if not gnss_data:
                return
            msg = NavSatFix()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.gnss_frame
            
            msg.status.status = NavSatStatus.STATUS_FIX
            msg.status.service = NavSatStatus.SERVICE_GPS
            
            # Apply artificial scaling and offset
            scale_factor = 1000.0  # exaggerate motion
            base_lat = 37.4275     # fake base (Stanford)
            base_lon = -122.1697

            scaled_lat = base_lat + (gnss_data['latitude'] * scale_factor)
            scaled_lon = base_lon + (gnss_data['longitude'] * scale_factor)

            msg.latitude = scaled_lat
            msg.longitude = scaled_lon
            msg.altitude = gnss_data['altitude']
            
            position_covariance = [
                self.lon_stddev**2, 0.0, 0.0,
                0.0, self.lat_stddev**2, 0.0,
                0.0, 0.0, self.alt_stddev**2
            ]
            msg.position_covariance = position_covariance
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            
            self.gnss_pub.publish(msg)
            self.broadcast_tf(gnss_data['transform'])
            
        except Exception as e:
            self.get_logger().error(f"Error publishing GNSS data: {str(e)}")
    
    def broadcast_tf(self, gnss_transform=None):
        """Broadcast the transform from vehicle to GNSS sensor."""
        if not self.vehicle:
            return
            
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.vehicle_frame
            t.child_frame_id = self.gnss_frame
            
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.5
            
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().error(f"Error broadcasting transform: {str(e)}")
    
    def status_callback(self):
        """Report status information."""
        self.get_logger().info(f"Status: GNSS frames={self.gnss_count}")
        if self.vehicle:
            try:
                if self.vehicle.is_alive:
                    self.get_logger().info(f"Vehicle {self.vehicle.id} is active")
                    
                    loc = self.vehicle.get_location()
                    self.get_logger().info(f"Vehicle location: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")
                    
                    if self.latest_gnss_data:
                        lat = self.latest_gnss_data['latitude']
                        lon = self.latest_gnss_data['longitude']
                        alt = self.latest_gnss_data['altitude']
                        self.get_logger().info(f"GNSS: Lat: {lat:.6f}, Lon: {lon:.6f}, Alt: {alt:.2f}m")
                else:
                    self.get_logger().warn(f"Vehicle {self.vehicle.id} is no longer alive")
                    self.vehicle = None
                    self.find_vehicle()
            except Exception:
                self.get_logger().warn("Could not check vehicle status, attempting to find a new one")
                self.vehicle = None
                self.find_vehicle()
    
    def destroy_node(self):
        """Clean up resources before shutting down."""
        self.get_logger().info("Shutting down GNSS node")
        
        if self.gnss_sensor:
            try:
                self.gnss_sensor.stop()
                self.gnss_sensor.destroy()
                self.get_logger().info("GNSS sensor destroyed")
            except Exception as e:
                self.get_logger().error(f"Error destroying GNSS sensor: {str(e)}")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CarlaGNSSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()