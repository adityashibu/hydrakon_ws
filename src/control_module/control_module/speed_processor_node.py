import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import time
import carla

class CarlaSpeedProcessorNode(Node):
    """
    Fixed Speed Processor using direct CARLA API
    Publishes to: /current_speed, /current_acceleration, /velocity_vector
    """
    
    def __init__(self):
        super().__init__('speed_processor')
        
        # Parameters
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('publish_frequency', 50.0)
        
        # Get parameters
        self.carla_host = self.get_parameter('carla_host').value
        self.carla_port = self.get_parameter('carla_port').value
        self.carla_timeout = self.get_parameter('carla_timeout').value
        self.pub_freq = self.get_parameter('publish_frequency').value
        
        # Publishers
        self.speed_pub = self.create_publisher(Float64, '/current_speed', 10)
        self.accel_pub = self.create_publisher(Vector3Stamped, '/current_acceleration', 10)
        self.velocity_pub = self.create_publisher(Vector3Stamped, '/velocity_vector', 10)
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        
        # State tracking
        self.last_velocity = None
        self.last_time = time.time()
        
        # Connect to CARLA
        self.connect_to_carla()
        
        # Publishing timer
        self.pub_timer = self.create_timer(1.0/self.pub_freq, self.publish_vehicle_data)
        
        self.get_logger().info("CARLA Speed Processor initialized")
        self.get_logger().info(f"Connected to CARLA at {self.carla_host}:{self.carla_port}")
        if self.vehicle:
            self.get_logger().info(f"Using vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
        else:
            self.get_logger().warn("No vehicle found - will publish zero speed")
    
    def connect_to_carla(self):
        """Connect to CARLA and find the vehicle"""
        try:
            # Connect to CARLA
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(self.carla_timeout)
            
            # Get world
            self.world = self.client.get_world()
            
            # Find vehicle (use first available vehicle)
            vehicles = self.world.get_actors().filter('vehicle.*')
            if vehicles:
                self.vehicle = vehicles[0]
                self.get_logger().info(f"Found vehicle: {self.vehicle.type_id}")
            else:
                self.get_logger().error("No vehicles found in CARLA world")
                self.vehicle = None
                
        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA: {e}")
            self.client = None
            self.world = None
            self.vehicle = None
    
    def publish_vehicle_data(self):
        """Get vehicle data from CARLA and publish"""
        try:
            current_time = time.time()
            dt = current_time - self.last_time
            
            if not self.vehicle:
                # No vehicle - publish zero values
                self.publish_zero_data()
                return
            
            # Get velocity directly from CARLA vehicle
            carla_velocity = self.vehicle.get_velocity()
            
            # Convert CARLA velocity to ROS coordinate system
            # CARLA: X=forward, Y=right, Z=up
            # ROS: X=forward, Y=left, Z=up
            velocity_x = carla_velocity.x  # Forward (same)
            velocity_y = -carla_velocity.y  # Left (flip sign)
            velocity_z = carla_velocity.z  # Up (same)
            
            # Calculate speed magnitude
            speed = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
            
            # Calculate acceleration if we have previous velocity
            acceleration_x = 0.0
            acceleration_y = 0.0
            acceleration_z = 0.0
            
            if self.last_velocity is not None and dt > 0:
                acceleration_x = (velocity_x - self.last_velocity[0]) / dt
                acceleration_y = (velocity_y - self.last_velocity[1]) / dt
                acceleration_z = (velocity_z - self.last_velocity[2]) / dt
            
            # Store current velocity for next iteration
            self.last_velocity = (velocity_x, velocity_y, velocity_z)
            self.last_time = current_time
            
            # Publish speed
            speed_msg = Float64()
            speed_msg.data = float(speed)
            self.speed_pub.publish(speed_msg)
            
            # Publish velocity vector
            vel_msg = Vector3Stamped()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.header.frame_id = "base_link"
            vel_msg.vector.x = velocity_x
            vel_msg.vector.y = velocity_y
            vel_msg.vector.z = velocity_z
            self.velocity_pub.publish(vel_msg)
            
            # Publish acceleration
            accel_msg = Vector3Stamped()
            accel_msg.header.stamp = self.get_clock().now().to_msg()
            accel_msg.header.frame_id = "base_link"
            accel_msg.vector.x = acceleration_x
            accel_msg.vector.y = acceleration_y
            accel_msg.vector.z = acceleration_z
            self.accel_pub.publish(accel_msg)
            
            # Debug logging (every 2 seconds)
            if int(current_time * 0.5) % 2 == 0:
                self.get_logger().debug(
                    f"CARLA Speed: {speed:.2f} m/s, "
                    f"Velocity: ({velocity_x:.2f}, {velocity_y:.2f}, {velocity_z:.2f}), "
                    f"Accel: ({acceleration_x:.2f}, {acceleration_y:.2f}, {acceleration_z:.2f})"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error getting CARLA vehicle data: {e}")
            # Try to reconnect
            if not self.vehicle:
                self.connect_to_carla()
            else:
                # Publish last known good data or zeros
                self.publish_zero_data()
    
    def publish_zero_data(self):
        """Publish zero values when no vehicle data available"""
        current_time = self.get_clock().now()
        
        # Publish zero speed
        speed_msg = Float64()
        speed_msg.data = 0.0
        self.speed_pub.publish(speed_msg)
        
        # Publish zero velocity
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = current_time.to_msg()
        vel_msg.header.frame_id = "base_link"
        vel_msg.vector.x = 0.0
        vel_msg.vector.y = 0.0
        vel_msg.vector.z = 0.0
        self.velocity_pub.publish(vel_msg)
        
        # Publish zero acceleration
        accel_msg = Vector3Stamped()
        accel_msg.header.stamp = current_time.to_msg()
        accel_msg.header.frame_id = "base_link"
        accel_msg.vector.x = 0.0
        accel_msg.vector.y = 0.0
        accel_msg.vector.z = 0.0
        self.accel_pub.publish(accel_msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = CarlaSpeedProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()