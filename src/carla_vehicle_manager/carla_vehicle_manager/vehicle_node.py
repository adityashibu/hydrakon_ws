#!/usr/bin/env python3
"""
CARLA Vehicle Manager with External Control Support
This node spawns a vehicle in CARLA and allows controlling it via ROS topics.
"""

import rclpy
from rclpy.node import Node
import carla
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32


class VehicleNode(Node):
    def __init__(self):
        super().__init__('carla_vehicle_manager')
        
        self.declare_parameter("carla_host", "localhost")
        self.declare_parameter("carla_port", 2000)
        
        self.host = self.get_parameter("carla_host").get_parameter_value().string_value
        self.port = self.get_parameter("carla_port").get_parameter_value().integer_value
        
        self.vehicle = None
        self.world = None
        
        self.control = carla.VehicleControl()
        self.control.throttle = 0.0
        self.control.steer = 0.0
        self.control.brake = 0.0
        self.control.hand_brake = False
        self.control.reverse = False
        
        self.cmd_vel_sub = self.create_subscription(
            Twist, 
            '/carla/vehicle/cmd_vel', 
            self.cmd_vel_callback, 
            10)
            
        self.throttle_sub = self.create_subscription(
            Float32, 
            '/carla/vehicle/throttle', 
            self.throttle_callback, 
            10)
            
        self.brake_sub = self.create_subscription(
            Float32, 
            '/carla/vehicle/brake', 
            self.brake_callback, 
            10)
            
        self.handbrake_sub = self.create_subscription(
            Bool, 
            '/carla/vehicle/handbrake', 
            self.handbrake_callback, 
            10)
            
        self.reverse_sub = self.create_subscription(
            Bool, 
            '/carla/vehicle/reverse', 
            self.reverse_callback, 
            10)
        
        self.setup()
        
        self.timer = self.create_timer(0.05, self.apply_control)  # 20Hz control update
        
        self.get_logger().info("Vehicle node initialized. Listening for control commands.")
    
    def setup(self):
        """Connect to CARLA and spawn a vehicle."""
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(5.0)
            self.get_logger().info(f"Connecting to Carla at {self.host}:{self.port}")
            
            self.world = client.get_world()
            self.vehicle = self.spawn_vehicle()
            
            if self.vehicle:
                self.get_logger().info(f"Vehicle successfully spawned with ID: {self.vehicle.id}")
            else:
                self.get_logger().error("Vehicle could not be spawned.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA: {str(e)}")
    
    def spawn_vehicle(self):
        """Spawn a vehicle in the CARLA world."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.*')[2]
            
            if not vehicle_bp:
                self.get_logger().error("No vehicle blueprints found")
                return None
                
            self.get_logger().info(f"Using vehicle blueprint: {vehicle_bp.id}")
            
            # Trackdrive spawn point
            spawn_transform = carla.Transform(
                carla.Location(x=-35.0, y=0.0, z=5.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )

            # Autocross spawn point
            # spawn_transform = carla.Transform(
            #     carla.Location(x=95.0, y=-2.0, z=5.0),
            #     carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            # )

            # Skidpad spawn point
            # spawn_transform = carla.Transform(
            #     carla.Location(x=-90.0, y=-36.0, z=5.0),
            #     carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            # )

            # Acceleration spawn point
            # spawn_transform = carla.Transform(
            #     carla.Location(x=-90.0, y=-92.0, z=5.0),
            #     carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            # )
            
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if vehicle:
                self.get_logger().info(f"Vehicle spawned at {spawn_transform.location}")
                time.sleep(2.0)
                
                if not vehicle.is_alive:
                    self.get_logger().error("Vehicle failed to spawn or is not alive")
                    return None
                return vehicle
            else:
                self.get_logger().error("Spawn location occupied. Vehicle not spawned.")
                return None
        except Exception as e:
            self.get_logger().error(f"Error spawning vehicle: {str(e)}")
            return None
    
    def cmd_vel_callback(self, msg):
        """Handle cmd_vel messages for controlling the vehicle."""
        if msg.linear.x >= 0:
            self.control.throttle = float(msg.linear.x)
            self.control.brake = 0.0
        else:
            self.control.throttle = 0.0
            self.control.brake = float(-msg.linear.x)
        
        self.control.steer = float(msg.angular.z)
    
    def throttle_callback(self, msg):
        """Handle throttle control messages."""
        self.control.throttle = float(msg.data)
    
    def brake_callback(self, msg):
        """Handle brake control messages."""
        self.control.brake = float(msg.data)
    
    def handbrake_callback(self, msg):
        """Handle handbrake control messages."""
        self.control.hand_brake = bool(msg.data)
    
    def reverse_callback(self, msg):
        """Handle reverse gear control messages."""
        self.control.reverse = bool(msg.data)
    
    def apply_control(self):
        """Apply the current control values to the vehicle."""
        if self.vehicle and self.vehicle.is_alive:
            try:
                self.vehicle.apply_control(self.control)
                
                velocity = self.vehicle.get_velocity()
                speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                
                if hasattr(self, 'log_counter'):
                    self.log_counter += 1
                else:
                    self.log_counter = 0
                    
                if self.log_counter % 20 == 0:
                    self.get_logger().info(
                        f"Speed: {speed_kmh:.1f} km/h, "
                        f"Throttle: {self.control.throttle:.1f}, "
                        f"Steer: {self.control.steer:.1f}, "
                        f"Brake: {self.control.brake:.1f}"
                    )
                      
            except Exception as e:
                self.get_logger().error(f"Error applying control: {str(e)}")
                
    def destroy_node(self):
        """Clean up resources when the node is destroyed."""
        self.get_logger().info("Shutting down vehicle node...")
            
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            self.get_logger().info("Vehicle destroyed")
            
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VehicleNode()
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
