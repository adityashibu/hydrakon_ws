import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import carla
import time
import numpy as np
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist

class VehicleInterfaceNode(Node):
    """
    Vehicle Interface that routes PID commands through CARLA Vehicle Manager
    AND gets real feedback from CARLA vehicle state
    
    Subscribes to: /throttle_cmd, /brake_cmd, /steering_cmd (from PID controller)
    Publishes to: /carla/vehicle/cmd_vel, /carla/vehicle/throttle, /carla/vehicle/brake (to Vehicle Manager)
    Also publishes: /actual_throttle, /actual_brake, /actual_steering (REAL feedback from CARLA)
    """
    
    def __init__(self):
        super().__init__('vehicle_interface')
        
        # Parameters
        self.declare_parameter('use_carla', True)
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('command_timeout', 0.5)  
        self.declare_parameter('feedback_enabled', True)
        self.declare_parameter('max_speed_scale', 8.0)  # Maximum speed scaling for cmd_vel
        
        # Real car parameters (commented for CARLA setup)
        # self.declare_parameter('can_bus_enabled', False)
        # self.declare_parameter('can_interface', 'can0')
        # self.declare_parameter('serial_port', '/dev/ttyUSB0')
        # self.declare_parameter('sensor_feedback_topics', True)
        
        self.use_carla = self.get_parameter('use_carla').value
        self.carla_host = self.get_parameter('carla_host').value
        self.carla_port = self.get_parameter('carla_port').value
        self.carla_timeout = self.get_parameter('carla_timeout').value
        self.cmd_timeout = self.get_parameter('command_timeout').value
        self.feedback_enabled = self.get_parameter('feedback_enabled').value
        self.max_speed_scale = self.get_parameter('max_speed_scale').value
        
        # Subscribers (from PID controller)
        self.throttle_sub = self.create_subscription(
            Float64, '/throttle_cmd', self.throttle_callback, 10)
        self.brake_sub = self.create_subscription(
            Float64, '/brake_cmd', self.brake_callback, 10)
        self.steering_sub = self.create_subscription(
            Float64, '/steering_cmd', self.steering_callback, 10)
        
        # Publishers to CARLA Vehicle Manager topics
        self.carla_throttle_pub = self.create_publisher(Float32, '/carla/vehicle/throttle', 10)
        self.carla_brake_pub = self.create_publisher(Float32, '/carla/vehicle/brake', 10)
        self.carla_cmd_vel_pub = self.create_publisher(Twist, '/carla/vehicle/cmd_vel', 10)
        
        # REAL Feedback publishers (for PID controller) - now gets real CARLA data
        if self.feedback_enabled:
            self.actual_throttle_pub = self.create_publisher(Float64, '/actual_throttle', 10)
            self.actual_brake_pub = self.create_publisher(Float64, '/actual_brake', 10)
            self.actual_steering_pub = self.create_publisher(Float64, '/actual_steering', 10)
        
        # Command state (what we're commanding)
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_steering = 0.0
        
        # REAL state (what CARLA is actually doing)
        self.actual_throttle = 0.0
        self.actual_brake = 0.0
        self.actual_steering = 0.0
        
        # Command timing for timeout detection
        self.last_throttle_time = time.time()
        self.last_brake_time = time.time()
        self.last_steering_time = time.time()
        
        # CARLA connection for REAL feedback
        self.carla_client = None
        self.carla_world = None
        self.carla_vehicle = None
        
        if self.use_carla:
            self.setup_carla_connection()
        
        # Control timer to publish vehicle manager commands
        self.command_timer = self.create_timer(0.05, self.publish_vehicle_commands)  # 20Hz
        
        # Feedback timer to get real CARLA state
        if self.feedback_enabled and self.use_carla:
            self.feedback_timer = self.create_timer(0.02, self.update_carla_feedback)  # 50Hz feedback
        
        # Real car setup (commented out for CARLA)
        # if not self.use_carla:
        #     self.setup_real_car_interface()
        
        self.get_logger().info("Vehicle Interface Node initialized - REAL CARLA feedback enabled")
        self.get_logger().info(f"CARLA connection: {self.carla_host}:{self.carla_port}")
        self.get_logger().info(f"Max speed scale: {self.max_speed_scale} m/s")
        self.get_logger().info(f"Real feedback: {'ENABLED' if self.feedback_enabled else 'DISABLED'}")
    
    def setup_carla_connection(self):
        """Setup CARLA connection to get real vehicle feedback"""
        try:
            self.get_logger().info(f"Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
            self.carla_client = carla.Client(self.carla_host, self.carla_port)
            self.carla_client.set_timeout(self.carla_timeout)
            
            self.carla_world = self.carla_client.get_world()
            
            # Find the ego vehicle (assuming it's the first vehicle)
            vehicles = self.carla_world.get_actors().filter('vehicle.*')
            if vehicles:
                self.carla_vehicle = vehicles[0]  # Get first vehicle as ego
                self.get_logger().info(f"Found CARLA vehicle: {self.carla_vehicle.type_id} (ID: {self.carla_vehicle.id})")
            else:
                self.get_logger().error("No vehicles found in CARLA world!")
                self.carla_vehicle = None
                
        except Exception as e:
            self.get_logger().error(f"Failed to connect to CARLA: {e}")
            self.carla_client = None
            self.carla_world = None
            self.carla_vehicle = None
    
    # ============================================================================
    # REAL CAR INTERFACE (commented out - would be used for actual Formula Student car)
    # ============================================================================
    
    # def setup_real_car_interface(self):
    #     """Setup interface for real Formula Student car"""
    #     try:
    #         # CAN bus setup
    #         if self.get_parameter('can_bus_enabled').value:
    #             import can
    #             self.can_bus = can.interface.Bus(
    #                 interface='socketcan',
    #                 channel=self.get_parameter('can_interface').value,
    #                 bitrate=500000
    #             )
    #             self.get_logger().info("CAN bus interface initialized")
    #         
    #         # Serial interface setup (for Arduino/ECU communication)
    #         if self.get_parameter('serial_enabled').value:
    #             import serial
    #             self.serial_port = serial.Serial(
    #                 self.get_parameter('serial_port').value,
    #                 baudrate=115200,
    #                 timeout=0.1
    #             )
    #             self.get_logger().info("Serial interface initialized")
    #         
    #         # Setup subscribers for real sensor feedback
    #         if self.get_parameter('sensor_feedback_topics').value:
    #             self.real_throttle_sub = self.create_subscription(
    #                 Float64, '/sensors/throttle_position', self.real_throttle_callback, 10)
    #             self.real_brake_sub = self.create_subscription(
    #                 Float64, '/sensors/brake_pressure', self.real_brake_callback, 10)
    #             self.real_steering_sub = self.create_subscription(
    #                 Float64, '/sensors/steering_angle', self.real_steering_callback, 10)
    #         
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to setup real car interface: {e}")
    
    # def real_throttle_callback(self, msg):
    #     """Receive real throttle position from vehicle sensors"""
    #     self.actual_throttle = msg.data
    
    # def real_brake_callback(self, msg):
    #     """Receive real brake pressure from vehicle sensors"""
    #     self.actual_brake = msg.data
    
    # def real_steering_callback(self, msg):
    #     """Receive real steering angle from vehicle sensors"""
    #     self.actual_steering = msg.data
    
    # def send_can_commands(self):
    #     """Send commands via CAN bus to real vehicle ECU"""
    #     try:
    #         if hasattr(self, 'can_bus'):
    #             # Throttle command (CAN ID: 0x200)
    #             throttle_data = int(self.current_throttle * 255).to_bytes(1, 'big')
    #             throttle_msg = can.Message(arbitration_id=0x200, data=throttle_data)
    #             self.can_bus.send(throttle_msg)
    #             
    #             # Brake command (CAN ID: 0x201)
    #             brake_data = int(self.current_brake * 255).to_bytes(1, 'big')
    #             brake_msg = can.Message(arbitration_id=0x201, data=brake_data)
    #             self.can_bus.send(brake_msg)
    #             
    #             # Steering command (CAN ID: 0x202)
    #             steering_data = int((self.current_steering + 1.0) * 127.5).to_bytes(1, 'big')
    #             steering_msg = can.Message(arbitration_id=0x202, data=steering_data)
    #             self.can_bus.send(steering_msg)
    #     except Exception as e:
    #         self.get_logger().error(f"CAN bus communication error: {e}")
    
    # def send_serial_commands(self):
    #     """Send commands via serial to Arduino/microcontroller"""
    #     try:
    #         if hasattr(self, 'serial_port'):
    #             # Format: "T:0.75,B:0.00,S:0.25\n"
    #             command = f"T:{self.current_throttle:.2f},B:{self.current_brake:.2f},S:{self.current_steering:.2f}\n"
    #             self.serial_port.write(command.encode())
    #     except Exception as e:
    #         self.get_logger().error(f"Serial communication error: {e}")
    
    # ============================================================================
    # END REAL CAR INTERFACE
    # ============================================================================
    
    def throttle_callback(self, msg):
        """Receive throttle command from PID controller"""
        self.current_throttle = np.clip(msg.data, 0.0, 1.0)
        self.last_throttle_time = time.time()
        self.get_logger().debug(f"Throttle command: {self.current_throttle:.3f}")
    
    def brake_callback(self, msg):
        """Receive brake command from PID controller"""
        self.current_brake = np.clip(msg.data, 0.0, 1.0)
        self.last_brake_time = time.time()
        self.get_logger().debug(f"Brake command: {self.current_brake:.3f}")
    
    def steering_callback(self, msg):
        """Receive steering command from PID controller"""
        self.current_steering = np.clip(msg.data, -1.0, 1.0)
        self.last_steering_time = time.time()
        self.get_logger().debug(f"Steering command: {self.current_steering:.3f}")
    
    def check_command_timeouts(self):
        """Check for command timeouts and apply safety measures"""
        current_time = time.time()
        timeout_detected = False
        
        if current_time - self.last_throttle_time > self.cmd_timeout:
            self.current_throttle = 0.0
            timeout_detected = True
        
        if current_time - self.last_brake_time > self.cmd_timeout:
            # Don't reset brake on timeout - maintain last command for safety
            pass
        
        if current_time - self.last_steering_time > self.cmd_timeout:
            # Gradually return to center on steering timeout
            self.current_steering *= 0.9
            timeout_detected = True
        
        if timeout_detected:
            self.get_logger().warn("Command timeout detected - applying safety measures")
    
    def update_carla_feedback(self):
        """Get REAL feedback from CARLA vehicle state"""
        try:
            if self.carla_vehicle is not None:
                # Get REAL control values from CARLA vehicle
                carla_control = self.carla_vehicle.get_control()
                
                # Update actual values with REAL CARLA data
                self.actual_throttle = carla_control.throttle
                self.actual_brake = carla_control.brake
                self.actual_steering = carla_control.steer
                
                # Publish real feedback
                self.publish_real_feedback()
                
        except Exception as e:
            self.get_logger().error(f"Error getting CARLA feedback: {e}")
            # Fallback to commanded values if CARLA fails
            self.actual_throttle = self.current_throttle
            self.actual_brake = self.current_brake
            self.actual_steering = self.current_steering
    
    def publish_vehicle_commands(self):
        """Publish commands to CARLA Vehicle Manager with amplified steering"""
        try:
            # Check for command timeouts
            self.check_command_timeouts()
            
            if self.use_carla:
                # AMPLIFY: Increase steering command for testing
                amplified_steering = self.current_steering * 5.0  # 5x amplification
                amplified_steering = np.clip(amplified_steering, -1.0, 1.0)
                
                # Convert amplified steering to radians
                max_steering_rad = np.radians(41.0)  # Your vehicle's 41° limit
                steering_rad = amplified_steering * max_steering_rad
                
                # Method 1: Use cmd_vel (primary method)
                cmd_vel_msg = Twist()
                
                # Convert throttle/brake to linear velocity command
                if self.current_throttle > 0.01:  # Small threshold to avoid noise
                    # Scale throttle (0-1) to velocity (0 to max_speed_scale m/s)
                    cmd_vel_msg.linear.x = float(self.current_throttle * self.max_speed_scale)
                    cmd_vel_msg.linear.y = 0.0
                    cmd_vel_msg.linear.z = 0.0
                elif self.current_brake > 0.01:  # Brake command
                    # Negative velocity for braking
                    cmd_vel_msg.linear.x = float(-self.current_brake * self.max_speed_scale * 0.5)
                    cmd_vel_msg.linear.y = 0.0
                    cmd_vel_msg.linear.z = 0.0
                else:
                    # No throttle or brake - coast/stop
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.linear.y = 0.0
                    cmd_vel_msg.linear.z = 0.0
                
                # Set angular velocity in radians
                cmd_vel_msg.angular.x = 0.0
                cmd_vel_msg.angular.y = 0.0
                cmd_vel_msg.angular.z = float(steering_rad)  # Convert to radians!
                
                # Publish to vehicle manager
                self.carla_cmd_vel_pub.publish(cmd_vel_msg)
                
                # Method 2: Also publish individual commands (backup method)
                throttle_msg = Float32()
                throttle_msg.data = float(self.current_throttle)
                self.carla_throttle_pub.publish(throttle_msg)
                
                brake_msg = Float32()
                brake_msg.data = float(self.current_brake)
                self.carla_brake_pub.publish(brake_msg)
                
                # Debug logging with degrees for easy verification
                if hasattr(self, 'log_counter'):
                    self.log_counter += 1
                else:
                    self.log_counter = 0
                
                if self.log_counter % 20 == 0:  # Every 20 cycles (1 second at 20Hz)
                    self.get_logger().info(
                        f"Commands -> T: {self.current_throttle:.3f}, "
                        f"B: {self.current_brake:.3f}, "
                        f"S: {self.current_steering:.3f} -> {amplified_steering:.3f} ({np.degrees(steering_rad):.1f}°) | "
                        f"REAL -> T: {self.actual_throttle:.3f}, "
                        f"B: {self.actual_brake:.3f}, "
                        f"S: {self.actual_steering:.3f}"
                    )
            
            # For real car, use CAN/Serial instead of CARLA
            # else:
            #     self.send_can_commands()
            #     self.send_serial_commands()
            
        except Exception as e:
            self.get_logger().error(f"Error publishing vehicle commands: {e}")
    
    def publish_real_feedback(self):
        """Publish REAL feedback from CARLA to PID controller"""
        if not self.feedback_enabled:
            return
        
        try:
            # Publish REAL values from CARLA (not commanded values!)
            throttle_msg = Float64()
            throttle_msg.data = float(self.actual_throttle)  # REAL CARLA value
            self.actual_throttle_pub.publish(throttle_msg)
            
            brake_msg = Float64()
            brake_msg.data = float(self.actual_brake)        # REAL CARLA value
            self.actual_brake_pub.publish(brake_msg)
            
            steering_msg = Float64()
            steering_msg.data = float(self.actual_steering)   # REAL CARLA value
            self.actual_steering_pub.publish(steering_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing real feedback: {e}")
    
    def destroy_node(self):
        """Clean shutdown"""
        # Close real car interfaces
        # if hasattr(self, 'can_bus'):
        #     self.can_bus.shutdown()
        # if hasattr(self, 'serial_port'):
        #     self.serial_port.close()
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VehicleInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nVehicle Interface shutting down...")
    except Exception as e:
        print(f"Error in Vehicle Interface: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()