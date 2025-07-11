import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
import carla
import time
import numpy as np

class VehicleInterfaceNode(Node):
    """
    Vehicle Interface for Formula Student - Direct CARLA Control
    
    Mimics ADS-DV CAN API interface:
    - Accepts acceleration commands (m/s²) from PID controller
    - Accepts steering commands (radians) from planner
    - Directly controls CARLA vehicle via carla.VehicleControl()
    
    Subscribes to: 
    - /acceleration_cmd (Float64) - from PID controller
    - /planning/reference_steering (Float64) - from planner (in degrees)
    - /mission_complete (Bool)
    
    Publishes to: 
    - /actual_acceleration (Float64) - real feedback from CARLA
    - /actual_steering (Float64) - real feedback from CARLA
    - /current_speed (Float64) - for PID controller
    """
    
    def __init__(self):
        super().__init__('vehicle_interface')
        
        # Parameters
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('command_timeout', 0.5)
        self.declare_parameter('feedback_enabled', True)
        self.declare_parameter('control_frequency', 50.0)  # Hz
        
        # Vehicle dynamics parameters
        self.declare_parameter('max_steering_angle', 0.7)    # radians (≈40°) - internal limit
        self.declare_parameter('max_throttle', 1.0)          # 0-1
        self.declare_parameter('max_brake', 1.0)             # 0-1
        self.declare_parameter('accel_to_throttle_gain', 0.25) # Convert accel to throttle
        self.declare_parameter('decel_to_brake_gain', 0.15)    # Convert decel to brake
        
        # Get parameters
        self.carla_host = self.get_parameter('carla_host').value
        self.carla_port = self.get_parameter('carla_port').value
        self.carla_timeout = self.get_parameter('carla_timeout').value
        self.cmd_timeout = self.get_parameter('command_timeout').value
        self.feedback_enabled = self.get_parameter('feedback_enabled').value
        self.control_freq = self.get_parameter('control_frequency').value
        
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.max_throttle = self.get_parameter('max_throttle').value
        self.max_brake = self.get_parameter('max_brake').value
        self.accel_gain = self.get_parameter('accel_to_throttle_gain').value
        self.decel_gain = self.get_parameter('decel_to_brake_gain').value
        
        # Subscribers - NEW INTERFACE
        self.acceleration_sub = self.create_subscription(
            Float64, '/acceleration_cmd', self.acceleration_callback, 10)
        self.steering_sub = self.create_subscription(
            Float64, '/planning/reference_steering', self.steering_callback, 10)
        self.mission_complete_sub = self.create_subscription(
            Bool, '/mission_complete', self.mission_complete_callback, 10)
        
        # Publishers - Feedback to other nodes
        if self.feedback_enabled:
            self.actual_acceleration_pub = self.create_publisher(Float64, '/actual_acceleration', 10)
            self.actual_steering_pub = self.create_publisher(Float64, '/actual_steering', 10)
        
        # Speed feedback for PID controller
        self.current_speed_pub = self.create_publisher(Float64, '/current_speed', 10)
        
        # Command state
        self.current_acceleration = 0.0  # m/s²
        self.current_steering = 0.0      # radians
        self.mission_complete = False
        
        # Command timing for timeout detection
        self.last_acceleration_time = time.time()
        self.last_steering_time = time.time()
        
        # CARLA connection
        self.carla_client = None
        self.carla_world = None
        self.carla_vehicle = None
        self.vehicle_control = None
        
        # Vehicle state tracking
        self.previous_velocity = 0.0
        self.current_velocity = 0.0
        self.actual_acceleration = 0.0
        self.actual_steering = 0.0
        
        # Setup CARLA connection
        self.setup_carla_connection()
        
        # Control timer - apply commands to CARLA
        self.control_timer = self.create_timer(1.0/self.control_freq, self.apply_vehicle_control)
        
        # Feedback timer - get real state from CARLA
        if self.feedback_enabled:
            self.feedback_timer = self.create_timer(0.02, self.update_vehicle_feedback)  # 50Hz
        
        self.get_logger().info("🏎️  Vehicle Interface Node - Direct CARLA Control")
        self.get_logger().info(f"CARLA: {self.carla_host}:{self.carla_port}")
        self.get_logger().info(f"Max steering: ±{np.degrees(self.max_steering_angle):.1f}° (input from planner in degrees)")
        self.get_logger().info(f"Accel gain: {self.accel_gain}, Decel gain: {self.decel_gain}")
        self.get_logger().info(f"Control frequency: {self.control_freq} Hz")
    
    def setup_carla_connection(self):
        """Setup CARLA connection and find ego vehicle"""
        try:
            self.get_logger().info(f"Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
            self.carla_client = carla.Client(self.carla_host, self.carla_port)
            self.carla_client.set_timeout(self.carla_timeout)
            
            self.carla_world = self.carla_client.get_world()
            
            # Find the ego vehicle (first vehicle or vehicle with specific role_name)
            vehicles = self.carla_world.get_actors().filter('vehicle.*')
            if vehicles:
                # Try to find vehicle with role_name 'ego' first
                ego_vehicles = [v for v in vehicles if v.attributes.get('role_name') == 'ego']
                if ego_vehicles:
                    self.carla_vehicle = ego_vehicles[0]
                    self.get_logger().info(f"Found EGO vehicle: {self.carla_vehicle.type_id}")
                else:
                    # Fallback to first vehicle
                    self.carla_vehicle = vehicles[0]
                    self.get_logger().info(f"Using first vehicle as EGO: {self.carla_vehicle.type_id}")
                
                # Initialize vehicle control
                self.vehicle_control = carla.VehicleControl()
                self.get_logger().info(f"Vehicle ID: {self.carla_vehicle.id}")
                
            else:
                self.get_logger().error("❌ No vehicles found in CARLA world!")
                self.carla_vehicle = None
                
        except Exception as e:
            self.get_logger().error(f"❌ Failed to connect to CARLA: {e}")
            self.carla_client = None
            self.carla_world = None
            self.carla_vehicle = None
    
    def acceleration_callback(self, msg):
        """Receive acceleration command from PID controller"""
        self.current_acceleration = msg.data  # m/s²
        self.last_acceleration_time = time.time()
        self.get_logger().debug(f"Acceleration command: {self.current_acceleration:+.2f} m/s²")
    
    def steering_callback(self, msg):
        """Receive steering command from planner (in degrees) and convert to radians"""
        steering_degrees = msg.data
        steering_radians = np.radians(steering_degrees)  # Convert degrees to radians
        self.current_steering = np.clip(steering_radians, -self.max_steering_angle, self.max_steering_angle)
        self.last_steering_time = time.time()
        self.get_logger().debug(f"Steering command: {steering_degrees:+.1f}° -> {self.current_steering:+.3f} rad")
    
    def mission_complete_callback(self, msg):
        """Receive mission status"""
        self.mission_complete = msg.data
        if self.mission_complete:
            self.get_logger().info("Mission complete - stopping vehicle")
    
    def check_command_timeouts(self):
        """Check for command timeouts and apply safety measures"""
        current_time = time.time()
        
        if current_time - self.last_acceleration_time > self.cmd_timeout:
            self.current_acceleration = -2.0  # Emergency deceleration
            self.get_logger().warn("Acceleration command timeout - emergency braking")
        
        if current_time - self.last_steering_time > self.cmd_timeout:
            # Gradually return steering to center
            self.current_steering *= 0.95
            if abs(self.current_steering) < 0.01:
                self.current_steering = 0.0
    
    def acceleration_to_throttle_brake(self, acceleration):
        """Convert acceleration command to throttle/brake values"""
        if self.mission_complete:
            return 0.0, 1.0  # Emergency stop
        
        if acceleration > 0.1:  # Accelerating
            throttle = min(acceleration * self.accel_gain, self.max_throttle)
            brake = 0.0
        elif acceleration < -0.1:  # Decelerating
            throttle = 0.0
            brake = min(abs(acceleration) * self.decel_gain, self.max_brake)
        else:  # Coasting
            throttle = 0.0
            brake = 0.0
        
        return throttle, brake
    
    def apply_vehicle_control(self):
        """Apply control commands directly to CARLA vehicle"""
        try:
            if self.carla_vehicle is None:
                return
            
            # Check for timeouts
            self.check_command_timeouts()
            
            # Convert acceleration to throttle/brake
            throttle, brake = self.acceleration_to_throttle_brake(self.current_acceleration)
            
            # Convert steering from radians to CARLA's -1 to 1 range
            steering_normalized = self.current_steering / self.max_steering_angle
            steering_normalized = np.clip(steering_normalized, -1.0, 1.0)
            
            # Create and apply vehicle control
            if self.vehicle_control is None:
                self.vehicle_control = carla.VehicleControl()
            
            self.vehicle_control.throttle = float(throttle)
            self.vehicle_control.brake = float(brake)
            self.vehicle_control.steer = float(steering_normalized)
            self.vehicle_control.hand_brake = self.mission_complete
            self.vehicle_control.reverse = False
            
            # Apply control to vehicle
            self.carla_vehicle.apply_control(self.vehicle_control)
            
            # Debug logging
            if hasattr(self, 'log_counter'):
                self.log_counter += 1
            else:
                self.log_counter = 0
            
            if self.log_counter % 50 == 0:  # Every 50 cycles (1 second at 50Hz)
                self.get_logger().info(
                    f"🚗 Control -> Accel: {self.current_acceleration:+.2f} m/s² "
                    f"-> T: {throttle:.3f}, B: {brake:.3f} | "
                    f"Steer: {self.current_steering:+.3f} rad ({np.degrees(self.current_steering):+.1f}°) "
                    f"-> {steering_normalized:+.3f} | Speed: {self.current_velocity:.1f} m/s"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error applying vehicle control: {e}")
    
    def update_vehicle_feedback(self):
        """Get real feedback from CARLA vehicle and publish"""
        try:
            if self.carla_vehicle is None:
                return
            
            # Get vehicle velocity
            velocity = self.carla_vehicle.get_velocity()
            self.current_velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Calculate actual acceleration (numerical derivative)
            dt = 0.02  # 50Hz = 0.02s
            self.actual_acceleration = (self.current_velocity - self.previous_velocity) / dt
            self.previous_velocity = self.current_velocity
            
            # Get actual steering from vehicle control
            actual_control = self.carla_vehicle.get_control()
            self.actual_steering = actual_control.steer * self.max_steering_angle  # Convert back to radians
            
            # Publish feedback
            if self.feedback_enabled:
                # Actual acceleration
                accel_msg = Float64()
                accel_msg.data = float(self.actual_acceleration)
                self.actual_acceleration_pub.publish(accel_msg)
                
                # Actual steering
                steering_msg = Float64()
                steering_msg.data = float(self.actual_steering)
                self.actual_steering_pub.publish(steering_msg)
            
            # Publish current speed for PID controller
            speed_msg = Float64()
            speed_msg.data = float(self.current_velocity)
            self.current_speed_pub.publish(speed_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error getting vehicle feedback: {e}")
    
    def destroy_node(self):
        """Clean shutdown"""
        if self.carla_vehicle is not None:
            # Stop the vehicle
            stop_control = carla.VehicleControl()
            stop_control.throttle = 0.0
            stop_control.brake = 1.0
            stop_control.steer = 0.0
            stop_control.hand_brake = True
            self.carla_vehicle.apply_control(stop_control)
            self.get_logger().info("Vehicle stopped and handbrake applied")
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VehicleInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 Vehicle Interface shutting down...")
    except Exception as e:
        print(f"❌ Error in Vehicle Interface: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


     
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