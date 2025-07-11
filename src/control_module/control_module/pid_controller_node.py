import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Point
import numpy as np
import time

class PIDAccelerationController(Node):
    """
    Simplified PID Controller for Formula Student vehicle - ACCELERATION ONLY
    
    Subscribes to: 
    - /planning/target_speed (Float64)
    - /planning/target_position (Point) - for future use with position control
    - /current_speed (Float64)
    - /mission_complete (Bool)
    
    Publishes to: 
    - /acceleration_cmd (Float64) - in m/s²
    """
    
    def __init__(self):
        super().__init__('pid_acceleration_controller')
        
        # Declare parameters - simplified for acceleration control only
        self.declare_parameter('speed_kp', 1.2)              # Proportional gain
        self.declare_parameter('speed_ki', 0.3)              # Integral gain  
        self.declare_parameter('speed_kd', 0.1)              # Derivative gain
        self.declare_parameter('speed_integral_limit', 3.0)  # Integral windup limit
        
        self.declare_parameter('max_acceleration', 4.0)      # Max acceleration m/s²
        self.declare_parameter('max_deceleration', -6.0)     # Max deceleration m/s²
        self.declare_parameter('control_frequency', 50.0)    # Control loop frequency Hz
        self.declare_parameter('emergency_decel', -8.0)      # Emergency brake deceleration
        self.declare_parameter('max_speed_limit', 15.0)      # Maximum allowed speed m/s
        
        # Get parameters
        self.speed_kp = self.get_parameter('speed_kp').value
        self.speed_ki = self.get_parameter('speed_ki').value
        self.speed_kd = self.get_parameter('speed_kd').value
        self.speed_int_limit = self.get_parameter('speed_integral_limit').value
        
        self.max_accel = self.get_parameter('max_acceleration').value
        self.max_decel = self.get_parameter('max_deceleration').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.emergency_decel = self.get_parameter('emergency_decel').value
        self.max_speed = self.get_parameter('max_speed_limit').value
        
        # Subscribers
        self.target_speed_sub = self.create_subscription(
            Float64, '/planning/target_speed', self.target_speed_callback, 10)
        self.target_position_sub = self.create_subscription(
            Point, '/planning/target_position', self.target_position_callback, 10)
        self.current_speed_sub = self.create_subscription(
            Float64, '/current_speed', self.current_speed_callback, 10)
        self.mission_complete_sub = self.create_subscription(
            Bool, '/mission_complete', self.mission_complete_callback, 10)
        
        # Publisher - ONLY acceleration command
        self.acceleration_pub = self.create_publisher(Float64, '/acceleration_cmd', 10)
        
        # Control state
        self.target_speed = 0.0
        self.target_position = Point()  # For future position control
        self.current_speed = 0.0
        self.mission_complete = False
        self.last_target_time = time.time()
        self.target_timeout = 2.0
        
        # PID state for speed control
        self.speed_integral = 0.0
        self.speed_prev_error = 0.0
        self.last_time = time.time()
        
        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_freq, self.control_loop)
        
        self.get_logger().info("🏎️  PID Acceleration Controller initialized")
        self.get_logger().info(f"Speed PID: Kp={self.speed_kp}, Ki={self.speed_ki}, Kd={self.speed_kd}")
        self.get_logger().info(f"Max acceleration: {self.max_accel} m/s², Max deceleration: {self.max_decel} m/s²")
        self.get_logger().info(f"Control frequency: {self.control_freq} Hz")
    
    def target_speed_callback(self, msg):
        """Receive target speed from planning module"""
        raw_speed = msg.data
        self.target_speed = min(raw_speed, self.max_speed)  # Apply speed limit
        self.last_target_time = time.time()
        
        if abs(raw_speed - self.target_speed) > 0.1:
            self.get_logger().debug(f"Target speed limited: {raw_speed:.2f} -> {self.target_speed:.2f} m/s")
    
    def target_position_callback(self, msg):
        """Receive target position from planning module (for future use)"""
        self.target_position = msg
        # TODO: Implement position-based control in future versions
    
    def current_speed_callback(self, msg):
        """Receive current speed from speed processing node"""
        self.current_speed = msg.data
    
    def mission_complete_callback(self, msg):
        """Receive mission status"""
        self.mission_complete = msg.data
        if self.mission_complete:
            self.get_logger().info("Mission complete - emergency stop")
    
    def control_loop(self):
        """Main PID control loop - outputs acceleration command only"""
        try:
            current_time = time.time()
            dt = current_time - self.last_time
            dt = max(dt, 0.001)  # Prevent division by zero
            
            # Emergency stop conditions
            if self.mission_complete:
                self.emergency_stop()
                return
            
            # Check timeout
            target_age = current_time - self.last_target_time
            if target_age > self.target_timeout:
                self.get_logger().warn(f"Target timeout ({target_age:.1f}s)! Emergency stop.")
                self.emergency_stop()
                return
            
            # Calculate speed error
            speed_error = self.target_speed - self.current_speed
            
            # PID calculation for acceleration
            acceleration_cmd = self.speed_pid_update(speed_error, dt)
            
            # Apply acceleration limits
            acceleration_cmd = np.clip(acceleration_cmd, self.max_decel, self.max_accel)
            
            # Publish acceleration command
            self.publish_acceleration_command(acceleration_cmd)
            
            self.last_time = current_time
            
            # Logging every 1 second
            if int(current_time * 1) % 1 == 0:
                self.log_control_status(speed_error, acceleration_cmd)
            
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            self.emergency_stop()
    
    def speed_pid_update(self, error, dt):
        """PID controller for speed -> acceleration"""
        # Proportional term
        P = self.speed_kp * error
        
        # Integral term with windup protection
        self.speed_integral += error * dt
        self.speed_integral = np.clip(self.speed_integral, -self.speed_int_limit, self.speed_int_limit)
        I = self.speed_ki * self.speed_integral
        
        # Derivative term
        derivative = (error - self.speed_prev_error) / dt
        D = self.speed_kd * derivative
        
        # Total PID output (acceleration in m/s²)
        acceleration = P + I + D
        
        self.speed_prev_error = error
        return acceleration
    
    def publish_acceleration_command(self, acceleration):
        """Publish acceleration command"""
        accel_msg = Float64()
        accel_msg.data = float(acceleration)
        self.acceleration_pub.publish(accel_msg)
    
    def log_control_status(self, speed_error, acceleration_cmd):
        """Log control status"""
        accel_direction = "ACCEL" if acceleration_cmd > 0 else "DECEL" if acceleration_cmd < 0 else "COAST"
        
        log_msg = (
            f"PID Acceleration - Target: {self.target_speed:.1f} m/s, "
            f"Current: {self.current_speed:.1f} m/s, "
            f"Error: {speed_error:+.2f} m/s, "
            f"Accel: {acceleration_cmd:+.2f} m/s² ({accel_direction})"
        )
        
        self.get_logger().info(log_msg)
    
    def emergency_stop(self):
        """Emergency stop - maximum deceleration"""
        accel_msg = Float64()
        accel_msg.data = float(self.emergency_decel)  # Maximum deceleration
        self.acceleration_pub.publish(accel_msg)
        
        # Reset PID state
        self.speed_integral = 0.0
        self.speed_prev_error = 0.0
        
        self.get_logger().warn(f"🚨 EMERGENCY STOP - Deceleration: {self.emergency_decel} m/s²")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PIDAccelerationController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()