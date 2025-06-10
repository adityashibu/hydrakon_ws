import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import time
from collections import deque

class PIDControllerNode(Node):
    """
    Advanced PID Controller with Complex Rate Limiting for Formula Student vehicle.
    
    Features sophisticated rate limiting including:
    - Jerk limiting (second derivative control)
    - Context-aware rate limiting based on speed/situation
    - Coordinated throttle/brake transitions
    - Emergency override capabilities
    - Multi-variable constraint handling
    
    Subscribes to: /target_speed, /target_steering, /current_speed, /path_curvature
    Publishes to: /throttle_cmd, /brake_cmd, /steering_cmd
    """
    
    def __init__(self):
        super().__init__('pid_controller')
        
        # Declare basic parameters
        self.declare_parameter('speed_kp', 0.8)
        self.declare_parameter('speed_ki', 0.15)
        self.declare_parameter('speed_kd', 0.05)
        self.declare_parameter('speed_integral_limit', 2.0)
        
        self.declare_parameter('steering_kp', 0.8)
        self.declare_parameter('steering_ki', 0.05)
        self.declare_parameter('steering_kd', 0.12)
        self.declare_parameter('steering_integral_limit', 0.8)
        
        self.declare_parameter('max_throttle', 0.8)
        self.declare_parameter('max_brake', 0.8)
        self.declare_parameter('max_steering', 1.0)
        self.declare_parameter('control_frequency', 100.0)
        self.declare_parameter('emergency_brake_decel', 5.0)
        self.declare_parameter('max_speed_limit', 15.0)
        
        # =================================================================
        # COMPLEX RATE LIMITING PARAMETERS
        # =================================================================
        
        # Steering rate limiting
        self.declare_parameter('steering_rate_limiting_enabled', True)
        self.declare_parameter('max_steering_rate', 0.15)              # rad/s - first derivative limit
        self.declare_parameter('max_steering_jerk', 0.8)               # rad/s² - second derivative limit (jerk)
        self.declare_parameter('steering_rate_speed_factor', 0.5)      # Reduce rate limits at high speed
        
        # Throttle rate limiting
        self.declare_parameter('throttle_rate_limiting_enabled', True)
        self.declare_parameter('max_throttle_rate', 0.3)               # /s - prevent sudden acceleration
        self.declare_parameter('max_throttle_jerk', 1.2)               # /s² - smooth throttle application
        self.declare_parameter('throttle_buildup_factor', 0.7)         # Slower initial throttle application
        
        # Brake rate limiting
        self.declare_parameter('brake_rate_limiting_enabled', True)
        self.declare_parameter('max_brake_rate', 0.4)                  # /s - prevent brake spikes
        self.declare_parameter('max_brake_jerk', 2.0)                  # /s² - smooth brake application
        self.declare_parameter('brake_release_factor', 0.3)            # Slow brake release to prevent rebound
        
        # Coordinated control
        self.declare_parameter('coordinated_control_enabled', True)
        self.declare_parameter('throttle_brake_transition_time', 0.2)  # Smooth transitions between T/B
        self.declare_parameter('coordinated_steering_factor', 0.8)     # Reduce steering during accel/decel
        
        # Emergency override
        self.declare_parameter('emergency_override_enabled', True)
        self.declare_parameter('emergency_speed_threshold', 0.5)       # Below this speed, relax limits
        self.declare_parameter('emergency_decel_threshold', 8.0)       # Above this decel, bypass limits
        
        # Speed-based steering limits (existing)
        self.declare_parameter('speed_based_steering_limits', True)
        self.declare_parameter('high_speed_threshold', 3.0)
        self.declare_parameter('medium_speed_threshold', 1.5)
        self.declare_parameter('high_speed_steering_limit', 0.3)
        self.declare_parameter('medium_speed_steering_limit', 0.6)
        self.declare_parameter('low_speed_steering_limit', 1.0)
        
        # Curvature-based speed adaptation (existing)
        self.declare_parameter('curvature_speed_adaptation', True)
        self.declare_parameter('max_curvature_speed_reduction', 0.5)
        self.declare_parameter('curvature_threshold', 0.1)
        
        # Get parameters
        self.speed_kp = self.get_parameter('speed_kp').value
        self.speed_ki = self.get_parameter('speed_ki').value
        self.speed_kd = self.get_parameter('speed_kd').value
        self.speed_int_limit = self.get_parameter('speed_integral_limit').value
        
        self.steer_kp = self.get_parameter('steering_kp').value
        self.steer_ki = self.get_parameter('steering_ki').value
        self.steer_kd = self.get_parameter('steering_kd').value
        self.steer_int_limit = self.get_parameter('steering_integral_limit').value
        
        self.max_throttle = self.get_parameter('max_throttle').value
        self.max_brake = self.get_parameter('max_brake').value
        self.max_steering = self.get_parameter('max_steering').value
        self.control_freq = self.get_parameter('control_frequency').value
        self.emergency_decel = self.get_parameter('emergency_brake_decel').value
        self.max_speed = self.get_parameter('max_speed_limit').value
        
        # Complex rate limiting parameters
        self.steering_rate_enabled = self.get_parameter('steering_rate_limiting_enabled').value
        self.max_steering_rate = self.get_parameter('max_steering_rate').value
        self.max_steering_jerk = self.get_parameter('max_steering_jerk').value
        self.steering_rate_speed_factor = self.get_parameter('steering_rate_speed_factor').value
        
        self.throttle_rate_enabled = self.get_parameter('throttle_rate_limiting_enabled').value
        self.max_throttle_rate = self.get_parameter('max_throttle_rate').value
        self.max_throttle_jerk = self.get_parameter('max_throttle_jerk').value
        self.throttle_buildup_factor = self.get_parameter('throttle_buildup_factor').value
        
        self.brake_rate_enabled = self.get_parameter('brake_rate_limiting_enabled').value
        self.max_brake_rate = self.get_parameter('max_brake_rate').value
        self.max_brake_jerk = self.get_parameter('max_brake_jerk').value
        self.brake_release_factor = self.get_parameter('brake_release_factor').value
        
        self.coordinated_control = self.get_parameter('coordinated_control_enabled').value
        self.tb_transition_time = self.get_parameter('throttle_brake_transition_time').value
        self.coordinated_steering_factor = self.get_parameter('coordinated_steering_factor').value
        
        self.emergency_override = self.get_parameter('emergency_override_enabled').value
        self.emergency_speed_threshold = self.get_parameter('emergency_speed_threshold').value
        self.emergency_decel_threshold = self.get_parameter('emergency_decel_threshold').value
        
        # Speed-based steering limits
        self.speed_steering_limits = self.get_parameter('speed_based_steering_limits').value
        self.high_speed_threshold = self.get_parameter('high_speed_threshold').value
        self.medium_speed_threshold = self.get_parameter('medium_speed_threshold').value
        self.high_speed_limit = self.get_parameter('high_speed_steering_limit').value
        self.medium_speed_limit = self.get_parameter('medium_speed_steering_limit').value
        self.low_speed_limit = self.get_parameter('low_speed_steering_limit').value
        
        # Curvature adaptation
        self.curvature_adaptation = self.get_parameter('curvature_speed_adaptation').value
        self.max_curvature_reduction = self.get_parameter('max_curvature_speed_reduction').value
        self.curvature_threshold = self.get_parameter('curvature_threshold').value
        
        # Subscribers
        self.target_speed_sub = self.create_subscription(
            Float64, '/target_speed', self.target_speed_callback, 10)
        self.target_steering_sub = self.create_subscription(
            Float64, '/target_steering', self.target_steering_callback, 10)
        self.current_speed_sub = self.create_subscription(
            Float64, '/current_speed', self.current_speed_callback, 10)
        self.mission_complete_sub = self.create_subscription(
            Bool, '/mission_complete', self.mission_complete_callback, 10)
        self.path_curvature_sub = self.create_subscription(
            Float64, '/path_curvature', self.path_curvature_callback, 10)
        
        # Publishers
        self.throttle_pub = self.create_publisher(Float64, '/throttle_cmd', 10)
        self.brake_pub = self.create_publisher(Float64, '/brake_cmd', 10)
        self.steering_pub = self.create_publisher(Float64, '/steering_cmd', 10)
        
        # Control state
        self.target_speed = 0.0
        self.target_steering = 0.0
        self.current_speed = 0.0
        self.mission_complete = False
        self.last_target_time = time.time()
        self.target_timeout = 2.0
        self.path_curvature = 0.0
        
        # PID state
        self.speed_integral = 0.0
        self.speed_prev_error = 0.0
        self.steer_integral = 0.0
        self.steer_prev_error = 0.0
        
        # =================================================================
        # COMPLEX RATE LIMITING STATE
        # =================================================================
        
        # Output history for rate and jerk calculation
        self.throttle_history = deque(maxlen=5)  # Store last few values for derivative calc
        self.brake_history = deque(maxlen=5)
        self.steering_history = deque(maxlen=5)
        
        # Rate tracking (first derivatives)
        self.throttle_rate = 0.0
        self.brake_rate = 0.0
        self.steering_rate = 0.0
        
        # Previous rates for jerk calculation (second derivatives)
        self.prev_throttle_rate = 0.0
        self.prev_brake_rate = 0.0
        self.prev_steering_rate = 0.0
        
        # Final limited outputs
        self.limited_throttle = 0.0
        self.limited_brake = 0.0
        self.limited_steering = 0.0
        
        # Transition state tracking
        self.last_throttle_brake_mode = 'none'  # 'throttle', 'brake', 'none'
        self.transition_start_time = 0.0
        
        # Emergency state
        self.emergency_active = False
        
        # Timing
        self.last_time = time.time()
        self.dt = 1.0 / self.control_freq
        
        # Initialize histories with zeros
        for _ in range(5):
            self.throttle_history.append(0.0)
            self.brake_history.append(0.0)
            self.steering_history.append(0.0)
        
        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_freq, self.control_loop)
        
        self.get_logger().info("🏎️  Advanced PID Controller with Complex Rate Limiting")
        self.get_logger().info(f"Speed PID: Kp={self.speed_kp}, Ki={self.speed_ki}, Kd={self.speed_kd}")
        self.get_logger().info(f"Steering PID: Kp={self.steer_kp}, Ki={self.steer_ki}, Kd={self.steer_kd}")
        self.get_logger().info("🔄 Complex Rate Limiting Features:")
        self.get_logger().info(f"  Steering: Rate={self.max_steering_rate:.2f}rad/s, Jerk={self.max_steering_jerk:.2f}rad/s²")
        self.get_logger().info(f"  Throttle: Rate={self.max_throttle_rate:.2f}/s, Jerk={self.max_throttle_jerk:.2f}/s²")
        self.get_logger().info(f"  Brake: Rate={self.max_brake_rate:.2f}/s, Jerk={self.max_brake_jerk:.2f}/s²")
        self.get_logger().info(f"  Coordinated Control: {'ENABLED' if self.coordinated_control else 'DISABLED'}")
        self.get_logger().info(f"  Emergency Override: {'ENABLED' if self.emergency_override else 'DISABLED'}")
    
    def target_speed_callback(self, msg):
        """Receive target speed with curvature adaptation"""
        raw_speed = msg.data
        
        if self.curvature_adaptation:
            adapted_speed = self.apply_curvature_speed_adaptation(raw_speed)
        else:
            adapted_speed = raw_speed
            
        self.target_speed = min(adapted_speed, self.max_speed)
        self.last_target_time = time.time()
        
        if abs(raw_speed - self.target_speed) > 0.1:
            self.get_logger().debug(f"Speed adapted: {raw_speed:.2f} -> {self.target_speed:.2f}")
    
    def target_steering_callback(self, msg):
        """Receive target steering with speed-based safety limits"""
        raw_steering = msg.data
        
        if self.speed_steering_limits:
            steering_limit = self.get_speed_based_steering_limit()
            limited_steering = np.clip(raw_steering, -steering_limit, steering_limit)
        else:
            limited_steering = np.clip(raw_steering, -self.max_steering, self.max_steering)
        
        self.target_steering = limited_steering
        self.last_target_time = time.time()
        
        if abs(raw_steering - self.target_steering) > 0.05:
            limit_reason = f"speed-based limit ({self.current_speed:.1f}m/s)" if self.speed_steering_limits else "physical limit"
            self.get_logger().warn(f"Steering limited by {limit_reason}: {raw_steering:.3f} -> {self.target_steering:.3f}")
    
    def get_speed_based_steering_limit(self):
        """Calculate maximum steering angle based on current speed"""
        try:
            if self.current_speed >= self.high_speed_threshold:
                return self.high_speed_limit
            elif self.current_speed >= self.medium_speed_threshold:
                return self.medium_speed_limit
            else:
                return self.low_speed_limit
        except Exception as e:
            self.get_logger().error(f"Error calculating speed-based steering limit: {e}")
            return self.max_steering
    
    def current_speed_callback(self, msg):
        """Receive current speed"""
        self.current_speed = msg.data
    
    def mission_complete_callback(self, msg):
        """Receive mission status"""
        self.mission_complete = msg.data
        if self.mission_complete:
            self.get_logger().info("Mission complete - emergency stop")
    
    def path_curvature_callback(self, msg):
        """Receive path curvature for speed adaptation"""
        self.path_curvature = msg.data
    
    def apply_curvature_speed_adaptation(self, target_speed):
        """Apply curvature-based speed adaptation"""
        try:
            if abs(self.path_curvature) < self.curvature_threshold:
                return target_speed
            
            curvature_severity = min(abs(self.path_curvature) / 0.5, 1.0)
            speed_reduction_factor = 1.0 - (curvature_severity * self.max_curvature_reduction)
            adapted_speed = target_speed * speed_reduction_factor
            adapted_speed = max(adapted_speed, target_speed * 0.3)
            
            return adapted_speed
        except Exception as e:
            self.get_logger().error(f"Curvature adaptation error: {e}")
            return target_speed
    
    def check_emergency_conditions(self):
        """Check if emergency override should be activated"""
        # Low speed emergency - relax limits for maneuvering
        low_speed_emergency = self.current_speed < self.emergency_speed_threshold
        
        # High deceleration emergency - immediate braking needed
        required_decel = abs(self.current_speed - self.target_speed) / self.dt
        high_decel_emergency = required_decel > self.emergency_decel_threshold
        
        # Mission complete emergency
        mission_emergency = self.mission_complete
        
        # Timeout emergency
        timeout_emergency = (time.time() - self.last_target_time) > self.target_timeout
        
        self.emergency_active = (low_speed_emergency or high_decel_emergency or 
                               mission_emergency or timeout_emergency)
        
        return self.emergency_active
    
    def apply_complex_rate_limiting(self, raw_throttle, raw_brake, raw_steering, dt):
        """
        🧠 COMPLEX RATE LIMITING CORE ALGORITHM
        
        Applies sophisticated rate and jerk limiting with coordination
        """
        current_time = time.time()
        
        # Check emergency conditions
        emergency = self.check_emergency_conditions() if self.emergency_override else False
        
        if emergency:
            # Emergency mode - bypass most limits but keep basic safety
            limited_throttle = np.clip(raw_throttle, 0.0, self.max_throttle)
            limited_brake = np.clip(raw_brake, 0.0, self.max_brake)
            limited_steering = np.clip(raw_steering, -self.max_steering, self.max_steering)
            
            self.get_logger().debug("🚨 Emergency mode - rate limits bypassed")
            
        else:
            # Normal mode - apply full complex rate limiting
            
            # 1. STEERING RATE AND JERK LIMITING
            limited_steering = self.apply_steering_rate_limiting(raw_steering, dt)
            
            # 2. THROTTLE/BRAKE COORDINATION AND RATE LIMITING
            limited_throttle, limited_brake = self.apply_throttle_brake_coordination(
                raw_throttle, raw_brake, dt, current_time)
            
            # 3. COORDINATED STEERING REDUCTION DURING ACCELERATION/DECELERATION
            if self.coordinated_control and (limited_throttle > 0.1 or limited_brake > 0.1):
                steering_reduction = self.coordinated_steering_factor
                limited_steering *= steering_reduction
                self.get_logger().debug(f"Coordinated steering reduction: {steering_reduction:.2f}")
        
        # Update histories for next iteration
        self.update_control_histories(limited_throttle, limited_brake, limited_steering)
        
        # Store final outputs
        self.limited_throttle = limited_throttle
        self.limited_brake = limited_brake
        self.limited_steering = limited_steering
        
        return limited_throttle, limited_brake, limited_steering
    
    def apply_steering_rate_limiting(self, raw_steering, dt):
        """Apply steering rate and jerk limiting with speed-based adaptation"""
        if not self.steering_rate_enabled:
            return raw_steering
        
        # Get previous values
        prev_steering = self.steering_history[-1]
        prev_prev_steering = self.steering_history[-2] if len(self.steering_history) >= 2 else prev_steering
        
        # Calculate current rate request
        desired_rate = (raw_steering - prev_steering) / dt
        
        # Speed-based rate limit adaptation
        speed_factor = 1.0
        if self.current_speed > self.medium_speed_threshold:
            speed_factor = self.steering_rate_speed_factor
        
        max_rate = self.max_steering_rate * speed_factor
        
        # Rate limiting
        if abs(desired_rate) > max_rate:
            limited_rate = np.sign(desired_rate) * max_rate
            limited_steering = prev_steering + limited_rate * dt
        else:
            limited_rate = desired_rate
            limited_steering = raw_steering
        
        # Jerk limiting (second derivative)
        prev_rate = (prev_steering - prev_prev_steering) / dt
        desired_jerk = (limited_rate - prev_rate) / dt
        
        if abs(desired_jerk) > self.max_steering_jerk:
            limited_jerk = np.sign(desired_jerk) * self.max_steering_jerk
            final_rate = prev_rate + limited_jerk * dt
            limited_steering = prev_steering + final_rate * dt
            
            self.get_logger().debug(f"Steering jerk limited: {desired_jerk:.2f} -> {limited_jerk:.2f} rad/s²")
        
        # Final range check
        limited_steering = np.clip(limited_steering, -self.max_steering, self.max_steering)
        
        return limited_steering
    
    def apply_throttle_brake_coordination(self, raw_throttle, raw_brake, dt, current_time):
        """Apply coordinated throttle/brake rate limiting with smooth transitions"""
        
        # Determine current mode
        if raw_throttle > 0.05 and raw_brake < 0.05:
            current_mode = 'throttle'
        elif raw_brake > 0.05 and raw_throttle < 0.05:
            current_mode = 'brake'
        else:
            current_mode = 'none'
        
        # Check for mode transition
        if current_mode != self.last_throttle_brake_mode and self.last_throttle_brake_mode != 'none':
            self.transition_start_time = current_time
            self.get_logger().debug(f"Mode transition: {self.last_throttle_brake_mode} -> {current_mode}")
        
        # Handle transitions with smooth blending
        transition_active = (current_time - self.transition_start_time) < self.tb_transition_time
        
        if transition_active and self.coordinated_control:
            # During transition, blend gradually
            transition_progress = (current_time - self.transition_start_time) / self.tb_transition_time
            blend_factor = min(transition_progress, 1.0)
            
            if self.last_throttle_brake_mode == 'throttle' and current_mode == 'brake':
                # Throttle to brake - ease off throttle before applying brake
                raw_throttle *= (1.0 - blend_factor)
                raw_brake *= blend_factor
            elif self.last_throttle_brake_mode == 'brake' and current_mode == 'throttle':
                # Brake to throttle - ease off brake before applying throttle  
                raw_brake *= (1.0 - blend_factor)
                raw_throttle *= blend_factor
        
        # Apply individual rate limiting
        limited_throttle = self.apply_throttle_rate_limiting(raw_throttle, dt)
        limited_brake = self.apply_brake_rate_limiting(raw_brake, dt)
        
        # Store mode for next iteration
        self.last_throttle_brake_mode = current_mode
        
        return limited_throttle, limited_brake
    
    def apply_throttle_rate_limiting(self, raw_throttle, dt):
        """Apply throttle-specific rate and jerk limiting"""
        if not self.throttle_rate_enabled:
            return np.clip(raw_throttle, 0.0, self.max_throttle)
        
        prev_throttle = self.throttle_history[-1]
        prev_prev_throttle = self.throttle_history[-2] if len(self.throttle_history) >= 2 else prev_throttle
        
        # Calculate desired rate
        desired_rate = (raw_throttle - prev_throttle) / dt
        
        # Apply buildup factor for initial acceleration
        max_rate = self.max_throttle_rate
        if prev_throttle < 0.1 and raw_throttle > prev_throttle:  # Starting acceleration
            max_rate *= self.throttle_buildup_factor
        
        # Rate limiting
        if abs(desired_rate) > max_rate:
            limited_rate = np.sign(desired_rate) * max_rate
            limited_throttle = prev_throttle + limited_rate * dt
        else:
            limited_rate = desired_rate
            limited_throttle = raw_throttle
        
        # Jerk limiting
        prev_rate = (prev_throttle - prev_prev_throttle) / dt
        desired_jerk = (limited_rate - prev_rate) / dt
        
        if abs(desired_jerk) > self.max_throttle_jerk:
            limited_jerk = np.sign(desired_jerk) * self.max_throttle_jerk
            final_rate = prev_rate + limited_jerk * dt
            limited_throttle = prev_throttle + final_rate * dt
        
        return np.clip(limited_throttle, 0.0, self.max_throttle)
    
    def apply_brake_rate_limiting(self, raw_brake, dt):
        """Apply brake-specific rate and jerk limiting"""
        if not self.brake_rate_enabled:
            return np.clip(raw_brake, 0.0, self.max_brake)
        
        prev_brake = self.brake_history[-1]
        prev_prev_brake = self.brake_history[-2] if len(self.brake_history) >= 2 else prev_brake
        
        # Calculate desired rate
        desired_rate = (raw_brake - prev_brake) / dt
        
        # Apply release factor for brake release
        max_rate = self.max_brake_rate
        if prev_brake > 0.1 and raw_brake < prev_brake:  # Releasing brake
            max_rate *= self.brake_release_factor
        
        # Rate limiting
        if abs(desired_rate) > max_rate:
            limited_rate = np.sign(desired_rate) * max_rate
            limited_brake = prev_brake + limited_rate * dt
        else:
            limited_rate = desired_rate
            limited_brake = raw_brake
        
        # Jerk limiting
        prev_rate = (prev_brake - prev_prev_brake) / dt
        desired_jerk = (limited_rate - prev_rate) / dt
        
        if abs(desired_jerk) > self.max_brake_jerk:
            limited_jerk = np.sign(desired_jerk) * self.max_brake_jerk
            final_rate = prev_rate + limited_jerk * dt
            limited_brake = prev_brake + final_rate * dt
        
        return np.clip(limited_brake, 0.0, self.max_brake)
    
    def update_control_histories(self, throttle, brake, steering):
        """Update control histories for rate/jerk calculation"""
        self.throttle_history.append(throttle)
        self.brake_history.append(brake)
        self.steering_history.append(steering)
    
    def control_loop(self):
        """Main PID control loop with complex rate limiting"""
        try:
            current_time = time.time()
            dt = current_time - self.last_time
            dt = max(dt, 0.001)
            
            if self.mission_complete:
                self.emergency_stop()
                return
            
            # Check timeout
            target_age = current_time - self.last_target_time
            if target_age > self.target_timeout:
                self.get_logger().warn(f"Target timeout! Emergency stop.")
                self.emergency_stop()
                return
            
            # Basic PID calculations
            speed_error = self.target_speed - self.current_speed
            raw_throttle_brake = self.speed_pid_update(speed_error, dt)
            
            steering_error = self.target_steering - 0.0  # Open loop
            raw_steering = self.steering_pid_update(steering_error, dt)
            
            # Convert throttle/brake command
            if raw_throttle_brake >= 0.05:
                raw_throttle = min(raw_throttle_brake, self.max_throttle)
                raw_brake = 0.0
            elif raw_throttle_brake <= -0.05:
                raw_throttle = 0.0
                raw_brake = min(abs(raw_throttle_brake), self.max_brake)
            else:
                raw_throttle = 0.0
                raw_brake = 0.0
            
            # 🧠 APPLY COMPLEX RATE LIMITING
            limited_throttle, limited_brake, limited_steering = self.apply_complex_rate_limiting(
                raw_throttle, raw_brake, raw_steering, dt)
            
            # Publish final commands
            self.publish_commands(limited_throttle, limited_brake, limited_steering)
            
            self.last_time = current_time
            
            # Enhanced logging
            if int(current_time * 2) % 2 == 0:
                self.log_control_status(speed_error, steering_error, raw_throttle_brake, limited_steering)
            
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
    
    def speed_pid_update(self, error, dt):
        """Basic speed PID controller"""
        P = self.speed_kp * error
        
        self.speed_integral += error * dt
        self.speed_integral = np.clip(self.speed_integral, -self.speed_int_limit, self.speed_int_limit)
        I = self.speed_ki * self.speed_integral
        
        derivative = (error - self.speed_prev_error) / dt
        D = self.speed_kd * derivative
        
        output = P + I + D
        output = np.clip(output, -1.0, 1.0)
        
        self.speed_prev_error = error
        return output
    
    def steering_pid_update(self, error, dt):
        """Basic steering PID controller (rate limiting applied separately)"""
        P = self.steer_kp * error
        
        self.steer_integral += error * dt
        self.steer_integral = np.clip(self.steer_integral, -self.steer_int_limit, self.steer_int_limit)
        I = self.steer_ki * self.steer_integral
        
        derivative = (error - self.steer_prev_error) / dt
        D = self.steer_kd * derivative
        
        output = P + I + D
        output = np.clip(output, -self.max_steering, self.max_steering)
        
        self.steer_prev_error = error
        return output
    
    def publish_commands(self, throttle, brake, steering):
        """Publish final control commands"""
        throttle_msg = Float64()
        throttle_msg.data = float(throttle)
        self.throttle_pub.publish(throttle_msg)
        
        brake_msg = Float64()
        brake_msg.data = float(brake)
        self.brake_pub.publish(brake_msg)
        
        steering_msg = Float64()
        steering_msg.data = float(steering)
        self.steering_pub.publish(steering_msg)
    
    def log_control_status(self, speed_error, steering_error, raw_throttle_brake, final_steering):
        """Enhanced logging with rate limiting info"""
        direction = 'LEFT' if final_steering > 0 else 'RIGHT' if final_steering < 0 else 'STRAIGHT'
        
        log_msg = (
            f"Advanced PID - Target: {self.target_speed:.1f}m/s, {self.target_steering:.2f}rad | "
            f"Current: {self.current_speed:.1f}m/s | "
            f"T/B: {self.limited_throttle:.2f}/{self.limited_brake:.2f}, "
            f"S: {final_steering:.3f}({direction})"
        )
        
        if self.emergency_active:
            log_msg += " | 🚨EMERGENCY"
        
        if self.coordinated_control and (self.limited_throttle > 0.1 or self.limited_brake > 0.1):
            log_msg += " | 🔄COORD"
        
        self.get_logger().debug(log_msg)
    
    def emergency_stop(self):
        """Emergency stop with immediate command override"""
        # Bypass rate limiting for emergency stop
        throttle_msg = Float64()
        throttle_msg.data = 0.0
        self.throttle_pub.publish(throttle_msg)
        
        brake_msg = Float64()
        brake_msg.data = 1.0
        self.brake_pub.publish(brake_msg)
        
        steering_msg = Float64()
        steering_msg.data = 0.0
        self.steering_pub.publish(steering_msg)
        
        # Reset states
        self.speed_integral = 0.0
        self.steer_integral = 0.0
        
        # Reset rate limiting histories
        for i in range(len(self.throttle_history)):
            self.throttle_history[i] = 0.0
            self.brake_history[i] = 1.0
            self.steering_history[i] = 0.0

def main(args=None):
    rclpy.init(args=args)
    try:
        node = PIDControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()