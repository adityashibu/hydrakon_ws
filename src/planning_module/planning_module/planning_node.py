import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
import carla
import numpy as np
import cv2
import time
import threading
import signal
import sys
from collections import deque
from .zed_2i import Zed2iCamera

class LapCounter:
    def __init__(self):
        self.laps_completed = 0
        self.last_orange_gate_time = 0
        self.cooldown_duration = 3.0  # 3 seconds cooldown between lap counts
        self.orange_gate_passed_threshold = 2.0  # Distance threshold for passing through orange gate
        
    def check_orange_gate_passage(self, orange_cones, vehicle_position):
        """Check if vehicle has passed between two orange cones"""
        current_time = time.time()
        
        # Cooldown check to prevent multiple counts for same gate
        if current_time - self.last_orange_gate_time < self.cooldown_duration:
            return False
        
        # Check single orange cone or pair
        if len(orange_cones) < 1:
            return False
        
        # Find the closest orange gate (pair of orange cones) or single cone
        best_gate = self.find_closest_orange_gate(orange_cones)
        
        if not best_gate:
            # If no gate found, try single closest orange cone
            if len(orange_cones) >= 1:
                closest_orange = min(orange_cones, key=lambda c: c['depth'])
                if closest_orange['depth'] < 3.0:  # Very close to single orange cone
                    self.laps_completed += 1
                    self.last_orange_gate_time = current_time
                    print(f"🏁 LAP {self.laps_completed} COMPLETED! Passed single orange cone!")
                    return True
            return False
        
        # Check if vehicle is close enough to the gate center
        gate_center_x = best_gate['midpoint_x']
        gate_center_y = best_gate['midpoint_y']
        
        # Convert to vehicle-relative coordinates for distance check
        distance_to_gate = np.sqrt(gate_center_x**2 + gate_center_y**2)
        
        print(f"DEBUG: Orange gate distance: {distance_to_gate:.2f}m, threshold: {self.orange_gate_passed_threshold:.2f}m")
        print(f"DEBUG: Gate center: ({gate_center_x:.2f}, {gate_center_y:.2f})")
        
        if distance_to_gate < self.orange_gate_passed_threshold:
            self.laps_completed += 1
            self.last_orange_gate_time = current_time
            print(f"🏁 LAP {self.laps_completed} COMPLETED! Passed through orange gate!")
            return True
        
        return False
    
    def find_closest_orange_gate(self, orange_cones):
        """Find the closest valid orange gate (pair of orange cones)"""
        if len(orange_cones) < 2:
            return None
        
        # Sort orange cones by depth (closest first)
        orange_cones.sort(key=lambda c: c['depth'])
        
        # Try to pair cones to form a gate
        for i in range(len(orange_cones)):
            for j in range(i + 1, len(orange_cones)):
                cone1 = orange_cones[i]
                cone2 = orange_cones[j]
                
                # Check if cones can form a valid gate
                if self.is_valid_orange_gate(cone1, cone2):
                    gate = {
                        'cone1': cone1,
                        'cone2': cone2,
                        'midpoint_x': (cone1['x'] + cone2['x']) / 2,
                        'midpoint_y': (cone1['y'] + cone2['y']) / 2,
                        'width': abs(cone1['x'] - cone2['x']),
                        'avg_depth': (cone1['depth'] + cone2['depth']) / 2
                    }
                    print(f"DEBUG: Found orange gate - Width: {gate['width']:.2f}m, Depth: {gate['avg_depth']:.2f}m")
                    return gate
        
        return None
    
    def is_valid_orange_gate(self, cone1, cone2):
        """Check if two orange cones can form a valid gate"""
        # Check depth similarity
        depth_diff = abs(cone1['depth'] - cone2['depth'])
        if depth_diff > 3.0:  # More lenient for orange cones
            print(f"DEBUG: Orange gate rejected - depth diff: {depth_diff:.2f}m")
            return False
        
        # Check gate width (should be reasonable for a lap marker)
        width = abs(cone1['x'] - cone2['x'])
        if width < 1.5 or width > 12.0:  # More lenient width range
            print(f"DEBUG: Orange gate rejected - width: {width:.2f}m")
            return False
        
        # Check if gate is close enough
        avg_depth = (cone1['depth'] + cone2['depth']) / 2
        if avg_depth > 15.0:  # Allow farther orange gates
            print(f"DEBUG: Orange gate rejected - too far: {avg_depth:.2f}m")
            return False
        
        print(f"DEBUG: Valid orange gate found - Width: {width:.2f}m, Depth: {avg_depth:.2f}m")
        return True

class PurePursuitController(Node):
    def __init__(self, vehicle, lookahead_distance=4.0):
        super().__init__('carla_racing_controller')
        
        # Store vehicle reference (for position/velocity sensing only, NO CONTROL)
        self.vehicle = vehicle
        
        # Declare ROS2 parameters
        self.declare_parameter('max_speed', 8.0)
        self.declare_parameter('min_speed', 2.0)
        self.declare_parameter('wheelbase', 2.7)
        self.declare_parameter('lookahead_distance', lookahead_distance)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('pid_enabled', True)
        self.declare_parameter('show_visualization', True)
        
        # Physics-based constraint parameters (loosened for PID)
        self.declare_parameter('physics_constraints_enabled', False)
        self.declare_parameter('max_steering_at_speed', False)
        self.declare_parameter('coordinated_speed_steering', False)
        
        # Get parameters
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.pid_enabled = self.get_parameter('pid_enabled').value
        self.show_visualization = self.get_parameter('show_visualization').value
        
        # Physics constraint parameters
        self.physics_constraints_enabled = self.get_parameter('physics_constraints_enabled').value
        self.max_steering_at_speed = self.get_parameter('max_steering_at_speed').value
        self.coordinated_speed_steering = self.get_parameter('coordinated_speed_steering').value
        
        # ROS2 Publishers - Commands to Control Module (NO CARLA CONTROL)
        self.target_speed_pub = self.create_publisher(Float64, '/target_speed', 10)
        self.target_steering_pub = self.create_publisher(Float64, '/target_steering', 10)
        
        # Publishers for PID information (one-way communication TO PID)
        self.path_curvature_pub = self.create_publisher(Float64, '/path_curvature', 10)
        self.lookahead_distance_pub = self.create_publisher(Float64, '/current_lookahead', 10)
        self.turn_type_pub = self.create_publisher(Float64, '/turn_type', 10)  # 0=straight, 1=gentle, 2=sharp, 3=u_turn
        
        # ROS2 Subscribers - Get current state for situational awareness only
        self.current_speed_sub = self.create_subscription(
            Float64, '/current_speed', self.current_speed_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/carla/imu_sensor', self.imu_callback, 10)
        
        # Planning state (for situational awareness only, no PID feedback)
        self.current_speed = 0.0
        self.current_imu_data = None
        
        # Physics-based constraints tracking (loosened)
        self.last_physics_warning_time = 0.0
        self.physics_warning_interval = 2.0  # Less frequent warnings
        self.consecutive_sharp_turns = 0
        self.max_consecutive_sharp_turns = 5  # More lenient
        
        # Control parameters - much more restrictive
        self.safety_offset = 0.5  # meters from cones
        self.max_depth = 8.0   # maximum cone detection range - reduced significantly
        self.min_depth = 1.5   # minimum cone detection range - increased
        self.max_lateral_distance = 4.0  # maximum lateral distance from vehicle center
        
        # NEW: Turn radius and path widening parameters
        self.min_turn_radius = 3.5  # Minimum safe turning radius (meters)
        self.path_widening_factor = 1.8  # How much to widen the path in turns
        self.sharp_turn_threshold = 25.0  # Angle threshold for sharp turns (degrees)
        self.u_turn_threshold = 60.0  # Angle threshold for U-turns (degrees)
        self.turn_detection_distance = 6.0  # Distance to look ahead for turn detection
        
        # State tracking
        self.last_steering = 0.0
        self.steering_history = deque(maxlen=5)
        
        # NEW: Turn state tracking
        self.current_turn_type = "straight"  # "straight", "gentle", "sharp", "u_turn"
        self.turn_direction = "none"  # "left", "right", "none"
        self.path_offset = 0.0  # Current path offset for wider turns
        self.gate_sequence = deque(maxlen=5)  # Track recent gates for turn prediction
        
        # Track following parameters
        self.track_width_min = 3.0  # Minimum track width (meters)
        self.track_width_max = 5.0  # Maximum track width (meters) - reduced
        self.max_depth_diff = 1.0   # Maximum depth difference between gate cones - reduced
        self.max_lateral_jump = 1.5 # Maximum lateral movement between consecutive gates - reduced
        self.forward_focus_angle = 30.0  # Only consider cones within this angle (degrees) from vehicle heading
        
        # Backup navigation when no gates found
        self.lost_track_counter = 0
        self.max_lost_track_frames = 20
        
        # Distance tracking for basic stats
        self.distance_traveled = 0.0
        self.last_position = None
        
        # Initialize lap counter
        self.lap_counter = LapCounter()
        
        # Planning state for information sharing (from first node)
        self.planning_state = {
            'target_speed': self.min_speed,
            'target_steering': 0.0,
            'path_curvature': 0.0,
            'current_lookahead': self.lookahead_distance,
            'turn_type_numeric': 0.0,  # 0=straight, 1=gentle, 2=sharp, 3=u_turn
            'cone_visibility_risk': 0.0,  # 0=low, 1=moderate, 2=high
            'track_confidence': 1.0  # 0-1, how confident we are in the current path
        }
        
        # Store current target for visibility calculations
        self.current_target_x = 0.0
        self.current_target_y = 0.0
        
        self.get_logger().info("CARLA Racing Controller Node initialized with ROS2 interface")
        self.get_logger().info("🚫 NO DIRECT CARLA CONTROL - Publishing ROS2 topics only")
        self.get_logger().info(f"PID Information Publishing: {'ENABLED' if self.pid_enabled else 'DISABLED'}")
        self.get_logger().info(f"Physics Constraints: {'ENABLED' if self.physics_constraints_enabled else 'DISABLED'}")
        self.get_logger().info(f"Control frequency: {self.control_frequency} Hz")
    
    def current_speed_callback(self, msg):
        """Receive current speed for situational awareness only"""
        self.current_speed = msg.data
    
    def imu_callback(self, msg):
        """Store IMU data for planning calculations"""
        self.current_imu_data = msg
    
    def apply_physics_constraints(self, raw_speed, raw_steering):
        """Apply loosened physics-based constraints (since PID already has constraints)"""
        if not self.physics_constraints_enabled:
            return raw_speed, raw_steering
        
        try:
            constrained_speed = raw_speed
            constrained_steering = raw_steering
            current_time = time.time()
            
            # CONSTRAINT 1: Speed-dependent steering limits (loosened)
            if self.max_steering_at_speed:
                if self.current_speed > 4.0:  # Higher threshold for high speed
                    max_steering = 0.4  # More lenient at high speed
                elif self.current_speed > 2.0:  # Higher threshold for medium speed  
                    max_steering = 0.7  # More lenient at medium speed
                else:  # Low speed
                    max_steering = 1.0  # Full steering at low speed
                
                if abs(raw_steering) > max_steering:
                    constrained_steering = np.clip(raw_steering, -max_steering, max_steering)
                    
                    # Log warning (but not too frequently)
                    if current_time - self.last_physics_warning_time > self.physics_warning_interval:
                        self.get_logger().warn(f"PHYSICS: Steering limited for speed {self.current_speed:.1f}m/s: {raw_steering:.3f} -> {constrained_steering:.3f}")
                        self.last_physics_warning_time = current_time
            
            # CONSTRAINT 2: Coordinated speed-steering commands (loosened)
            if self.coordinated_speed_steering:
                # If we need to make a sharp turn, reduce speed first
                if abs(raw_steering) > 0.6:  # Higher threshold for speed reduction
                    # Reduce speed for sharp turns
                    max_speed_for_turn = 3.0 - (abs(raw_steering) - 0.6) * 2.5  # More lenient relationship
                    max_speed_for_turn = max(max_speed_for_turn, self.min_speed * 0.8)  # Higher minimum
                    
                    if constrained_speed > max_speed_for_turn:
                        constrained_speed = max_speed_for_turn
                        
                        # Track consecutive sharp turns
                        self.consecutive_sharp_turns += 1
                        
                        if current_time - self.last_physics_warning_time > self.physics_warning_interval:
                            self.get_logger().warn(f"PHYSICS: Speed reduced for sharp turn: {raw_speed:.2f} -> {constrained_speed:.2f} (steering: {raw_steering:.3f})")
                            self.last_physics_warning_time = current_time
                else:
                    # Reset consecutive sharp turn counter
                    self.consecutive_sharp_turns = 0
            
            # CONSTRAINT 3: Prevent excessive consecutive sharp turns (loosened)
            if self.consecutive_sharp_turns > self.max_consecutive_sharp_turns:
                # Force a gentler approach
                constrained_steering = constrained_steering * 0.8  # Less aggressive reduction
                constrained_speed = min(constrained_speed, self.min_speed * 1.5)  # Higher minimum speed
                
                if current_time - self.last_physics_warning_time > self.physics_warning_interval:
                    self.get_logger().warn(f"PHYSICS: Gentler approach forced after {self.consecutive_sharp_turns} consecutive sharp turns")
                    self.last_physics_warning_time = current_time
            
            # CONSTRAINT 4: Rate limiting for steering changes (loosened)
            max_steering_change = 0.3  # Higher maximum change per planning cycle
            if hasattr(self, 'last_constrained_steering'):
                steering_change = constrained_steering - self.last_constrained_steering
                if abs(steering_change) > max_steering_change:
                    if steering_change > 0:
                        constrained_steering = self.last_constrained_steering + max_steering_change
                    else:
                        constrained_steering = self.last_constrained_steering - max_steering_change
                    
                    if current_time - self.last_physics_warning_time > self.physics_warning_interval:
                        self.get_logger().warn(f"PHYSICS: Steering rate limited: change {steering_change:.3f} -> {max_steering_change:.3f}")
                        self.last_physics_warning_time = current_time
            
            # Store for next iteration
            self.last_constrained_steering = constrained_steering
            
            # CONSTRAINT 5: Emergency coordination (loosened)
            speed_for_steering = 2.5 + (1.0 - abs(constrained_steering)) * 5.0  # More speed allowed
            if self.current_speed > speed_for_steering + 1.5:  # Higher tolerance
                # Emergency speed reduction
                constrained_speed = min(constrained_speed, speed_for_steering)
                
                if current_time - self.last_physics_warning_time > self.physics_warning_interval:
                    self.get_logger().warn(f"PHYSICS: Emergency speed coordination: speed {self.current_speed:.1f} too high for steering {constrained_steering:.3f}")
                    self.last_physics_warning_time = current_time
            
            return constrained_speed, constrained_steering
            
        except Exception as e:
            self.get_logger().error(f"Error in physics constraints: {e}")
            return raw_speed, raw_steering
    
    def publish_control_targets(self, target_speed, target_steering):
        """Publish control targets to PID controllers (NO CARLA CONTROL)"""
        speed_msg = Float64()
        speed_msg.data = float(target_speed)
        self.target_speed_pub.publish(speed_msg)
        
        steering_msg = Float64()
        steering_msg.data = float(target_steering)
        self.target_steering_pub.publish(steering_msg)
    
    def publish_planning_state(self):
        """Publish additional planning state for PID information"""
        if not self.pid_enabled:
            return
            
        # Publish path curvature
        curvature_msg = Float64()
        curvature_msg.data = float(self.planning_state['path_curvature'])
        self.path_curvature_pub.publish(curvature_msg)
        
        # Publish current lookahead distance
        lookahead_msg = Float64()
        lookahead_msg.data = float(self.planning_state['current_lookahead'])
        self.lookahead_distance_pub.publish(lookahead_msg)
        
        # Publish turn type
        turn_msg = Float64()
        turn_msg.data = float(self.planning_state['turn_type_numeric'])
        self.turn_type_pub.publish(turn_msg)
    
    def calculate_path_curvature(self, target_x, target_y):
        """Calculate path curvature for PID controller information"""
        try:
            # Calculate curvature based on steering geometry
            lookahead_dist = np.sqrt(target_x**2 + target_y**2)
            if lookahead_dist < 0.1:
                return 0.0
            
            # Simple curvature calculation: curvature = 2 * sin(alpha) / lookahead
            alpha = np.arctan2(target_x, target_y)
            curvature = 2.0 * np.sin(alpha) / lookahead_dist
            
            return float(curvature)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating path curvature: {e}")
            return 0.0
    
    def detect_turn_type(self, current_gate, blue_cones, yellow_cones):
        """Detect the type of turn and calculate appropriate path widening"""
        if not current_gate:
            return "straight", "none", 0.0
        
        try:
            # Add current gate to sequence
            self.gate_sequence.append({
                'midpoint_x': current_gate['midpoint_x'],
                'midpoint_y': current_gate['midpoint_y'],
                'width': current_gate['width']
            })
            
            if len(self.gate_sequence) < 2:
                return "straight", "none", 0.0
            
            # Calculate turn angle based on gate sequence
            current_pos = self.gate_sequence[-1]
            previous_pos = self.gate_sequence[-2]
            
            # Calculate direction change
            angle_change = np.degrees(np.arctan2(current_pos['midpoint_x'], current_pos['midpoint_y']) - 
                                    np.arctan2(previous_pos['midpoint_x'], previous_pos['midpoint_y']))
            
            # Normalize angle to [-180, 180]
            while angle_change > 180:
                angle_change -= 360
            while angle_change < -180:
                angle_change += 360
            
            abs_angle = abs(angle_change)
            
            # Look ahead for upcoming turns by analyzing cone distribution
            upcoming_turn_severity = self.analyze_upcoming_turn(blue_cones, yellow_cones)
            
            # Determine turn type and direction with updated planning state
            if abs_angle < 10 and upcoming_turn_severity < 20:
                turn_type = "straight"
                direction = "none"
                path_offset = 0.0
                self.planning_state['turn_type_numeric'] = 0.0
            elif abs_angle < self.sharp_turn_threshold and upcoming_turn_severity < 40:
                turn_type = "gentle"
                direction = "left" if angle_change > 0 else "right"
                path_offset = 0.3 * self.path_widening_factor
                self.planning_state['turn_type_numeric'] = 1.0
            elif abs_angle < self.u_turn_threshold and upcoming_turn_severity < 70:
                turn_type = "sharp"
                direction = "left" if angle_change > 0 else "right"
                path_offset = 0.6 * self.path_widening_factor
                self.planning_state['turn_type_numeric'] = 2.0
            else:
                turn_type = "u_turn"
                direction = "left" if angle_change > 0 else "right"
                path_offset = 1.0 * self.path_widening_factor
                self.planning_state['turn_type_numeric'] = 3.0
            
            print(f"DEBUG: Turn analysis - Type: {turn_type}, Direction: {direction}, Angle change: {angle_change:.1f}°, Upcoming severity: {upcoming_turn_severity:.1f}°")
            
            return turn_type, direction, path_offset
            
        except Exception as e:
            print(f"ERROR in turn detection: {e}")
            return "straight", "none", 0.0
    
    def analyze_upcoming_turn(self, blue_cones, yellow_cones):
        """Analyze cone distribution to predict upcoming turn severity"""
        try:
            if not blue_cones or not yellow_cones:
                return 0.0
            
            # Look at cones within turn detection distance
            nearby_blue = [c for c in blue_cones if c['depth'] <= self.turn_detection_distance]
            nearby_yellow = [c for c in yellow_cones if c['depth'] <= self.turn_detection_distance]
            
            if len(nearby_blue) < 2 or len(nearby_yellow) < 2:
                return 0.0
            
            # Calculate the curvature based on lateral position changes
            blue_lateral_change = 0.0
            yellow_lateral_change = 0.0
            
            for i in range(1, min(len(nearby_blue), 3)):
                blue_lateral_change += abs(nearby_blue[i]['x'] - nearby_blue[i-1]['x'])
            
            for i in range(1, min(len(nearby_yellow), 3)):
                yellow_lateral_change += abs(nearby_yellow[i]['x'] - nearby_yellow[i-1]['x'])
            
            # Convert lateral changes to approximate turn angle
            max_lateral_change = max(blue_lateral_change, yellow_lateral_change)
            turn_severity = np.degrees(np.arctan2(max_lateral_change, self.turn_detection_distance))
            
            return min(turn_severity, 90.0)  # Cap at 90 degrees
            
        except Exception as e:
            print(f"ERROR in upcoming turn analysis: {e}")
            return 0.0
    
    def calculate_turn_radius(self, steering_angle):
        """Calculate the turning radius based on steering angle and wheelbase"""
        if abs(steering_angle) < 0.01:
            return float('inf')  # Straight line
        
        # Bicycle model turning radius
        turn_radius = self.wheelbase / np.tan(abs(steering_angle))
        return max(turn_radius, self.min_turn_radius)
    
    def adjust_target_for_turn(self, target_x, target_y, turn_type, turn_direction, path_offset):
        """Adjust the target point to create a wider path around turns"""
        try:
            if turn_type == "straight":
                return target_x, target_y
            
            # Calculate the vector from vehicle to target
            target_distance = np.sqrt(target_x**2 + target_y**2)
            
            if target_distance < 0.1:
                return target_x, target_y
            
            # Normalize the target vector
            target_unit_x = target_x / target_distance
            target_unit_y = target_y / target_distance
            
            # Create a perpendicular vector for path offset
            perp_x = -target_unit_y  # Perpendicular vector
            perp_y = target_unit_x
            
            # Determine offset direction based on turn
            if turn_direction == "left":
                # For left turns, offset to the right (away from inner cones)
                offset_multiplier = -path_offset
            elif turn_direction == "right":
                # For right turns, offset to the left (away from inner cones)
                offset_multiplier = path_offset
            else:
                offset_multiplier = 0.0
            
            # Apply the offset
            adjusted_x = target_x + perp_x * offset_multiplier
            adjusted_y = target_y + perp_y * offset_multiplier
            
            # Additional forward offset for sharp turns and U-turns
            if turn_type in ["sharp", "u_turn"]:
                forward_offset = 0.5 if turn_type == "sharp" else 1.0
                adjusted_y += forward_offset
            
            print(f"DEBUG: Path adjustment - Original: ({target_x:.2f}, {target_y:.2f}), "
                  f"Adjusted: ({adjusted_x:.2f}, {adjusted_y:.2f}), "
                  f"Offset: {offset_multiplier:.2f}, Turn: {turn_type}-{turn_direction}")
            
            return adjusted_x, adjusted_y
            
        except Exception as e:
            print(f"ERROR in target adjustment: {e}")
            return target_x, target_y
    
    def calculate_adaptive_speed(self, turn_type, steering_angle, current_depth):
        """Calculate speed based on turn type and conditions"""
        base_speed = self.min_speed + (self.max_speed - self.min_speed) * 0.7
        
        # Speed reduction based on turn type
        if turn_type == "gentle":
            speed_factor = 0.8
        elif turn_type == "sharp":
            speed_factor = 0.6
        elif turn_type == "u_turn":
            speed_factor = 0.4
        else:
            speed_factor = 1.0
        
        # Additional speed reduction based on steering angle
        steering_factor = 1.0 - 0.7 * abs(steering_angle)
        
        # Speed reduction when approaching targets
        if current_depth < 6.0:
            distance_factor = 0.7
        elif current_depth < 3.0:
            distance_factor = 0.5
        else:
            distance_factor = 1.0
        
        # Apply tracking confidence factor (internal planning assessment)
        confidence_factor = self.planning_state.get('track_confidence', 1.0)
        if confidence_factor < 0.7:
            confidence_speed_factor = 0.8  # Reduce speed when confidence is low
        else:
            confidence_speed_factor = 1.0
        
        # Combine all factors
        final_speed = base_speed * speed_factor * steering_factor * distance_factor * confidence_speed_factor
        
        return max(final_speed, self.min_speed * 0.7)  # Minimum speed limit
    
    def image_to_world_coords(self, center_x, center_y, depth):
        """Convert image coordinates to world coordinates (vehicle reference frame)"""
        # Camera parameters for ZED 2i simulation
        image_width = 1280
        fov_horizontal = 90.0  # degrees
        
        # Calculate angle from image center
        angle = ((center_x - image_width / 2) / (image_width / 2)) * (fov_horizontal / 2)
        
        # Convert to world coordinates relative to vehicle
        world_x = depth * np.tan(np.radians(angle))
        world_y = depth
        
        return world_x, world_y
    
    def process_cone_detections(self, cone_detections):
        """Process cone detections with strict spatial filtering to focus on immediate track"""
        if not cone_detections:
            return [], [], []
            
        blue_cones = []    # Class 1 - LEFT side
        yellow_cones = []  # Class 0 - RIGHT side
        orange_cones = []  # Class 2 - ORANGE (lap markers)
        
        try:
            for detection in cone_detections:
                if not isinstance(detection, dict):
                    continue
                    
                if 'box' not in detection or 'cls' not in detection or 'depth' not in detection:
                    continue
                    
                # Handle different box formats
                box = detection['box']
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                else:
                    continue
                    
                cls = detection['cls']
                depth = detection['depth']
                
                # Ensure numeric values
                try:
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    cls = int(cls)
                    depth = float(depth)
                except (ValueError, TypeError):
                    continue
                
                # Filter by depth range - more lenient for orange cones
                if cls == 2:  # Orange cone - allow farther detection
                    if depth < 1.0 or depth > 15.0:
                        continue
                else:  # Blue/Yellow cones - strict filtering
                    if depth < self.min_depth or depth > self.max_depth:
                        continue
                    
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to world coordinates
                world_x, world_y = self.image_to_world_coords(center_x, center_y, depth)
                
                # CRITICAL: Filter by lateral distance - more lenient for orange cones
                if cls == 2:  # Orange cone - allow wider lateral range
                    if abs(world_x) > 8.0:  # Wider range for orange cones
                        continue
                else:  # Blue/Yellow cones - strict filtering
                    if abs(world_x) > self.max_lateral_distance:
                        continue
                
                # CRITICAL: Filter by forward focus angle - more lenient for orange cones
                angle_to_cone = np.degrees(abs(np.arctan2(world_x, world_y)))
                if cls == 2:  # Orange cone - allow wider angle
                    if angle_to_cone > 60.0:  # Wider angle for orange cones
                        continue
                else:  # Blue/Yellow cones - strict filtering
                    if angle_to_cone > self.forward_focus_angle:
                        continue
                
                # Only consider cones that are reasonably positioned for track boundaries
                # Blue cones should be on the left (negative x), yellow on the right (positive x)
                # Orange cones can be anywhere (lap markers)
                if cls == 1 and world_x > 1.0:  # Blue cone too far right
                    continue
                if cls == 0 and world_x < -1.0:  # Yellow cone too far left
                    continue
                
                cone_data = {
                    'x': world_x,
                    'y': world_y,
                    'depth': depth,
                    'center_x': center_x,
                    'center_y': center_y,
                    'original_box': (x1, y1, x2, y2),
                    'confidence': detection.get('conf', 1.0),
                    'angle_from_center': angle_to_cone
                }
                
                if cls == 1:  # Blue cone - LEFT side
                    blue_cones.append(cone_data)
                elif cls == 0:  # Yellow cone - RIGHT side  
                    yellow_cones.append(cone_data)
                elif cls == 2:  # Orange cone - LAP MARKER
                    orange_cones.append(cone_data)
                    
        except Exception as e:
            print(f"ERROR processing cone detections: {e}")
            return [], [], []
        
        # Sort by depth (closest first), then by angle (most centered first)
        blue_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        yellow_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        orange_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        
        # Debug filtered cones
        print(f"DEBUG: After spatial filtering - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
        if blue_cones:
            closest_blue = blue_cones[0]
            print(f"  Closest blue: x={closest_blue['x']:.2f}, y={closest_blue['y']:.2f}, angle={closest_blue['angle_from_center']:.1f}°")
        if yellow_cones:
            closest_yellow = yellow_cones[0]
            print(f"  Closest yellow: x={closest_yellow['x']:.2f}, y={closest_yellow['y']:.2f}, angle={closest_yellow['angle_from_center']:.1f}°")
        if orange_cones:
            closest_orange = orange_cones[0]
            print(f"  Closest orange: x={closest_orange['x']:.2f}, y={closest_orange['y']:.2f}, angle={closest_orange['angle_from_center']:.1f}°")
        
        return blue_cones, yellow_cones, orange_cones
    
    def is_valid_track_segment(self, blue, yellow):
        """Strict validation for immediate track segments"""
        try:
            # Ensure blue is on left, yellow on right
            if blue['x'] >= yellow['x']:
                return False
            
            # Depth similarity - much stricter
            depth_diff = abs(blue['depth'] - yellow['depth'])
            if depth_diff > self.max_depth_diff:
                return False
            
            # Track width validation - strict Formula Student standards
            width = abs(blue['x'] - yellow['x'])
            if width < self.track_width_min or width > self.track_width_max:
                return False
            
            # Both cones must be reasonably close (immediate track section)
            avg_depth = (blue['depth'] + yellow['depth']) / 2
            if avg_depth > 6.0:  # Very close focus
                return False
            
            # Both cones should be roughly aligned laterally (not offset significantly)
            lateral_center = (blue['x'] + yellow['x']) / 2
            if abs(lateral_center) > 1.0:  # Gate center should be roughly in front of vehicle
                return False
                
            return True
            
        except Exception as e:
            print(f"ERROR in track segment validation: {e}")
            return False
    
    def find_best_track_segment(self, blue_cones, yellow_cones):
        """Find the best immediate track segment with strict proximity focus"""
        if not blue_cones or not yellow_cones:
            print(f"DEBUG: Cannot form track segment - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}")
            return None
            
        print(f"DEBUG: Finding immediate track segment from {len(blue_cones)} blue and {len(yellow_cones)} yellow cones")
        
        # Only consider the closest 3 cones of each color to focus on immediate track
        blue_candidates = blue_cones[:3]
        yellow_candidates = yellow_cones[:3]
        
        valid_segments = []
        
        try:
            for blue in blue_candidates:
                for yellow in yellow_candidates:
                    
                    if not self.is_valid_track_segment(blue, yellow):
                        continue
                    
                    # Create track segment
                    segment = {
                        'blue': blue,
                        'yellow': yellow,
                        'midpoint_x': (blue['x'] + yellow['x']) / 2,
                        'midpoint_y': (blue['y'] + yellow['y']) / 2,
                        'width': abs(blue['x'] - yellow['x']),
                        'avg_depth': (blue['depth'] + yellow['depth']) / 2,
                        'confidence': (blue.get('confidence', 1.0) + yellow.get('confidence', 1.0)) / 2
                    }
                    
                    # Additional validation: segment should be roughly centered in front of vehicle
                    if abs(segment['midpoint_x']) < 2.0:  # Segment center within 2m of vehicle centerline
                        valid_segments.append(segment)
            
            if not valid_segments:
                print("DEBUG: No valid immediate track segments found")
                return None
            
            # Sort by distance (closest first), heavily prioritize centerline alignment
            def segment_score(s):
                distance_score = s['avg_depth']
                centerline_score = abs(s['midpoint_x']) * 3.0  # Heavy penalty for off-center segments
                return distance_score + centerline_score
            
            valid_segments.sort(key=segment_score)
            best_segment = valid_segments[0]
            
            print(f"DEBUG: Found immediate track segment:")
            print(f"  Blue cone (LEFT):  x={best_segment['blue']['x']:6.2f}, y={best_segment['blue']['y']:6.2f}")
            print(f"  Yellow cone (RIGHT): x={best_segment['yellow']['x']:6.2f}, y={best_segment['yellow']['y']:6.2f}")
            print(f"  Segment midpoint: x={best_segment['midpoint_x']:6.2f}, y={best_segment['midpoint_y']:6.2f}")
            print(f"  Segment width: {best_segment['width']:.2f}m, Average depth: {best_segment['avg_depth']:.2f}m")
            print(f"  Centerline offset: {abs(best_segment['midpoint_x']):.2f}m")
            
            return best_segment
            
        except Exception as e:
            print(f"ERROR in track segment finding: {e}")
            return None
    
    def follow_cone_line(self, blue_cones, yellow_cones):
        """Fallback: follow immediate cones when no track segments can be formed"""
        # Only consider the closest cones that are directly ahead
        immediate_cones = []
        
        for cone in blue_cones[:2] + yellow_cones[:2]:
            if cone['depth'] < 5.0 and abs(cone['x']) < 3.0:  # Very close and centered
                immediate_cones.append(cone)
        
        if not immediate_cones:
            return None
            
        # Sort by distance
        immediate_cones.sort(key=lambda c: c['depth'])
        closest_cone = immediate_cones[0]
        
        # Create a very conservative target point
        if closest_cone in blue_cones:
            # Blue cone on left, aim slightly right but stay close
            target_x = closest_cone['x'] + 1.5
        else:
            # Yellow cone on right, aim slightly left but stay close
            target_x = closest_cone['x'] - 1.5
            
        target_y = closest_cone['y']
        
        # Don't follow cone line if it's too far from center
        if abs(target_x) > 3.0:
            return None
        
        print(f"DEBUG: Following immediate cone line - target ({target_x:.2f}, {target_y:.2f})")
        
        return {
            'midpoint_x': target_x,
            'midpoint_y': target_y,
            'avg_depth': target_y,
            'width': 3.0,  # Conservative assumed width
            'type': 'cone_line'
        }
    
    def calculate_pure_pursuit_steering(self, target_x, target_y):
        """Calculate steering angle using pure pursuit algorithm with enhanced cone visibility preservation"""
        try:
            print(f"DEBUG: Pure pursuit calculation for target ({target_x:.2f}, {target_y:.2f})")
            
            # Calculate angle to target
            alpha = np.arctan2(target_x, target_y)
            print(f"DEBUG: Alpha (angle to target): {np.degrees(alpha):.1f}°")
            
            # Calculate lookahead distance
            lookahead_dist = np.sqrt(target_x**2 + target_y**2)
            print(f"DEBUG: Lookahead distance: {lookahead_dist:.2f}m")
            
            # NEW: Calculate steering aggressiveness based on cone visibility risk
            lateral_offset = abs(target_x)
            visibility_risk_factor = 1.0
            
            # If target is getting close to our field of view limits, increase steering aggressiveness
            if lateral_offset > 2.5:  # Getting close to losing sight
                visibility_risk_factor = 1.4
                self.planning_state['cone_visibility_risk'] = 2.0  # High risk
                print(f"DEBUG: High visibility risk - increasing steering aggressiveness by 40%")
            elif lateral_offset > 1.8:  # Moderate risk
                visibility_risk_factor = 1.2
                self.planning_state['cone_visibility_risk'] = 1.0  # Moderate risk
                print(f"DEBUG: Moderate visibility risk - increasing steering aggressiveness by 20%")
            else:
                self.planning_state['cone_visibility_risk'] = 0.0  # Low risk
            
            # NEW: Adaptive lookahead with steering aggressiveness consideration
            # For sharp turns, reduce lookahead to make steering more responsive
            base_adaptive_lookahead = max(self.lookahead_distance, min(lookahead_dist, 6.0))
            
            # If we detect we're in a turn or approaching one, reduce lookahead for quicker response
            if self.current_turn_type in ["sharp", "u_turn"]:
                adaptive_lookahead = base_adaptive_lookahead * 0.7  # 30% reduction for sharp turns
                print(f"DEBUG: Sharp/U-turn detected - reducing lookahead for quicker response")
            elif self.current_turn_type == "gentle":
                adaptive_lookahead = base_adaptive_lookahead * 0.85  # 15% reduction for gentle turns
                print(f"DEBUG: Gentle turn detected - slightly reducing lookahead")
            else:
                adaptive_lookahead = base_adaptive_lookahead
            
            # Apply visibility risk factor to lookahead (shorter lookahead = more aggressive steering)
            adaptive_lookahead = adaptive_lookahead / visibility_risk_factor
            
            # Store current lookahead in planning state
            self.planning_state['current_lookahead'] = adaptive_lookahead
            
            print(f"DEBUG: Adjusted lookahead: {adaptive_lookahead:.2f}m (visibility factor: {visibility_risk_factor:.2f})")
            
            # Pure pursuit steering calculation with enhanced responsiveness
            base_steering_angle = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), adaptive_lookahead)
            
            # NEW: Apply additional steering enhancement for cone visibility preservation
            # If the lateral offset is significant, add extra steering bias
            if lateral_offset > 1.5:
                # Calculate additional steering needed to keep cones in sight
                visibility_steering_boost = np.arctan2(lateral_offset - 1.5, lookahead_dist) * 0.6
                if target_x > 0:  # Target to the right
                    base_steering_angle += visibility_steering_boost
                else:  # Target to the left
                    base_steering_angle -= visibility_steering_boost
                
                print(f"DEBUG: Applied visibility steering boost: {np.degrees(visibility_steering_boost):.1f}°")
            
            steering_angle = base_steering_angle
            
            # Calculate the required turn radius and check if it's feasible
            required_turn_radius = self.calculate_turn_radius(steering_angle)
            
            # If the required turn radius is too small, adjust the steering
            if required_turn_radius < self.min_turn_radius:
                # Recalculate steering for minimum safe turn radius
                max_safe_steering = np.arctan(self.wheelbase / self.min_turn_radius)
                if steering_angle > 0:
                    steering_angle = min(steering_angle, max_safe_steering)
                else:
                    steering_angle = max(steering_angle, -max_safe_steering)
                
                print(f"DEBUG: Adjusted steering for minimum turn radius: {np.degrees(steering_angle):.1f}°")
            
            print(f"DEBUG: Final steering angle: {np.degrees(steering_angle):.1f}°")
            print(f"DEBUG: Turn radius: {required_turn_radius:.2f}m")
            
            # Convert to normalized steering [-1, 1] with enhanced sensitivity
            max_steering_rad = np.radians(30.0)  # Max 30 degrees
            normalized_steering = np.clip(steering_angle / max_steering_rad, -1.0, 1.0)
            
            # NEW: Apply final visibility preservation enhancement
            # If we're at risk of losing cones and steering is not aggressive enough, boost it
            if lateral_offset > 2.0 and abs(normalized_steering) < 0.4:
                steering_boost = min(0.2, (lateral_offset - 2.0) * 0.3)
                if normalized_steering > 0:
                    normalized_steering = min(1.0, normalized_steering + steering_boost)
                else:
                    normalized_steering = max(-1.0, normalized_steering - steering_boost)
                
                print(f"DEBUG: Applied final steering boost for cone visibility: {steering_boost:.3f}")
            
            print(f"DEBUG: Normalized steering: {normalized_steering:.3f}")
            direction = 'LEFT' if normalized_steering > 0 else 'RIGHT' if normalized_steering < 0 else 'STRAIGHT'
            print(f"DEBUG: Steering direction: {direction}")
            
            return normalized_steering
            
        except Exception as e:
            print(f"ERROR in pure pursuit calculation: {e}")
            return 0.0
    
    def smooth_steering(self, raw_steering):
        """Apply steering smoothing with enhanced responsiveness for cone visibility"""
        try:
            self.steering_history.append(raw_steering)
            
            # NEW: Adaptive smoothing based on visibility risk
            lateral_offset = abs(getattr(self, 'current_target_x', 0.0))
            tracking_confidence = self.planning_state.get('track_confidence', 1.0)
            
            if len(self.steering_history) >= 3:
                # If we're at high risk of losing cones, use less smoothing (more responsive)
                if lateral_offset > 2.5 or self.current_turn_type in ["sharp", "u_turn"] or tracking_confidence < 0.5:
                    # More aggressive weighting for recent steering inputs
                    weights = np.array([0.7, 0.2, 0.1])  # Heavy emphasis on current steering
                    print(f"DEBUG: High visibility risk - using aggressive steering smoothing")
                elif lateral_offset > 1.8 or self.current_turn_type == "gentle" or tracking_confidence < 0.8:
                    # Moderate smoothing
                    weights = np.array([0.6, 0.25, 0.15])
                    print(f"DEBUG: Moderate visibility risk - using moderate steering smoothing")
                else:
                    # Normal smoothing for straight sections
                    weights = np.array([0.5, 0.3, 0.2])
                
                recent_steering = np.array(list(self.steering_history)[-3:])
                smoothed = np.average(recent_steering, weights=weights)
            else:
                smoothed = raw_steering
            
            # NEW: Adaptive rate limiting based on cone visibility risk
            if lateral_offset > 2.5 or tracking_confidence < 0.5:
                # High risk - allow more aggressive steering changes
                max_change = 0.25
                print(f"DEBUG: High visibility risk - allowing max steering change: {max_change}")
            elif lateral_offset > 1.8 or tracking_confidence < 0.8:
                # Moderate risk - slightly more aggressive
                max_change = 0.2
            elif self.current_turn_type in ["sharp", "u_turn"]:
                # Sharp turns - more responsive
                max_change = 0.18
            else:
                # Normal rate limiting
                max_change = 0.15
            
            # Apply rate limiting
            if abs(smoothed - self.last_steering) > max_change:
                if smoothed > self.last_steering:
                    smoothed = self.last_steering + max_change
                else:
                    smoothed = self.last_steering - max_change
                print(f"DEBUG: Applied steering rate limiting: {max_change}")
            
            self.last_steering = smoothed
            return smoothed
            
        except Exception as e:
            print(f"ERROR in steering smoothing: {e}")
            return self.last_steering
    
    def update_distance_traveled(self):
        """Update distance traveled for basic tracking"""
        try:
            transform = self.vehicle.get_transform()
            location = transform.location
            
            if self.last_position is not None:
                distance_delta = np.sqrt(
                    (location.x - self.last_position.x)**2 + 
                    (location.y - self.last_position.y)**2 + 
                    (location.z - self.last_position.z)**2
                )
                self.distance_traveled += distance_delta
            
            self.last_position = location
            
        except Exception as e:
            print(f"Error updating distance: {e}")
    
    def plan_path(self, cone_detections):
        """Main path planning function - ONLY PUBLISHES ROS2 TOPICS (NO CARLA CONTROL)"""
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG: PLANNING CYCLE - {len(cone_detections) if cone_detections else 0} detections")
            print(f"Laps completed: {self.lap_counter.laps_completed}")
            print(f"Current turn type: {self.current_turn_type}, Direction: {self.turn_direction}")
            print(f"Lost track counter: {self.lost_track_counter}")
            print(f"🚫 ROS2 TOPICS ONLY - NO CARLA CONTROL")
            print(f"{'='*60}")
            
            # Update distance traveled
            self.update_distance_traveled()
            
            # Process cone detections (includes orange cones)
            blue_cones, yellow_cones, orange_cones = self.process_cone_detections(cone_detections)
            print(f"DEBUG: Processed cones - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
            
            # Check for lap completion through orange gate
            if orange_cones:
                transform = self.vehicle.get_transform()
                vehicle_position = (transform.location.x, transform.location.y, transform.location.z)
                self.lap_counter.check_orange_gate_passage(orange_cones, vehicle_position)
            
            # NEW: Enhanced lost track detection with recovery planning
            if len(blue_cones) == 0 and len(yellow_cones) == 0:
                self.lost_track_counter += 1
                self.planning_state['track_confidence'] = 0.1
                print(f"DEBUG: NO CONES DETECTED - lost track for {self.lost_track_counter} frames")
                
                # Plan recovery maneuvers (publish to ROS2, no direct control)
                if self.lost_track_counter <= 10:
                    # Try to steer in the direction we were last going
                    recovery_steering = self.last_steering * 1.5  # Amplify last steering
                    recovery_steering = np.clip(recovery_steering, -0.8, 0.8)
                    recovery_speed = 0.2
                    print(f"DEBUG: Planning recovery steering: {recovery_steering:.3f}")
                    
                    # Apply physics constraints to recovery commands
                    if self.physics_constraints_enabled:
                        recovery_speed, recovery_steering = self.apply_physics_constraints(recovery_speed, recovery_steering)
                    
                    # Publish ROS2 commands (NO CARLA CONTROL)
                    self.planning_state['target_speed'] = recovery_speed
                    self.planning_state['target_steering'] = recovery_steering
                    self.publish_control_targets(recovery_speed, recovery_steering)
                    self.publish_planning_state()
                    
                    return recovery_steering, recovery_speed
                elif self.lost_track_counter <= 20:
                    # More aggressive search pattern
                    search_steering = 0.6 * np.sin(self.lost_track_counter * 0.3)
                    search_speed = 0.15
                    print(f"DEBUG: Planning aggressive search pattern: {search_steering:.3f}")
                    
                    # Apply physics constraints
                    if self.physics_constraints_enabled:
                        search_speed, search_steering = self.apply_physics_constraints(search_speed, search_steering)
                    
                    # Publish ROS2 commands (NO CARLA CONTROL)
                    self.planning_state['target_speed'] = search_speed
                    self.planning_state['target_steering'] = search_steering
                    self.publish_control_targets(search_speed, search_steering)
                    self.publish_planning_state()
                    
                    return search_steering, search_speed
                else:
                    # Last resort - wide search
                    search_steering = 0.8 * np.sin(self.lost_track_counter * 0.2)
                    search_speed = 0.1
                    
                    # Apply physics constraints
                    if self.physics_constraints_enabled:
                        search_speed, search_steering = self.apply_physics_constraints(search_speed, search_steering)
                    
                    # Publish ROS2 commands (NO CARLA CONTROL)
                    self.planning_state['target_speed'] = search_speed
                    self.planning_state['target_steering'] = search_steering
                    self.publish_control_targets(search_speed, search_steering)
                    self.publish_planning_state()
                    
                    return search_steering, search_speed
            
            # Try to find a track segment
            track_segment = self.find_best_track_segment(blue_cones, yellow_cones)
            
            # If no track segment found, try cone line following
            if not track_segment and (blue_cones or yellow_cones):
                track_segment = self.follow_cone_line(blue_cones, yellow_cones)
                print("DEBUG: Using cone line following")
            
            if not track_segment:
                self.lost_track_counter += 1
                self.planning_state['track_confidence'] = 0.3
                print(f"DEBUG: No navigation target found - lost track for {self.lost_track_counter} frames")
                
                # If lost for too long, implement search pattern
                if self.lost_track_counter > self.max_lost_track_frames:
                    print("DEBUG: Lost track for too long - planning search pattern")
                    search_steering = 0.3 * np.sin(self.lost_track_counter * 0.1)  # Gentle search pattern
                    search_speed = 0.15
                    
                    # Apply physics constraints
                    if self.physics_constraints_enabled:
                        search_speed, search_steering = self.apply_physics_constraints(search_speed, search_steering)
                    
                    # Publish ROS2 commands (NO CARLA CONTROL)
                    self.planning_state['target_speed'] = search_speed
                    self.planning_state['target_steering'] = search_steering
                    self.publish_control_targets(search_speed, search_steering)
                    self.publish_planning_state()
                    
                    return search_steering, search_speed
                else:
                    # Move forward slowly while searching
                    recovery_steering = self.last_steering * 0.5
                    recovery_speed = 0.15
                    
                    # Apply physics constraints
                    if self.physics_constraints_enabled:
                        recovery_speed, recovery_steering = self.apply_physics_constraints(recovery_speed, recovery_steering)
                    
                    # Publish ROS2 commands (NO CARLA CONTROL)
                    self.planning_state['target_speed'] = recovery_speed
                    self.planning_state['target_steering'] = recovery_steering
                    self.publish_control_targets(recovery_speed, recovery_steering)
                    self.publish_planning_state()
                    
                    return recovery_steering, recovery_speed
            
            # Reset lost track counter if we found something
            self.lost_track_counter = 0
            # Restore confidence gradually
            self.planning_state['track_confidence'] = min(
                self.planning_state['track_confidence'] + 0.1, 1.0
            )
            
            # Detect turn type and calculate path widening
            turn_type, turn_direction, path_offset = self.detect_turn_type(track_segment, blue_cones, yellow_cones)
            self.current_turn_type = turn_type
            self.turn_direction = turn_direction
            self.path_offset = path_offset
            
            # Adjust target point for wider turns
            original_target_x = track_segment['midpoint_x']
            original_target_y = track_segment['midpoint_y']
            
            # NEW: Store current target for visibility calculations
            self.current_target_x = original_target_x
            self.current_target_y = original_target_y
            
            adjusted_target_x, adjusted_target_y = self.adjust_target_for_turn(
                original_target_x, original_target_y, turn_type, turn_direction, path_offset
            )
            
            # Navigate towards the adjusted target
            raw_steering = self.calculate_pure_pursuit_steering(adjusted_target_x, adjusted_target_y)
            smooth_steering = self.smooth_steering(raw_steering)
            
            # Calculate adaptive speed based on turn type
            current_depth = track_segment['avg_depth']
            raw_speed = self.calculate_adaptive_speed(turn_type, smooth_steering, current_depth)
            
            # Calculate and store path curvature for PID information
            self.planning_state['path_curvature'] = self.calculate_path_curvature(adjusted_target_x, adjusted_target_y)
            
            # Apply physics constraints (loosened for PID integration)
            if self.physics_constraints_enabled:
                target_speed, target_steering = self.apply_physics_constraints(raw_speed, smooth_steering)
            else:
                target_speed, target_steering = raw_speed, smooth_steering
            
            # Update planning state
            self.planning_state['target_speed'] = target_speed
            self.planning_state['target_steering'] = target_steering
            
            # Publish ROS2 commands and planning state (NO CARLA CONTROL)
            self.publish_control_targets(target_speed, target_steering)
            self.publish_planning_state()
            
            # Enhanced debug output
            direction = 'LEFT' if target_steering > 0 else 'RIGHT' if target_steering < 0 else 'STRAIGHT'
            
            print(f"DEBUG: PLANNED CONTROL (ROS2 ONLY):")
            print(f"  Navigation: {track_segment.get('type', 'track_segment')}_{turn_type}")
            print(f"  Turn Analysis: {turn_type}-{turn_direction} (offset: {path_offset:.2f}m)")
            print(f"  Original target: ({original_target_x:.2f}, {original_target_y:.2f})")
            print(f"  Adjusted target: ({adjusted_target_x:.2f}, {adjusted_target_y:.2f})")
            print(f"  Target distance: {current_depth:.2f}m")
            print(f"  Turn radius: {self.calculate_turn_radius(np.radians(target_steering * 30)):.2f}m")
            print(f"  Raw->Final: Speed {raw_speed:.2f}->{target_speed:.2f}, Steering {smooth_steering:.3f}->{target_steering:.3f}")
            print(f"  Steering: {target_steering:.3f} ({direction})")
            print(f"  Cone visibility risk: {'HIGH' if abs(original_target_x) > 2.5 else 'MODERATE' if abs(original_target_x) > 1.8 else 'LOW'}")
            print(f"  Published Speed: {target_speed:.1f} m/s ({target_speed*3.6:.1f} km/h)")
            print(f"  Published Steering: {target_steering:.3f}")
            print(f"  Distance: {self.distance_traveled:.1f}m")
            print(f"  Laps: {self.lap_counter.laps_completed}")
            print(f"  Track Confidence: {self.planning_state['track_confidence']:.2f}")
            print(f"  Physics Constraints: {'APPLIED' if self.physics_constraints_enabled else 'OFF'}")
            print(f"  🚫 NO CARLA CONTROL - ROS2 TOPICS ONLY")
            print(f"{'='*60}\n")
            
            return target_steering, target_speed
            
        except Exception as e:
            print(f"ERROR in path planning: {e}")
            import traceback
            traceback.print_exc()
            # Safe fallback - publish safe commands to ROS2 (NO CARLA CONTROL)
            self.publish_control_targets(0.0, 0.0)
            
            return 0.0, 0.0

class CarlaRacingSystem(Node):
    def __init__(self):
        super().__init__('carla_racing_system')
        
        # Declare ROS2 parameters
        self.declare_parameter('model_path', '/home/legion5/hydrakon_ws/src/planning_module/planning_module/best.pt')
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('control_frequency', 10.0)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.show_visualization = self.get_parameter('show_visualization').value
        self.control_frequency = self.get_parameter('control_frequency').value
        
        # CARLA components
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.controller = None
        self.running = True
        
        # Threading
        self.display_thread = None
        
        # ROS2 timer for planning loop (NO CARLA CONTROL)
        timer_period = 1.0 / self.control_frequency
        self.planning_timer = self.create_timer(timer_period, self.planning_loop_ros)
        
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.get_logger().info("CARLA Racing System Node initialized with ROS2 interface")
        self.get_logger().info("🚫 NO CARLA CONTROL - Planning and publishing ROS2 topics only")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Control frequency: {self.control_frequency} Hz")
        self.get_logger().info(f"Visualization: {'ENABLED' if self.show_visualization else 'DISABLED'}")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C for clean shutdown"""
        self.get_logger().info("Received shutdown signal")
        self.running = False
    
    def setup_carla(self):
        """Initialize CARLA connection and find existing vehicle"""
        try:
            # Connect to CARLA
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.get_logger().info("Connected to CARLA server")
            
            # Get world
            self.world = self.client.get_world()
            
            # Find existing vehicle instead of spawning new one
            self.vehicle = self.find_existing_vehicle()
            if not self.vehicle:
                raise RuntimeError("No existing vehicle found in the world")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error setting up CARLA: {e}")
            return False
    
    def find_existing_vehicle(self):
        """Find an existing vehicle in the CARLA world"""
        try:
            # Get all actors in the world
            all_actors = self.world.get_actors()
            
            # Filter for vehicles
            vehicles = all_actors.filter('vehicle.*')
            
            if not vehicles:
                self.get_logger().error("No vehicles found in the world")
                self.get_logger().info("Available actors:")
                for actor in all_actors:
                    self.get_logger().info(f"  - {actor.type_id} (ID: {actor.id})")
                return None
            
            # Use the first vehicle found
            vehicle = vehicles[0]
            transform = vehicle.get_transform()
            location = transform.location
            
            self.get_logger().info(f"Found existing vehicle: {vehicle.type_id} (ID: {vehicle.id})")
            self.get_logger().info(f"Vehicle location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
            self.get_logger().info("🚫 Vehicle will NOT be controlled by this node - ROS2 topics only")
            
            # Check if vehicle is alive
            if not vehicle.is_alive:
                self.get_logger().error("Found vehicle is not alive")
                return None
                
            # List all available vehicles for reference
            self.get_logger().info(f"Available vehicles in world ({len(vehicles)} total):")
            for i, v in enumerate(vehicles):
                loc = v.get_transform().location
                self.get_logger().info(f"  {i+1}. {v.type_id} (ID: {v.id}) at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
                
            return vehicle
            
        except Exception as e:
            self.get_logger().error(f"Error finding existing vehicle: {str(e)}")
            return None
    
    def setup_camera_and_controller(self):
        """Setup ZED 2i camera and robust controller with ROS2 integration"""
        try:
            # Setup camera
            self.camera = Zed2iCamera(
                world=self.world,
                vehicle=self.vehicle,
                resolution=(1280, 720),
                fps=30,
                model_path=self.model_path
            )
            
            if not self.camera.setup():
                raise RuntimeError("Failed to setup camera")
            
            # Setup robust controller with ROS2 integration (NO CARLA CONTROL)
            self.controller = PurePursuitController(self.vehicle)
            self.get_logger().info("Camera and planning controller with ROS2 interface setup complete")
            self.get_logger().info("🚫 Controller will NOT control vehicle - ROS2 topics only")
            self.get_logger().info("Lap counter enabled - orange cones will be detected for lap counting")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error setting up camera and controller: {e}")
            return False
    
    def planning_loop_ros(self):
        """ROS2 timer-based planning loop (NO CARLA CONTROL)"""
        try:
            if not self.running or not self.camera or not self.controller:
                return
            
            # Process camera frame
            self.camera.process_frame()
            
            # Get cone detections
            cone_detections = getattr(self.camera, 'cone_detections', [])
            
            # Plan path using controller (publishes ROS2 topics, NO CARLA CONTROL)
            steering, speed = self.controller.plan_path(cone_detections)
            
        except Exception as e:
            self.get_logger().error(f"Error in ROS2 planning loop: {e}")
    
    def display_loop(self):
        """Display camera feed with detections and YOLO bounding boxes"""
        if not self.show_visualization:
            return
            
        self.get_logger().info("Starting display loop with YOLO visualization...")
        
        while self.running:
            try:
                if hasattr(self.camera, 'rgb_image') and self.camera.rgb_image is not None:
                    # Create visualization with YOLO bounding boxes
                    viz_image = self.create_visualization_with_yolo()
                    
                    cv2.imshow('ROS2 CARLA Planning - YOLO Detection & Lap Counter (NO CARLA CONTROL)', viz_image)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                time.sleep(0.033)  # ~30 FPS display
                
            except Exception as e:
                self.get_logger().error(f"Error in display loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def create_visualization_with_yolo(self):
        """Create visualization with YOLO bounding boxes and enhanced ROS2 status"""
        try:
            if not hasattr(self.camera, 'rgb_image') or self.camera.rgb_image is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
                
            viz_image = self.camera.rgb_image.copy()
            
            # Add vehicle info at the top
            vehicle_info = f"ROS2 Planning Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id}) - NO CARLA CONTROL"
            cv2.putText(viz_image, vehicle_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Get cone detections for YOLO visualization
            cone_detections = getattr(self.camera, 'cone_detections', [])
            
            # Draw YOLO bounding boxes around ALL detected cones
            if cone_detections:
                self.draw_yolo_detections(viz_image, cone_detections)
                
                # Process detections for planning visualization
                blue_cones, yellow_cones, orange_cones = self.controller.process_cone_detections(cone_detections)
                
                # Try to find current target
                track_segment = self.controller.find_best_track_segment(blue_cones, yellow_cones)
                if not track_segment and (blue_cones or yellow_cones):
                    track_segment = self.controller.follow_cone_line(blue_cones, yellow_cones)
                
                # Draw planning target if found
                if track_segment:
                    self.draw_planning_target(viz_image, track_segment)
                    
                    # Draw track line if it's a proper track segment
                    if track_segment.get('type') != 'cone_line' and 'blue' in track_segment and 'yellow' in track_segment:
                        self.draw_track_line(viz_image, track_segment)
                
                # Draw orange cone lap markers
                self.draw_orange_lap_markers(viz_image, orange_cones)
            
            # Add enhanced status text with ROS2 information
            self.add_ros2_status_overlay(viz_image, cone_detections)
            
            return viz_image
            
        except Exception as e:
            self.get_logger().error(f"Error creating YOLO visualization: {e}")
            # Return a blank image if visualization fails
            blank_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_image, "Visualization Error", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank_image
    
    def draw_yolo_detections(self, viz_image, cone_detections):
        """Draw YOLO bounding boxes around all detected cones"""
        try:
            for detection in cone_detections:
                if not isinstance(detection, dict):
                    continue
                    
                # Get bounding box coordinates
                box = detection.get('box', [])
                if len(box) < 4:
                    continue
                    
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls = detection.get('cls', -1)
                conf = detection.get('conf', 0.0)
                depth = detection.get('depth', 0.0)
                
                # Class names and colors
                class_names = {0: 'Yellow', 1: 'Blue', 2: 'Orange'}
                class_colors = {0: (0, 255, 255), 1: (255, 0, 0), 2: (0, 165, 255)}  # BGR format
                
                class_name = class_names.get(cls, f'Class_{cls}')
                color = class_colors.get(cls, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class, confidence, and depth
                label = f"{class_name}: {conf:.2f} ({depth:.1f}m)"
                
                # Get text size for background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw background for text
                cv2.rectangle(viz_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # Draw text
                cv2.putText(viz_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(viz_image, (center_x, center_y), 3, color, -1)
                
        except Exception as e:
            self.get_logger().error(f"Error drawing YOLO detections: {e}")
    
    def draw_planning_target(self, viz_image, track_segment):
        """Draw the planning target point"""
        try:
            # Draw target point
            depth = track_segment.get('midpoint_y', 5.0)
            angle = np.arctan2(track_segment.get('midpoint_x', 0), depth)
            px = int(640 + (angle / np.radians(45)) * 640)
            py = int(720 - 100 - depth * 25)
            py = max(50, min(py, 720))
            px = max(0, min(px, 1280))
            
            # Different colors for different navigation types
            if track_segment.get('type') == 'cone_line':
                color = (255, 0, 255)  # Magenta for cone line following
                text = "CONE LINE TARGET"
            else:
                color = (0, 0, 255)    # Red for track segment
                text = "TRACK TARGET"
            
            # Draw target circle
            cv2.circle(viz_image, (px, py), 15, color, -1)
            cv2.circle(viz_image, (px, py), 15, (255, 255, 255), 2)  # White border
            
            # Draw target text
            cv2.putText(viz_image, text, (px+20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw distance to target
            distance_text = f"{depth:.1f}m"
            cv2.putText(viz_image, distance_text, (px-30, py+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            self.get_logger().error(f"Error drawing planning target: {e}")
    
    def draw_track_line(self, viz_image, track_segment):
        """Draw track line between blue and yellow cones"""
        try:
            blue_cone = track_segment['blue']
            yellow_cone = track_segment['yellow']
            
            # Blue cone position
            blue_angle = np.arctan2(blue_cone['x'], blue_cone['y'])
            blue_px = int(640 + (blue_angle / np.radians(45)) * 640)
            blue_py = int(720 - 100 - blue_cone['y'] * 25)
            blue_py = max(50, min(blue_py, 720))
            blue_px = max(0, min(blue_px, 1280))
            
            # Yellow cone position  
            yellow_angle = np.arctan2(yellow_cone['x'], yellow_cone['y'])
            yellow_px = int(640 + (yellow_angle / np.radians(45)) * 640)
            yellow_py = int(720 - 100 - yellow_cone['y'] * 25)
            yellow_py = max(50, min(yellow_py, 720))
            yellow_px = max(0, min(yellow_px, 1280))
            
            # Draw track line
            cv2.line(viz_image, (blue_px, blue_py), (yellow_px, yellow_py), (0, 255, 0), 4)
            
            # Draw cone markers
            cv2.circle(viz_image, (blue_px, blue_py), 8, (255, 0, 0), -1)
            cv2.circle(viz_image, (yellow_px, yellow_py), 8, (0, 255, 255), -1)
            
            # Add distance labels
            cv2.putText(viz_image, f"{blue_cone['depth']:.1f}m", 
                       (blue_px + 10, blue_py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(viz_image, f"{yellow_cone['depth']:.1f}m", 
                       (yellow_px + 10, yellow_py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
        except Exception as e:
            self.get_logger().error(f"Error drawing track line: {e}")
    
    def draw_orange_lap_markers(self, viz_image, orange_cones):
        """Draw orange cone lap markers"""
        try:
            for orange in orange_cones:
                orange_angle = np.arctan2(orange['x'], orange['y'])
                orange_px = int(640 + (orange_angle / np.radians(45)) * 640)
                orange_py = int(720 - 100 - orange['y'] * 25)
                orange_py = max(50, min(orange_py, 720))
                orange_px = max(0, min(orange_px, 1280))
                
                # Draw orange cone with distinct marker
                cv2.circle(viz_image, (orange_px, orange_py), 12, (0, 165, 255), -1)  # Orange color
                cv2.circle(viz_image, (orange_px, orange_py), 12, (255, 255, 255), 2)  # White border
                cv2.putText(viz_image, "LAP", (orange_px-15, orange_py-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(viz_image, f"{orange['depth']:.1f}m", (orange_px-20, orange_py+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
        except Exception as e:
            self.get_logger().error(f"Error drawing orange lap markers: {e}")
    
    def add_ros2_status_overlay(self, viz_image, cone_detections):
        """Add comprehensive ROS2 status information overlay"""
        try:
            # Get current vehicle velocity (for display only)
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            status_text = [
                f"🚫 ROS2 PLANNING NODE - NO CARLA CONTROL",
                f"Publishing Topics: /target_speed, /target_steering, /path_curvature",
                f"Laps Completed: {self.controller.lap_counter.laps_completed}",
                f"YOLO Detections: Blue: {len([d for d in cone_detections if d.get('cls') == 1])}, "
                f"Yellow: {len([d for d in cone_detections if d.get('cls') == 0])}, "
                f"Orange: {len([d for d in cone_detections if d.get('cls') == 2])}",
                f"Current Speed: {current_speed:.1f} m/s ({current_speed*3.6:.1f} km/h)",
                f"Target Speed: {self.controller.planning_state['target_speed']:.1f} m/s (ROS2)",
                f"Target Steering: {self.controller.planning_state['target_steering']:.3f} (ROS2)",
                f"Turn Type: {self.controller.current_turn_type.upper()} ({self.controller.turn_direction})",
                f"Track Confidence: {self.controller.planning_state['track_confidence']:.2f}",
                f"Path Curvature: {self.controller.planning_state['path_curvature']:.3f} (ROS2)",
                f"Cone Visibility Risk: {'HIGH' if self.controller.planning_state['cone_visibility_risk'] >= 2.0 else 'MODERATE' if self.controller.planning_state['cone_visibility_risk'] >= 1.0 else 'LOW'}",
                f"Physics Constraints: {'ON' if self.controller.physics_constraints_enabled else 'OFF'} (Loosened)",
                f"PID Info Publishing: {'ON' if self.controller.pid_enabled else 'OFF'}",
                f"Distance: {self.controller.distance_traveled:.1f}m",
                f"Lost Track: {self.controller.lost_track_counter}",
                f"Planning Frequency: {self.control_frequency} Hz"
            ]
            
            for i, text in enumerate(status_text):
                y_pos = 50 + i*18  # Tighter spacing for more info
                
                # Color coding for different types of information
                if i == 0:  # Header - NO CONTROL warning
                    color = (0, 0, 255)  # Red for warning
                elif i == 1:  # ROS2 topics
                    color = (255, 255, 0)  # Yellow for ROS2 info
                elif i == 2:  # Lap counter
                    color = (0, 255, 0)  # Green for lap counter
                elif i == 3:  # YOLO detections
                    color = (255, 0, 255)  # Magenta for YOLO
                elif i == 4:  # Current speed (display only)
                    color = (0, 255, 255)  # Cyan for current speed
                elif i == 5 or i == 6:  # Target commands (ROS2)
                    color = (255, 255, 0)  # Yellow for ROS2 commands
                elif i == 7:  # Turn type
                    color = (255, 0, 255)  # Magenta for turn type
                elif i == 8:  # Track confidence
                    color = (0, 255, 0) if self.controller.planning_state['track_confidence'] > 0.7 else (255, 255, 0) if self.controller.planning_state['track_confidence'] > 0.4 else (255, 0, 0)
                elif i == 9:  # Path curvature (ROS2)
                    color = (255, 255, 0)  # Yellow for ROS2 data
                elif i == 10:  # Visibility risk
                    risk = self.controller.planning_state['cone_visibility_risk']
                    color = (255, 0, 0) if risk >= 2.0 else (255, 255, 0) if risk >= 1.0 else (0, 255, 0)
                elif i == 11:  # Physics constraints
                    color = (0, 255, 0) if self.controller.physics_constraints_enabled else (255, 0, 0)
                elif i == 12:  # PID info
                    color = (0, 255, 0) if self.controller.pid_enabled else (255, 0, 0)
                elif i == 14:  # Lost track counter
                    color = (255, 0, 0) if self.controller.lost_track_counter > 10 else (0, 255, 0)
                elif i == 15:  # Planning frequency
                    color = (255, 255, 0)  # Yellow for system info
                else:
                    color = (255, 255, 255)  # White for other info
                
                cv2.putText(viz_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
        except Exception as e:
            self.get_logger().error(f"Error adding ROS2 status overlay: {e}")
    
    def run(self):
        """Main execution function with ROS2 integration (NO CARLA CONTROL)"""
        try:
            # Setup CARLA
            if not self.setup_carla():
                return False
            
            # Setup camera and controller
            if not self.setup_camera_and_controller():
                return False
            
            self.get_logger().info("🚫 ROS2 CARLA Planning System ready! NO CARLA CONTROL - ROS2 topics only")
            self.get_logger().info(f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id}) - SENSING ONLY")
            self.get_logger().info("🟠 Orange cones will be detected for lap counting")
            self.get_logger().info("📡 Publishing ROS2 topics: /target_speed, /target_steering, /path_curvature, /turn_type")
            self.get_logger().info("🎯 YOLO bounding boxes will be displayed in visualization")
            self.get_logger().info("Press Ctrl+C to stop or 'q' in the display window")
            
            # Start display thread if visualization enabled
            if self.show_visualization:
                self.display_thread = threading.Thread(target=self.display_loop)
                self.display_thread.start()
            
            # ROS2 timer handles the planning loop, so we just spin
            try:
                rclpy.spin(self)
            except KeyboardInterrupt:
                self.get_logger().info("Received keyboard interrupt")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error running ROS2 planning system: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all resources without affecting the existing vehicle"""
        self.get_logger().info("Cleaning up ROS2 CARLA Planning System...")
        
        self.running = False
        
        # NO VEHICLE CONTROL - just log the final state
        self.get_logger().info("🚫 NO CARLA CONTROL performed - vehicle unaffected")
        
        # Print final lap count
        if self.controller and hasattr(self.controller, 'lap_counter'):
            self.get_logger().info(f"🏁 Final lap count: {self.controller.lap_counter.laps_completed} laps completed")
        
        # Print final planning state
        if self.controller:
            self.get_logger().info(f"📊 Final planning state:")
            self.get_logger().info(f"   Target Speed: {self.controller.planning_state['target_speed']:.2f} m/s")
            self.get_logger().info(f"   Target Steering: {self.controller.planning_state['target_steering']:.3f}")
            self.get_logger().info(f"   Track Confidence: {self.controller.planning_state['track_confidence']:.2f}")
            self.get_logger().info(f"   Distance Traveled: {self.controller.distance_traveled:.1f}m")
        
        # Wait for display thread to finish
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        # Cleanup camera
        if self.camera:
            try:
                self.camera.shutdown()
            except:
                pass
        
        # Close CV2 windows
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        self.get_logger().info("🚫 ROS2 CARLA Planning System cleanup complete - vehicle preserved and uncontrolled")


def main(args=None):
    """Main function with ROS2 integration"""
    rclpy.init(args=args)
    
    try:
        # Create and run the enhanced planning system with ROS2 interface (NO CARLA CONTROL)
        planning_system = CarlaRacingSystem()
        
        success = planning_system.run()
        if success:
            planning_system.get_logger().info("🚫 ROS2 CARLA Planning system completed successfully - NO CARLA CONTROL")
        else:
            planning_system.get_logger().error("ROS2 CARLA Planning system failed to start")
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()