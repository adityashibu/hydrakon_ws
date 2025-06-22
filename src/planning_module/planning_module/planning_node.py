#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import cv2
import time
import threading
import signal
import sys
from collections import deque
from std_msgs.msg import Float64, Int32, Bool, Header
from geometry_msgs.msg import Point, Vector3
from .zed_2i import Zed2iCamera
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor  


class LapCounter:
    def __init__(self, node):
        self.node = node
        self.laps_completed = 0
        self.last_orange_gate_time = 0
        self.cooldown_duration = 3.0  # 3 seconds cooldown between lap counts
        self.orange_gate_passed_threshold = 2.0  # Distance threshold for passing through orange gate
        
        # ROS2 publisher for lap events
        self.lap_pub = self.node.create_publisher(Int32, '/planning/lap_count', 10)
        
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
                    self.publish_lap_completion()
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
            self.publish_lap_completion()
            print(f"🏁 LAP {self.laps_completed} COMPLETED! Passed through orange gate!")
            return True
        
        return False
    
    def publish_lap_completion(self):
        """Publish lap completion event to ROS2 topic"""
        try:
            msg = Int32()
            msg.data = self.laps_completed
            self.lap_pub.publish(msg)
            self.node.get_logger().info(f"Published lap completion: {self.laps_completed}")
        except Exception as e:
            print(f"Error publishing lap completion: {e}")
    
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

class PurePursuitController:
    def __init__(self, vehicle, node, lookahead_distance=4.0):
        self.vehicle = vehicle
        self.node = node
        self.lookahead_distance = lookahead_distance
        
        # Initialize with default values first (in case parameter loading fails)
        # Vehicle parameters
        self.wheelbase = 2.7  # meters
        self.max_speed = 8.0  # m/s
        self.min_speed = 2.0  # m/s
        
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
        
        # Track following parameters
        self.track_width_min = 3.0  # Minimum track width (meters)
        self.track_width_max = 5.0  # Maximum track width (meters) - reduced
        self.max_depth_diff = 1.0   # Maximum depth difference between gate cones - reduced
        self.max_lateral_jump = 1.5 # Maximum lateral movement between consecutive gates - reduced
        self.forward_focus_angle = 30.0  # Only consider cones within this angle (degrees) from vehicle heading
        
        # Backup navigation when no gates found
        self.max_lost_track_frames = 20
        
        # Visibility parameters (defaults)
        self.high_risk_threshold = 2.5
        self.moderate_risk_threshold = 1.8
        self.visibility_steering_boost = 0.6
        self.high_risk_factor = 1.4
        self.moderate_risk_factor = 1.2
        
        # Recovery parameters (defaults)
        self.recovery_steering_multiplier = 1.5
        self.aggressive_search_amplitude = 18.0
        self.wide_search_amplitude = 24.0
        
        # Camera parameters (defaults)
        self.fov_horizontal = 110.0  # ZED 2i FOV
        self.max_steering_degrees = 30.0
        
        # Load parameters from ROS2 parameter server (will override defaults)
        self.load_parameters(node)
        
        # State tracking
        self.last_steering = 0.0
        self.steering_history = deque(maxlen=5)
        
        # NEW: Turn state tracking
        self.current_turn_type = "straight"  # "straight", "gentle", "sharp", "u_turn"
        self.turn_direction = "none"  # "left", "right", "none"
        self.path_offset = 0.0  # Current path offset for wider turns
        self.gate_sequence = deque(maxlen=5)  # Track recent gates for turn prediction
        
        # Backup navigation when no gates found
        self.lost_track_counter = 0
        self.max_lost_track_frames = 20
        
        # Distance tracking for basic stats
        self.distance_traveled = 0.0
        self.last_position = None
        
        # Initialize lap counter
        self.lap_counter = LapCounter(node)
        
        # ROS2 Publishers for control commands
        self.setup_ros_publishers()
        
    def setup_ros_publishers(self):
        """Setup ROS2 publishers for control commands"""
        try:
            # QoS profile for reliable communication
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            # Publishers for control commands using standard ROS2 messages
            self.target_speed_pub = self.node.create_publisher(Float64, '/planning/target_speed', qos_profile)
            self.target_position_pub = self.node.create_publisher(Point, '/planning/target_position', qos_profile)
            self.reference_steering_pub = self.node.create_publisher(Float64, '/planning/reference_steering', qos_profile)
            self.lookahead_distance_pub = self.node.create_publisher(Float64, '/planning/lookahead_distance', qos_profile)
            self.emergency_stop_pub = self.node.create_publisher(Bool, '/planning/emergency_stop', qos_profile)
            
            # Debug publishers
            self.debug_info_pub = self.node.create_publisher(Vector3, '/planning/debug_info', qos_profile)
            self.turn_info_pub = self.node.create_publisher(Vector3, '/planning/turn_info', qos_profile)
            
            self.node.get_logger().info("ROS2 publishers initialized for planning outputs")
            
        except Exception as e:
            print(f"Error setting up ROS2 publishers: {e}")
    
    def load_parameters(self, node):
        """Load parameters from ROS2 parameter server"""
        try:
            # Vehicle parameters
            self.wheelbase = node.get_parameter('vehicle.wheelbase').get_parameter_value().double_value
            self.max_steering_degrees = node.get_parameter('vehicle.max_steering_degrees').get_parameter_value().double_value
            
            # Control parameters
            self.max_speed = node.get_parameter('control.max_speed').get_parameter_value().double_value
            self.min_speed = node.get_parameter('control.min_speed').get_parameter_value().double_value
            self.lookahead_distance = node.get_parameter('control.lookahead_distance').get_parameter_value().double_value
            
            # Camera parameters
            self.fov_horizontal = node.get_parameter('camera.fov_horizontal').get_parameter_value().double_value
            
            # Detection parameters
            self.safety_offset = node.get_parameter('detection.safety_offset').get_parameter_value().double_value
            self.max_depth = node.get_parameter('detection.max_depth').get_parameter_value().double_value
            self.min_depth = node.get_parameter('detection.min_depth').get_parameter_value().double_value
            self.max_lateral_distance = node.get_parameter('detection.max_lateral_distance').get_parameter_value().double_value
            
            # Turn parameters
            self.min_turn_radius = node.get_parameter('turns.min_turn_radius').get_parameter_value().double_value
            self.path_widening_factor = node.get_parameter('turns.path_widening_factor').get_parameter_value().double_value
            self.sharp_turn_threshold = node.get_parameter('turns.sharp_turn_threshold').get_parameter_value().double_value
            self.u_turn_threshold = node.get_parameter('turns.u_turn_threshold').get_parameter_value().double_value
            self.turn_detection_distance = node.get_parameter('turns.turn_detection_distance').get_parameter_value().double_value
            
            # Track parameters
            self.track_width_min = node.get_parameter('track.width_min').get_parameter_value().double_value
            self.track_width_max = node.get_parameter('track.width_max').get_parameter_value().double_value
            self.max_depth_diff = node.get_parameter('track.max_depth_diff').get_parameter_value().double_value
            self.max_lateral_jump = node.get_parameter('track.max_lateral_jump').get_parameter_value().double_value
            self.forward_focus_angle = node.get_parameter('track.forward_focus_angle').get_parameter_value().double_value
            
            # Visibility parameters
            self.high_risk_threshold = node.get_parameter('visibility.high_risk_threshold').get_parameter_value().double_value
            self.moderate_risk_threshold = node.get_parameter('visibility.moderate_risk_threshold').get_parameter_value().double_value
            self.visibility_steering_boost = node.get_parameter('visibility.visibility_steering_boost').get_parameter_value().double_value
            self.high_risk_factor = node.get_parameter('visibility.high_risk_factor').get_parameter_value().double_value
            self.moderate_risk_factor = node.get_parameter('visibility.moderate_risk_factor').get_parameter_value().double_value
            
            # Recovery parameters
            self.max_lost_track_frames = node.get_parameter('recovery.max_lost_track_frames').get_parameter_value().integer_value
            self.recovery_steering_multiplier = node.get_parameter('recovery.recovery_steering_multiplier').get_parameter_value().double_value
            self.aggressive_search_amplitude = node.get_parameter('recovery.aggressive_search_amplitude').get_parameter_value().double_value
            self.wide_search_amplitude = node.get_parameter('recovery.wide_search_amplitude').get_parameter_value().double_value
            
            node.get_logger().info("✅ Parameters loaded successfully from parameter server")
            
        except Exception as e:
            node.get_logger().warn(f"⚠️ Failed to load some parameters: {e}")
            node.get_logger().info("Using default hardcoded values")
    
    def publish_control_targets(self, target_x, target_y, target_speed, steering_angle):
        """Publish control targets to ROS2 topics instead of applying to CARLA vehicle"""
        try:
            # Publish target speed
            speed_msg = Float64()
            speed_msg.data = target_speed
            self.target_speed_pub.publish(speed_msg)
            
            # Publish target position (in vehicle frame)
            pos_msg = Point()
            pos_msg.x = target_x
            pos_msg.y = target_y
            pos_msg.z = 0.0
            self.target_position_pub.publish(pos_msg)
            
            # Publish reference steering
            steering_msg = Float64()
            steering_msg.data = steering_angle  # Already in degrees
            self.reference_steering_pub.publish(steering_msg)
            
            # Publish lookahead distance
            lookahead_msg = Float64()
            lookahead_dist = np.sqrt(target_x**2 + target_y**2)
            lookahead_msg.data = lookahead_dist
            self.lookahead_distance_pub.publish(lookahead_msg)
            
            # Publish turn information (encoded as Vector3)
            turn_msg = Vector3()
            # Encode turn type as number: 0=straight, 1=gentle, 2=sharp, 3=u_turn
            turn_type_map = {"straight": 0, "gentle": 1, "sharp": 2, "u_turn": 3}
            turn_msg.x = float(turn_type_map.get(self.current_turn_type, 0))
            # Encode direction: -1=left, 0=none, 1=right
            direction_map = {"left": -1, "none": 0, "right": 1}
            turn_msg.y = float(direction_map.get(self.turn_direction, 0))
            turn_msg.z = float(self.path_offset)
            self.turn_info_pub.publish(turn_msg)
            
            # Publish debug info (cone counts)
            debug_msg = Vector3()
            debug_msg.x = float(self.last_blue_cone_count if hasattr(self, 'last_blue_cone_count') else 0)
            debug_msg.y = float(self.last_yellow_cone_count if hasattr(self, 'last_yellow_cone_count') else 0)
            debug_msg.z = float(self.last_orange_cone_count if hasattr(self, 'last_orange_cone_count') else 0)
            self.debug_info_pub.publish(debug_msg)
            
        except Exception as e:
            print(f"Error publishing control targets: {e}")
    
    def publish_emergency_stop(self):
        """Publish emergency stop signal to ROS2 topics"""
        try:
            # Publish emergency stop flag
            emergency_msg = Bool()
            emergency_msg.data = True
            self.emergency_stop_pub.publish(emergency_msg)
            
            # Publish zero targets
            speed_msg = Float64()
            speed_msg.data = 0.0
            self.target_speed_pub.publish(speed_msg)
            
            pos_msg = Point()
            pos_msg.x = 0.0
            pos_msg.y = 5.0  # Look ahead point
            pos_msg.z = 0.0
            self.target_position_pub.publish(pos_msg)
            
            steering_msg = Float64()
            steering_msg.data = 0.0
            self.reference_steering_pub.publish(steering_msg)
            
            print("DEBUG: Published emergency stop to ROS2 topics")
            
        except Exception as e:
            print(f"Error publishing emergency stop: {e}")
    
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
            
            # Determine turn type and direction
            if abs_angle < 10 and upcoming_turn_severity < 20:
                turn_type = "straight"
                direction = "none"
                path_offset = 0.0
            elif abs_angle < self.sharp_turn_threshold and upcoming_turn_severity < 40:
                turn_type = "gentle"
                direction = "left" if angle_change > 0 else "right"
                path_offset = 0.3 * self.path_widening_factor
            elif abs_angle < self.u_turn_threshold and upcoming_turn_severity < 70:
                turn_type = "sharp"
                direction = "left" if angle_change > 0 else "right"
                path_offset = 0.6 * self.path_widening_factor
            else:
                turn_type = "u_turn"
                direction = "left" if angle_change > 0 else "right"
                path_offset = 1.0 * self.path_widening_factor
            
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
        
        # Combine all factors
        final_speed = base_speed * speed_factor * steering_factor * distance_factor
        
        return max(final_speed, self.min_speed * 0.7)  # Minimum speed limit
    
    def image_to_world_coords(self, center_x, center_y, depth):
        """Convert image coordinates to world coordinates (vehicle reference frame)"""
        # Camera parameters for ZED 2i simulation
        image_width = 1280
        # Use parameter instead of hardcoded FOV
        fov_horizontal = getattr(self, 'fov_horizontal', 110.0)  # Default to 110.0 if not loaded
        
        # Calculate angle from image center
        angle = ((center_x - image_width / 2) / (image_width / 2)) * (fov_horizontal / 2)
        
        # Convert to world coordinates relative to vehicle
        world_x = depth * np.tan(np.radians(angle))
        world_y = depth
        
        return world_x, world_y
    
    def process_cone_detections(self, cone_detections):
        """Process cone detections - FIXED FOR ZED 2I FORMAT"""
        if not cone_detections:
            return [], [], []
            
        blue_cones = []    # Class 1 - LEFT side
        yellow_cones = []  # Class 0 - RIGHT side
        orange_cones = []  # Class 2 - ORANGE (lap markers)
        
        try:
            print(f"DEBUG: Processing {len(cone_detections)} cone detections from ZED 2i")
            
            for detection in cone_detections:
                if not isinstance(detection, dict):
                    continue
                    
                # ZED 2i format: {'box': (x1, y1, x2, y2), 'cls': cls, 'depth': depth, 'y_pos': y_pos}
                if 'box' not in detection or 'cls' not in detection or 'depth' not in detection:
                    print(f"DEBUG: Skipping detection missing required fields: {detection.keys()}")
                    continue
                    
                # Handle box format from ZED 2i
                box = detection['box']
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                else:
                    print(f"DEBUG: Invalid box format: {box}")
                    continue
                    
                cls = detection['cls']
                depth = detection['depth']
                
                # Ensure numeric values
                try:
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    cls = int(cls)
                    depth = float(depth)
                except (ValueError, TypeError):
                    print(f"DEBUG: Error converting detection values: box={box}, cls={cls}, depth={depth}")
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
                
                # Convert to world coordinates using ZED 2i FOV
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
                    print(f"DEBUG: Blue cone at ({world_x:.2f}, {world_y:.2f}) depth={depth:.2f}m")
                elif cls == 0:  # Yellow cone - RIGHT side  
                    yellow_cones.append(cone_data)
                    print(f"DEBUG: Yellow cone at ({world_x:.2f}, {world_y:.2f}) depth={depth:.2f}m")
                elif cls == 2:  # Orange cone - LAP MARKER
                    orange_cones.append(cone_data)
                    print(f"DEBUG: Orange cone at ({world_x:.2f}, {world_y:.2f}) depth={depth:.2f}m")
                    
        except Exception as e:
            print(f"ERROR processing cone detections: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
        
        # Sort by depth (closest first), then by angle (most centered first)
        blue_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        yellow_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        orange_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        
        # Store counts for debug publishing
        self.last_blue_cone_count = len(blue_cones)
        self.last_yellow_cone_count = len(yellow_cones)
        self.last_orange_cone_count = len(orange_cones)
        
        # Debug filtered cones
        print(f"DEBUG: After spatial filtering - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
        
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
            
            # Use parameters instead of hardcoded values
            high_risk_threshold = getattr(self, 'high_risk_threshold', 2.5)
            moderate_risk_threshold = getattr(self, 'moderate_risk_threshold', 1.8)
            high_risk_factor = getattr(self, 'high_risk_factor', 1.4)
            moderate_risk_factor = getattr(self, 'moderate_risk_factor', 1.2)
            
            # If target is getting close to our field of view limits, increase steering aggressiveness
            if lateral_offset > high_risk_threshold:  # Getting close to losing sight
                visibility_risk_factor = high_risk_factor
                print(f"DEBUG: High visibility risk - increasing steering aggressiveness by {(high_risk_factor-1)*100:.0f}%")
            elif lateral_offset > moderate_risk_threshold:  # Moderate risk
                visibility_risk_factor = moderate_risk_factor
                print(f"DEBUG: Moderate visibility risk - increasing steering aggressiveness by {(moderate_risk_factor-1)*100:.0f}%")
            
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
            
            # Convert to degrees for ROS2 output
            steering_degrees = np.degrees(steering_angle)
            
            # Apply steering limits (keep in degrees)
            max_steering_degrees = 30.0
            steering_degrees = np.clip(steering_degrees, -max_steering_degrees, max_steering_degrees)
            
            # NEW: Apply final visibility preservation enhancement
            # If we're at risk of losing cones and steering is not aggressive enough, boost it
            if lateral_offset > 2.0 and abs(steering_degrees) < 12.0:  # 12 degrees = 0.4 * 30
                steering_boost = min(6.0, (lateral_offset - 2.0) * 9.0)  # Up to 6 degrees boost
                if steering_degrees > 0:
                    steering_degrees = min(30.0, steering_degrees + steering_boost)
                else:
                    steering_degrees = max(-30.0, steering_degrees - steering_boost)
                
                print(f"DEBUG: Applied final steering boost for cone visibility: {steering_boost:.1f}°")
            
            print(f"DEBUG: Final steering command: {steering_degrees:.3f}°")
            direction = 'LEFT' if steering_degrees > 0 else 'RIGHT' if steering_degrees < 0 else 'STRAIGHT'
            print(f"DEBUG: Steering direction: {direction}")
            
            return steering_degrees
            
        except Exception as e:
            print(f"ERROR in pure pursuit calculation: {e}")
            return 0.0
    
    def smooth_steering(self, raw_steering):
        """Apply steering smoothing with enhanced responsiveness for cone visibility"""
        try:
            self.steering_history.append(raw_steering)
            
            # NEW: Adaptive smoothing based on visibility risk
            lateral_offset = abs(getattr(self, 'current_target_x', 0.0))
            
            if len(self.steering_history) >= 3:
                # If we're at high risk of losing cones, use less smoothing (more responsive)
                if lateral_offset > 2.5 or self.current_turn_type in ["sharp", "u_turn"]:
                    # More aggressive weighting for recent steering inputs
                    weights = np.array([0.7, 0.2, 0.1])  # Heavy emphasis on current steering
                    print(f"DEBUG: High visibility risk - using aggressive steering smoothing")
                elif lateral_offset > 1.8 or self.current_turn_type == "gentle":
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
            if lateral_offset > 2.5:
                # High risk - allow more aggressive steering changes
                max_change = 7.5  # degrees
                print(f"DEBUG: High visibility risk - allowing max steering change: {max_change}°")
            elif lateral_offset > 1.8:
                # Moderate risk - slightly more aggressive
                max_change = 6.0  # degrees
            elif self.current_turn_type in ["sharp", "u_turn"]:
                # Sharp turns - more responsive
                max_change = 5.5  # degrees
            else:
                # Normal rate limiting
                max_change = 4.5  # degrees
            
            # Apply rate limiting
            if abs(smoothed - self.last_steering) > max_change:
                if smoothed > self.last_steering:
                    smoothed = self.last_steering + max_change
                else:
                    smoothed = self.last_steering - max_change
                print(f"DEBUG: Applied steering rate limiting: {max_change}°")
            
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
    
    def control_vehicle(self, cone_detections):
        """Main control function - NOW PUBLISHES TO ROS2 TOPICS with enhanced lost track recovery"""
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG: CONTROL CYCLE - {len(cone_detections) if cone_detections else 0} detections from ZED 2i")
            print(f"Laps completed: {self.lap_counter.laps_completed}")
            print(f"Current turn type: {self.current_turn_type}, Direction: {self.turn_direction}")
            print(f"Lost track counter: {self.lost_track_counter}")
            print(f"{'='*60}")
            
            # Update distance traveled
            self.update_distance_traveled()
            
            # Process cone detections (includes orange cones) - FIXED FOR ZED 2I
            blue_cones, yellow_cones, orange_cones = self.process_cone_detections(cone_detections)
            print(f"DEBUG: Processed cones - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
            
            # Check for lap completion through orange gate
            if orange_cones:
                transform = self.vehicle.get_transform()
                vehicle_position = (transform.location.x, transform.location.y, transform.location.z)
                self.lap_counter.check_orange_gate_passage(orange_cones, vehicle_position)
            
            # NEW: Enhanced lost track detection with immediate recovery steering
            if len(blue_cones) == 0 and len(yellow_cones) == 0:
                self.lost_track_counter += 1
                print(f"DEBUG: NO CONES DETECTED - lost track for {self.lost_track_counter} frames")
                
                # Immediate aggressive steering to try to find cones again
                if self.lost_track_counter <= 10:
                    # Try to steer in the direction we were last going
                    recovery_steering = self.last_steering * 1.5  # Amplify last steering
                    recovery_steering = np.clip(recovery_steering, -24.0, 24.0)
                    print(f"DEBUG: Applying recovery steering: {recovery_steering:.3f}°")
                    
                    self.publish_control_targets(0.0, 5.0, 1.5, recovery_steering)
                    return recovery_steering, 1.5
                elif self.lost_track_counter <= 20:
                    # More aggressive search pattern
                    search_steering = 18.0 * np.sin(self.lost_track_counter * 0.3)
                    print(f"DEBUG: Applying aggressive search pattern: {search_steering:.3f}°")
                    
                    self.publish_control_targets(0.0, 5.0, 1.2, search_steering)
                    return search_steering, 1.2
                else:
                    # Last resort - wide search
                    search_steering = 24.0 * np.sin(self.lost_track_counter * 0.2)
                    print(f"DEBUG: Applying wide search pattern: {search_steering:.3f}°")
                    
                    self.publish_control_targets(0.0, 5.0, 0.8, search_steering)
                    return search_steering, 0.8
            
            # Try to find a track segment
            track_segment = self.find_best_track_segment(blue_cones, yellow_cones)
            
            # If no track segment found, try cone line following
            if not track_segment and (blue_cones or yellow_cones):
                track_segment = self.follow_cone_line(blue_cones, yellow_cones)
                print("DEBUG: Using cone line following")
            
            if not track_segment:
                self.lost_track_counter += 1
                print(f"DEBUG: No navigation target found - lost track for {self.lost_track_counter} frames")
                
                # If lost for too long, implement search pattern
                if self.lost_track_counter > self.max_lost_track_frames:
                    print("DEBUG: Lost track for too long - implementing emergency stop")
                    self.publish_emergency_stop()
                    return 0.0, 0.0
                else:
                    # Move forward slowly while searching
                    search_steering = self.last_steering * 0.5
                    self.publish_control_targets(0.0, 5.0, 1.2, search_steering)
                    return search_steering, 1.2
            
            # Reset lost track counter if we found something
            self.lost_track_counter = 0
            
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
            
            adjusted_target_x, adjusted_target_y = self.adjust_target_for_turn(
                original_target_x, original_target_y, turn_type, turn_direction, path_offset
            )
            
            # Navigate towards the adjusted target
            raw_steering = self.calculate_pure_pursuit_steering(adjusted_target_x, adjusted_target_y)
            smooth_steering = self.smooth_steering(raw_steering)
            
            # Calculate adaptive speed based on turn type
            current_depth = track_segment['avg_depth']
            target_speed = self.calculate_adaptive_speed(turn_type, smooth_steering, current_depth)
            
            # PUBLISH TO ROS2 TOPICS INSTEAD OF APPLYING TO CARLA VEHICLE
            self.publish_control_targets(adjusted_target_x, adjusted_target_y, target_speed, smooth_steering)
            
            # Enhanced debug output
            direction = 'LEFT' if smooth_steering > 0 else 'RIGHT' if smooth_steering < 0 else 'STRAIGHT'
            
            print(f"DEBUG: PUBLISHED CONTROL TARGETS TO ROS2:")
            print(f"  Navigation: {track_segment.get('type', 'track_segment')}_{turn_type}")
            print(f"  Turn Analysis: {turn_type}-{turn_direction} (offset: {path_offset:.2f}m)")
            print(f"  Original target: ({original_target_x:.2f}, {original_target_y:.2f})")
            print(f"  Adjusted target: ({adjusted_target_x:.2f}, {adjusted_target_y:.2f})")
            print(f"  Target distance: {current_depth:.2f}m")
            print(f"  Turn radius: {self.calculate_turn_radius(np.radians(smooth_steering)):.2f}m")
            print(f"  Steering: {smooth_steering:.3f}° ({direction})")
            print(f"  Cone visibility risk: {'HIGH' if abs(original_target_x) > 2.5 else 'MODERATE' if abs(original_target_x) > 1.8 else 'LOW'}")
            print(f"  Target Speed: {target_speed:.1f} m/s ({target_speed*3.6:.1f} km/h)")
            print(f"  Distance: {self.distance_traveled:.1f}m")
            print(f"  Laps: {self.lap_counter.laps_completed}")
            print(f"  PUBLISHED TO ROS2 TOPICS")
            print(f"{'='*60}\n")
            
            return smooth_steering, target_speed
            
        except Exception as e:
            print(f"ERROR in vehicle control: {e}")
            import traceback
            traceback.print_exc()
            # Safe fallback - publish emergency stop
            self.publish_emergency_stop()
            return 0.0, 0.0

class CarlaRacingSystemROS2(Node):
    def __init__(self, model_path=None):
        super().__init__('carla_racing_system')
        
        # Declare all parameters with default values and descriptions
        self.declare_all_parameters()
        
        # CARLA components
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.controller = None
        self.running = True
        self.model_path = model_path
        
        # Threading
        self.control_thread = None
        self.display_thread = None
        
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.get_logger().info("CARLA Racing System ROS2 Node initialized")
        self.get_logger().info(f"YOLO model path: {model_path}")
        
    def declare_all_parameters(self):
        """Declare all ROS2 parameters with default values and descriptions"""
        
        # Vehicle parameters
        self.declare_parameter('vehicle.wheelbase', 2.7, 
                              descriptor=ParameterDescriptor(description='Vehicle wheelbase in meters'))
        self.declare_parameter('vehicle.max_steering_degrees', 30.0,
                              descriptor=ParameterDescriptor(description='Maximum steering angle in degrees'))
        
        # Control parameters
        self.declare_parameter('control.max_speed', 8.0,
                              descriptor=ParameterDescriptor(description='Maximum vehicle speed in m/s'))
        self.declare_parameter('control.min_speed', 2.0,
                              descriptor=ParameterDescriptor(description='Minimum vehicle speed in m/s'))
        self.declare_parameter('control.lookahead_distance', 4.0,
                              descriptor=ParameterDescriptor(description='Pure pursuit lookahead distance in meters'))
        
        # Camera parameters
        self.declare_parameter('camera.fov_horizontal', 110.0,
                              descriptor=ParameterDescriptor(description='Camera horizontal field of view in degrees'))
        self.declare_parameter('camera.resolution_width', 1280,
                              descriptor=ParameterDescriptor(description='Camera resolution width in pixels'))
        self.declare_parameter('camera.resolution_height', 720,
                              descriptor=ParameterDescriptor(description='Camera resolution height in pixels'))
        self.declare_parameter('camera.fps', 30,
                              descriptor=ParameterDescriptor(description='Camera frames per second'))
        
        # Detection parameters
        self.declare_parameter('detection.safety_offset', 0.5,
                              descriptor=ParameterDescriptor(description='Safety offset from cones in meters'))
        self.declare_parameter('detection.max_depth', 8.0,
                              descriptor=ParameterDescriptor(description='Maximum cone detection range in meters'))
        self.declare_parameter('detection.min_depth', 1.5,
                              descriptor=ParameterDescriptor(description='Minimum cone detection range in meters'))
        self.declare_parameter('detection.max_lateral_distance', 4.0,
                              descriptor=ParameterDescriptor(description='Maximum lateral distance from vehicle center in meters'))
        
        # Turn parameters
        self.declare_parameter('turns.min_turn_radius', 3.5,
                              descriptor=ParameterDescriptor(description='Minimum safe turning radius in meters'))
        self.declare_parameter('turns.path_widening_factor', 1.8,
                              descriptor=ParameterDescriptor(description='How much to widen the path in turns'))
        self.declare_parameter('turns.sharp_turn_threshold', 25.0,
                              descriptor=ParameterDescriptor(description='Angle threshold for sharp turns in degrees'))
        self.declare_parameter('turns.u_turn_threshold', 60.0,
                              descriptor=ParameterDescriptor(description='Angle threshold for U-turns in degrees'))
        self.declare_parameter('turns.turn_detection_distance', 6.0,
                              descriptor=ParameterDescriptor(description='Distance to look ahead for turn detection in meters'))
        
        # Track parameters
        self.declare_parameter('track.width_min', 3.0,
                              descriptor=ParameterDescriptor(description='Minimum track width in meters'))
        self.declare_parameter('track.width_max', 5.0,
                              descriptor=ParameterDescriptor(description='Maximum track width in meters'))
        self.declare_parameter('track.max_depth_diff', 1.0,
                              descriptor=ParameterDescriptor(description='Maximum depth difference between gate cones in meters'))
        self.declare_parameter('track.max_lateral_jump', 1.5,
                              descriptor=ParameterDescriptor(description='Maximum lateral movement between consecutive gates in meters'))
        self.declare_parameter('track.forward_focus_angle', 30.0,
                              descriptor=ParameterDescriptor(description='Only consider cones within this angle from vehicle heading in degrees'))
        
        # Visibility parameters
        self.declare_parameter('visibility.high_risk_threshold', 2.5,
                              descriptor=ParameterDescriptor(description='High risk threshold for cone visibility in meters'))
        self.declare_parameter('visibility.moderate_risk_threshold', 1.8,
                              descriptor=ParameterDescriptor(description='Moderate risk threshold for cone visibility in meters'))
        self.declare_parameter('visibility.visibility_steering_boost', 0.6,
                              descriptor=ParameterDescriptor(description='Steering boost factor for cone visibility preservation'))
        self.declare_parameter('visibility.high_risk_factor', 1.4,
                              descriptor=ParameterDescriptor(description='High risk steering factor multiplier'))
        self.declare_parameter('visibility.moderate_risk_factor', 1.2,
                              descriptor=ParameterDescriptor(description='Moderate risk steering factor multiplier'))
        
        # Recovery parameters
        self.declare_parameter('recovery.max_lost_track_frames', 20,
                              descriptor=ParameterDescriptor(description='Maximum frames before triggering emergency stop'))
        self.declare_parameter('recovery.recovery_steering_multiplier', 1.5,
                              descriptor=ParameterDescriptor(description='Steering multiplier for track recovery'))
        self.declare_parameter('recovery.aggressive_search_amplitude', 18.0,
                              descriptor=ParameterDescriptor(description='Amplitude for aggressive search pattern in degrees'))
        self.declare_parameter('recovery.wide_search_amplitude', 24.0,
                              descriptor=ParameterDescriptor(description='Amplitude for wide search pattern in degrees'))
        
        # Lap counter parameters
        self.declare_parameter('lap_counter.cooldown_duration', 3.0,
                              descriptor=ParameterDescriptor(description='Cooldown duration between lap counts in seconds'))
        self.declare_parameter('lap_counter.orange_gate_threshold', 2.0,
                              descriptor=ParameterDescriptor(description='Distance threshold for passing through orange gate in meters'))
        
        # System parameters
        self.declare_parameter('system.control_loop_hz', 20.0,
                              descriptor=ParameterDescriptor(description='Control loop frequency in Hz'))
        self.declare_parameter('system.display_loop_hz', 30.0,
                              descriptor=ParameterDescriptor(description='Display loop frequency in Hz'))
        self.declare_parameter('system.enable_visualization', True,
                              descriptor=ParameterDescriptor(description='Enable OpenCV visualization window'))
        
        # CARLA connection parameters
        self.declare_parameter('carla.host', 'localhost',
                              descriptor=ParameterDescriptor(description='CARLA server host address'))
        self.declare_parameter('carla.port', 2000,
                              descriptor=ParameterDescriptor(description='CARLA server port number'))
        self.declare_parameter('carla.timeout', 10.0,
                              descriptor=ParameterDescriptor(description='CARLA connection timeout in seconds'))
        
        # Debug parameters
        self.declare_parameter('debug.enable_debug_output', True,
                              descriptor=ParameterDescriptor(description='Enable debug console output'))
        self.declare_parameter('debug.log_level', 'INFO',
                              descriptor=ParameterDescriptor(description='ROS logging level (DEBUG, INFO, WARN, ERROR)'))
        
        self.get_logger().info("✅ All ROS2 parameters declared successfully")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C for clean shutdown"""
        print("\nShutting down gracefully...")
        self.running = False
        
    def setup_carla(self):
        """Initialize CARLA connection and find existing vehicle"""
        try:
            # Connect to CARLA using parameters
            import carla
            host = self.get_parameter('carla.host').get_parameter_value().string_value
            port = self.get_parameter('carla.port').get_parameter_value().integer_value
            timeout = self.get_parameter('carla.timeout').get_parameter_value().double_value
            
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.get_logger().info(f"Connected to CARLA server at {host}:{port}")
            
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
                return None
            
            # Use the first vehicle found
            vehicle = vehicles[0]
            transform = vehicle.get_transform()
            location = transform.location
            
            self.get_logger().info(f"Found existing vehicle: {vehicle.type_id} (ID: {vehicle.id})")
            self.get_logger().info(f"Vehicle location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
            
            return vehicle
            
        except Exception as e:
            self.get_logger().error(f"Error finding existing vehicle: {str(e)}")
            return None
    
    def setup_camera_and_controller(self):
        """Setup ZED 2i camera and robust controller with YOLO model"""
        try:
            # Get camera parameters
            resolution_width = self.get_parameter('camera.resolution_width').get_parameter_value().integer_value
            resolution_height = self.get_parameter('camera.resolution_height').get_parameter_value().integer_value
            fps = self.get_parameter('camera.fps').get_parameter_value().integer_value
            
            # Setup camera with YOLO model
            self.camera = Zed2iCamera(
                world=self.world,
                vehicle=self.vehicle,
                resolution=(resolution_width, resolution_height),
                fps=fps,
                model_path=self.model_path  # Pass YOLO model path
            )
            
            if not self.camera.setup():
                raise RuntimeError("Failed to setup camera")
                
            # Setup robust controller with ROS2 node reference
            lookahead_distance = self.get_parameter('control.lookahead_distance').get_parameter_value().double_value
            self.controller = PurePursuitController(self.vehicle, self, lookahead_distance)
            self.get_logger().info("Camera with YOLO model and robust controller setup complete")
            self.get_logger().info(f"YOLO model loaded: {self.model_path}")
            self.get_logger().info("CONTROLLER NOW PUBLISHES TO ROS2 TOPICS")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error setting up camera and controller: {e}")
            return False
    
    def control_loop(self):
        """Main control loop for vehicle with YOLO detection"""
        self.get_logger().info("Starting control loop with YOLO detection...")
        
        # Get control loop frequency from parameters
        control_hz = self.get_parameter('system.control_loop_hz').get_parameter_value().double_value
        control_period = 1.0 / control_hz
        
        while self.running and rclpy.ok():
            try:
                # Process camera frame with YOLO detection
                self.camera.process_frame()
                
                # Get cone detections from ZED 2i camera (different format than before)
                cone_detections = getattr(self.camera, 'cone_detections', [])
                
                # Log detection count periodically
                if len(cone_detections) > 0:
                    self.get_logger().debug(f"ZED 2i detected {len(cone_detections)} cones")
                    # Debug: Print detection format
                    if len(cone_detections) > 0:
                        sample_detection = cone_detections[0]
                        self.get_logger().debug(f"Sample detection format: {sample_detection}")
                
                # Control vehicle using robust controller - NOW PUBLISHES TO ROS2
                steering, speed = self.controller.control_vehicle(cone_detections)
                
                time.sleep(control_period)
                
            except Exception as e:
                self.get_logger().error(f"Error in control loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def display_loop(self):
        """Display camera feed with detections"""
        self.get_logger().info("Starting display loop...")
        
        # Check if visualization is enabled
        enable_visualization = self.get_parameter('system.enable_visualization').get_parameter_value().bool_value
        if not enable_visualization:
            self.get_logger().info("Visualization disabled by parameter")
            return
        
        # Get display loop frequency from parameters
        display_hz = self.get_parameter('system.display_loop_hz').get_parameter_value().double_value
        display_period = 1.0 / display_hz
        
        while self.running and rclpy.ok():
            try:
                if hasattr(self.camera, 'rgb_image') and self.camera.rgb_image is not None:
                    # Create visualization
                    viz_image = self.create_visualization()
                    
                    cv2.imshow('CARLA Racing with ROS2 Output - ZED 2i Enhanced Controller', viz_image)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                time.sleep(display_period)
                
            except Exception as e:
                self.get_logger().error(f"Error in display loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def create_visualization(self):
        """Create visualization with YOLO detections and enhanced error handling"""
        try:
            if not hasattr(self.camera, 'rgb_image') or self.camera.rgb_image is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
                
            viz_image = self.camera.rgb_image.copy()
            
            # Add vehicle info at the top
            vehicle_info = f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id}) - ROS2 OUTPUT MODE - ZED 2i"
            cv2.putText(viz_image, vehicle_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process detections for visualization - ZED 2i format
            cone_detections = getattr(self.camera, 'cone_detections', [])
            if cone_detections:
                blue_cones, yellow_cones, orange_cones = self.controller.process_cone_detections(cone_detections)
                
                # The ZED 2i already draws detection boxes in its own visualization
                # Just add our additional tracking info
                
                # Try to find current target
                track_segment = self.controller.find_best_track_segment(blue_cones, yellow_cones)
                if not track_segment and (blue_cones or yellow_cones):
                    track_segment = self.controller.follow_cone_line(blue_cones, yellow_cones)
                
                # Draw target if found
                if track_segment:
                    # Draw target point on visualization
                    depth = track_segment.get('midpoint_y', 5.0)
                    angle = np.arctan2(track_segment.get('midpoint_x', 0), depth)
                    # Updated for ZED 2i FOV (110 degrees)
                    px = int(640 + (angle / np.radians(55)) * 640)  # 55 = 110/2
                    py = int(720 - 100 - depth * 25)
                    py = max(50, min(py, 720))
                    px = max(0, min(px, 1280))
                    
                    # Different colors for different navigation types
                    if track_segment.get('type') == 'cone_line':
                        color = (255, 0, 255)  # Magenta for cone line following
                        text = "CONE LINE"
                    else:
                        color = (0, 0, 255)    # Red for track segment
                        text = "TRACK TARGET"
                    
                    cv2.circle(viz_image, (px, py), 15, color, -1)
                    cv2.putText(viz_image, text, (px+20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add enhanced status text with ZED 2i info and visibility metrics
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Calculate cone visibility risk
            cone_detections = getattr(self.camera, 'cone_detections', [])
            blue_cones, yellow_cones, orange_cones = self.controller.process_cone_detections(cone_detections)
            
            visibility_risk = "LOW"
            if hasattr(self.controller, 'current_target_x'):
                lateral_offset = abs(self.controller.current_target_x)
                if lateral_offset > 2.5:
                    visibility_risk = "HIGH"
                elif lateral_offset > 1.8:
                    visibility_risk = "MODERATE"
            
            status_text = [
                f"Mode: ROS2 OUTPUT MODE (ZED 2i Enhanced)",
                f"YOLO Model: {self.model_path.split('/')[-1] if self.model_path else 'Not loaded'}",
                f"ZED 2i Detections: {len(cone_detections)} total",
                f"Blue: {len([d for d in cone_detections if d.get('cls') == 1])}, "
                f"Yellow: {len([d for d in cone_detections if d.get('cls') == 0])}, "
                f"Orange: {len([d for d in cone_detections if d.get('cls') == 2])}",
                f"Laps Completed: {self.controller.lap_counter.laps_completed}",
                f"Current Speed: {current_speed:.1f} m/s ({current_speed*3.6:.1f} km/h)",
                f"Distance: {self.controller.distance_traveled:.1f}m",
                f"Last Steering: {self.controller.last_steering:.3f}°",
                f"Cone Visibility Risk: {visibility_risk}",
                f"Lost Track: {self.controller.lost_track_counter}",
                f"Publishing to ROS2 Topics"
            ]
            
            for i, text in enumerate(status_text):
                y_pos = 50 + i*20  # Start lower to avoid vehicle info
                color = (0, 255, 0)
                if i == 0:  # Mode info
                    color = (255, 0, 255)  # Magenta for ROS2 mode
                elif i == 1:  # YOLO model info
                    color = (0, 255, 255)  # Cyan for YOLO
                elif i == 2 or i == 3:  # Detection info
                    color = (255, 255, 0)  # Yellow for detections
                elif i == 4:  # Lap counter
                    color = (0, 255, 0)  # Green for lap counter
                elif i == 8:  # Cone visibility risk
                    if "HIGH" in text:
                        color = (0, 0, 255)  # Red for high risk
                    elif "MODERATE" in text:
                        color = (0, 255, 255)  # Cyan for moderate risk
                    else:
                        color = (0, 255, 0)  # Green for low risk
                elif i == 9:  # Lost track counter
                    color = (255, 0, 0) if self.controller.lost_track_counter > 10 else (0, 255, 0)
                elif i == 10:  # ROS output indicator
                    color = (255, 255, 0)  # Yellow for ROS output
                    
                cv2.putText(viz_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return viz_image
            
        except Exception as e:
            self.get_logger().error(f"Error creating visualization: {e}")
            # Return a blank image if visualization fails
            blank_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_image, "Visualization Error", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank_image
    
    def run_system(self):
        """Main execution function"""
        try:
            # Setup CARLA
            if not self.setup_carla():
                return False
            
            # Setup camera and controller with YOLO
            if not self.setup_camera_and_controller():
                return False
            
            # Log all parameter values for debugging
            self.log_parameter_values()
            
            self.get_logger().info("System ready! Using existing vehicle for racing with ROS2 output.")
            self.get_logger().info(f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
            self.get_logger().info(f"YOLO model: {self.model_path}")
            self.get_logger().info("🟠 Orange cones will be detected for lap counting")
            self.get_logger().info("🔍 ZED 2i YOLO model running for cone detection")
            self.get_logger().info("🚗 Enhanced steering with cone visibility preservation")
            self.get_logger().info("🎯 Advanced lost track recovery patterns")
            self.get_logger().info("🤖 Controller publishes to ROS2 topics:")
            self.get_logger().info("   - /planning/target_speed (std_msgs/Float64)")
            self.get_logger().info("   - /planning/target_position (geometry_msgs/Point)")
            self.get_logger().info("   - /planning/reference_steering (std_msgs/Float64)")
            self.get_logger().info("   - /planning/emergency_stop (std_msgs/Bool)")
            self.get_logger().info("   - /planning/lap_count (std_msgs/Int32)")
            
            # Start threads
            self.control_thread = threading.Thread(target=self.control_loop)
            self.display_thread = threading.Thread(target=self.display_loop)
            
            self.control_thread.start()
            self.display_thread.start()
            
            # ROS2 spin in main thread
            rclpy.spin(self)
            
            # Wait for threads to complete
            self.control_thread.join()
            self.display_thread.join()
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error running system: {e}")
            return False
        finally:
            self.cleanup()
    
    def log_parameter_values(self):
        """Log all parameter values for debugging"""
        self.get_logger().info("📋 Current parameter values:")
        
        # Get all parameters
        param_names = [
            'vehicle.wheelbase', 'vehicle.max_steering_degrees',
            'control.max_speed', 'control.min_speed', 'control.lookahead_distance',
            'camera.fov_horizontal', 'camera.resolution_width', 'camera.resolution_height', 'camera.fps',
            'detection.safety_offset', 'detection.max_depth', 'detection.min_depth', 'detection.max_lateral_distance',
            'turns.min_turn_radius', 'turns.path_widening_factor', 'turns.sharp_turn_threshold', 'turns.u_turn_threshold', 'turns.turn_detection_distance',
            'track.width_min', 'track.width_max', 'track.max_depth_diff', 'track.max_lateral_jump', 'track.forward_focus_angle',
            'visibility.high_risk_threshold', 'visibility.moderate_risk_threshold', 'visibility.visibility_steering_boost', 'visibility.high_risk_factor', 'visibility.moderate_risk_factor',
            'recovery.max_lost_track_frames', 'recovery.recovery_steering_multiplier', 'recovery.aggressive_search_amplitude', 'recovery.wide_search_amplitude',
            'lap_counter.cooldown_duration', 'lap_counter.orange_gate_threshold',
            'system.control_loop_hz', 'system.display_loop_hz', 'system.enable_visualization',
            'carla.host', 'carla.port', 'carla.timeout',
            'debug.enable_debug_output', 'debug.log_level'
        ]
        
        for param_name in param_names:
            try:
                param = self.get_parameter(param_name)
                self.get_logger().info(f"  {param_name}: {param.value}")
            except Exception as e:
                self.get_logger().warn(f"  {param_name}: Failed to get value - {e}")
    
    def cleanup(self):
        """Clean up all resources"""
        self.get_logger().info("Cleaning up resources...")
        
        self.running = False
        
        # Publish final emergency stop
        if self.controller and hasattr(self.controller, 'publish_emergency_stop'):
            self.controller.publish_emergency_stop()
            self.get_logger().info("Published final emergency stop to ROS2 topics")
        
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
        
        self.get_logger().info("Cleanup complete - ROS2 topics will stop publishing")

def main(args=None):
    """Main function for ROS2"""
    rclpy.init(args=args)
    
    # Default YOLO model path
    model_path = '/home/legion5/hydrakon_ws/src/planning_module/planning_module/best.pt'
    
    print(f"Using YOLO model: {model_path}")
    
    racing_system = CarlaRacingSystemROS2(model_path=model_path)
    
    try:
        success = racing_system.run_system()
        if success:
            racing_system.get_logger().info("Racing system with ROS2 output completed successfully")
        else:
            racing_system.get_logger().error("Racing system with ROS2 output failed to start")
    except KeyboardInterrupt:
        racing_system.get_logger().info("Received interrupt signal")
    except Exception as e:
        racing_system.get_logger().error(f"Unexpected error: {e}")
    finally:
        racing_system.cleanup()
        rclpy.shutdown()

if __name__ == "__main__":
    main()