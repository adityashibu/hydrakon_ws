import carla
import numpy as np
import cv2
import time
import threading
import signal
import sys
import csv
import os
from datetime import datetime
from collections import deque
from .zed_2i import Zed2iCamera

class SpeedLogger:
    def __init__(self, log_dir="speed_logs"):
        self.log_dir = log_dir
        self.log_file = None
        self.csv_writer = None
        self.session_start_time = None
        self.speed_history = deque(maxlen=100)  # Keep last 100 speed readings
        self.distance_traveled = 0.0
        self.last_position = None
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.init_log_file()
    
    def init_log_file(self):
        """Initialize CSV log file with headers"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"speed_log_{timestamp}.csv"
        self.log_file_path = os.path.join(self.log_dir, log_filename)
        
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        
        # Write CSV headers
        headers = [
            'timestamp',
            'elapsed_time',
            'speed_mps',
            'speed_kmh',
            'speed_mph',
            'target_speed',
            'throttle',
            'brake',
            'steering',
            'position_x',
            'position_y',
            'position_z',
            'distance_traveled',
            'gates_completed',
            'navigation_type'
        ]
        self.csv_writer.writerow(headers)
        self.log_file.flush()
        
        self.session_start_time = time.time()
        print(f"Speed logger initialized: {self.log_file_path}")
    
    def log_speed_data(self, vehicle, target_speed, throttle, brake, steering, 
                      gates_completed, navigation_type="unknown"):
        """Log current speed and vehicle data"""
        try:
            # Get current time
            current_time = time.time()
            elapsed_time = current_time - self.session_start_time
            
            # Get vehicle velocity and position
            velocity = vehicle.get_velocity()
            transform = vehicle.get_transform()
            location = transform.location
            
            # Calculate current speed (m/s)
            current_speed_mps = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Convert to other units
            current_speed_kmh = current_speed_mps * 3.6
            current_speed_mph = current_speed_mps * 2.237
            
            # Calculate distance traveled
            if self.last_position is not None:
                distance_delta = np.sqrt(
                    (location.x - self.last_position.x)**2 + 
                    (location.y - self.last_position.y)**2 + 
                    (location.z - self.last_position.z)**2
                )
                self.distance_traveled += distance_delta
            
            self.last_position = location
            
            # Add to speed history
            self.speed_history.append(current_speed_mps)
            
            # Write to CSV
            row_data = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # timestamp with milliseconds
                f"{elapsed_time:.3f}",
                f"{current_speed_mps:.3f}",
                f"{current_speed_kmh:.3f}",
                f"{current_speed_mph:.3f}",
                f"{target_speed:.3f}",
                f"{throttle:.3f}",
                f"{brake:.3f}",
                f"{steering:.3f}",
                f"{location.x:.3f}",
                f"{location.y:.3f}",
                f"{location.z:.3f}",
                f"{self.distance_traveled:.3f}",
                gates_completed,
                navigation_type
            ]
            
            self.csv_writer.writerow(row_data)
            self.log_file.flush()  # Ensure data is written immediately
            
        except Exception as e:
            print(f"Error logging speed data: {e}")
    
    def get_speed_stats(self):
        """Get current speed statistics"""
        if not self.speed_history:
            return {
                'current': 0.0,
                'average': 0.0,
                'max': 0.0,
                'min': 0.0
            }
        
        speeds = list(self.speed_history)
        return {
            'current': speeds[-1],
            'average': np.mean(speeds),
            'max': np.max(speeds),
            'min': np.min(speeds)
        }
    
    def close(self):
        """Close the log file and print summary"""
        if self.log_file:
            try:
                # Write summary at the end
                stats = self.get_speed_stats()
                elapsed_time = time.time() - self.session_start_time
                
                print(f"\n{'='*50}")
                print("SPEED LOGGING SESSION SUMMARY")
                print(f"{'='*50}")
                print(f"Log file: {self.log_file_path}")
                print(f"Session duration: {elapsed_time:.1f} seconds")
                print(f"Distance traveled: {self.distance_traveled:.2f} meters")
                print(f"Average speed: {stats['average']:.2f} m/s ({stats['average']*3.6:.2f} km/h)")
                print(f"Maximum speed: {stats['max']:.2f} m/s ({stats['max']*3.6:.2f} km/h)")
                print(f"Minimum speed: {stats['min']:.2f} m/s ({stats['min']*3.6:.2f} km/h)")
                print(f"Speed readings: {len(self.speed_history)}")
                print(f"{'='*50}")
                
                self.log_file.close()
                self.log_file = None
                
            except Exception as e:
                print(f"Error closing speed logger: {e}")


class PurePursuitController:
    def __init__(self, vehicle, lookahead_distance=4.0):
        self.vehicle = vehicle
        self.lookahead_distance = lookahead_distance
        
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
        
        # State tracking
        self.last_steering = 0.0
        self.steering_history = deque(maxlen=5)
        self.all_gates_completed = False
        
        # NEW: Turn state tracking
        self.current_turn_type = "straight"  # "straight", "gentle", "sharp", "u_turn"
        self.turn_direction = "none"  # "left", "right", "none"
        self.path_offset = 0.0  # Current path offset for wider turns
        self.gate_sequence = deque(maxlen=5)  # Track recent gates for turn prediction
        
        # Gate tracking
        self.gates_completed = 0
        self.target_gates = 3  # Number of gates to complete
        self.current_gate = None
        self.previous_gate = None
        self.gate_passed_threshold = 2.0  # Distance threshold for passing through gate
        self.cooldown_counter = 0
        self.cooldown_duration = 15  # Frames to wait after passing a gate
        
        # Much more restrictive track following parameters
        self.track_width_min = 3.0  # Minimum track width (meters)
        self.track_width_max = 5.0  # Maximum track width (meters) - reduced
        self.max_depth_diff = 1.0   # Maximum depth difference between gate cones - reduced
        self.max_lateral_jump = 1.5 # Maximum lateral movement between consecutive gates - reduced
        self.forward_focus_angle = 30.0  # Only consider cones within this angle (degrees) from vehicle heading
        
        # Gate validation - simplified
        self.gate_history = deque(maxlen=3)
        self.required_consecutive_gates = 2  # Reduced for faster response
        self.consecutive_valid_gates = 0
        
        # Backup navigation when no gates found
        self.lost_track_counter = 0
        self.max_lost_track_frames = 20
        
        # Initialize speed logger
        self.speed_logger = SpeedLogger()
    
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
        
        # Speed reduction when approaching gates
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
            return [], []
            
        blue_cones = []    # Class 1 - LEFT side
        yellow_cones = []  # Class 0 - RIGHT side
        
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
                
                # Filter by depth range - much more restrictive
                if depth < self.min_depth or depth > self.max_depth:
                    continue
                    
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to world coordinates
                world_x, world_y = self.image_to_world_coords(center_x, center_y, depth)
                
                # CRITICAL: Filter by lateral distance - only consider cones near the vehicle path
                if abs(world_x) > self.max_lateral_distance:
                    continue
                
                # CRITICAL: Filter by forward focus angle - only consider cones roughly ahead
                angle_to_cone = np.degrees(abs(np.arctan2(world_x, world_y)))
                if angle_to_cone > self.forward_focus_angle:
                    continue
                
                # Only consider cones that are reasonably positioned for track boundaries
                # Blue cones should be on the left (negative x), yellow on the right (positive x)
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
                    
        except Exception as e:
            print(f"ERROR processing cone detections: {e}")
            return [], []
        
        # Sort by depth (closest first), then by angle (most centered first)
        blue_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        yellow_cones.sort(key=lambda c: (c['depth'], c['angle_from_center']))
        
        # Debug filtered cones
        print(f"DEBUG: After spatial filtering - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}")
        if blue_cones:
            closest_blue = blue_cones[0]
            print(f"  Closest blue: x={closest_blue['x']:.2f}, y={closest_blue['y']:.2f}, angle={closest_blue['angle_from_center']:.1f}°")
        if yellow_cones:
            closest_yellow = yellow_cones[0]
            print(f"  Closest yellow: x={closest_yellow['x']:.2f}, y={closest_yellow['y']:.2f}, angle={closest_yellow['angle_from_center']:.1f}°")
        
        return blue_cones, yellow_cones
    
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
    
    def find_best_gate(self, blue_cones, yellow_cones):
        """Find the best immediate gate with strict proximity focus"""
        if not blue_cones or not yellow_cones:
            print(f"DEBUG: Cannot form gate - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}")
            return None
            
        print(f"DEBUG: Finding immediate gate from {len(blue_cones)} blue and {len(yellow_cones)} yellow cones")
        
        # Only consider the closest 3 cones of each color to focus on immediate track
        blue_candidates = blue_cones[:3]
        yellow_candidates = yellow_cones[:3]
        
        valid_gates = []
        
        try:
            for blue in blue_candidates:
                for yellow in yellow_candidates:
                    
                    if not self.is_valid_track_segment(blue, yellow):
                        continue
                    
                    # Create gate
                    gate = {
                        'blue': blue,
                        'yellow': yellow,
                        'midpoint_x': (blue['x'] + yellow['x']) / 2,
                        'midpoint_y': (blue['y'] + yellow['y']) / 2,
                        'width': abs(blue['x'] - yellow['x']),
                        'avg_depth': (blue['depth'] + yellow['depth']) / 2,
                        'confidence': (blue.get('confidence', 1.0) + yellow.get('confidence', 1.0)) / 2
                    }
                    
                    # Additional validation: gate should be roughly centered in front of vehicle
                    if abs(gate['midpoint_x']) < 2.0:  # Gate center within 2m of vehicle centerline
                        valid_gates.append(gate)
            
            if not valid_gates:
                print("DEBUG: No valid immediate gates found")
                return None
            
            # Sort by distance (closest first), heavily prioritize centerline alignment
            def gate_score(g):
                distance_score = g['avg_depth']
                centerline_score = abs(g['midpoint_x']) * 3.0  # Heavy penalty for off-center gates
                return distance_score + centerline_score
            
            valid_gates.sort(key=gate_score)
            best_gate = valid_gates[0]
            
            print(f"DEBUG: Found immediate track gate:")
            print(f"  Blue cone (LEFT):  x={best_gate['blue']['x']:6.2f}, y={best_gate['blue']['y']:6.2f}")
            print(f"  Yellow cone (RIGHT): x={best_gate['yellow']['x']:6.2f}, y={best_gate['yellow']['y']:6.2f}")
            print(f"  Gate midpoint: x={best_gate['midpoint_x']:6.2f}, y={best_gate['midpoint_y']:6.2f}")
            print(f"  Gate width: {best_gate['width']:.2f}m, Average depth: {best_gate['avg_depth']:.2f}m")
            print(f"  Centerline offset: {abs(best_gate['midpoint_x']):.2f}m")
            
            return best_gate
            
        except Exception as e:
            print(f"ERROR in gate finding: {e}")
            return None
    
    def follow_cone_line(self, blue_cones, yellow_cones):
        """Fallback: follow immediate cones when no gates can be formed"""
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
                print(f"DEBUG: High visibility risk - increasing steering aggressiveness by 40%")
            elif lateral_offset > 1.8:  # Moderate risk
                visibility_risk_factor = 1.2
                print(f"DEBUG: Moderate visibility risk - increasing steering aggressiveness by 20%")
            
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
                max_change = 0.25
                print(f"DEBUG: High visibility risk - allowing max steering change: {max_change}")
            elif lateral_offset > 1.8:
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
    
    def control_vehicle(self, cone_detections):
        """Main control function with robust error handling and enhanced cone visibility preservation"""
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG: CONTROL CYCLE - {len(cone_detections) if cone_detections else 0} detections")
            print(f"Gates completed: {self.gates_completed}/{self.target_gates}")
            print(f"All gates completed: {self.all_gates_completed}")
            print(f"Current turn type: {self.current_turn_type}, Direction: {self.turn_direction}")
            print(f"Cooldown counter: {self.cooldown_counter}")
            print(f"Lost track counter: {self.lost_track_counter}")
            print(f"{'='*60}")
            
            # If all gates completed, stop the vehicle
            if self.all_gates_completed:
                print("DEBUG: All gates completed - stopping vehicle")
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                self.vehicle.apply_control(control)
                return 0.0, 0.0
            
            # Handle cooldown period after passing a gate
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                print(f"DEBUG: Cooldown period - moving forward slowly ({self.cooldown_counter} frames remaining)")
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.12
                control.brake = 0.0
                self.vehicle.apply_control(control)
                
                # Reset gate validation during cooldown
                if self.cooldown_counter == 0:
                    self.gate_history.clear()
                    self.consecutive_valid_gates = 0
                    self.current_gate = None
                    self.previous_gate = None
                    self.lost_track_counter = 0
                    self.current_turn_type = "straight"
                    self.turn_direction = "none"
                    print("DEBUG: Cooldown complete - ready for next gate")
                
                return 0.0, 0.12
            
            # Process cone detections
            blue_cones, yellow_cones = self.process_cone_detections(cone_detections)
            print(f"DEBUG: Processed cones - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}")
            
            # NEW: Enhanced lost track detection with immediate recovery steering
            if len(blue_cones) == 0 and len(yellow_cones) == 0:
                self.lost_track_counter += 1
                print(f"DEBUG: NO CONES DETECTED - lost track for {self.lost_track_counter} frames")
                
                # Immediate aggressive steering to try to find cones again
                if self.lost_track_counter <= 10:
                    # Try to steer in the direction we were last going
                    recovery_steering = self.last_steering * 1.5  # Amplify last steering
                    recovery_steering = np.clip(recovery_steering, -0.8, 0.8)
                    print(f"DEBUG: Applying recovery steering: {recovery_steering:.3f}")
                    
                    control = carla.VehicleControl()
                    control.steer = recovery_steering
                    control.throttle = 0.2  # Slower while recovering
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    return recovery_steering, 0.2
                elif self.lost_track_counter <= 20:
                    # More aggressive search pattern
                    search_steering = 0.6 * np.sin(self.lost_track_counter * 0.3)
                    print(f"DEBUG: Applying aggressive search pattern: {search_steering:.3f}")
                    
                    control = carla.VehicleControl()
                    control.steer = search_steering
                    control.throttle = 0.15
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    return search_steering, 0.15
                else:
                    # Last resort - wide search
                    search_steering = 0.8 * np.sin(self.lost_track_counter * 0.2)
                    control = carla.VehicleControl()
                    control.steer = search_steering
                    control.throttle = 0.1
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    return search_steering, 0.1
            
            # Try to find a gate
            gate = self.find_best_gate(blue_cones, yellow_cones)
            
            # If no gate found, try cone line following
            if not gate and (blue_cones or yellow_cones):
                gate = self.follow_cone_line(blue_cones, yellow_cones)
                print("DEBUG: Using cone line following")
            
            if not gate:
                self.lost_track_counter += 1
                print(f"DEBUG: No navigation target found - lost track for {self.lost_track_counter} frames")
                
                # If lost for too long, implement search pattern
                if self.lost_track_counter > self.max_lost_track_frames:
                    print("DEBUG: Lost track for too long - implementing search pattern")
                    search_steering = 0.3 * np.sin(self.lost_track_counter * 0.1)  # Gentle search pattern
                    control = carla.VehicleControl()
                    control.steer = search_steering
                    control.throttle = 0.15
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    return search_steering, 0.15
                else:
                    # Move forward slowly while searching
                    control = carla.VehicleControl()
                    control.steer = self.last_steering * 0.5
                    control.throttle = 0.15
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    return self.last_steering * 0.5, 0.15
            
            # Reset lost track counter if we found something
            self.lost_track_counter = 0
            
            # Detect turn type and calculate path widening
            turn_type, turn_direction, path_offset = self.detect_turn_type(gate, blue_cones, yellow_cones)
            self.current_turn_type = turn_type
            self.turn_direction = turn_direction
            self.path_offset = path_offset
            
            # Update current gate
            self.current_gate = gate
            
            # Check if we've passed through the current gate
            current_depth = gate['avg_depth']
            if current_depth < self.gate_passed_threshold and gate.get('type') != 'cone_line':
                self.gates_completed += 1
                print(f"DEBUG: 🎉 GATE {self.gates_completed} PASSED! 🎉")
                print(f"DEBUG: Total gates completed: {self.gates_completed}/{self.target_gates}")
                
                # Check if all gates are completed
                if self.gates_completed >= self.target_gates:
                    print(f"DEBUG: 🏁 ALL {self.target_gates} GATES COMPLETED! MISSION ACCOMPLISHED! 🏁")
                    self.all_gates_completed = True
                    control = carla.VehicleControl()
                    control.steer = 0.0
                    control.throttle = 0.0
                    control.brake = 1.0
                    self.vehicle.apply_control(control)
                    return 0.0, 0.0
                else:
                    print(f"DEBUG: Looking for next gate... ({self.target_gates - self.gates_completed} remaining)")
                    self.cooldown_counter = self.cooldown_duration
                    
                    # Move forward during cooldown
                    control = carla.VehicleControl()
                    control.steer = 0.0
                    control.throttle = 0.12
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    return 0.0, 0.12
            
            # Adjust target point for wider turns
            original_target_x = gate['midpoint_x']
            original_target_y = gate['midpoint_y']
            
            # NEW: Store current target for visibility calculations
            self.current_target_x = original_target_x
            
            adjusted_target_x, adjusted_target_y = self.adjust_target_for_turn(
                original_target_x, original_target_y, turn_type, turn_direction, path_offset
            )
            
            # Navigate towards the adjusted target
            raw_steering = self.calculate_pure_pursuit_steering(adjusted_target_x, adjusted_target_y)
            smooth_steering = self.smooth_steering(raw_steering)
            
            # Calculate adaptive speed based on turn type
            target_speed = self.calculate_adaptive_speed(turn_type, smooth_steering, current_depth)
            
            # Get current speed
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Apply control
            control = carla.VehicleControl()
            control.steer = float(smooth_steering)
            
            speed_diff = target_speed - current_speed
            if speed_diff > 0.5:
                control.throttle = min(0.5, 0.2 + 0.3 * (speed_diff / self.max_speed))
                control.brake = 0.0
            elif speed_diff < -0.5:
                control.throttle = 0.0
                control.brake = min(0.4, 0.2 * abs(speed_diff) / self.max_speed)
            else:
                control.throttle = 0.3
                control.brake = 0.0
            
            self.vehicle.apply_control(control)
            
            # Log speed data
            navigation_type = f"{gate.get('type', 'gate')}_{turn_type}"
            self.speed_logger.log_speed_data(
                self.vehicle, target_speed, control.throttle, control.brake, 
                smooth_steering, self.gates_completed, navigation_type
            )
            
            # Enhanced debug output
            direction = 'LEFT' if smooth_steering > 0 else 'RIGHT' if smooth_steering < 0 else 'STRAIGHT'
            speed_stats = self.speed_logger.get_speed_stats()
            
            print(f"DEBUG: APPLIED CONTROL:")
            print(f"  Navigation: {navigation_type}")
            print(f"  Turn Analysis: {turn_type}-{turn_direction} (offset: {path_offset:.2f}m)")
            print(f"  Original target: ({original_target_x:.2f}, {original_target_y:.2f})")
            print(f"  Adjusted target: ({adjusted_target_x:.2f}, {adjusted_target_y:.2f})")
            print(f"  Target distance: {current_depth:.2f}m")
            print(f"  Turn radius: {self.calculate_turn_radius(np.radians(smooth_steering * 30)):.2f}m")
            print(f"  Steering: {smooth_steering:.3f} ({direction})")
            print(f"  Cone visibility risk: {'HIGH' if abs(original_target_x) > 2.5 else 'MODERATE' if abs(original_target_x) > 1.8 else 'LOW'}")
            print(f"  Throttle: {control.throttle:.2f}")
            print(f"  Brake: {control.brake:.2f}")
            print(f"  Current Speed: {speed_stats['current']:.1f} m/s ({speed_stats['current']*3.6:.1f} km/h)")
            print(f"  Target Speed: {target_speed:.1f} m/s ({target_speed*3.6:.1f} km/h)")
            print(f"  Avg Speed: {speed_stats['average']:.1f} m/s")
            print(f"  Distance: {self.speed_logger.distance_traveled:.1f}m")
            print(f"{'='*60}\n")
            
            return smooth_steering, target_speed
            
        except Exception as e:
            print(f"ERROR in vehicle control: {e}")
            import traceback
            traceback.print_exc()
            # Safe fallback - also log the emergency stop
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.5
            self.vehicle.apply_control(control)
            
            # Log emergency stop
            self.speed_logger.log_speed_data(
                self.vehicle, 0.0, 0.0, 0.5, 0.0, 
                self.gates_completed, "emergency_stop"
            )
            
            return 0.0, 0.0
    
    def cleanup(self):
        """Clean up controller resources"""
        if hasattr(self, 'speed_logger'):
            self.speed_logger.close()

class CarlaGateRacingSystem:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.controller = None
        self.running = True
        
        # Threading
        self.control_thread = None
        self.display_thread = None
        
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C for clean shutdown"""
        print("\nShutting down gracefully...")
        self.running = False
        
    def setup_carla(self):
        """Initialize CARLA connection and spawn vehicle"""
        try:
            # Connect to CARLA
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            print("Connected to CARLA server")
            
            # Get world
            self.world = self.client.get_world()
            
            # Spawn vehicle using your specific method
            self.vehicle = self.spawn_vehicle()
            if not self.vehicle:
                raise RuntimeError("Failed to spawn vehicle")
            
            return True
            
        except Exception as e:
            print(f"Error setting up CARLA: {e}")
            return False
    
    def spawn_vehicle(self):
        """Spawn a vehicle in the CARLA world using your exact method."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.*')[2]
            if not vehicle_bp:
                print("ERROR: No vehicle blueprints found")
                return None
                
            print(f"Using vehicle blueprint: {vehicle_bp.id}")
            
            spawn_transform = carla.Transform(
                carla.Location(x=170.0, y=0.0, z=2.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
            print(f"Vehicle spawned at {spawn_transform.location}")
            
            time.sleep(2.0)
            if not vehicle.is_alive:
                print("ERROR: Vehicle failed to spawn or is not alive")
                return None
            return vehicle
        except Exception as e:
            print(f"Error spawning vehicle: {str(e)}")
            return None
    
    def setup_camera_and_controller(self, model_path=None):
        """Setup ZED 2i camera and robust controller"""
        try:
            # Setup camera
            self.camera = Zed2iCamera(
                world=self.world,
                vehicle=self.vehicle,
                resolution=(1280, 720),
                fps=30,
                model_path=model_path
            )
            
            if not self.camera.setup():
                raise RuntimeError("Failed to setup camera")
                
            # Setup robust controller
            self.controller = PurePursuitController(self.vehicle)
            print("Camera and robust controller setup complete")
            print(f"Speed logging enabled - logs will be saved to: speed_logs/")
            
            return True
            
        except Exception as e:
            print(f"Error setting up camera and controller: {e}")
            return False
    
    def control_loop(self):
        """Main control loop for vehicle"""
        print("Starting control loop...")
        
        while self.running:
            try:
                # Process camera frame
                self.camera.process_frame()
                
                # Get cone detections
                cone_detections = getattr(self.camera, 'cone_detections', [])
                
                # Control vehicle using robust controller
                steering, speed = self.controller.control_vehicle(cone_detections)
                
                # If all gates completed, stop the system
                if self.controller.all_gates_completed:
                    print(f"All {self.controller.target_gates} gates completed! Stopping system in 5 seconds...")
                    time.sleep(5.0)
                    self.running = False
                    break
                
                time.sleep(0.05)  # 20 Hz control loop
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Brief pause before retrying
    
    def display_loop(self):
        """Display camera feed with detections"""
        print("Starting display loop...")
        
        while self.running:
            try:
                if hasattr(self.camera, 'rgb_image') and self.camera.rgb_image is not None:
                    # Create visualization
                    viz_image = self.create_visualization()
                    
                    cv2.imshow('CARLA Gate Racing - Robust Controller', viz_image)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                time.sleep(0.033)  # ~30 FPS display
                
            except Exception as e:
                print(f"Error in display loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def create_visualization(self):
        """Create visualization with enhanced error handling"""
        try:
            if not hasattr(self.camera, 'rgb_image') or self.camera.rgb_image is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
                
            viz_image = self.camera.rgb_image.copy()
            
            # Process detections for visualization
            cone_detections = getattr(self.camera, 'cone_detections', [])
            if cone_detections and not self.controller.all_gates_completed:
                blue_cones, yellow_cones = self.controller.process_cone_detections(cone_detections)
                
                # Try to find current target
                gate = self.controller.find_best_gate(blue_cones, yellow_cones)
                if not gate and (blue_cones or yellow_cones):
                    gate = self.controller.follow_cone_line(blue_cones, yellow_cones)
                
                # Draw target if found
                if gate:
                    # Draw target point
                    depth = gate.get('midpoint_y', 5.0)
                    angle = np.arctan2(gate.get('midpoint_x', 0), depth)
                    px = int(640 + (angle / np.radians(45)) * 640)
                    py = int(720 - 100 - depth * 25)
                    py = max(50, min(py, 720))
                    px = max(0, min(px, 1280))
                    
                    # Different colors for different navigation types
                    if gate.get('type') == 'cone_line':
                        color = (255, 0, 255)  # Magenta for cone line following
                        text = "CONE LINE"
                    else:
                        color = (0, 0, 255)    # Red for gate
                        text = "GATE TARGET"
                    
                    cv2.circle(viz_image, (px, py), 15, color, -1)
                    cv2.putText(viz_image, text, (px+20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw gate line if it's a proper gate
                    if gate.get('type') != 'cone_line' and 'blue' in gate and 'yellow' in gate:
                        blue_cone = gate['blue']
                        yellow_cone = gate['yellow']
                        
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
                        
                        # Draw gate line and cones
                        cv2.line(viz_image, (blue_px, blue_py), (yellow_px, yellow_py), (0, 255, 0), 4)
                        cv2.circle(viz_image, (blue_px, blue_py), 8, (255, 0, 0), -1)
                        cv2.circle(viz_image, (yellow_px, yellow_py), 8, (0, 255, 255), -1)
            
            # Add enhanced status text with speed information
            speed_stats = self.controller.speed_logger.get_speed_stats()
            status_text = [
                f"Gates Completed: {self.controller.gates_completed}/{self.controller.target_gates}",
                f"Mission Status: {'COMPLETED' if self.controller.all_gates_completed else 'IN PROGRESS'}",
                f"Blue Cones: {len([d for d in cone_detections if d.get('cls') == 1])}",
                f"Yellow Cones: {len([d for d in cone_detections if d.get('cls') == 0])}",
                f"Current Speed: {speed_stats['current']:.1f} m/s ({speed_stats['current']*3.6:.1f} km/h)",
                f"Avg Speed: {speed_stats['average']:.1f} m/s ({speed_stats['average']*3.6:.1f} km/h)",
                f"Max Speed: {speed_stats['max']:.1f} m/s ({speed_stats['max']*3.6:.1f} km/h)",
                f"Distance: {self.controller.speed_logger.distance_traveled:.1f}m",
                f"Steering: {self.controller.last_steering:.3f}",
                f"Lost Track: {self.controller.lost_track_counter}",
                f"Cooldown: {self.controller.cooldown_counter if self.controller.cooldown_counter > 0 else 'None'}"
            ]
            
            for i, text in enumerate(status_text):
                color = (0, 255, 0) if not self.controller.all_gates_completed else (0, 255, 255)
                if i == 0:  # Gate counter
                    color = (0, 255, 255) if self.controller.all_gates_completed else (255, 255, 0)
                elif i == 4:  # Current speed
                    color = (0, 255, 255)  # Cyan for current speed
                elif i == 5:  # Average speed
                    color = (255, 255, 0)  # Yellow for average speed
                elif i == 6:  # Max speed
                    color = (255, 0, 255)  # Magenta for max speed
                elif i == 9:  # Lost track counter
                    color = (255, 0, 0) if self.controller.lost_track_counter > 10 else (0, 255, 0)
                cv2.putText(viz_image, text, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return viz_image
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            # Return a blank image if visualization fails
            blank_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_image, "Visualization Error", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank_image
    
    def run(self, model_path=None):
        """Main execution function"""
        try:
            # Setup CARLA
            if not self.setup_carla():
                return False
            
            # Setup camera and controller
            if not self.setup_camera_and_controller(model_path):
                return False
            
            print("System ready! The vehicle will drive through multiple gates using robust control.")
            print(f"Target: {self.controller.target_gates} gates")
            print("Press Ctrl+C to stop or 'q' in the display window")
            
            # Start threads
            self.control_thread = threading.Thread(target=self.control_loop)
            self.display_thread = threading.Thread(target=self.display_loop)
            
            self.control_thread.start()
            self.display_thread.start()
            
            # Wait for threads to complete
            self.control_thread.join()
            self.display_thread.join()
            
            return True
            
        except Exception as e:
            print(f"Error running system: {e}")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up resources...")
        
        self.running = False
        
        # Stop vehicle
        if self.vehicle:
            try:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                self.vehicle.apply_control(control)
                time.sleep(1.0)
            except:
                pass
        
        # Cleanup controller (including speed logger)
        if self.controller:
            try:
                self.controller.cleanup()
            except:
                pass
        
        # Cleanup camera
        if self.camera:
            try:
                self.camera.shutdown()
            except:
                pass
        
        # Destroy vehicle
        if self.vehicle:
            try:
                self.vehicle.destroy()
                print("Vehicle destroyed")
            except:
                pass
        
        # Close CV2 windows
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("Cleanup complete")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        model_path = '/home/aditya/hydrakon_ws/src/planning_module/planning_module/best.pt'
        print(f"Using YOLO model: {model_path}")
    else:
        # Default model path based on your structure
        model_path = '/home/aditya/hydrakon_ws/src/planning_module/planning_module/best.pt'
        print(f"Using default YOLO model: {model_path}")
    
    # Create and run the robust racing system
    racing_system = CarlaGateRacingSystem()
    
    try:
        success = racing_system.run(model_path)
        if success:
            print("Robust gate racing system completed successfully")
        else:
            print("Robust gate racing system failed to start")
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        racing_system.cleanup()


if __name__ == "__main__":
    main()