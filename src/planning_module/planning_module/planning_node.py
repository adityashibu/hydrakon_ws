# =============================================================================
# RACING SYSTEM CONFIGURATION - CHANGE THESE VALUES
# =============================================================================
target_laps = 10  # Set to None for unlimited laps, or any number (e.g., 1, 3, 5, 10)

# Lap validation parameters
MIN_LAP_TIME = 150.0  # Minimum seconds for a lap to be considered valid (default: 3.0)

# Vehicle speed parameters  
MIN_SPEED = 2.0   # Minimum vehicle speed in m/s (default: 70.0)
MAX_SPEED = 3.0  # Maximum vehicle speed in m/s (default: 100.0)

# Acceleration
# MIN_SPEED = 100.0   # Minimum vehicle speed in m/s (default: 70.0)
# MAX_SPEED = 150.0  # Maximum vehicle speed in m/s (default: 100.0)

# Orange cone detection parameters
ORANGE_GATE_THRESHOLD = 2.0  # Distance threshold for orange gate passage in meters (default: 2.0)
ORANGE_COOLDOWN = 3.0        # Cooldown between orange gate detections in seconds (default: 3.0)
# =============================================================================

import carla
import numpy as np
import cv2
import time
import threading
import signal
import os
from collections import deque

# Set matplotlib backend before importing pyplot to avoid Qt issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .zed_2i import Zed2iCamera

class LapCounter:
    def __init__(self, target_laps=None):
        self.laps_completed = 0
        self.last_orange_gate_time = 0
        self.cooldown_duration = ORANGE_COOLDOWN  # Use configurable parameter
        self.orange_gate_passed_threshold = ORANGE_GATE_THRESHOLD  # Use configurable parameter
        
        # Target laps functionality
        self.target_laps = target_laps
        self.target_reached = False
        
        # Lap timing functionality
        self.race_start_time = time.time()
        self.lap_start_time = time.time()
        self.lap_times = []  # Store individual lap times
        self.current_lap_time = 0.0
        self.best_lap_time = float('inf')
        self.last_lap_time = 0.0
        
        # Turn tracking for each lap
        self.current_lap_turns = {
            'straight': 0,
            'gentle': 0,
            'sharp': 0
        }
        self.lap_turn_data = []  # Store turn counts for each completed lap
        self.last_turn_type = "straight"
        self.turn_change_cooldown = 1.0  # 1 second cooldown to prevent rapid turn type changes
        self.last_turn_change_time = 0
        
        # Speed tracking for each lap
        self.current_lap_speeds = []  # Store speeds during current lap
        self.lap_speed_data = []  # Store speed statistics for each completed lap
        self.speed_sample_interval = 0.5  # Sample speed every 0.5 seconds
        self.last_speed_sample_time = 0
        
        print(f"🎯 Lap Counter initialized:")
        print(f"   Target: {target_laps if target_laps else 'UNLIMITED'} valid laps")
        print(f"   Min lap time: {MIN_LAP_TIME}s")
        print(f"   Orange gate threshold: {ORANGE_GATE_THRESHOLD}m")
        print(f"   Orange cooldown: {ORANGE_COOLDOWN}s")
        
    def record_speed(self, speed_ms):
        """Record speed sample for current lap"""
        current_time = time.time()
        
        # Sample speed at regular intervals
        if current_time - self.last_speed_sample_time >= self.speed_sample_interval:
            speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
            self.current_lap_speeds.append(speed_kmh)
            self.last_speed_sample_time = current_time
            
    def record_turn(self, turn_type):
        """Record a turn for the current lap"""
        current_time = time.time()
        
        # Only record if turn type has changed and cooldown has passed
        if (turn_type != self.last_turn_type and 
            current_time - self.last_turn_change_time > self.turn_change_cooldown):
            
            if turn_type in self.current_lap_turns:
                self.current_lap_turns[turn_type] += 1
                self.last_turn_type = turn_type
                self.last_turn_change_time = current_time
                print(f"DEBUG: Recorded {turn_type} turn. Current lap turns: {self.current_lap_turns}")
        
    def get_current_lap_time(self):
        """Get the current lap time in progress"""
        return time.time() - self.lap_start_time
    
    def get_total_race_time(self):
        """Get total race time since start"""
        return time.time() - self.race_start_time
    
    def format_time(self, time_seconds):
        """Format time in MM:SS.mmm format"""
        if time_seconds == float('inf'):
            return "--:--.---"
        
        minutes = int(time_seconds // 60)
        seconds = time_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def get_lap_time_stats(self):
        """Get comprehensive lap time statistics"""
        current_lap = self.get_current_lap_time()
        total_race = self.get_total_race_time()
        
        stats = {
            'current_lap': current_lap,
            'total_race': total_race,
            'laps_completed': self.laps_completed,
            'valid_laps_completed': len(self.lap_times),  # Only count valid laps (>MIN_LAP_TIME)
            'best_lap': self.best_lap_time,
            'last_lap': self.last_lap_time,
            'lap_times': self.lap_times.copy(),
            'average_lap': sum(self.lap_times) / len(self.lap_times) if self.lap_times else 0.0,
            'lap_turn_data': self.lap_turn_data.copy(),
            'current_lap_turns': self.current_lap_turns.copy(),
            'lap_speed_data': self.lap_speed_data.copy(),
            'current_lap_speeds': self.current_lap_speeds.copy(),
            'target_laps': self.target_laps,
            'target_reached': self.target_reached
        }
        
        return stats
        
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
                    self._complete_lap(current_time)
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
            self._complete_lap(current_time)
            return True
        
        return False
    
    def _complete_lap(self, current_time):
        """Complete a lap and update timing statistics"""
        # Calculate lap time
        lap_time = current_time - self.lap_start_time
        
        # Skip first "lap" if it's too short (race start)
        if self.laps_completed == 0 and lap_time < 10.0:
            print(f"🏁 RACE STARTED! Starting lap timing...")
            print(f"   Minimum lap time for counting: {self.format_time(MIN_LAP_TIME)}")
            if self.target_laps:
                print(f"🎯 Target: {self.target_laps} valid laps")
        else:
            # Check if lap time meets minimum threshold
            if lap_time < MIN_LAP_TIME:
                print(f"⚠️  FALSE LAP DETECTED - IGNORED!")
                print(f"   Lap time: {self.format_time(lap_time)} (under {self.format_time(MIN_LAP_TIME)} minimum)")
                print(f"   This was likely a false detection from orange cone positioning")
                print(f"   Continuing current lap timing...")
                
                # Update cooldown but don't count the lap or restart timing
                self.last_orange_gate_time = current_time
                return  # Exit without counting this lap
            
            # Valid lap - record the lap time, turn data, and speed data
            self.lap_times.append(lap_time)
            self.last_lap_time = lap_time
            valid_lap_number = len(self.lap_times)
            
            # Record turn data for this lap
            lap_turn_summary = self.current_lap_turns.copy()
            self.lap_turn_data.append(lap_turn_summary)
            
            # Record speed data for this lap
            if self.current_lap_speeds:
                speed_stats = {
                    'max_speed': max(self.current_lap_speeds),
                    'min_speed': min(self.current_lap_speeds),
                    'avg_speed': np.mean(self.current_lap_speeds),
                    'std_speed': np.std(self.current_lap_speeds),
                    'speed_samples': len(self.current_lap_speeds)
                }
                self.lap_speed_data.append(speed_stats)
                print(f"   Speed Summary: Avg:{speed_stats['avg_speed']:.1f} km/h, Max:{speed_stats['max_speed']:.1f} km/h")
            else:
                # Fallback if no speed data collected
                self.lap_speed_data.append({
                    'max_speed': 0,
                    'min_speed': 0, 
                    'avg_speed': 0,
                    'std_speed': 0,
                    'speed_samples': 0
                })
            
            # Update best lap time
            if lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
                print(f"🏆 NEW BEST LAP TIME: {self.format_time(lap_time)}!")
            
            print(f"🏁 VALID LAP {valid_lap_number} COMPLETED!")
            print(f"   Lap Time: {self.format_time(lap_time)}")
            print(f"   Best Lap: {self.format_time(self.best_lap_time)}")
            print(f"   Turn Summary: Straight:{lap_turn_summary['straight']}, Gentle:{lap_turn_summary['gentle']}, Sharp:{lap_turn_summary['sharp']}")
            if len(self.lap_times) > 1:
                avg_time = sum(self.lap_times) / len(self.lap_times)
                print(f"   Average: {self.format_time(avg_time)}")
            
            # Check if target laps reached
            if self.target_laps and valid_lap_number >= self.target_laps:
                self.target_reached = True
                print(f"🎯 TARGET REACHED! Completed {valid_lap_number}/{self.target_laps} valid laps")
                print(f"🏁 Race will end after this lap!")
            elif self.target_laps:
                remaining_laps = self.target_laps - valid_lap_number
                print(f"🎯 Progress: {valid_lap_number}/{self.target_laps} valid laps ({remaining_laps} remaining)")
            
            # Reset counters for next lap
            self.current_lap_turns = {
                'straight': 0,
                'gentle': 0,
                'sharp': 0
            }
            self.current_lap_speeds = []  # Reset speed tracking
            
            # Only restart lap timing for valid laps
            self.lap_start_time = current_time
        
        # Update counters and cooldown
        self.laps_completed += 1
        self.last_orange_gate_time = current_time
    
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
    def __init__(self, vehicle, lookahead_distance=4.0, target_laps=None):
        self.vehicle = vehicle
        self.lookahead_distance = lookahead_distance
        
        # Vehicle parameters - use configurable speeds
        self.wheelbase = 2.7  # meters
        self.max_speed = MAX_SPEED  # Use configurable parameter
        self.min_speed = MIN_SPEED  # Use configurable parameter
        
        # Control parameters - optimized for immediate track section focus
        self.safety_offset = 1.75  # meters from cones - standard track width estimation
        self.max_depth = 8.0   # maximum cone detection range - reduced for immediate focus
        self.min_depth = 1.5   # minimum cone detection range
        self.max_lateral_distance = 3.0  # maximum lateral distance - reduced for immediate track
        
        # Turn radius and path parameters - improved for smoother cone following
        self.min_turn_radius = 3.5  # Minimum safe turning radius (meters)
        self.lookahead_for_turns = 8.0  # Look ahead distance for turn detection - increased
        self.sharp_turn_threshold = 25.0  # Angle threshold for sharp turns (degrees)
        self.u_turn_threshold = 60.0  # Angle threshold for U-turns (degrees)
        self.turn_detection_distance = 8.0  # Distance to look ahead for turn detection - increased
        
        # State tracking
        self.last_steering = 0.0
        self.steering_history = deque(maxlen=5)
        
        # Turn state tracking
        self.current_turn_type = "straight"  # "straight", "gentle", "sharp", "u_turn"
        self.turn_direction = "none"  # "left", "right", "none"
        self.path_offset = 0.0  # Current path offset for wider turns
        self.cone_sequence = deque(maxlen=3)  # Track recent cones for turn prediction - limited to 3
        
        # Cone following parameters - optimized for immediate midpoint following
        self.cone_follow_lookahead = 4.0  # Reduced lookahead for immediate focus
        self.early_turn_factor = 1.0  # Reduced factor for more precise control
        self.smoothing_factor = 0.3  # Reduced smoothing for more responsiveness
        
        # Backup navigation when no cones found
        self.lost_track_counter = 0
        self.max_lost_track_frames = 20
        
        # Distance tracking for basic stats
        self.distance_traveled = 0.0
        self.last_position = None
        
        # ACCELERATION TRACKING - NEW FEATURE
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.last_velocity = 0.0
        self.acceleration_history = deque(maxlen=10)  # Store last 10 acceleration values
        self.control_history = deque(maxlen=10)  # Store last 10 control inputs
        self.requested_acceleration = 0.0  # Current requested acceleration
        self.actual_acceleration = 0.0  # Actual measured acceleration
        
        # Initialize lap counter with enhanced timing, speed tracking, and target laps
        self.lap_counter = LapCounter(target_laps=target_laps)
        
        print(f"🚗 Controller initialized:")
        print(f"   Speed range: {MIN_SPEED:.1f} - {MAX_SPEED:.1f} m/s ({MIN_SPEED*3.6:.1f} - {MAX_SPEED*3.6:.1f} km/h)")
        print(f"   🚀 Acceleration tracking enabled")
    
    def is_target_reached(self):
        """Check if target laps have been reached"""
        return self.lap_counter.target_reached
    
    def calculate_requested_acceleration(self, throttle, brake, current_speed):
        """Calculate the acceleration being requested from CARLA based on control inputs"""
        # Approximate CARLA vehicle acceleration characteristics
        max_acceleration = 3.0  # m/s² (typical car acceleration)
        max_deceleration = -8.0  # m/s² (typical car braking)
        
        if throttle > 0 and brake == 0:
            # Throttle applied - calculate forward acceleration
            # Consider speed-dependent acceleration (less acceleration at higher speeds)
            speed_factor = max(0.1, 1.0 - (current_speed / self.max_speed) * 0.7)
            requested_accel = throttle * max_acceleration * speed_factor
        elif brake > 0 and throttle == 0:
            # Brake applied - calculate deceleration
            requested_accel = -brake * abs(max_deceleration)
        elif throttle > 0 and brake > 0:
            # Both applied (shouldn't happen in normal operation)
            net_input = throttle - brake
            if net_input > 0:
                speed_factor = max(0.1, 1.0 - (current_speed / self.max_speed) * 0.7)
                requested_accel = net_input * max_acceleration * speed_factor
            else:
                requested_accel = net_input * abs(max_deceleration)
        else:
            # Neither applied - engine braking/coast
            if current_speed > 0:
                requested_accel = -0.5  # Light deceleration due to drag/engine braking
            else:
                requested_accel = 0.0
        
        return requested_accel
    
    def calculate_actual_acceleration(self, current_velocity):
        """Calculate actual acceleration from velocity changes"""
        if self.last_velocity is not None:
            dt = 0.05  # Assuming 20 Hz control loop
            accel = (current_velocity - self.last_velocity) / dt
            self.acceleration_history.append(accel)
            
            # Smooth the acceleration measurement
            if len(self.acceleration_history) >= 3:
                smoothed_accel = np.mean(list(self.acceleration_history)[-3:])
            else:
                smoothed_accel = accel
            
            self.actual_acceleration = smoothed_accel
        
        self.last_velocity = current_velocity
        return self.actual_acceleration
    
    def update_acceleration_tracking(self, throttle, brake, current_velocity):
        """Update acceleration tracking with current control inputs and velocity"""
        # Calculate requested acceleration based on control inputs
        self.current_throttle = throttle
        self.current_brake = brake
        self.requested_acceleration = self.calculate_requested_acceleration(throttle, brake, current_velocity)
        
        # Calculate actual acceleration from velocity changes
        self.calculate_actual_acceleration(current_velocity)
        
        # Store control history for analysis
        self.control_history.append({
            'throttle': throttle,
            'brake': brake,
            'requested_accel': self.requested_acceleration,
            'actual_accel': self.actual_acceleration,
            'timestamp': time.time()
        })
    
    def get_acceleration_stats(self):
        """Get acceleration statistics for display"""
        return {
            'current_throttle': self.current_throttle,
            'current_brake': self.current_brake,
            'requested_acceleration': self.requested_acceleration,
            'actual_acceleration': self.actual_acceleration,
            'avg_requested_accel': np.mean([h['requested_accel'] for h in self.control_history]) if self.control_history else 0.0,
            'avg_actual_accel': np.mean([h['actual_accel'] for h in self.control_history]) if self.control_history else 0.0,
            'acceleration_efficiency': (self.actual_acceleration / self.requested_acceleration * 100) if self.requested_acceleration != 0 else 100.0
        }
    
    def detect_turn_type_from_cones(self, blue_cones, yellow_cones):
        """Detect turn type and direction from cone patterns - limited to first 3 pairs"""
        # Limit to first 3 cones of each side for immediate focus
        limited_blue = blue_cones[:3]
        limited_yellow = yellow_cones[:3]
        all_cones = limited_blue + limited_yellow
        
        if len(all_cones) < 2:
            return "straight", "none", 0.0
        
        try:
            # Sort cones by depth (closest first)
            all_cones.sort(key=lambda c: c['depth'])
            
            # Add recent cones to sequence for pattern analysis - limited to 3 pairs
            for cone in all_cones[:3]:  # Use only closest 3 cones
                self.cone_sequence.append({
                    'x': cone['x'],
                    'y': cone['y'],
                    'depth': cone['depth'],
                    'side': 'left' if cone in limited_blue else 'right'
                })
            
            if len(self.cone_sequence) < 2:
                return "straight", "none", 0.0
            
            # Analyze cone sequence for turn patterns - use only last 3 cones
            recent_cones = list(self.cone_sequence)[-3:]  # Use only last 3 cones
            
            # Calculate lateral movement trend from first 3 cone pairs
            left_cones = [c for c in recent_cones if c['side'] == 'left']
            right_cones = [c for c in recent_cones if c['side'] == 'right']
            
            left_trend = 0.0
            right_trend = 0.0
            
            if len(left_cones) >= 2:
                left_positions = [c['x'] for c in left_cones]
                left_trend = (left_positions[-1] - left_positions[0]) / len(left_positions)
            
            if len(right_cones) >= 2:
                right_positions = [c['x'] for c in right_cones]
                right_trend = (right_positions[-1] - right_positions[0]) / len(right_positions)
            
            # Determine turn characteristics
            avg_trend = (left_trend + right_trend) / 2 if left_cones and right_cones else (left_trend or right_trend)
            turn_magnitude = abs(avg_trend)
            
            # Classify turn type with focus on first 3 pairs
            if turn_magnitude < 0.3:
                turn_type = "straight"
                direction = "none"
                path_offset = 0.0
            elif turn_magnitude < 0.8:
                turn_type = "gentle"
                direction = "left" if avg_trend > 0 else "right"
                path_offset = 0.4
            else:
                turn_type = "sharp"
                direction = "left" if avg_trend > 0 else "right"
                path_offset = 0.8
            
            print(f"DEBUG: Cone pattern analysis (first 3 pairs) - Type: {turn_type}, Direction: {direction}, Trend: {avg_trend:.3f}")
            
            return turn_type, direction, path_offset
            
        except Exception as e:
            print(f"ERROR in cone turn detection: {e}")
            return "straight", "none", 0.0
    
    def calculate_adaptive_speed(self, turn_type, steering_angle, current_depth):
        """Calculate speed based on turn type and conditions"""
        base_speed = self.min_speed + (self.max_speed - self.min_speed) * 0.7
        
        # Speed reduction based on turn type
        if turn_type == "gentle":
            speed_factor = 0.8
        elif turn_type == "sharp":
            speed_factor = 0.6
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
        fov_horizontal = 90.0  # degrees
        
        # Calculate angle from image center
        angle = ((center_x - image_width / 2) / (image_width / 2)) * (fov_horizontal / 2)
        
        # Convert to world coordinates relative to vehicle
        world_x = depth * np.tan(np.radians(angle))
        world_y = depth
        
        return world_x, world_y
    
    def process_cone_detections(self, cone_detections):
        """Process cone detections with strict spatial filtering focused on immediate track section"""
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
                
                # STRICT depth filtering for immediate track focus
                if cls == 2:  # Orange cone - allow farther detection
                    if depth < 1.0 or depth > 15.0:
                        continue
                else:  # Blue/Yellow cones - very strict for immediate section only
                    if depth < 1.5 or depth > 8.0:  # Reduced from 12.0 to 8.0
                        continue
                    
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to world coordinates
                world_x, world_y = self.image_to_world_coords(center_x, center_y, depth)
                
                # STRICT lateral distance filtering for immediate track section
                if cls == 2:  # Orange cone - allow wider lateral range
                    if abs(world_x) > 8.0:
                        continue
                else:  # Blue/Yellow cones - very strict for immediate track
                    if abs(world_x) > 3.0:  # Reduced from 4.0 to 3.0
                        continue
                
                # STRICT forward focus angle for immediate track section
                angle_to_cone = np.degrees(abs(np.arctan2(world_x, world_y)))
                if cls == 2:  # Orange cone - allow wider angle
                    if angle_to_cone > 60.0:
                        continue
                else:  # Blue/Yellow cones - very strict for immediate track
                    if angle_to_cone > 30.0:  # Reduced from 45.0 to 30.0
                        continue
                
                # Additional filtering: Only accept cones that are clearly part of immediate track
                # Ensure blue cones are on the left and yellow on the right for immediate section
                if cls == 1 and world_x > 0.5:  # Blue cone too far right for immediate track
                    continue
                if cls == 0 and world_x < -0.5:  # Yellow cone too far left for immediate track
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
        print(f"DEBUG: After STRICT filtering - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
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
    
    def calculate_smooth_cone_target(self, blue_cones, yellow_cones):
        """Calculate precise midpoint target for immediate track section with priority on both-side pairs"""
        if not blue_cones and not yellow_cones:
            return None
        
        try:
            # Limit to first 2 cones of each side for immediate track focus
            limited_blue = blue_cones[:2]
            limited_yellow = yellow_cones[:2]
            
            target_x = 0.0
            target_y = 4.0  # Default forward target
            
            print(f"DEBUG: Limited cones - Blue: {len(limited_blue)}, Yellow: {len(limited_yellow)}")
            
            # PRIORITY 1: If we have both blue and yellow cones, find the best immediate pair
            if limited_blue and limited_yellow:
                best_pair = None
                min_depth_diff = float('inf')
                
                # Find the best matching pair with similar depths
                for blue_cone in limited_blue:
                    for yellow_cone in limited_yellow:
                        depth_diff = abs(blue_cone['depth'] - yellow_cone['depth'])
                        avg_depth = (blue_cone['depth'] + yellow_cone['depth']) / 2
                        
                        # Only consider pairs that are close and reasonable
                        if depth_diff < 2.0 and avg_depth < 6.0:  # Immediate track section
                            if depth_diff < min_depth_diff:
                                min_depth_diff = depth_diff
                                best_pair = (blue_cone, yellow_cone)
                
                if best_pair:
                    blue_cone, yellow_cone = best_pair
                    # Calculate precise midpoint
                    target_x = (blue_cone['x'] + yellow_cone['x']) / 2
                    target_y = (blue_cone['y'] + yellow_cone['y']) / 2
                    
                    track_width = abs(blue_cone['x'] - yellow_cone['x'])
                    print(f"DEBUG: MIDPOINT from immediate pair - Blue: ({blue_cone['x']:.2f}, {blue_cone['y']:.2f}), Yellow: ({yellow_cone['x']:.2f}, {yellow_cone['y']:.2f})")
                    print(f"DEBUG: MIDPOINT target: ({target_x:.2f}, {target_y:.2f}), Width: {track_width:.2f}m")
                    
                    # Ensure we're following the centerline
                    if abs(target_x) > 1.5:  # If midpoint is too far off center, adjust
                        target_x = np.clip(target_x, -1.5, 1.5)
                        print(f"DEBUG: Adjusted midpoint to stay centered: ({target_x:.2f}, {target_y:.2f})")
                        
                else:
                    # No good pairs found, use average positions with centerline bias
                    blue_avg_x = np.mean([c['x'] for c in limited_blue])
                    yellow_avg_x = np.mean([c['x'] for c in limited_yellow])
                    target_x = (blue_avg_x + yellow_avg_x) / 2
                    target_y = np.mean([c['y'] for c in limited_blue + limited_yellow])
                    print(f"DEBUG: MIDPOINT from averages: ({target_x:.2f}, {target_y:.2f})")
            
            # PRIORITY 2: Only one side available - follow with centerline offset
            elif limited_blue:
                # Only blue cones available - aim for centerline with right offset
                closest_blue = limited_blue[0]
                # Estimate track width and aim for center
                estimated_track_width = 3.5  # Standard Formula Student track width
                target_x = closest_blue['x'] + (estimated_track_width / 2)
                target_y = closest_blue['y']
                print(f"DEBUG: Following blue cones with centerline estimation: ({target_x:.2f}, {target_y:.2f})")
                
            elif limited_yellow:
                # Only yellow cones available - aim for centerline with left offset
                closest_yellow = limited_yellow[0]
                # Estimate track width and aim for center
                estimated_track_width = 3.5  # Standard Formula Student track width
                target_x = closest_yellow['x'] - (estimated_track_width / 2)
                target_y = closest_yellow['y']
                print(f"DEBUG: Following yellow cones with centerline estimation: ({target_x:.2f}, {target_y:.2f})")
            
            # Apply minimal smoothing to avoid oscillation
            if hasattr(self, 'last_target_x') and hasattr(self, 'last_target_y'):
                # Use lighter smoothing to be more responsive
                smooth_factor = 0.3  # Reduced from 0.7 for more responsiveness
                target_x = smooth_factor * target_x + (1 - smooth_factor) * self.last_target_x
                target_y = smooth_factor * target_y + (1 - smooth_factor) * self.last_target_y
                print(f"DEBUG: Lightly smoothed target: ({target_x:.2f}, {target_y:.2f})")
            
            # Store for next iteration
            self.last_target_x = target_x
            self.last_target_y = target_y
            
            # Ensure target is within reasonable bounds
            target_y = max(target_y, 2.0)  # Minimum lookahead
            target_y = min(target_y, 6.0)  # Maximum lookahead for immediate focus
            target_x = np.clip(target_x, -2.0, 2.0)  # Reasonable lateral bounds
            
            return {
                'midpoint_x': target_x,
                'midpoint_y': target_y,
                'avg_depth': target_y,
                'width': abs(target_x) * 2,
                'type': 'immediate_midpoint'
            }
            
        except Exception as e:
            print(f"ERROR in immediate midpoint calculation: {e}")
            return None
    
    def calculate_pure_pursuit_steering(self, target_x, target_y):
        """Calculate steering angle using pure pursuit algorithm optimized for cone following"""
        try:
            print(f"DEBUG: Pure pursuit calculation for target ({target_x:.2f}, {target_y:.2f})")
            
            # Calculate angle to target
            alpha = np.arctan2(target_x, target_y)
            print(f"DEBUG: Alpha (angle to target): {np.degrees(alpha):.1f}°")
            
            # Calculate lookahead distance
            lookahead_dist = np.sqrt(target_x**2 + target_y**2)
            print(f"DEBUG: Lookahead distance: {lookahead_dist:.2f}m")
            
            # Adaptive lookahead based on turn type and cone visibility
            lateral_offset = abs(target_x)
            
            # For cone following, use more responsive steering
            if self.current_turn_type == "sharp":
                adaptive_lookahead = lookahead_dist * 0.6  # More responsive for sharp turns
                print(f"DEBUG: Sharp turn - using responsive lookahead")
            elif self.current_turn_type == "gentle":
                adaptive_lookahead = lookahead_dist * 0.8  # Slightly more responsive
                print(f"DEBUG: Gentle turn - using moderate lookahead")
            else:
                adaptive_lookahead = lookahead_dist  # Normal lookahead for straight
            
            # Ensure minimum and maximum lookahead
            adaptive_lookahead = max(adaptive_lookahead, 2.0)
            adaptive_lookahead = min(adaptive_lookahead, 8.0)
            
            print(f"DEBUG: Adaptive lookahead: {adaptive_lookahead:.2f}m")
            
            # Pure pursuit steering calculation
            steering_angle = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), adaptive_lookahead)
            
            # Apply early steering enhancement for turns
            if lateral_offset > 1.0:
                # Calculate additional steering for early turn initiation
                early_steering_factor = min(lateral_offset / 2.0, 1.0)  # Scale factor based on lateral offset
                early_steering_boost = np.arctan2(lateral_offset * early_steering_factor, lookahead_dist) * 0.4
                
                if target_x > 0:  # Target to the right
                    steering_angle += early_steering_boost
                else:  # Target to the left
                    steering_angle -= early_steering_boost
                
                print(f"DEBUG: Applied early steering boost: {np.degrees(early_steering_boost):.1f}°")
            
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
            
            # Convert to normalized steering [-1, 1]
            max_steering_rad = np.radians(30.0)  # Max 30 degrees
            normalized_steering = np.clip(steering_angle / max_steering_rad, -1.0, 1.0)
            
            print(f"DEBUG: Normalized steering: {normalized_steering:.3f}")
            direction = 'LEFT' if normalized_steering > 0 else 'RIGHT' if normalized_steering < 0 else 'STRAIGHT'
            print(f"DEBUG: Steering direction: {direction}")
            
            return normalized_steering
            
        except Exception as e:
            print(f"ERROR in pure pursuit calculation: {e}")
            return 0.0
    
    def calculate_turn_radius(self, steering_angle):
        """Calculate the turning radius based on steering angle and wheelbase"""
        if abs(steering_angle) < 0.01:
            return float('inf')  # Straight line
        
        # Bicycle model turning radius
        turn_radius = self.wheelbase / np.tan(abs(steering_angle))
        return max(turn_radius, self.min_turn_radius)
    
    def smooth_steering(self, raw_steering):
        """Apply steering smoothing optimized for cone following"""
        try:
            self.steering_history.append(raw_steering)
            
            # Adaptive smoothing based on turn requirements and lateral offset
            lateral_offset = abs(getattr(self, 'current_target_x', 0.0))
            
            if len(self.steering_history) >= 3:
                # For cone following, use balanced smoothing that responds to turns
                if self.current_turn_type == "sharp":
                    # More responsive for sharp turns
                    weights = np.array([0.6, 0.25, 0.15])
                    print(f"DEBUG: Sharp turn - using responsive steering smoothing")
                elif lateral_offset > 2.0:
                    # Moderate smoothing for significant lateral movement
                    weights = np.array([0.55, 0.3, 0.15])
                    print(f"DEBUG: High lateral offset - using moderate smoothing")
                else:
                    # Balanced smoothing for normal following
                    weights = np.array([0.5, 0.3, 0.2])
                
                recent_steering = np.array(list(self.steering_history)[-3:])
                smoothed = np.average(recent_steering, weights=weights)
            else:
                smoothed = raw_steering
            
            # Adaptive rate limiting for cone following
            if self.current_turn_type == "sharp":
                max_change = 0.2  # Allow more change for sharp turns
            elif lateral_offset > 1.5:
                max_change = 0.18  # Moderate change for turns
            else:
                max_change = 0.15  # Normal rate limiting
            
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
    
    def control_vehicle(self, cone_detections):
        """Main control function with smooth cone line following, target lap checking, and acceleration tracking"""
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG: CONTROL CYCLE - {len(cone_detections) if cone_detections else 0} detections")
            
            # Get lap statistics for display
            lap_stats = self.lap_counter.get_lap_time_stats()
            print(f"Total orange gate detections: {lap_stats['laps_completed']}")
            print(f"Valid laps (>{MIN_LAP_TIME}s): {lap_stats['valid_laps_completed']}")
            if lap_stats['target_laps']:
                print(f"Target: {lap_stats['target_laps']} laps | Reached: {lap_stats['target_reached']}")
            print(f"Current turn type: {self.current_turn_type}, Direction: {self.turn_direction}")
            print(f"Lost track counter: {self.lost_track_counter}")
            print(f"{'='*60}")
            
            # Check if target laps reached - if so, stop the vehicle safely
            if self.is_target_reached():
                print(f"🎯 TARGET LAPS REACHED! Stopping vehicle safely...")
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                self.vehicle.apply_control(control)
                
                # Update acceleration tracking for the stop command
                velocity = self.vehicle.get_velocity()
                current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                self.update_acceleration_tracking(0.0, 1.0, current_speed)
                
                return 0.0, 0.0  # Return zero values to indicate stopping
            
            # Update distance traveled
            self.update_distance_traveled()
            
            # Get current speed and record it for lap statistics
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            self.lap_counter.record_speed(current_speed)
            
            # Process cone detections (includes orange cones)
            blue_cones, yellow_cones, orange_cones = self.process_cone_detections(cone_detections)
            print(f"DEBUG: Processed cones - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
            
            # Check for lap completion through orange gate
            if orange_cones:
                transform = self.vehicle.get_transform()
                vehicle_position = (transform.location.x, transform.location.y, transform.location.z)
                self.lap_counter.check_orange_gate_passage(orange_cones, vehicle_position)
            
            # Enhanced lost track detection with immediate recovery steering
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
                    
                    # Update acceleration tracking
                    self.update_acceleration_tracking(0.2, 0.0, current_speed)
                    
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
                    
                    # Update acceleration tracking
                    self.update_acceleration_tracking(0.15, 0.0, current_speed)
                    
                    return search_steering, 0.15
                else:
                    # Last resort - wide search
                    search_steering = 0.8 * np.sin(self.lost_track_counter * 0.2)
                    control = carla.VehicleControl()
                    control.steer = search_steering
                    control.throttle = 0.1
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    
                    # Update acceleration tracking
                    self.update_acceleration_tracking(0.1, 0.0, current_speed)
                    
                    return search_steering, 0.1
            
            # Calculate smooth cone following target
            navigation_target = self.calculate_smooth_cone_target(blue_cones, yellow_cones)
            
            if not navigation_target:
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
                    
                    # Update acceleration tracking
                    self.update_acceleration_tracking(0.15, 0.0, current_speed)
                    
                    return search_steering, 0.15
                else:
                    # Move forward slowly while searching
                    control = carla.VehicleControl()
                    control.steer = self.last_steering * 0.5
                    control.throttle = 0.15
                    control.brake = 0.0
                    self.vehicle.apply_control(control)
                    
                    # Update acceleration tracking
                    self.update_acceleration_tracking(0.15, 0.0, current_speed)
                    
                    return self.last_steering * 0.5, 0.15
            
            # Reset lost track counter if we found something
            self.lost_track_counter = 0
            
            # Detect turn type from cone patterns
            turn_type, turn_direction, path_offset = self.detect_turn_type_from_cones(blue_cones, yellow_cones)
            self.current_turn_type = turn_type
            self.turn_direction = turn_direction
            self.path_offset = path_offset
            
            # Record the turn for lap statistics
            self.lap_counter.record_turn(turn_type)
            
            # Get target point from smooth cone following
            target_x = navigation_target['midpoint_x']
            target_y = navigation_target['midpoint_y']
            
            # Store current target for calculations
            self.current_target_x = target_x
            
            # Navigate towards the target
            raw_steering = self.calculate_pure_pursuit_steering(target_x, target_y)
            smooth_steering = self.smooth_steering(raw_steering)
            
            # Calculate adaptive speed based on turn type
            current_depth = navigation_target['avg_depth']
            target_speed = self.calculate_adaptive_speed(turn_type, smooth_steering, current_depth)
            
            speed_diff = target_speed - current_speed
            if speed_diff > 0.5:
                control = carla.VehicleControl()
                control.steer = float(smooth_steering)
                control.throttle = min(0.5, 0.2 + 0.3 * (speed_diff / self.max_speed))
                control.brake = 0.0
            elif speed_diff < -0.5:
                control = carla.VehicleControl()
                control.steer = float(smooth_steering)
                control.throttle = 0.0
                control.brake = min(0.4, 0.2 * abs(speed_diff) / self.max_speed)
            else:
                control = carla.VehicleControl()
                control.steer = float(smooth_steering)
                control.throttle = 0.3
                control.brake = 0.0
            
            self.vehicle.apply_control(control)
            
            # UPDATE ACCELERATION TRACKING - NEW FEATURE
            self.update_acceleration_tracking(control.throttle, control.brake, current_speed)
            
            # Enhanced debug output with acceleration info
            direction = 'LEFT' if smooth_steering > 0 else 'RIGHT' if smooth_steering < 0 else 'STRAIGHT'
            accel_stats = self.get_acceleration_stats()
            
            print(f"DEBUG: APPLIED CONTROL:")
            print(f"  Navigation: {navigation_target.get('type', 'cone_following')}_{turn_type}")
            print(f"  Turn Analysis: {turn_type}-{turn_direction}")
            print(f"  Target: ({target_x:.2f}, {target_y:.2f})")
            print(f"  Target distance: {current_depth:.2f}m")
            print(f"  Turn radius: {self.calculate_turn_radius(np.radians(smooth_steering * 30)):.2f}m")
            print(f"  Steering: {smooth_steering:.3f} ({direction})")
            print(f"  Throttle: {control.throttle:.2f}")
            print(f"  Brake: {control.brake:.2f}")
            print(f"  🚀 Requested Acceleration: {accel_stats['requested_acceleration']:.2f} m/s²")
            print(f"  📊 Actual Acceleration: {accel_stats['actual_acceleration']:.2f} m/s²")
            print(f"  Current Speed: {current_speed:.1f} m/s ({current_speed*3.6:.1f} km/h)")
            print(f"  Target Speed: {target_speed:.1f} m/s ({target_speed*3.6:.1f} km/h)")
            print(f"  Distance: {self.distance_traveled:.1f}m")
            if lap_stats['target_laps']:
                print(f"  Target Progress: {lap_stats['valid_laps_completed']}/{lap_stats['target_laps']} laps")
            print(f"{'='*60}\n")
            
            return smooth_steering, target_speed
            
        except Exception as e:
            print(f"ERROR in vehicle control: {e}")
            import traceback
            traceback.print_exc()
            # Safe fallback
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.5
            self.vehicle.apply_control(control)
            
            # Update acceleration tracking for safe fallback
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            self.update_acceleration_tracking(0.0, 0.5, current_speed)
            
            return 0.0, 0.0

class CarlaRacingSystem:
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
        
        # Initialize display flag
        self.display_enabled = True
        
        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C for clean shutdown"""
        print("\nShutting down gracefully...")
        self.running = False
        
    def plot_comprehensive_analysis(self, lap_stats):
        """Create a comprehensive 5x2 grid analysis with speed tracking and consistency metrics"""
        lap_times = lap_stats['lap_times']
        lap_turn_data = lap_stats['lap_turn_data']
        lap_speed_data = lap_stats['lap_speed_data']
        
        if not lap_times:
            print("No lap times to plot.")
            return
        
        try:
            # Ensure we're using the Agg backend
            matplotlib.use('Agg')
            
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Create 5x2 grid of subplots
            fig = plt.figure(figsize=(20, 16))
            fig.patch.set_facecolor('#1b2a39')
            
            # Grid layout: 2 rows, 5 columns
            gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
            
            # Define colors for consistent theming
            primary_color = '#da940b'
            secondary_color = '#ffd700'
            text_color = 'white'
            background_color = '#1b2a39'
            
            lap_numbers = list(range(1, len(lap_times) + 1))
            
            # ===================
            # TOP ROW (5 graphs)
            # ===================
            
            # 1. LAP TIMES
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_facecolor(background_color)
            
            bars = ax1.bar(lap_numbers, lap_times, color=primary_color, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Highlight best lap
            if lap_stats['best_lap'] != float('inf'):
                best_lap_index = lap_times.index(lap_stats['best_lap'])
                bars[best_lap_index].set_color(secondary_color)
                bars[best_lap_index].set_linewidth(2)
            
            ax1.set_title('Lap Times', fontsize=12, fontweight='bold', color=text_color)
            ax1.set_xlabel('Lap', fontsize=10, color=text_color)
            ax1.set_ylabel('Time (s)', fontsize=10, color=text_color)
            ax1.tick_params(colors=text_color, which='both')
            ax1.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 2. SPEED PER LAP
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_facecolor(background_color)
            
            if lap_speed_data and len(lap_speed_data) > 0:
                avg_speeds = [speed_data['avg_speed'] for speed_data in lap_speed_data]
                max_speeds = [speed_data['max_speed'] for speed_data in lap_speed_data]
                min_speeds = [speed_data['min_speed'] for speed_data in lap_speed_data]
                
                # Plot average speed as bars
                speed_bars = ax2.bar(lap_numbers, avg_speeds, color='#2ecc71', alpha=0.7, label='Avg Speed')
                
                # Plot max and min as error bars
                ax2.errorbar(lap_numbers, avg_speeds, 
                           yerr=[np.array(avg_speeds) - np.array(min_speeds), 
                                 np.array(max_speeds) - np.array(avg_speeds)],
                           fmt='none', ecolor='#e74c3c', alpha=0.8, capsize=3)
                
                ax2.set_title('Speed per Lap', fontsize=12, fontweight='bold', color=text_color)
                ax2.set_xlabel('Lap', fontsize=10, color=text_color)
                ax2.set_ylabel('Speed (km/h)', fontsize=10, color=text_color)
            else:
                ax2.text(0.5, 0.5, 'No Speed Data', transform=ax2.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax2.set_title('Speed per Lap', fontsize=12, fontweight='bold', color=text_color)
            
            ax2.tick_params(colors=text_color, which='both')
            ax2.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 3. TURN ANALYSIS
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.set_facecolor(background_color)
            
            if lap_turn_data and len(lap_turn_data) > 0:
                turn_types = ['straight', 'gentle', 'sharp']
                turn_colors = ['#2ecc71', '#f39c12', '#e74c3c']
                
                bottoms = np.zeros(len(lap_turn_data))
                for i, (turn_type, color) in enumerate(zip(turn_types, turn_colors)):
                    values = [lap_data.get(turn_type, 0) for lap_data in lap_turn_data]
                    ax3.bar(lap_numbers, values, bottom=bottoms, color=color, alpha=0.8)
                    bottoms += values
                
                ax3.set_title('Turn Distribution', fontsize=12, fontweight='bold', color=text_color)
                ax3.set_xlabel('Lap', fontsize=10, color=text_color)
                ax3.set_ylabel('Turn Count', fontsize=10, color=text_color)
            else:
                ax3.text(0.5, 0.5, 'No Turn Data', transform=ax3.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax3.set_title('Turn Distribution', fontsize=12, fontweight='bold', color=text_color)
            
            ax3.tick_params(colors=text_color, which='both')
            ax3.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 4. LAP TIME TREND
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.set_facecolor(background_color)
            
            if len(lap_times) > 1:
                ax4.plot(lap_numbers, lap_times, marker='o', color=primary_color, linewidth=2, markersize=6)
                
                # Add trend line
                z = np.polyfit(lap_numbers, lap_times, 1)
                p = np.poly1d(z)
                trend_color = '#2ecc71' if z[0] < 0 else '#e74c3c'  # Green if improving, red if getting worse
                ax4.plot(lap_numbers, p(lap_numbers), "--", color=trend_color, alpha=0.8, linewidth=2)
                
                # Add best lap horizontal line
                ax4.axhline(y=lap_stats['best_lap'], color=secondary_color, linestyle=':', alpha=0.8, linewidth=2)
                
                ax4.set_title('Lap Time Trend', fontsize=12, fontweight='bold', color=text_color)
                ax4.set_xlabel('Lap', fontsize=10, color=text_color)
                ax4.set_ylabel('Time (s)', fontsize=10, color=text_color)
            else:
                ax4.text(0.5, 0.5, 'Need More Laps', transform=ax4.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax4.set_title('Lap Time Trend', fontsize=12, fontweight='bold', color=text_color)
            
            ax4.tick_params(colors=text_color, which='both')
            ax4.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 5. PERFORMANCE RADAR
            ax5 = fig.add_subplot(gs[0, 4], projection='polar')
            ax5.set_facecolor(background_color)
            
            if len(lap_times) > 1 and lap_speed_data:
                # Calculate performance metrics
                time_consistency = (1 - (np.std(lap_times) / np.mean(lap_times))) * 100
                time_consistency = max(0, min(100, time_consistency))
                
                speed_consistency = 0
                if lap_speed_data:
                    speed_stds = [data['std_speed'] for data in lap_speed_data if data['std_speed'] > 0]
                    if speed_stds:
                        avg_speed_std = np.mean(speed_stds)
                        avg_speeds = [data['avg_speed'] for data in lap_speed_data]
                        if avg_speeds:
                            speed_consistency = (1 - (avg_speed_std / np.mean(avg_speeds))) * 100
                            speed_consistency = max(0, min(100, speed_consistency))
                
                best_lap_performance = ((max(lap_times) - lap_stats['best_lap']) / max(lap_times)) * 100
                
                # Radar chart data
                categories = ['Time\nConsistency', 'Speed\nConsistency', 'Best Lap\nPerformance']
                values = [time_consistency, speed_consistency, best_lap_performance]
                
                # Number of variables
                N = len(categories)
                
                # Compute angle for each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Complete the circle
                
                # Close the plot
                values += values[:1]
                
                ax5.plot(angles, values, 'o-', linewidth=2, color=primary_color)
                ax5.fill(angles, values, alpha=0.25, color=primary_color)
                ax5.set_xticks(angles[:-1])
                ax5.set_xticklabels(categories, color=text_color, fontsize=9)
                ax5.set_ylim(0, 100)
                ax5.set_title('Performance Radar', fontsize=12, fontweight='bold', color=text_color, pad=20)
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Need More Data', transform=ax5.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax5.set_title('Performance Radar', fontsize=12, fontweight='bold', color=text_color)
            
            # ===================
            # BOTTOM ROW (5 consistency metrics)
            # ===================
            
            # 1. LAP TIME CONSISTENCY
            ax6 = fig.add_subplot(gs[1, 0])
            ax6.set_facecolor(background_color)
            
            if len(lap_times) > 1:
                std_dev = np.std(lap_times)
                mean_time = np.mean(lap_times)
                consistency_pct = (1 - (std_dev / mean_time)) * 100
                
                # Color based on consistency
                if consistency_pct > 95:
                    gauge_color = '#2ecc71'
                    status = 'Excellent'
                elif consistency_pct > 90:
                    gauge_color = '#f39c12'
                    status = 'Good'
                else:
                    gauge_color = '#e74c3c'
                    status = 'Poor'
                
                ax6.bar([0], [consistency_pct], color=gauge_color, alpha=0.8, width=0.8)
                ax6.set_ylim(0, 100)
                ax6.set_title('Time Consistency', fontsize=12, fontweight='bold', color=text_color)
                ax6.set_ylabel('Consistency %', fontsize=10, color=text_color)
                ax6.text(0, consistency_pct + 5, f'{consistency_pct:.1f}%\n({status})', 
                        ha='center', va='bottom', color=text_color, fontweight='bold')
            else:
                ax6.text(0.5, 0.5, 'Need More Laps', transform=ax6.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax6.set_title('Time Consistency', fontsize=12, fontweight='bold', color=text_color)
            
            ax6.tick_params(colors=text_color, which='both')
            ax6.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 2. SPEED CONSISTENCY
            ax7 = fig.add_subplot(gs[1, 1])
            ax7.set_facecolor(background_color)
            
            if lap_speed_data and len(lap_speed_data) > 1:
                # Calculate speed consistency across laps
                avg_speeds = [data['avg_speed'] for data in lap_speed_data]
                speed_consistency = 0
                if avg_speeds:
                    speed_std = np.std(avg_speeds)
                    speed_mean = np.mean(avg_speeds)
                    if speed_mean > 0:
                        speed_consistency = (1 - (speed_std / speed_mean)) * 100
                        speed_consistency = max(0, min(100, speed_consistency))
                
                # Color based on consistency
                if speed_consistency > 95:
                    gauge_color = '#2ecc71'
                    status = 'Excellent'
                elif speed_consistency > 90:
                    gauge_color = '#f39c12'
                    status = 'Good'
                else:
                    gauge_color = '#e74c3c'
                    status = 'Poor'
                
                ax7.bar([0], [speed_consistency], color=gauge_color, alpha=0.8, width=0.8)
                ax7.set_ylim(0, 100)
                ax7.set_title('Speed Consistency', fontsize=12, fontweight='bold', color=text_color)
                ax7.set_ylabel('Consistency %', fontsize=10, color=text_color)
                ax7.text(0, speed_consistency + 5, f'{speed_consistency:.1f}%\n({status})', 
                        ha='center', va='bottom', color=text_color, fontweight='bold')
            else:
                ax7.text(0.5, 0.5, 'No Speed Data', transform=ax7.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax7.set_title('Speed Consistency', fontsize=12, fontweight='bold', color=text_color)
            
            ax7.tick_params(colors=text_color, which='both')
            ax7.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 3. TURN CONSISTENCY
            ax8 = fig.add_subplot(gs[1, 2])
            ax8.set_facecolor(background_color)
            
            if lap_turn_data and len(lap_turn_data) > 1:
                # Calculate turn pattern consistency
                turn_consistency_scores = []
                
                for turn_type in ['straight', 'gentle', 'sharp']:
                    turn_counts = [lap_data.get(turn_type, 0) for lap_data in lap_turn_data]
                    if any(count > 0 for count in turn_counts):
                        std_dev_turns = np.std(turn_counts)
                        mean_turns = np.mean(turn_counts)
                        if mean_turns > 0:
                            consistency = (1 - (std_dev_turns / mean_turns)) * 100
                            consistency = max(0, min(100, consistency))
                            turn_consistency_scores.append(consistency)
                
                if turn_consistency_scores:
                    overall_turn_consistency = np.mean(turn_consistency_scores)
                    
                    # Color based on consistency
                    if overall_turn_consistency > 85:
                        gauge_color = '#2ecc71'
                        status = 'Excellent'
                    elif overall_turn_consistency > 70:
                        gauge_color = '#f39c12'
                        status = 'Good'
                    else:
                        gauge_color = '#e74c3c'
                        status = 'Poor'
                    
                    ax8.bar([0], [overall_turn_consistency], color=gauge_color, alpha=0.8, width=0.8)
                    ax8.set_ylim(0, 100)
                    ax8.text(0, overall_turn_consistency + 5, f'{overall_turn_consistency:.1f}%\n({status})', 
                            ha='center', va='bottom', color=text_color, fontweight='bold')
                else:
                    ax8.text(0.5, 0.5, 'Insufficient\nTurn Data', transform=ax8.transAxes, ha='center', va='center',
                            fontsize=10, color=text_color, fontweight='bold')
                
                ax8.set_title('Turn Consistency', fontsize=12, fontweight='bold', color=text_color)
                ax8.set_ylabel('Consistency %', fontsize=10, color=text_color)
            else:
                ax8.text(0.5, 0.5, 'No Turn Data', transform=ax8.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax8.set_title('Turn Consistency', fontsize=12, fontweight='bold', color=text_color)
            
            ax8.tick_params(colors=text_color, which='both')
            ax8.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 4. IMPROVEMENT TREND
            ax9 = fig.add_subplot(gs[1, 3])
            ax9.set_facecolor(background_color)
            
            if len(lap_times) > 2:
                # Calculate improvement trend (lower times = better)
                recent_laps = lap_times[-3:] if len(lap_times) >= 3 else lap_times
                early_laps = lap_times[:3] if len(lap_times) >= 3 else lap_times[:-1] if len(lap_times) > 1 else [lap_times[0]]
                
                recent_avg = np.mean(recent_laps)
                early_avg = np.mean(early_laps)
                
                improvement_pct = ((early_avg - recent_avg) / early_avg) * 100
                
                # Color based on improvement
                if improvement_pct > 2:
                    gauge_color = '#2ecc71'
                    status = 'Improving'
                elif improvement_pct > -2:
                    gauge_color = '#f39c12'
                    status = 'Stable'
                else:
                    gauge_color = '#e74c3c'
                    status = 'Declining'
                
                # Normalize for display (center at 0, range -10 to +10)
                display_value = np.clip(improvement_pct, -10, 10) + 10  # Convert to 0-20 range
                
                ax9.bar([0], [display_value], color=gauge_color, alpha=0.8, width=0.8)
                ax9.set_ylim(0, 20)
                ax9.set_title('Improvement Trend', fontsize=12, fontweight='bold', color=text_color)
                ax9.set_ylabel('Trend Score', fontsize=10, color=text_color)
                ax9.text(0, display_value + 1, f'{improvement_pct:+.1f}%\n({status})', 
                        ha='center', va='bottom', color=text_color, fontweight='bold')
                
                # Add reference line at center (0% improvement)
                ax9.axhline(y=10, color='white', linestyle='--', alpha=0.5)
            else:
                ax9.text(0.5, 0.5, 'Need More Laps', transform=ax9.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax9.set_title('Improvement Trend', fontsize=12, fontweight='bold', color=text_color)
            
            ax9.tick_params(colors=text_color, which='both')
            ax9.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # 5. OVERALL SCORE
            ax10 = fig.add_subplot(gs[1, 4])
            ax10.set_facecolor(background_color)
            
            if len(lap_times) > 1:
                # Calculate overall performance score
                scores = []
                
                # Time consistency score
                if len(lap_times) > 1:
                    time_consistency = (1 - (np.std(lap_times) / np.mean(lap_times))) * 100
                    scores.append(max(0, min(100, time_consistency)))
                
                # Speed consistency score
                if lap_speed_data:
                    avg_speeds = [data['avg_speed'] for data in lap_speed_data]
                    if len(avg_speeds) > 1:
                        speed_std = np.std(avg_speeds)
                        speed_mean = np.mean(avg_speeds)
                        if speed_mean > 0:
                            speed_consistency = (1 - (speed_std / speed_mean)) * 100
                            scores.append(max(0, min(100, speed_consistency)))
                
                # Best lap performance (relative to average)
                best_lap_score = ((np.mean(lap_times) - lap_stats['best_lap']) / np.mean(lap_times)) * 100
                scores.append(max(0, min(100, best_lap_score)))
                
                if scores:
                    overall_score = np.mean(scores)
                    
                    # Color and grade based on overall score
                    if overall_score > 85:
                        gauge_color = '#2ecc71'
                        grade = 'A'
                    elif overall_score > 75:
                        gauge_color = '#27ae60'
                        grade = 'B+'
                    elif overall_score > 65:
                        gauge_color = '#f39c12'
                        grade = 'B'
                    elif overall_score > 55:
                        gauge_color = '#e67e22'
                        grade = 'C+'
                    else:
                        gauge_color = '#e74c3c'
                        grade = 'C'
                    
                    ax10.bar([0], [overall_score], color=gauge_color, alpha=0.8, width=0.8)
                    ax10.set_ylim(0, 100)
                    ax10.set_title('Overall Score', fontsize=12, fontweight='bold', color=text_color)
                    ax10.set_ylabel('Score', fontsize=10, color=text_color)
                    ax10.text(0, overall_score + 5, f'{overall_score:.0f}\nGrade: {grade}', 
                            ha='center', va='bottom', color=text_color, fontweight='bold', fontsize=11)
                else:
                    ax10.text(0.5, 0.5, 'Calculating...', transform=ax10.transAxes, ha='center', va='center',
                            fontsize=12, color=text_color, fontweight='bold')
                    ax10.set_title('Overall Score', fontsize=12, fontweight='bold', color=text_color)
            else:
                ax10.text(0.5, 0.5, 'Need More Data', transform=ax10.transAxes, ha='center', va='center',
                        fontsize=12, color=text_color, fontweight='bold')
                ax10.set_title('Overall Score', fontsize=12, fontweight='bold', color=text_color)
            
            ax10.tick_params(colors=text_color, which='both')
            ax10.grid(True, alpha=0.3, color='white', linestyle='--')
            
            # Add overall title with target lap info and configuration
            target_info = f" (Target: {lap_stats.get('target_laps', 'Unlimited')} laps)" if lap_stats.get('target_laps') else ""
            config_info = f" | Min Lap: {MIN_LAP_TIME}s | Speed: {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s"
            fig.suptitle(f'Formula Student - Racing Performance Analysis{target_info}{config_info}', 
                        fontsize=16, fontweight='bold', color=text_color, y=0.95)
            
            # Add statistics summary box with target progress and configuration
            target_progress = ""
            if lap_stats.get('target_laps'):
                target_progress = f" | Target Progress: {len(lap_times)}/{lap_stats['target_laps']}"
                if lap_stats.get('target_reached'):
                    target_progress += " ✅ COMPLETED"
            
            stats_text = f"""Race Summary:
Total Laps: {len(lap_times)} | Best: {self.controller.lap_counter.format_time(lap_stats['best_lap'])}
Average: {self.controller.lap_counter.format_time(lap_stats['average_lap'])} | Total Time: {self.controller.lap_counter.format_time(lap_stats['total_race'])}{target_progress}
Config: Min Lap {MIN_LAP_TIME}s | Speed {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s | Orange Gate {ORANGE_GATE_THRESHOLD}m/{ORANGE_COOLDOWN}s"""
            
            fig.text(0.02, 0.02, stats_text, fontsize=9, color=text_color, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#2c3e50', alpha=0.9, edgecolor=primary_color))
            
            # Save the comprehensive plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            target_suffix = f"_target{lap_stats.get('target_laps', 'unlimited')}" if lap_stats.get('target_laps') else ""
            config_suffix = f"_minlap{MIN_LAP_TIME}s_speed{MIN_SPEED:.0f}-{MAX_SPEED:.0f}"
            filename = f"racing_analysis{target_suffix}{config_suffix}_{timestamp}.png"
            
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=background_color, edgecolor='none')
                print(f"\n📊 Comprehensive racing analysis saved as: {filename}")
                print(f"🎯 Configuration: Target {lap_stats.get('target_laps', 'unlimited')} laps | Min lap {MIN_LAP_TIME}s | Speed {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s")
                if lap_stats.get('target_laps'):
                    print(f"🏁 Status: {len(lap_times)}/{lap_stats['target_laps']} laps | {'✅ COMPLETED' if lap_stats.get('target_reached') else '🔄 IN PROGRESS'}")
                
                # Get absolute path for user convenience
                abs_path = os.path.abspath(filename)
                print(f"📂 Full path: {abs_path}")
                
            except Exception as save_error:
                print(f"❌ Error saving comprehensive plot: {save_error}")
                # Try saving to a different location
                try:
                    home_path = os.path.expanduser("~")
                    alt_filename = os.path.join(home_path, filename)
                    plt.savefig(alt_filename, dpi=300, bbox_inches='tight', facecolor=background_color, edgecolor='none')
                    print(f"📊 Comprehensive analysis saved to home directory: {alt_filename}")
                except Exception as alt_save_error:
                    print(f"❌ Failed to save to home directory: {alt_save_error}")
            
            # Close the plot to free memory
            plt.close(fig)
            plt.close('all')
            
            print("✅ Comprehensive racing analysis chart generation completed successfully!")
            
        except Exception as e:
            print(f"❌ Error creating comprehensive analysis visualization: {e}")
            import traceback
            traceback.print_exc()
            print("🔧 Try installing: pip install matplotlib pillow")
            
            # Try to close any open plots
            try:
                plt.close('all')
            except:
                pass
    
    def setup_carla(self):
        """Initialize CARLA connection and find existing vehicle"""
        try:
            # Connect to CARLA
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            print("Connected to CARLA server")
            
            # Get world
            self.world = self.client.get_world()
            
            # Find existing vehicle instead of spawning new one
            self.vehicle = self.find_existing_vehicle()
            if not self.vehicle:
                raise RuntimeError("No existing vehicle found in the world")
            
            return True
            
        except Exception as e:
            print(f"Error setting up CARLA: {e}")
            return False
    
    def find_existing_vehicle(self):
        """Find an existing vehicle in the CARLA world"""
        try:
            # Get all actors in the world
            all_actors = self.world.get_actors()
            
            # Filter for vehicles
            vehicles = all_actors.filter('vehicle.*')
            
            if not vehicles:
                print("ERROR: No vehicles found in the world")
                print("Available actors:")
                for actor in all_actors:
                    print(f"  - {actor.type_id} (ID: {actor.id})")
                return None
            
            # Use the first vehicle found
            vehicle = vehicles[0]
            transform = vehicle.get_transform()
            location = transform.location
            
            print(f"Found existing vehicle: {vehicle.type_id} (ID: {vehicle.id})")
            print(f"Vehicle location: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
            print(f"Vehicle rotation: pitch={transform.rotation.pitch:.2f}, yaw={transform.rotation.yaw:.2f}, roll={transform.rotation.roll:.2f}")
            
            # Check if vehicle is alive
            if not vehicle.is_alive:
                print("ERROR: Found vehicle is not alive")
                return None
                
            # List all available vehicles for reference
            print(f"Available vehicles in world ({len(vehicles)} total):")
            for i, v in enumerate(vehicles):
                loc = v.get_transform().location
                print(f"  {i+1}. {v.type_id} (ID: {v.id}) at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
                
            return vehicle
            
        except Exception as e:
            print(f"Error finding existing vehicle: {str(e)}")
            return None
    
    def setup_camera_and_controller(self, model_path=None):
        """Setup ZED 2i camera and robust controller with configurable parameters"""
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
                
            # Setup improved controller with configurable parameters
            self.controller = PurePursuitController(self.vehicle, target_laps=target_laps)
            print("Camera and improved cone following controller setup complete")
            print(f"Lap counter with timing and speed tracking enabled - orange cones will be detected for lap counting")
            print(f"Enhanced with smooth cone line following and early turn detection")
            print(f"🚀 NEW: Acceleration tracking and display enabled")
            if target_laps:
                print(f"🎯 TARGET SET: Will stop automatically after {target_laps} valid laps")
            else:
                print(f"🔄 UNLIMITED MODE: Will run until manually stopped")
            
            return True
            
        except Exception as e:
            print(f"Error setting up camera and controller: {e}")
            return False
    
    def control_loop(self):
        """Main control loop for vehicle with target lap checking"""
        print("Starting configurable cone following control loop with acceleration tracking...")
        print(f"🔧 Configuration: Target {target_laps if target_laps else 'unlimited'} laps | Min lap {MIN_LAP_TIME}s | Speed {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s")
        print(f"🚀 Acceleration tracking: Requested vs Actual acceleration will be monitored")
        
        while self.running:
            try:
                # Check if target laps reached
                if self.controller and self.controller.is_target_reached():
                    print(f"🎯 TARGET LAPS REACHED! Stopping control loop...")
                    self.running = False
                    break
                
                # Process camera frame
                self.camera.process_frame()
                
                # Get cone detections
                cone_detections = getattr(self.camera, 'cone_detections', [])
                
                # Control vehicle using improved smooth cone following
                steering, speed = self.controller.control_vehicle(cone_detections)
                
                # If controller returns zeros, it means target reached and vehicle stopped
                if steering == 0.0 and speed == 0.0 and self.controller.is_target_reached():
                    print(f"🏁 Vehicle stopped - target laps completed!")
                    self.running = False
                    break
                
                time.sleep(0.05)  # 20 Hz control loop
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Brief pause before retrying
    
    def display_loop(self):
        """Display camera feed with detections and enhanced speed HUD including target progress and acceleration"""
        print("Starting display loop with acceleration tracking...")
        
        # Set CV2 to not use Qt backend if possible
        try:
            cv2.namedWindow('CARLA Racing - Acceleration Tracking', cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"Warning: Could not create CV2 window: {e}")
            print("Running in headless mode - visualization disabled but lap timing, speed tracking and acceleration tracking will still work")
            # Set flag to disable display
            self.display_enabled = False
            return
        
        self.display_enabled = True
        
        while self.running:
            try:
                if hasattr(self.camera, 'rgb_image') and self.camera.rgb_image is not None:
                    # Create visualization
                    viz_image = self.create_visualization()
                    
                    cv2.imshow('CARLA Racing - Acceleration Tracking', viz_image)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                
                time.sleep(0.033)  # ~30 FPS display
                
            except Exception as e:
                print(f"Error in display loop: {e}")
                # If display fails, continue without visualization
                self.display_enabled = False
                print("Continuing in headless mode...")
                time.sleep(0.1)
    
    def create_visualization(self):
        """Create enhanced visualization with acceleration tracking shown in HUD"""
        try:
            if not hasattr(self.camera, 'rgb_image') or self.camera.rgb_image is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
                
            viz_image = self.camera.rgb_image.copy()
            
            # Add vehicle info at the top
            vehicle_info = f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})"
            cv2.putText(viz_image, vehicle_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process detections for visualization
            cone_detections = getattr(self.camera, 'cone_detections', [])
            if cone_detections:
                blue_cones, yellow_cones, orange_cones = self.controller.process_cone_detections(cone_detections)
                
                # Get current smooth cone following target
                navigation_target = self.controller.calculate_smooth_cone_target(blue_cones, yellow_cones)
                
                # Draw target if found
                if navigation_target:
                    # Draw target point
                    depth = navigation_target.get('midpoint_y', 5.0)
                    angle = np.arctan2(navigation_target.get('midpoint_x', 0), depth)
                    px = int(640 + (angle / np.radians(45)) * 640)
                    py = int(720 - 100 - depth * 25)
                    py = max(50, min(py, 720))
                    px = max(0, min(px, 1280))
                    
                    # Color for immediate midpoint following
                    color = (0, 255, 255)    # Cyan for immediate midpoint
                    text = "MIDPOINT TARGET"
                    
                    cv2.circle(viz_image, (px, py), 15, color, -1)
                    cv2.putText(viz_image, text, (px+20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw cone indicators - limited to first 2 of each side for immediate focus
                    for blue_cone in blue_cones[:2]:  # Show first 2 blue cones only
                        blue_angle = np.arctan2(blue_cone['x'], blue_cone['y'])
                        blue_px = int(640 + (blue_angle / np.radians(45)) * 640)
                        blue_py = int(720 - 100 - blue_cone['y'] * 25)
                        blue_py = max(50, min(blue_py, 720))
                        blue_px = max(0, min(blue_px, 1280))
                        cv2.circle(viz_image, (blue_px, blue_py), 8, (255, 0, 0), -1)  # Blue
                    
                    for yellow_cone in yellow_cones[:2]:  # Show first 2 yellow cones only
                        yellow_angle = np.arctan2(yellow_cone['x'], yellow_cone['y'])
                        yellow_px = int(640 + (yellow_angle / np.radians(45)) * 640)
                        yellow_py = int(720 - 100 - yellow_cone['y'] * 25)
                        yellow_py = max(50, min(yellow_py, 720))
                        yellow_px = max(0, min(yellow_px, 1280))
                        cv2.circle(viz_image, (yellow_px, yellow_py), 8, (0, 255, 255), -1)  # Yellow
                
                # Draw orange cones (lap markers)
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
            
            # Get vehicle stats
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Get lap timing and speed statistics
            lap_stats = self.controller.lap_counter.get_lap_time_stats()
            
            # Get acceleration statistics - NEW FEATURE
            accel_stats = self.controller.get_acceleration_stats()
            
            # LEFT SIDE STATUS WITH CONFIGURATION INFO AND ACCELERATION
            left_status_text = [
                f"Mode: Acceleration Tracking System",
                f"Config: Target {target_laps if target_laps else 'unlimited'} | MinLap {MIN_LAP_TIME}s | Speed {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s",
                f"Orange Gate: {ORANGE_GATE_THRESHOLD}m threshold | {ORANGE_COOLDOWN}s cooldown",
                f"Throttle: {accel_stats['current_throttle']:.3f} | Brake: {accel_stats['current_brake']:.3f}",
                f"Requested Accel: {accel_stats['requested_acceleration']:.2f}",
                f"Actual Accel: {accel_stats['actual_acceleration']:.2f}",
                f"Cones - Blue: {len([d for d in cone_detections if d.get('cls') == 1])} | Yellow: {len([d for d in cone_detections if d.get('cls') == 0])} | Orange: {len([d for d in cone_detections if d.get('cls') == 2])}",
                f"Current Speed: {current_speed:.1f} m/s ({current_speed*3.6:.1f} km/h)",
                f"Distance: {self.controller.distance_traveled:.1f}m"
            ]
            
            for i, text in enumerate(left_status_text):
                y_pos = 50 + i*22  # Start lower to avoid vehicle info
                color = (0, 255, 0)
                if i == 0:  # Mode info
                    color = (255, 255, 0)  # Yellow for acceleration tracking mode
                elif i == 1:  # Config line
                    color = (255, 165, 0)  # Orange for configuration
                elif i == 2:  # Orange gate config
                    color = (255, 100, 0)  # Orange for gate config
                elif i == 3:  # Throttle/Brake
                    color = (255, 0, 255)  # Magenta for control inputs
                elif i == 4:  # Requested acceleration
                    color = (0, 255, 255)  # Cyan for requested acceleration
                elif i == 5:  # Actual acceleration
                    # Color code actual acceleration based on value
                    if accel_stats['actual_acceleration'] > 2.0:
                        color = (0, 255, 0)  # Green for strong acceleration
                    elif accel_stats['actual_acceleration'] > 0.5:
                        color = (0, 255, 255)  # Cyan for moderate acceleration
                    elif accel_stats['actual_acceleration'] < -3.0:
                        color = (0, 0, 255)  # Red for strong braking
                    elif accel_stats['actual_acceleration'] < -1.0:
                        color = (0, 165, 255)  # Orange for moderate braking
                    else:
                        color = (255, 255, 255)  # White for minimal acceleration
                elif i == 7:  # Current speed
                    color = (0, 255, 255)  # Cyan for current speed
                cv2.putText(viz_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            # RIGHT SIDE ENHANCED HUD WITH TARGET PROGRESS AND ACCELERATION DETAILS
            right_x_start = 850  # Start position for right-side HUD
            
            # Draw semi-transparent background for enhanced HUD section
            overlay = viz_image.copy()
            cv2.rectangle(overlay, (right_x_start - 10, 40), (1270, 620), (0, 0, 0), -1)  # Made taller for acceleration data
            cv2.addWeighted(overlay, 0.7, viz_image, 0.3, 0, viz_image)
            
            # ENHANCED HUD HEADER WITH TARGET INFO
            if lap_stats['target_laps']:
                header_text = f"TARGET: {lap_stats['target_laps']} LAPS"
                header_color = (0, 255, 0) if lap_stats['target_reached'] else (255, 165, 0)
            else:
                header_text = "UNLIMITED MODE"
                header_color = (255, 255, 0)
            
            cv2.putText(viz_image, header_text, (right_x_start, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_color, 2)
            
            # TARGET PROGRESS BAR (if target is set)
            if lap_stats['target_laps']:
                progress = min(lap_stats['valid_laps_completed'] / lap_stats['target_laps'], 1.0)
                bar_width = 300
                bar_height = 20
                bar_x = right_x_start
                bar_y = 75
                
                # Background bar
                cv2.rectangle(viz_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress bar
                progress_width = int(bar_width * progress)
                progress_color = (0, 255, 0) if lap_stats['target_reached'] else (255, 165, 0)
                cv2.rectangle(viz_image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), progress_color, -1)
                # Border
                cv2.rectangle(viz_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                
                # Progress text
                progress_text = f"{lap_stats['valid_laps_completed']}/{lap_stats['target_laps']} laps"
                if lap_stats['target_reached']:
                    progress_text += " ✅ COMPLETE"
                cv2.putText(viz_image, progress_text, (bar_x + 10, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset = 110  # Adjust subsequent elements down
            else:
                y_offset = 85
            
            # ACCELERATION SECTION - NEW FEATURE
            cv2.putText(viz_image, "ACCELERATION TRACKING:", (right_x_start, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Acceleration bars visualization
            accel_bar_y = y_offset + 35
            accel_bar_height = 15
            accel_bar_width = 200
            
            # Requested acceleration bar
            cv2.putText(viz_image, "Requested:", (right_x_start, accel_bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            req_accel_normalized = np.clip((accel_stats['requested_acceleration'] + 8) / 16, 0, 1)  # Normalize from -8 to +8 m/s²
            req_bar_width = int(accel_bar_width * req_accel_normalized)
            req_color = (0, 255, 0) if accel_stats['requested_acceleration'] > 0 else (0, 0, 255)
            cv2.rectangle(viz_image, (right_x_start + 90, accel_bar_y), (right_x_start + 90 + req_bar_width, accel_bar_y + accel_bar_height), req_color, -1)
            cv2.rectangle(viz_image, (right_x_start + 90, accel_bar_y), (right_x_start + 90 + accel_bar_width, accel_bar_y + accel_bar_height), (255, 255, 255), 1)
            cv2.putText(viz_image, f"{accel_stats['requested_acceleration']:.2f}", (right_x_start + 300, accel_bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, req_color, 2)
            
            # Actual acceleration bar
            actual_bar_y = accel_bar_y + 25
            cv2.putText(viz_image, "Actual:", (right_x_start, actual_bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            actual_accel_normalized = np.clip((accel_stats['actual_acceleration'] + 8) / 16, 0, 1)  # Normalize from -8 to +8 m/s²
            actual_bar_width = int(accel_bar_width * actual_accel_normalized)
            actual_color = (0, 255, 0) if accel_stats['actual_acceleration'] > 0 else (0, 0, 255)
            cv2.rectangle(viz_image, (right_x_start + 90, actual_bar_y), (right_x_start + 90 + actual_bar_width, actual_bar_y + accel_bar_height), actual_color, -1)
            cv2.rectangle(viz_image, (right_x_start + 90, actual_bar_y), (right_x_start + 90 + accel_bar_width, actual_bar_y + accel_bar_height), (255, 255, 255), 1)
            cv2.putText(viz_image, f"{accel_stats['actual_acceleration']:.2f}", (right_x_start + 300, actual_bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, actual_color, 2)
            
            # Acceleration efficiency
            efficiency = accel_stats['acceleration_efficiency']
            efficiency_color = (0, 255, 0) if 90 <= efficiency <= 110 else (255, 165, 0) if 70 <= efficiency <= 130 else (0, 0, 255)
            cv2.putText(viz_image, f"Efficiency: {efficiency:.1f}%", (right_x_start, actual_bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, efficiency_color, 2)
            
            # Update y_offset for remaining elements
            y_offset += 105
            
            # Current lap time (large and prominent)
            current_lap_formatted = self.controller.lap_counter.format_time(lap_stats['current_lap'])
            cv2.putText(viz_image, "Current Lap:", (right_x_start, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(viz_image, current_lap_formatted, (right_x_start, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Current speed and lap speed stats
            cv2.putText(viz_image, f"Speed: {current_speed*3.6:.1f} km/h", (right_x_start, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Speed range indicator
            speed_range_text = f"Range: {MIN_SPEED*3.6:.0f}-{MAX_SPEED*3.6:.0f} km/h"
            cv2.putText(viz_image, speed_range_text, (right_x_start, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 2)
            
            # Current lap speed statistics if available
            if lap_stats['current_lap_speeds']:
                current_speeds = lap_stats['current_lap_speeds']
                current_avg_speed = np.mean(current_speeds)
                current_max_speed = max(current_speeds)
                cv2.putText(viz_image, f"Lap Avg: {current_avg_speed:.1f} km/h", (right_x_start, y_offset + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(viz_image, f"Lap Max: {current_max_speed:.1f} km/h", (right_x_start, y_offset + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Total race time
            total_race_formatted = self.controller.lap_counter.format_time(lap_stats['total_race'])
            cv2.putText(viz_image, "Total Race:", (right_x_start, y_offset + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(viz_image, total_race_formatted, (right_x_start, y_offset + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Laps completed (show both total detections and valid laps)
            cv2.putText(viz_image, f"Total Laps: {lap_stats['laps_completed']}", (right_x_start, y_offset + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            cv2.putText(viz_image, f"Valid Laps (>{MIN_LAP_TIME}s): {lap_stats['valid_laps_completed']}", (right_x_start, y_offset + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Best lap time and speed
            if lap_stats['best_lap'] != float('inf'):
                best_lap_formatted = self.controller.lap_counter.format_time(lap_stats['best_lap'])
                cv2.putText(viz_image, "Best Lap:", (right_x_start, y_offset + 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(viz_image, best_lap_formatted, (right_x_start, y_offset + 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show speed for best lap if available
                if lap_stats['lap_speed_data']:
                    best_lap_index = lap_stats['lap_times'].index(lap_stats['best_lap'])
                    if best_lap_index < len(lap_stats['lap_speed_data']):
                        best_lap_speed = lap_stats['lap_speed_data'][best_lap_index]['avg_speed']
                        cv2.putText(viz_image, f"(@ {best_lap_speed:.1f} km/h avg)", (right_x_start, y_offset + 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 255, 128), 2)
            else:
                cv2.putText(viz_image, "Best Lap: --:--.---", (right_x_start, y_offset + 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            
            # Last lap time and speed
            if lap_stats['last_lap'] > 0:
                last_lap_formatted = self.controller.lap_counter.format_time(lap_stats['last_lap'])
                cv2.putText(viz_image, "Last Lap:", (right_x_start, y_offset + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(viz_image, last_lap_formatted, (right_x_start, y_offset + 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show speed for last lap if available
                if lap_stats['lap_speed_data']:
                    last_speed_data = lap_stats['lap_speed_data'][-1]
                    last_avg_speed = last_speed_data['avg_speed']
                    cv2.putText(viz_image, f"(@ {last_avg_speed:.1f} km/h avg)", (right_x_start, y_offset + 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 128), 2)
            else:
                cv2.putText(viz_image, "Last Lap: --:--.---", (right_x_start, y_offset + 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            
            # Show turn statistics in compact format
            if lap_stats['current_lap_turns']:
                cv2.putText(viz_image, "Current Lap Turns:", (right_x_start, y_offset + 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                current_turns = lap_stats['current_lap_turns']
                turn_text = f"S:{current_turns['straight']} G:{current_turns['gentle']} Sh:{current_turns['sharp']}"
                cv2.putText(viz_image, turn_text, (right_x_start, y_offset + 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Target completion message if reached
            if lap_stats['target_reached']:
                cv2.putText(viz_image, "🎯 TARGET COMPLETED!", (right_x_start, y_offset + 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(viz_image, "Vehicle stopping...", (right_x_start, y_offset + 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return viz_image
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            # Return a blank image if visualization fails
            blank_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank_image, "Visualization Error", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank_image
    
    def run(self, model_path=None):
        """Main execution function with configurable parameters and acceleration tracking"""
        try:
            # Setup CARLA
            if not self.setup_carla():
                return False
            
            # Setup camera and controller
            if not self.setup_camera_and_controller(model_path):
                return False
            
            print("System ready! Using existing vehicle for racing with acceleration tracking.")
            print(f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
            print("🟠 Orange cones will be detected for lap counting")
            print("⏱️  Enhanced HUD with lap times and speed data")
            print("🚗 Speed tracking per lap for comprehensive analysis")
            print("🚀 NEW: Acceleration tracking enabled - monitoring requested vs actual acceleration")
            print("🔧 Configurable racing system enabled")
            print(f"📊 Configuration summary:")
            print(f"   🎯 Target laps: {target_laps if target_laps else 'unlimited'}")
            print(f"   ⏱️  Min lap time: {MIN_LAP_TIME}s")
            print(f"   🚗 Speed range: {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s ({MIN_SPEED*3.6:.0f}-{MAX_SPEED*3.6:.0f} km/h)")
            print(f"   🟠 Orange gate: {ORANGE_GATE_THRESHOLD}m threshold, {ORANGE_COOLDOWN}s cooldown")
            print(f"   🚀 Acceleration range: -8.0 to +3.0 m/s² (typical car performance)")
            print("📊 Press Ctrl+C to stop and generate comprehensive racing analysis visualization")
            print("Press 'q' in the display window to quit (if display is available)")
            print("💡 If running headless/SSH, only Ctrl+C will work to stop and generate charts")
            
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
        """Clean up all resources without destroying the existing vehicle and generate comprehensive analysis plot"""
        print("Cleaning up resources...")
        
        self.running = False
        
        # Stop vehicle but don't destroy it since it was pre-existing
        if self.vehicle:
            try:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                self.vehicle.apply_control(control)
                print("Vehicle stopped (not destroyed - was pre-existing)")
                time.sleep(1.0)
            except:
                pass
        
        # Print final race statistics and generate comprehensive plot
        if self.controller and hasattr(self.controller, 'lap_counter'):
            lap_stats = self.controller.lap_counter.get_lap_time_stats()
            accel_stats = self.controller.get_acceleration_stats()
            
            print(f"\n{'='*60}")
            print(f"🏁 FINAL RACING STATISTICS WITH ACCELERATION DATA")
            print(f"{'='*60}")
            print(f"🔧 Configuration Used:")
            print(f"   Target laps: {target_laps if target_laps else 'unlimited'}")
            print(f"   Min lap time: {MIN_LAP_TIME}s")
            print(f"   Speed range: {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s ({MIN_SPEED*3.6:.0f}-{MAX_SPEED*3.6:.0f} km/h)")
            print(f"   Orange gate: {ORANGE_GATE_THRESHOLD}m threshold, {ORANGE_COOLDOWN}s cooldown")
            print(f"🚀 Acceleration Performance:")
            print(f"   Average Requested: {accel_stats['avg_requested_accel']:.2f} m/s²")
            print(f"   Average Actual: {accel_stats['avg_actual_accel']:.2f} m/s²")
            print(f"   Overall Efficiency: {accel_stats['acceleration_efficiency']:.1f}%")
            print(f"   Final Throttle: {accel_stats['current_throttle']:.3f}")
            print(f"   Final Brake: {accel_stats['current_brake']:.3f}")
            print(f"📊 Race Results:")
            print(f"   Total Race Time: {self.controller.lap_counter.format_time(lap_stats['total_race'])}")
            print(f"   Total Orange Gate Detections: {lap_stats['laps_completed']}")
            print(f"   Valid Laps Completed (>{MIN_LAP_TIME}s): {lap_stats['valid_laps_completed']}")
            if lap_stats['target_laps']:
                print(f"   Target Status: {'✅ COMPLETED' if lap_stats['target_reached'] else '🔄 IN PROGRESS'}")
            print(f"   Distance Traveled: {self.controller.distance_traveled:.1f}m")
            
            if lap_stats['lap_times']:
                print(f"   Best Lap Time: {self.controller.lap_counter.format_time(lap_stats['best_lap'])}")
                print(f"   Average Lap Time: {self.controller.lap_counter.format_time(lap_stats['average_lap'])}")
                
                # Show speed statistics
                if lap_stats['lap_speed_data']:
                    avg_speeds = [data['avg_speed'] for data in lap_stats['lap_speed_data']]
                    max_speeds = [data['max_speed'] for data in lap_stats['lap_speed_data']]
                    print(f"   Average Speed: {np.mean(avg_speeds):.1f} km/h")
                    print(f"   Top Speed: {max(max_speeds):.1f} km/h")
                    print(f"   Speed Consistency: {np.std(avg_speeds):.2f} km/h std dev")
                
                # Show false detection statistics
                false_detections = lap_stats['laps_completed'] - lap_stats['valid_laps_completed']
                if false_detections > 0:
                    print(f"   False Gate Detections: {false_detections} (under {MIN_LAP_TIME}s, ignored)")
                
                # Generate the comprehensive racing analysis visualization
                print(f"\n📊 Generating comprehensive racing analysis visualization...")
                self.plot_comprehensive_analysis(lap_stats)
                
            else:
                print(f"   No valid laps completed (all were under {MIN_LAP_TIME}s minimum)")
                if lap_stats['laps_completed'] > 0:
                    print(f"   Had {lap_stats['laps_completed']} orange gate detections, but all were under {MIN_LAP_TIME}s")
                print("📊 No lap times to visualize - complete some valid laps first!")
            print(f"{'='*60}")
        
        # Cleanup camera
        if self.camera:
            try:
                self.camera.shutdown()
            except:
                pass
        
        # Close CV2 windows if display was enabled
        if hasattr(self, 'display_enabled') and self.display_enabled:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print("Cleanup complete - existing vehicle preserved")
        print("📊 Check your directory for the comprehensive racing analysis PNG file!")
        print("🚀 Acceleration tracking data has been integrated into the analysis!")


def main():
    """Main function with configurable parameters and acceleration tracking"""
    model_path = '/home/aditya/hydrakon_ws/src/planning_module/planning_module/best.pt'
    
    print(f"🚀 CARLA RACING SYSTEM WITH ACCELERATION TRACKING")
    print(f"={'='*60}")
    print(f"📋 Current Configuration:")
    print(f"   🎯 Target laps: {target_laps if target_laps else 'unlimited'}")
    print(f"   ⏱️  Min lap time: {MIN_LAP_TIME}s")
    print(f"   🚗 Speed range: {MIN_SPEED:.0f}-{MAX_SPEED:.0f} m/s ({MIN_SPEED*3.6:.0f}-{MAX_SPEED*3.6:.0f} km/h)")
    print(f"   🟠 Orange gate: {ORANGE_GATE_THRESHOLD}m threshold, {ORANGE_COOLDOWN}s cooldown")
    print(f"   🚀 Acceleration tracking: Requested vs Actual acceleration monitoring")
    print(f"   🚀 Control inputs: Throttle and Brake values displayed")
    print(f"   🚀 Efficiency calculation: Performance ratio between requested and actual")
    print(f"💡 To modify: Edit the configuration section at the top of this file")
    print(f"📊 NEW FEATURES:")
    print(f"   - Real-time acceleration display in HUD")
    print(f"   - Throttle/Brake input monitoring")
    print(f"   - Acceleration efficiency tracking")
    print(f"   - Color-coded acceleration visualization")
    print(f"={'='*60}")
    
    # Create and run the racing system with acceleration tracking
    racing_system = CarlaRacingSystem()
    
    try:
        success = racing_system.run(model_path)
        if success:
            print("Racing system with acceleration tracking completed successfully")
        else:
            print("Racing system with acceleration tracking failed to start")
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        racing_system.cleanup()


if __name__ == "__main__":
    main()