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
from std_msgs.msg import Float64, Int32, Bool
from geometry_msgs.msg import Point, Vector3
from .zed_2i import Zed2iCamera
from rcl_interfaces.msg import ParameterDescriptor  


class LapCounter:
    def __init__(self, node, target_laps=None):
        self.node = node
        self.laps_completed = 0
        self.last_orange_gate_time = 0
        self.cooldown_duration = 3.0  # 3 seconds cooldown between lap counts
        self.orange_gate_passed_threshold = 2.0  # Distance threshold for passing through orange gate
        
        # Target laps functionality (from Node 2)
        self.target_laps = target_laps
        self.target_reached = False
        
        # Lap timing functionality (from Node 2)
        self.race_start_time = time.time()
        self.lap_start_time = time.time()
        self.lap_times = []  # Store individual lap times
        self.current_lap_time = 0.0
        self.best_lap_time = float('inf')
        self.last_lap_time = 0.0
        
        # Turn tracking for each lap (from Node 2)
        self.current_lap_turns = {
            'straight': 0,
            'gentle': 0,
            'sharp': 0
        }
        self.lap_turn_data = []  # Store turn counts for each completed lap
        self.last_turn_type = "straight"
        self.turn_change_cooldown = 1.0  # 1 second cooldown to prevent rapid turn type changes
        self.last_turn_change_time = 0
        
        # Speed tracking for each lap (from Node 2)
        self.current_lap_speeds = []  # Store speeds during current lap
        self.lap_speed_data = []  # Store speed statistics for each completed lap
        self.speed_sample_interval = 0.5  # Sample speed every 0.5 seconds
        self.last_speed_sample_time = 0
        
        # ROS2 publisher for lap events
        self.lap_pub = self.node.create_publisher(Int32, '/planning/lap_count', 10)
        
        # NEW: Additional ROS2 publishers for enhanced telemetry
        self.lap_time_pub = self.node.create_publisher(Float64, '/planning/lap_time', 10)
        self.best_lap_pub = self.node.create_publisher(Float64, '/planning/best_lap_time', 10)
        self.lap_speed_pub = self.node.create_publisher(Vector3, '/planning/lap_speed_stats', 10)  # avg, max, min speeds
        self.turn_stats_pub = self.node.create_publisher(Vector3, '/planning/turn_stats', 10)  # straight, gentle, sharp counts
        self.target_progress_pub = self.node.create_publisher(Vector3, '/planning/target_progress', 10)  # completed, target, reached_flag
        
        print(f"🎯 Enhanced Lap Counter initialized:")
        print(f"   Target: {target_laps if target_laps else 'UNLIMITED'} valid laps")
        print(f"   Orange gate threshold: {self.orange_gate_passed_threshold}m")
        print(f"   Orange cooldown: {self.cooldown_duration}s")
        print(f"   📊 Enhanced telemetry publishing to ROS2 topics enabled")
        
    def record_speed(self, speed_ms):
        """Record speed sample for current lap (from Node 2)"""
        current_time = time.time()
        
        # Sample speed at regular intervals
        if current_time - self.last_speed_sample_time >= self.speed_sample_interval:
            speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
            self.current_lap_speeds.append(speed_kmh)
            self.last_speed_sample_time = current_time
            
    def record_turn(self, turn_type):
        """Record a turn for the current lap (from Node 2)"""
        current_time = time.time()
        
        # Only record if turn type has changed and cooldown has passed
        if (turn_type != self.last_turn_type and 
            current_time - self.last_turn_change_time > self.turn_change_cooldown):
            
            if turn_type in self.current_lap_turns:
                self.current_lap_turns[turn_type] += 1
                self.last_turn_type = turn_type
                self.last_turn_change_time = current_time
                
                # Publish turn statistics to ROS2
                self._publish_turn_stats()
                
                print(f"DEBUG: Recorded {turn_type} turn. Current lap turns: {self.current_lap_turns}")
        
    def get_current_lap_time(self):
        """Get the current lap time in progress (from Node 2)"""
        return time.time() - self.lap_start_time
    
    def get_total_race_time(self):
        """Get total race time since start (from Node 2)"""
        return time.time() - self.race_start_time
    
    def format_time(self, time_seconds):
        """Format time in MM:SS.mmm format (from Node 2)"""
        if time_seconds == float('inf'):
            return "--:--.---"
        
        minutes = int(time_seconds // 60)
        seconds = time_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"
    
    def get_lap_time_stats(self):
        """Get comprehensive lap time statistics (from Node 2)"""
        current_lap = self.get_current_lap_time()
        total_race = self.get_total_race_time()
        
        stats = {
            'current_lap': current_lap,
            'total_race': total_race,
            'laps_completed': self.laps_completed,
            'valid_laps_completed': len(self.lap_times),  # Only count valid laps
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
    
    def _publish_lap_time(self, lap_time):
        """Publish lap time to ROS2 topic"""
        try:
            msg = Float64()
            msg.data = lap_time
            self.lap_time_pub.publish(msg)
            self.node.get_logger().info(f"Published lap time: {self.format_time(lap_time)}")
        except Exception as e:
            print(f"Error publishing lap time: {e}")
    
    def _publish_best_lap_time(self):
        """Publish best lap time to ROS2 topic"""
        try:
            msg = Float64()
            msg.data = self.best_lap_time if self.best_lap_time != float('inf') else 0.0
            self.best_lap_pub.publish(msg)
            self.node.get_logger().info(f"Published best lap time: {self.format_time(self.best_lap_time)}")
        except Exception as e:
            print(f"Error publishing best lap time: {e}")
    
    def _publish_lap_speed_stats(self, speed_stats):
        """Publish lap speed statistics to ROS2 topic"""
        try:
            msg = Vector3()
            msg.x = speed_stats.get('avg_speed', 0.0)
            msg.y = speed_stats.get('max_speed', 0.0)
            msg.z = speed_stats.get('min_speed', 0.0)
            self.lap_speed_pub.publish(msg)
            self.node.get_logger().info(f"Published lap speed stats: avg={msg.x:.1f}, max={msg.y:.1f}, min={msg.z:.1f} km/h")
        except Exception as e:
            print(f"Error publishing lap speed stats: {e}")
    
    def _publish_turn_stats(self):
        """Publish current turn statistics to ROS2 topic"""
        try:
            msg = Vector3()
            msg.x = float(self.current_lap_turns['straight'])
            msg.y = float(self.current_lap_turns['gentle'])
            msg.z = float(self.current_lap_turns['sharp'])
            self.turn_stats_pub.publish(msg)
            self.node.get_logger().debug(f"Published turn stats: straight={msg.x}, gentle={msg.y}, sharp={msg.z}")
        except Exception as e:
            print(f"Error publishing turn stats: {e}")
    
    def _publish_target_progress(self):
        """Publish target progress to ROS2 topic"""
        try:
            msg = Vector3()
            msg.x = float(len(self.lap_times))  # Valid laps completed
            msg.y = float(self.target_laps) if self.target_laps else 0.0
            msg.z = 1.0 if self.target_reached else 0.0
            self.target_progress_pub.publish(msg)
            self.node.get_logger().info(f"Published target progress: {int(msg.x)}/{int(msg.y) if self.target_laps else 'unlimited'} laps")
        except Exception as e:
            print(f"Error publishing target progress: {e}")
            
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
        """Complete a lap and update timing statistics (enhanced from Node 2)"""
        # Calculate lap time
        lap_time = current_time - self.lap_start_time
        
        # Get minimum lap time from ROS2 parameters (default 3.0 if not set)
        try:
            min_lap_time = self.node.get_parameter('lap_counter.min_lap_time').get_parameter_value().double_value
        except:
            min_lap_time = 3.0  # Default minimum lap time
        
        # Skip first "lap" if it's too short (race start)
        if self.laps_completed == 0 and lap_time < 10.0:
            print(f"🏁 RACE STARTED! Starting lap timing...")
            print(f"   Minimum lap time for counting: {self.format_time(min_lap_time)}")
            if self.target_laps:
                print(f"🎯 Target: {self.target_laps} valid laps")
        else:
            # Check if lap time meets minimum threshold
            if lap_time < min_lap_time:
                print(f"⚠️  FALSE LAP DETECTED - IGNORED!")
                print(f"   Lap time: {self.format_time(lap_time)} (under {self.format_time(min_lap_time)} minimum)")
                print(f"   This was likely a false detection from orange cone positioning")
                print(f"   Continuing current lap timing...")
                
                # Update cooldown but don't count the lap or restart timing
                self.last_orange_gate_time = current_time
                return  # Exit without counting this lap
            
            # Valid lap - record the lap time, turn data, and speed data
            self.lap_times.append(lap_time)
            self.last_lap_time = lap_time
            valid_lap_number = len(self.lap_times)
            
            # Publish lap time to ROS2
            self._publish_lap_time(lap_time)
            
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
                
                # Publish speed stats to ROS2
                self._publish_lap_speed_stats(speed_stats)
                
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
                self._publish_best_lap_time()  # Publish new best lap to ROS2
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
            
            # Publish target progress to ROS2
            self._publish_target_progress()
            
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
        
        # Publish lap completion event to ROS2 topic
        self.publish_lap_completion()
    
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
        
        # ENHANCED: Control parameters from Node 1 - using Node 1's strict immediate focus approach
        self.safety_offset = 1.75  # meters from cones - standard track width estimation
        self.max_depth = 8.0   # maximum cone detection range - reduced for immediate focus
        self.min_depth = 1.5   # minimum cone detection range
        self.max_lateral_distance = 3.0  # maximum lateral distance - reduced for immediate track
        
        # ENHANCED: Turn radius and path parameters from Node 1 - immediate track section focus
        self.min_turn_radius = 3.5  # Minimum safe turning radius (meters)
        self.lookahead_for_turns = 8.0  # Look ahead distance for turn detection - increased
        self.sharp_turn_threshold = 25.0  # Angle threshold for sharp turns (degrees)
        self.u_turn_threshold = 60.0  # Angle threshold for U-turns (degrees)
        self.turn_detection_distance = 8.0  # Distance to look ahead for turn detection - increased
        
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
        
        # ENHANCED: Turn state tracking from Node 1 - limited cone sequence for immediate focus
        self.current_turn_type = "straight"  # "straight", "gentle", "sharp", "u_turn"
        self.turn_direction = "none"  # "left", "right", "none"
        self.path_offset = 0.0  # Current path offset for wider turns
        self.cone_sequence = deque(maxlen=3)  # Track recent cones for turn prediction - limited to 3
        
        # Backup navigation when no gates found
        self.lost_track_counter = 0
        self.max_lost_track_frames = 20
        
        # Distance tracking for basic stats
        self.distance_traveled = 0.0
        self.last_position = None
        
        # ENHANCED: Cone following parameters from Node 1 - optimized for immediate midpoint following
        self.cone_follow_lookahead = 4.0  # Reduced lookahead for immediate focus
        self.early_turn_factor = 1.0  # Reduced factor for more precise control
        self.smoothing_factor = 0.3  # Reduced smoothing for more responsiveness
        
        # Initialize lap counter
        self.lap_counter = LapCounter(node)
        
        # ROS2 Publishers for control commands
        self.setup_ros_publishers()
        
    def is_target_reached(self):
        """Check if target laps have been reached"""
        return self.lap_counter.target_reached
        
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
    
    def detect_turn_type_from_cones(self, blue_cones, yellow_cones):
        """ENHANCED: Detect turn type and direction from cone patterns - Node 1's limited immediate focus approach"""
        # Use Node 1's approach: limited to first 3 cones of each side for immediate focus
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
            
            print(f"DEBUG: Node 1 cone pattern analysis (first 3 pairs) - Type: {turn_type}, Direction: {direction}, Trend: {avg_trend:.3f}")
            
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
        """ENHANCED: Process cone detections with Node 1's strict spatial filtering focused on immediate track section"""
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
                
                # ENHANCED: STRICT depth filtering for immediate track focus (from Node 1)
                if cls == 2:  # Orange cone - allow farther detection
                    if depth < 1.0 or depth > 15.0:
                        continue
                else:  # Blue/Yellow cones - very strict for immediate section only
                    if depth < 1.5 or depth > 8.0:  # Reduced from 12.0 to 8.0
                        continue
                    
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to world coordinates using ZED 2i FOV
                world_x, world_y = self.image_to_world_coords(center_x, center_y, depth)
                
                # ENHANCED: STRICT lateral distance filtering for immediate track section (from Node 1)
                if cls == 2:  # Orange cone - allow wider lateral range
                    if abs(world_x) > 8.0:  # Wider range for orange cones
                        continue
                else:  # Blue/Yellow cones - very strict for immediate track
                    if abs(world_x) > 3.0:  # Reduced from 4.0 to 3.0
                        continue
                
                # ENHANCED: STRICT forward focus angle for immediate track section (from Node 1)
                angle_to_cone = np.degrees(abs(np.arctan2(world_x, world_y)))
                if cls == 2:  # Orange cone - allow wider angle
                    if angle_to_cone > 60.0:  # Wider angle for orange cones
                        continue
                else:  # Blue/Yellow cones - very strict for immediate track
                    if angle_to_cone > 30.0:  # Reduced from 45.0 to 30.0
                        continue
                
                # ENHANCED: Additional filtering from Node 1 - ensure cones are clearly part of immediate track
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
        print(f"DEBUG: After STRICT Node 1 filtering - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
        
        return blue_cones, yellow_cones, orange_cones
    
    def calculate_smooth_cone_target(self, blue_cones, yellow_cones):
        """ENHANCED: Calculate precise midpoint target for immediate track section with Node 1's approach"""
        if not blue_cones and not yellow_cones:
            return None
        
        try:
            # ENHANCED: Limit to first 2 cones of each side for immediate track focus (from Node 1)
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
                    
                    # ENHANCED: Node 1's centerline forcing - if midpoint is too far off center, adjust
                    if abs(target_x) > 1.5:  # If midpoint too far off center, adjust
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
            
            # ENHANCED: Apply minimal smoothing to avoid oscillation (from Node 1)
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
        """ENHANCED: Calculate steering angle using pure pursuit algorithm with Node 1's immediate response approach"""
        try:
            print(f"DEBUG: Pure pursuit calculation for target ({target_x:.2f}, {target_y:.2f})")
            
            # Calculate angle to target
            alpha = np.arctan2(target_x, target_y)
            print(f"DEBUG: Alpha (angle to target): {np.degrees(alpha):.1f}°")
            
            # Calculate lookahead distance
            lookahead_dist = np.sqrt(target_x**2 + target_y**2)
            print(f"DEBUG: Lookahead distance: {lookahead_dist:.2f}m")
            
            # ENHANCED: Adaptive lookahead based on turn type from Node 1
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
            
            # ENHANCED: Apply early steering enhancement for turns from Node 1
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
            
            # Convert to degrees for ROS2 output
            steering_degrees = np.degrees(steering_angle)
            
            # Apply steering limits (keep in degrees)
            max_steering_degrees = 30.0
            steering_degrees = np.clip(steering_degrees, -max_steering_degrees, max_steering_degrees)
            
            print(f"DEBUG: Final steering command: {steering_degrees:.3f}°")
            direction = 'LEFT' if steering_degrees > 0 else 'RIGHT' if steering_degrees < 0 else 'STRAIGHT'
            print(f"DEBUG: Steering direction: {direction}")
            
            return steering_degrees
            
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
        """ENHANCED: Apply steering smoothing with Node 1's turn-type based approach"""
        try:
            self.steering_history.append(raw_steering)
            
            # ENHANCED: Node 1's adaptive smoothing based on turn requirements and lateral offset
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
            
            # ENHANCED: Node 1's adaptive rate limiting for cone following
            if self.current_turn_type == "sharp":
                max_change = 6.0  # Allow more change for sharp turns (in degrees)
            elif lateral_offset > 1.5:
                max_change = 5.5  # Moderate change for turns
            else:
                max_change = 4.5  # Normal rate limiting
            
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
        """ENHANCED: Main control function with Node 1's enhanced lost track recovery patterns"""
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG: ENHANCED CONTROL CYCLE - {len(cone_detections) if cone_detections else 0} detections from ZED 2i")
            print(f"Laps completed: {self.lap_counter.laps_completed}")
            print(f"Current turn type: {self.current_turn_type}, Direction: {self.turn_direction}")
            print(f"Lost track counter: {self.lost_track_counter}")
            print(f"Using Node 1's strict filtering and immediate response approach")
            print(f"{'='*60}")
            
            # Update distance traveled
            self.update_distance_traveled()
            
            # ENHANCED: Process cone detections with Node 1's strict filtering
            blue_cones, yellow_cones, orange_cones = self.process_cone_detections(cone_detections)
            print(f"DEBUG: Processed cones - Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}")
            
            # Check for lap completion through orange gate
            if orange_cones:
                transform = self.vehicle.get_transform()
                vehicle_position = (transform.location.x, transform.location.y, transform.location.z)
                self.lap_counter.check_orange_gate_passage(orange_cones, vehicle_position)
            
            # ENHANCED: Lost track detection with Node 1's immediate recovery steering
            if len(blue_cones) == 0 and len(yellow_cones) == 0:
                self.lost_track_counter += 1
                print(f"DEBUG: NO CONES DETECTED - lost track for {self.lost_track_counter} frames")
                
                # Node 1's immediate aggressive steering to try to find cones again
                if self.lost_track_counter <= 10:
                    # Try to steer in the direction we were last going
                    recovery_steering = np.clip(recovery_steering, -24.0, 24.0)
                    print(f"DEBUG: Applying Node 1 recovery steering: {recovery_steering:.3f}°")
                    
                    self.publish_control_targets(0.0, 5.0, 1.5, recovery_steering)
                    return recovery_steering, 1.5
                elif self.lost_track_counter <= 20:
                    # More aggressive search pattern
                    search_steering = 18.0 * np.sin(self.lost_track_counter * 0.3)
                    print(f"DEBUG: Applying Node 1 aggressive search pattern: {search_steering:.3f}°")
                    
                    self.publish_control_targets(0.0, 5.0, 1.2, search_steering)
                    return search_steering, 1.2
                else:
                    # Last resort - wide search
                    search_steering = 24.0 * np.sin(self.lost_track_counter * 0.2)
                    print(f"DEBUG: Applying Node 1 wide search pattern: {search_steering:.3f}°")
                    
                    self.publish_control_targets(0.0, 5.0, 0.8, search_steering)
                    return search_steering, 0.8
            
            # ENHANCED: Try to find a track segment using Node 1's immediate midpoint approach
            track_segment = self.calculate_smooth_cone_target(blue_cones, yellow_cones)
            
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
            
            # ENHANCED: Detect turn type using Node 1's immediate cone pattern analysis
            turn_type, turn_direction, path_offset = self.detect_turn_type_from_cones(blue_cones, yellow_cones)
            self.current_turn_type = turn_type
            self.turn_direction = turn_direction
            self.path_offset = path_offset
            
            # Get target point from Node 1's smooth cone following
            target_x = track_segment['midpoint_x']
            target_y = track_segment['midpoint_y']
            
            # Store current target for calculations
            self.current_target_x = target_x
            
            # Navigate towards the target using Node 1's approach
            raw_steering = self.calculate_pure_pursuit_steering(target_x, target_y)
            smooth_steering = self.smooth_steering(raw_steering)
            
            # Calculate adaptive speed based on turn type
            current_depth = track_segment['avg_depth']
            target_speed = self.calculate_adaptive_speed(turn_type, smooth_steering, current_depth)
            
            # PUBLISH TO ROS2 TOPICS INSTEAD OF APPLYING TO CARLA VEHICLE
            self.publish_control_targets(target_x, target_y, target_speed, smooth_steering)
            
            # Enhanced debug output with Node 1 features
            direction = 'LEFT' if smooth_steering > 0 else 'RIGHT' if smooth_steering < 0 else 'STRAIGHT'
            
            print(f"DEBUG: PUBLISHED ENHANCED CONTROL TARGETS TO ROS2:")
            print(f"  Navigation: {track_segment.get('type', 'immediate_midpoint')}_{turn_type}")
            print(f"  Turn Analysis: {turn_type}-{turn_direction} (Node 1 limited 3-cone approach)")
            print(f"  Target: ({target_x:.2f}, {target_y:.2f})")
            print(f"  Target distance: {current_depth:.2f}m")
            print(f"  Turn radius: {self.calculate_turn_radius(np.radians(smooth_steering)):.2f}m")
            print(f"  Steering: {smooth_steering:.3f}° ({direction})")
            print(f"  Target Speed: {target_speed:.1f} m/s ({target_speed*3.6:.1f} km/h)")
            print(f"  Distance: {self.distance_traveled:.1f}m")
            print(f"  Laps: {self.lap_counter.laps_completed}")
            print(f"  PUBLISHED TO ROS2 TOPICS WITH NODE 1 ENHANCEMENTS")
            print(f"{'='*60}\n")
            
            return smooth_steering, target_speed
            
        except Exception as e:
            print(f"ERROR in enhanced vehicle control: {e}")
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
        
        self.get_logger().info("Enhanced CARLA Racing System ROS2 Node with Node 1 improvements initialized")
        self.get_logger().info(f"YOLO model path: {model_path}")
        
    def declare_all_parameters(self):
        """Declare all ROS2 parameters with default values and descriptions"""
        
        # Vehicle parameters
        self.declare_parameter('vehicle.wheelbase', 2.7, 
                              descriptor=ParameterDescriptor(description='Vehicle wheelbase in meters'))
        self.declare_parameter('vehicle.max_steering_degrees', 30.0,
                              descriptor=ParameterDescriptor(description='Maximum steering angle in degrees'))
        
        # Lap counter parameters
        self.declare_parameter('lap_counter.min_lap_time', 3.0,
                            descriptor=ParameterDescriptor(description='Minimum lap time for validation in seconds'))
        self.declare_parameter('lap_counter.target_laps', 0,
                            descriptor=ParameterDescriptor(description='Target number of laps (0 = unlimited)'))
        self.declare_parameter('lap_counter.speed_sample_interval', 0.5,
                            descriptor=ParameterDescriptor(description='Speed sampling interval in seconds'))
        
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
        
        # ENHANCED: Detection parameters with Node 1's strict values as defaults
        self.declare_parameter('detection.safety_offset', 1.75,
                              descriptor=ParameterDescriptor(description='Safety offset from cones in meters'))
        self.declare_parameter('detection.max_depth', 8.0,
                              descriptor=ParameterDescriptor(description='Maximum cone detection range in meters (Node 1: strict 8.0m)'))
        self.declare_parameter('detection.min_depth', 1.5,
                              descriptor=ParameterDescriptor(description='Minimum cone detection range in meters'))
        self.declare_parameter('detection.max_lateral_distance', 3.0,
                              descriptor=ParameterDescriptor(description='Maximum lateral distance from vehicle center in meters (Node 1: strict 3.0m)'))
        
        # Turn parameters
        self.declare_parameter('turns.min_turn_radius', 3.5,
                              descriptor=ParameterDescriptor(description='Minimum safe turning radius in meters'))
        self.declare_parameter('turns.path_widening_factor', 1.8,
                              descriptor=ParameterDescriptor(description='How much to widen the path in turns'))
        self.declare_parameter('turns.sharp_turn_threshold', 25.0,
                              descriptor=ParameterDescriptor(description='Angle threshold for sharp turns in degrees'))
        self.declare_parameter('turns.u_turn_threshold', 60.0,
                              descriptor=ParameterDescriptor(description='Angle threshold for U-turns in degrees'))
        self.declare_parameter('turns.turn_detection_distance', 8.0,
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
                              descriptor=ParameterDescriptor(description='Only consider cones within this angle from vehicle heading in degrees (Node 1: strict 30.0°)'))
        
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
        
        self.get_logger().info("✅ All ROS2 parameters declared successfully with Node 1 enhancements")
    
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
        """Setup ZED 2i camera and enhanced controller with YOLO model and Node 1 improvements"""
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
                
            # Setup enhanced controller with ROS2 node reference and Node 1 improvements
            lookahead_distance = self.get_parameter('control.lookahead_distance').get_parameter_value().double_value
            self.controller = PurePursuitController(self.vehicle, self, lookahead_distance)
            self.get_logger().info("Camera with YOLO model and enhanced controller with Node 1 improvements setup complete")
            self.get_logger().info(f"YOLO model loaded: {self.model_path}")
            self.get_logger().info("CONTROLLER NOW PUBLISHES TO ROS2 TOPICS WITH NODE 1 ENHANCEMENTS")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error setting up camera and controller: {e}")
            return False
    
    def control_loop(self):
        """Main control loop for vehicle with YOLO detection and Node 1 enhancements"""
        self.get_logger().info("Starting enhanced control loop with YOLO detection and Node 1 improvements...")
        
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
                
                # Control vehicle using enhanced controller with Node 1 improvements - NOW PUBLISHES TO ROS2
                steering, speed = self.controller.control_vehicle(cone_detections)
                
                time.sleep(control_period)
                
            except Exception as e:
                self.get_logger().error(f"Error in control loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def display_loop(self):
        """Display camera feed with detections and Node 1 enhancements"""
        self.get_logger().info("Starting display loop with Node 1 enhancements...")
        
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
                    
                    cv2.imshow('CARLA Racing with ROS2 Output - Enhanced with Node 1 Improvements', viz_image)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                time.sleep(display_period)
                
            except Exception as e:
                self.get_logger().error(f"Error in display loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
    
    def create_visualization(self):
        """Create visualization with YOLO detections and Node 1 enhancements"""
        try:
            if not hasattr(self.camera, 'rgb_image') or self.camera.rgb_image is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
                
            viz_image = self.camera.rgb_image.copy()
            
            # Add vehicle info at the top
            vehicle_info = f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id}) - ROS2 OUTPUT with Node 1 Enhancements - ZED 2i"
            cv2.putText(viz_image, vehicle_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process detections for visualization - ZED 2i format with Node 1 filtering
            cone_detections = getattr(self.camera, 'cone_detections', [])
            if cone_detections:
                blue_cones, yellow_cones, orange_cones = self.controller.process_cone_detections(cone_detections)
                
                # The ZED 2i already draws detection boxes in its own visualization
                # Just add our additional tracking info with Node 1 approach
                
                # Try to find current target using Node 1's immediate midpoint approach
                track_segment = self.controller.calculate_smooth_cone_target(blue_cones, yellow_cones)
                
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
                    
                    # Color for Node 1 enhanced navigation
                    color = (255, 0, 255)  # Magenta for Node 1 enhanced immediate midpoint
                    text = "NODE 1 TARGET"
                    
                    cv2.circle(viz_image, (px, py), 15, color, -1)
                    cv2.putText(viz_image, text, (px+20, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add enhanced status text with Node 1 info and cone filtering metrics
            velocity = self.vehicle.get_velocity()
            current_speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Calculate Node 1 cone filtering statistics
            cone_detections = getattr(self.camera, 'cone_detections', [])
            blue_cones, yellow_cones, orange_cones = self.controller.process_cone_detections(cone_detections)
            
            status_text = [
                f"Mode: ROS2 OUTPUT with Node 1 Enhancements (ZED 2i)",
                f"YOLO Model: {self.model_path.split('/')[-1] if self.model_path else 'Not loaded'}",
                f"ZED 2i Detections: {len(cone_detections)} total",
                f"Node 1 Filtered: Blue: {len(blue_cones)}, Yellow: {len(yellow_cones)}, Orange: {len(orange_cones)}",
                f"Node 1 Features: Strict 3.0m lateral, 30° angle, immediate focus",
                f"Laps Completed: {self.controller.lap_counter.laps_completed}",
                f"Current Speed: {current_speed:.1f} m/s ({current_speed*3.6:.1f} km/h)",
                f"Distance: {self.controller.distance_traveled:.1f}m",
                f"Last Steering: {self.controller.last_steering:.3f}°",
                f"Lost Track: {self.controller.lost_track_counter}",
                f"Publishing to ROS2 Topics with Node 1 Logic"
            ]
            
            for i, text in enumerate(status_text):
                y_pos = 50 + i*20  # Start lower to avoid vehicle info
                color = (0, 255, 0)
                if i == 0:  # Mode info
                    color = (255, 0, 255)  # Magenta for enhanced ROS2 mode
                elif i == 1:  # YOLO model info
                    color = (0, 255, 255)  # Cyan for YOLO
                elif i == 2 or i == 3:  # Detection info
                    color = (255, 255, 0)  # Yellow for detections
                elif i == 4:  # Node 1 features
                    color = (255, 100, 255)  # Pink for Node 1 features
                elif i == 5:  # Lap counter
                    color = (0, 255, 0)  # Green for lap counter
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
            cv2.putText(blank_image, "Enhanced Visualization Error", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank_image
    
    def run_system(self):
        """Main execution function with Node 1 enhancements"""
        try:
            # Setup CARLA
            if not self.setup_carla():
                return False
            
            # Setup camera and controller with YOLO and Node 1 enhancements
            if not self.setup_camera_and_controller():
                return False
            
            # Log all parameter values for debugging
            self.log_parameter_values()
            
            self.get_logger().info("Enhanced system ready! Using existing vehicle for racing with ROS2 output and Node 1 improvements.")
            self.get_logger().info(f"Vehicle: {self.vehicle.type_id} (ID: {self.vehicle.id})")
            self.get_logger().info(f"YOLO model: {self.model_path}")
            self.get_logger().info("🟠 Orange cones will be detected for lap counting")
            self.get_logger().info("🔍 ZED 2i YOLO model running for cone detection")
            self.get_logger().info("🎯 Enhanced with Node 1's strict filtering and immediate response")
            self.get_logger().info("🔧 Node 1 improvements: Immediate track focus, strict cone filtering, early turn detection")
            self.get_logger().info("🤖 Controller publishes to ROS2 topics with Node 1 logic:")
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
            self.get_logger().error(f"Error running enhanced system: {e}")
            return False
        finally:
            self.cleanup()
    
    def log_parameter_values(self):
        """Log all parameter values for debugging"""
        self.get_logger().info("📋 Current parameter values with Node 1 enhancements:")
        
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
        self.get_logger().info("Cleaning up enhanced system resources...")
        
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
        
        self.get_logger().info("Enhanced cleanup complete - ROS2 topics will stop publishing with Node 1 improvements")

def main(args=None):
    """Main function for enhanced ROS2 with Node 1 improvements"""
    rclpy.init(args=args)
    
    # Default YOLO model path
    model_path = '/home/aditya/hydrakon_ws/src/planning_module/planning_module/best.pt'
    
    print(f"Using YOLO model: {model_path}")
    print("Enhanced with Node 1's strict filtering and immediate response algorithms")
    
    racing_system = CarlaRacingSystemROS2(model_path=model_path)
    
    try:
        success = racing_system.run_system()
        if success:
            racing_system.get_logger().info("Enhanced racing system with ROS2 output and Node 1 improvements completed successfully")
        else:
            racing_system.get_logger().error("Enhanced racing system with ROS2 output and Node 1 improvements failed to start")
    except KeyboardInterrupt:
        racing_system.get_logger().info("Received interrupt signal")
    except Exception as e:
        racing_system.get_logger().error(f"Unexpected error: {e}")
    finally:
        racing_system.cleanup()
        rclpy.shutdown()

if __name__ == "__main__":
    main()