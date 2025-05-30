#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import queue
from collections import defaultdict
import signal
import sys

# Global variables for thread communication
cone_data_queue = queue.Queue()
current_cones = {'blue': [], 'yellow': [], 'orange': [], 'unknown': []}
node_instance = None

class ConeVisualizerNode(Node):
    def __init__(self):
        super().__init__('cone_visualizer_node')
        
        # Subscribe to cone markers
        self.subscription = self.create_subscription(
            MarkerArray,
            '/perception/cone_markers',
            self.cone_callback,
            10
        )
        
        # Color mapping for different cone types
        self.cone_colors = {
            'blue': 'blue',
            'yellow': 'yellow',
            'orange': 'orange',
            'unknown': 'gray'
        }
        
        self.get_logger().info('Cone Visualizer Node started. Listening to /perception/cone_markers')
    
    def cone_callback(self, msg):
        """Callback function for cone marker messages"""
        global cone_data_queue, current_cones
        
        cone_positions = defaultdict(list)
        
        for marker in msg.markers:
            if marker.action == marker.DELETE_ALL:
                # Clear all cones
                cone_positions = {'blue': [], 'yellow': [], 'orange': [], 'unknown': []}
                break
            elif marker.action == marker.ADD or marker.action == marker.MODIFY:
                # Extract position
                x = marker.pose.position.x
                y = marker.pose.position.y
                
                # Determine cone type based on marker properties
                cone_type = self.determine_cone_type(marker)
                cone_positions[cone_type].append([x, y])
        
        # Put data in queue for visualization thread
        while not cone_data_queue.empty():
            try:
                cone_data_queue.get_nowait()  # Remove old data
            except queue.Empty:
                break
        
        cone_data_queue.put(dict(cone_positions))
    
    def determine_cone_type(self, marker):
        """Determine cone type based on marker color or namespace"""
        # Check marker color (RGBA)
        r, g, b, a = marker.color.r, marker.color.g, marker.color.b, marker.color.a
        
        # Blue cone detection
        if b > 0.8 and r < 0.3 and g < 0.3:
            return 'blue'
        # Yellow cone detection
        elif r > 0.8 and g > 0.8 and b < 0.3:
            return 'yellow'
        # Orange cone detection
        elif r > 0.8 and g > 0.4 and g < 0.7 and b < 0.3:
            return 'orange'
        # Check namespace if color detection fails
        elif 'blue' in marker.ns.lower():
            return 'blue'
        elif 'yellow' in marker.ns.lower():
            return 'yellow'
        elif 'orange' in marker.ns.lower():
            return 'orange'
        else:
            return 'unknown'
    
def update_plot(frame):
    """Update the plot with new cone data"""
    global cone_data_queue, current_cones, ax
    
    try:
        # Get latest cone data
        new_data = cone_data_queue.get_nowait()
        current_cones.update(new_data)
    except queue.Empty:
        pass
    
    # Clear previous cone plots (keep vehicle)
    ax.clear()
    
    # Reset plot properties
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Real-time Cone Detection Visualization')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add vehicle representation
    vehicle = patches.Rectangle((-0.5, -1), 1, 2, linewidth=2, 
                              edgecolor='black', facecolor='gray', alpha=0.7)
    ax.add_patch(vehicle)
    
    # Plot cones by type
    cone_count = 0
    cone_colors = {
        'blue': 'blue',
        'yellow': 'yellow',
        'orange': 'orange',
        'unknown': 'gray'
    }
    
    for cone_type, positions in current_cones.items():
        if positions:
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=cone_colors[cone_type], s=100, 
                      label=f'{cone_type.capitalize()} Cones ({len(positions)})',
                      alpha=0.8, edgecolors='black', linewidth=1)
            cone_count += len(positions)
    
    # Add legend and info
    if cone_count > 0:
        ax.legend(loc='upper right')
    
    # Add coordinate system info
    ax.text(0.02, 0.98, f'Total Cones: {cone_count}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return []

def ros_thread_func():
    """Function to run ROS2 in a separate thread"""
    global node_instance
    
    try:
        node_instance = ConeVisualizerNode()
        rclpy.spin(node_instance)
    except Exception as e:
        print(f"ROS thread error: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nShutting down...')
    plt.close('all')
    if node_instance:
        node_instance.destroy_node()
    rclpy.shutdown()
    sys.exit(0)

def main(args=None):
    global ax
    
    rclpy.init(args=args)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start ROS2 in a separate thread
    ros_thread = threading.Thread(target=ros_thread_func, daemon=True)
    ros_thread.start()
    
    # Set up matplotlib in main thread
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create animation in main thread
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        if node_instance:
            node_instance.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()