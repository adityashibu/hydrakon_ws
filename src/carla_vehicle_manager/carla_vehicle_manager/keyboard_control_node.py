#!/usr/bin/env python3
"""
Keyboard Control Node for CARLA Vehicle

This node captures keyboard inputs and publishes vehicle control commands
as ROS topics. Run this in a separate terminal window while your vehicle
node is running through the launch file.
"""

import rclpy
from rclpy.node import Node
import sys
import termios
import tty
import select
import threading
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32

class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__('keyboard_control_node')
        
        self.declare_parameter("throttle_step", 0.1)
        self.declare_parameter("steering_step", 0.1)
        self.declare_parameter("brake_step", 0.2)
        
        self.throttle_step = self.get_parameter("throttle_step").get_parameter_value().double_value
        self.steering_step = self.get_parameter("steering_step").get_parameter_value().double_value
        self.brake_step = self.get_parameter("brake_step").get_parameter_value().double_value
        
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.reverse = False
        self.handbrake = False
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/carla/vehicle/cmd_vel', 10)
        self.throttle_pub = self.create_publisher(Float32, '/carla/vehicle/throttle', 10)
        self.brake_pub = self.create_publisher(Float32, '/carla/vehicle/brake', 10)
        self.handbrake_pub = self.create_publisher(Bool, '/carla/vehicle/handbrake', 10)
        self.reverse_pub = self.create_publisher(Bool, '/carla/vehicle/reverse', 10)
        
        self.exit_event = threading.Event()
        self.keyboard_thread = threading.Thread(target=self.keyboard_control_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        self.timer = self.create_timer(0.05, self.publish_controls)  # 20Hz update rate
        self.print_control_help()
    
    def print_control_help(self):
        """Print the keyboard control instructions."""
        help_text = """
CARLA Vehicle Keyboard Controls:
-------------------------------
W/S : Throttle/Brake
A/D : Steering Left/Right
Q   : Reset controls (zero throttle, steering, brake)
R   : Toggle reverse
SPACE : Toggle handbrake
ESC : Exit

Current control values will be displayed in real-time.
"""
        print(help_text)
    
    def get_key(self, timeout=0.1):
        """
        Get a single keypress with a timeout.
        Returns the key or None if timeout is reached.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
                return key
            else:
                return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def keyboard_control_loop(self):
        """Process keyboard inputs and update control values."""
        self.get_logger().info("Keyboard control active. Use WASD keys to drive.")
        
        while not self.exit_event.is_set():
            key = self.get_key()
            
            if key is None:
                continue
                
            if key == '\x1b':  # ESC key
                self.exit_event.set()
                self.get_logger().info("Exiting...")
                break
                
            if key == 'w':  # Throttle up
                self.throttle = min(1.0, self.throttle + self.throttle_step)
                self.brake = 0.0  # Release brake when accelerating
                
            elif key == 's':  # Brake/Throttle down
                if self.throttle > 0:
                    self.throttle = max(0.0, self.throttle - self.throttle_step)
                else:
                    self.brake = min(1.0, self.brake + self.brake_step)
                    
            elif key == 'a':  # Steer left
                self.steer = max(-1.0, self.steer - self.steering_step)
                
            elif key == 'd':  # Steer right
                self.steer = min(1.0, self.steer + self.steering_step)
                
            elif key == 'q':  # Reset controls
                self.throttle = 0.0
                self.steer = 0.0
                self.brake = 0.0
                
            elif key == 'r':  # Toggle reverse
                self.reverse = not self.reverse
                self.get_logger().info(f"Reverse: {'Enabled' if self.reverse else 'Disabled'}")
                
            elif key == ' ':  # Spacebar - Toggle handbrake
                self.handbrake = not self.handbrake
                self.get_logger().info(f"Handbrake: {'Enabled' if self.handbrake else 'Disabled'}")
            
            # Print current control values
            self.display_control_values()
    
    def display_control_values(self):
        """Display the current control values."""
        print(f"\rThrottle: {self.throttle:.1f} | "
              f"Steer: {self.steer:.1f} | "
              f"Brake: {self.brake:.1f} | "
              f"Reverse: {'On' if self.reverse else 'Off'} | "
              f"Handbrake: {'On' if self.handbrake else 'Off'}    ", 
              end='')
    
    def publish_controls(self):
        """Publish the control commands as ROS topics."""
        try:
            throttle_msg = Float32()
            throttle_msg.data = float(self.throttle)
            self.throttle_pub.publish(throttle_msg)
            
            brake_msg = Float32()
            brake_msg.data = float(self.brake)
            self.brake_pub.publish(brake_msg)
            
            cmd_vel = Twist()
            cmd_vel.linear.x = float(self.throttle - self.brake)
            cmd_vel.angular.z = float(self.steer)
            self.cmd_vel_pub.publish(cmd_vel)
            
            handbrake_msg = Bool()
            handbrake_msg.data = self.handbrake
            self.handbrake_pub.publish(handbrake_msg)
            
            reverse_msg = Bool()
            reverse_msg.data = self.reverse
            self.reverse_pub.publish(reverse_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing controls: {str(e)}")
    
    def destroy_node(self):
        """Clean up resources when the node is destroyed."""
        self.get_logger().info("Shutting down keyboard control node...")
        self.exit_event.set()
        
        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1.0)
            
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = KeyboardControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()
        
        # Reset terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    main()
