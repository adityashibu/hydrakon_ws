import rclpy
from rclpy.node import Node

class FusionTimerNode(Node):
    def __init__(self):
        super().__init__('fusion_timer_node')

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz
        self.get_logger().info("Fusion timer node initialized at 20Hz")

    def timer_callback(self):
        # Placeholder: fuse data or call fusion function here
        self.get_logger().info("Fusion timer callback triggered")

def main(args=None):
    rclpy.init(args=args)
    node = FusionTimerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()