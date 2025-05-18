"""
NavSat Transform Node for CARLA Simulation with ROS2

This node transforms GPS data from CARLA into odometry messages in the world frame.
It subscribes to IMU and GNSS topics from CARLA and publishes odometry that can be used
by robot_localization or other nodes.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import pyproj
import threading
from geometry_msgs.msg import TransformStamped, Quaternion, PoseStamped
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener



class NavsatTransformNode(Node):
    def __init__(self):
        super().__init__('navsat_transform_node')
        
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('magnetic_declination_radians', 0.0)
        self.declare_parameter('yaw_offset', 1.5707963)  # Default PI/2 (IMU 0 = north)
        self.declare_parameter('zero_altitude', True)
        self.declare_parameter('publish_filtered_gps', True)
        self.declare_parameter('broadcast_utm_transform', True)
        self.declare_parameter('use_odometry_yaw', False)
        self.declare_parameter('wait_for_datum', True)
        self.declare_parameter('datum_latitude', 0.0)
        self.declare_parameter('datum_longitude', 0.0)
        self.declare_parameter('datum_yaw', 0.0)
        
        self.frequency = self.get_parameter('frequency').value
        self.magnetic_declination = self.get_parameter('magnetic_declination_radians').value
        self.yaw_offset = self.get_parameter('yaw_offset').value
        self.zero_altitude = self.get_parameter('zero_altitude').value
        self.publish_filtered_gps = self.get_parameter('publish_filtered_gps').value
        self.broadcast_utm_transform = self.get_parameter('broadcast_utm_transform').value
        self.use_odometry_yaw = self.get_parameter('use_odometry_yaw').value
        self.wait_for_datum = self.get_parameter('wait_for_datum').value
        
        self.transform_good = False
        self.gps_frame_id = "gps"
        self.utm_frame_id = "utm"
        self.world_frame_id = "map"
        self.base_link_frame_id = "base_link"
        
        self.utm_proj = pyproj.Proj(proj='utm', zone=33, ellps='WGS84')  # Default UTM zone
        
        self.transform_lock = threading.Lock()
        
        self.datum_latitude = self.get_parameter('datum_latitude').value
        self.datum_longitude = self.get_parameter('datum_longitude').value
        self.datum_yaw = self.get_parameter('datum_yaw').value
        
        if not self.wait_for_datum and (self.datum_latitude != 0.0 or self.datum_longitude != 0.0):
            self.set_datum(self.datum_latitude, self.datum_longitude, self.datum_yaw)
        
        self.latest_imu_msg = None
        self.latest_gps_msg = None
        self.latest_odom_msg = None
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ROS subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            '/carla/imu_sensor',
            self.imu_callback,
            10)
            
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/carla/gnss',
            self.gps_callback,
            10)
            
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10)
        
        self.gps_odom_pub = self.create_publisher(
            Odometry,
            '/odometry/gps',
            10)
        
        self.pose_pub = self.create_publisher(
            PoseStamped, 
            '/gps/pose', 
            10)
            
        if self.publish_filtered_gps:
            self.filtered_gps_pub = self.create_publisher(
                NavSatFix,
                '/gps/filtered',
                10)
        
        self.timer = self.create_timer(1.0/self.frequency, self.publish_gps_odom)
        
        self.get_logger().info("NavSat Transform Node initialized")
    
    def set_datum(self, latitude, longitude, yaw):
        """
        Set the datum (reference point) for the UTM transformation.
        """
        with self.transform_lock:
            self.get_logger().info(f"Setting datum: lat={latitude}, lon={longitude}, yaw={yaw}")
            
            utm_x, utm_y = self.utm_proj(longitude, latitude)
            self.utm_origin_x = utm_x
            self.utm_origin_y = utm_y
            
            self.transform_yaw = yaw
            self.transform_good = True
    
    def imu_callback(self, msg):
        """
        Callback for processing IMU data.
        """
        self.latest_imu_msg = msg
        
    def gps_callback(self, msg):
        """
        Callback for processing GPS data.
        """
        self.latest_gps_msg = msg

        # self.get_logger().info(f"Received GNSS fix: status={msg.status.status}, lat={msg.latitude}, lon={msg.longitude}")
        
        # self.get_logger().info("Setting datum now...")
        if not self.wait_for_datum and not self.transform_good and msg.status.status >= 0:
            # Get yaw from IMU if available
            yaw = 0.0
            if self.latest_imu_msg is not None:
                q = self.latest_imu_msg.orientation
                yaw = self.get_yaw_from_quaternion(q)
            self.set_datum(msg.latitude, msg.longitude, yaw)
    
    def odom_callback(self, msg):
        """
        Callback for processing filtered odometry data.
        """
        self.latest_odom_msg = msg

        if self.use_odometry_yaw and not self.transform_good and self.latest_gps_msg is not None:
            q = msg.pose.pose.orientation
            yaw = self.get_yaw_from_quaternion(q)
            
            self.set_datum(self.latest_gps_msg.latitude, self.latest_gps_msg.longitude, yaw)
    
    def get_yaw_from_quaternion(self, q):
        """
        Extract yaw angle from quaternion.
        """
        # Convert quaternion to Euler angles
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(t3, t4)

        yaw = yaw + self.magnetic_declination + self.yaw_offset
        
        return yaw
    
    def publish_gps_odom(self):
        """
        Main function to publish the GPS odometry.
        """
        self.get_logger().info("Timer fired")

        if not self.transform_good or self.latest_gps_msg is None or self.latest_imu_msg is None:
            self.get_logger().warn(f"Transform status: {self.transform_good}, GPS: {self.latest_gps_msg is not None}, IMU: {self.latest_imu_msg is not None}")
            return
        
        # Get the latest messages
        gps_msg = self.latest_gps_msg
        imu_msg = self.latest_imu_msg
        
        # Check if GPS has a valid fix
        if gps_msg.status.status < 0:
            self.get_logger().warn("GPS status invalid, skipping transform")
            return
        
        try:
            # Convert GPS to UTM
            utm_x, utm_y = self.utm_proj(gps_msg.longitude, gps_msg.latitude)
            self.get_logger().info(f"UTM raw: x={utm_x}, y={utm_y}")
            
            map_x = utm_x - self.utm_origin_x
            map_y = utm_y - self.utm_origin_y
            self.get_logger().info(f"Delta map: x={map_x}, y={map_y}")
            
            cos_yaw = math.cos(self.transform_yaw)
            sin_yaw = math.sin(self.transform_yaw)
            map_x_rotated = map_x * cos_yaw + map_y * sin_yaw
            map_y_rotated = -map_x * sin_yaw + map_y * cos_yaw
            
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = self.world_frame_id
            odom_msg.child_frame_id = self.base_link_frame_id
            
            odom_msg.pose.pose.position.x = map_x_rotated
            odom_msg.pose.pose.position.y = map_y_rotated
            
            if self.zero_altitude:
                odom_msg.pose.pose.position.z = 0.0
            else:
                odom_msg.pose.pose.position.z = gps_msg.altitude
            
            if not self.use_odometry_yaw:
                odom_msg.pose.pose.orientation = imu_msg.orientation
            else:
                if self.latest_odom_msg is not None:
                    odom_msg.pose.pose.orientation = self.latest_odom_msg.pose.pose.orientation
                else:
                    odom_msg.pose.pose.orientation = imu_msg.orientation
            
            if self.latest_odom_msg is None:
                odom_msg.pose.covariance = [
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 99999.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 99999.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 99999.0
                ]
            else:
                h_acc = gps_msg.position_covariance[0]  # Horizontal accuracy
                v_acc = gps_msg.position_covariance[8]  # Vertical accuracy
                
                odom_msg.pose.covariance = [
                    h_acc, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, h_acc, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, v_acc, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 99999.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 99999.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 99999.0
                ]
            self.gps_odom_pub.publish(odom_msg)
            
            if self.broadcast_utm_transform:
                self.broadcast_utm_to_map_transform(utm_x, utm_y, gps_msg.altitude)
            
            if self.publish_filtered_gps:
                self.publish_filtered_gps_message(odom_msg)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = odom_msg.header.stamp
            pose_msg.header.frame_id = odom_msg.header.frame_id
            pose_msg.pose = odom_msg.pose.pose

            self.pose_pub.publish(pose_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error processing GPS data: {str(e)}")
    
    def broadcast_utm_to_map_transform(self, utm_x, utm_y, altitude):
        """
        Broadcast the transform from UTM to Map frame.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.utm_frame_id
        t.child_frame_id = self.world_frame_id
        
        t.transform.translation.x = self.utm_origin_x
        t.transform.translation.y = self.utm_origin_y
        
        if self.zero_altitude:
            t.transform.translation.z = 0.0
        else:
            t.transform.translation.z = altitude
        
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(self.transform_yaw / 2.0)
        q.w = math.cos(self.transform_yaw / 2.0)
        t.transform.rotation = q
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_filtered_gps_message(self, odom_msg):
        """
        Publish a filtered GPS message by transforming the odometry back to GPS coordinates.
        """
        filtered_gps = NavSatFix()
        filtered_gps.header.stamp = self.get_clock().now().to_msg()
        filtered_gps.header.frame_id = self.gps_frame_id
        
        cos_yaw = math.cos(-self.transform_yaw)  # Inverse transform
        sin_yaw = math.sin(-self.transform_yaw)  # Inverse transform
        
        x_utm = odom_msg.pose.pose.position.x * cos_yaw - odom_msg.pose.pose.position.y * sin_yaw
        y_utm = odom_msg.pose.pose.position.x * sin_yaw + odom_msg.pose.pose.position.y * cos_yaw
        
        x_utm += self.utm_origin_x
        y_utm += self.utm_origin_y
        
        # Transform from UTM back to GPS
        lon, lat = self.utm_proj(x_utm, y_utm, inverse=True)
        
        filtered_gps.latitude = lat
        filtered_gps.longitude = lon
        
        if self.zero_altitude:
            filtered_gps.altitude = 0.0
        else:
            filtered_gps.altitude = odom_msg.pose.pose.position.z
        
        if self.latest_gps_msg is not None:
            filtered_gps.status = self.latest_gps_msg.status
            filtered_gps.position_covariance = self.latest_gps_msg.position_covariance
            filtered_gps.position_covariance_type = self.latest_gps_msg.position_covariance_type
        
        self.filtered_gps_pub.publish(filtered_gps)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = NavsatTransformNode()
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


if __name__ == '__main__':
    main()
