import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    share_dir = get_package_share_directory('lio_sam')
    parameter_file = LaunchConfiguration('params_file')
    rviz_config_file = os.path.join(share_dir, 'config', 'rviz2.rviz')

    params_declare = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(share_dir, 'config', 'lio_sam_params.yaml'),
        description='Path to the ROS2 parameters file to use.'
    )

    return LaunchDescription([
        params_declare,

        # Static TF publisher (map -> odom)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),

        # IMU Preintegration Node
        Node(
            package='lio_sam',
            executable='imu_preintegration_node',
            name='lio_sam_imu_preintegration',
            parameters=[parameter_file],
            output='screen'
        ),

        # Image Projection Node
        Node(
            package='lio_sam',
            executable='image_projection_node',
            name='lio_sam_image_projection',
            remappings=[
            ('/imu/data', '/carla/imu_sensor'),
            ('points_raw', '/carla/lidar') 
        ],
            parameters=[parameter_file],
            output='screen'
        ),

        # Feature Extraction Node
        Node(
            package='lio_sam',
            executable='feature_extraction_node',
            name='lio_sam_feature_extraction',
            parameters=[parameter_file],
            output='screen'
        ),

        # Map Optimization Node
        Node(
            package='lio_sam',
            executable='map_optimization_node',
            name='lio_sam_map_optimization',
            parameters=[parameter_file],
            output='screen'
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
    ])
