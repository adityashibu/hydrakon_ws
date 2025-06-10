from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    return LaunchDescription([
        # CARLA cleanup script
        ExecuteProcess(
            cmd=['/usr/bin/python3', '/home/aditya/hydrakon_ws/scripts/carla_cleanup.py'],
            shell=False,
            output='screen'
        ),

        # Static Transform Publishers
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_tf_pub',
            arguments=['0', '0', '3.5', '0', '0', '0', 'base_link', 'lidar_link'],
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_tf_pub',
            arguments=['0', '0', '0.5', '0', '0', '0', 'base_link', 'imu_link'],
            output='screen'
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),

        # Vehicle Manager
        Node(
            package='carla_vehicle_manager',
            executable='vehicle_node',
            name='carla_vehicle_manager',
            output='screen',
            parameters=['config/vehicle_params.yaml']
        ),

        # IMU Node (delayed start)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='carla_vehicle_manager',
                    executable='imu_node',
                    name='carla_imu_node',
                    output='screen',
                    parameters=['config/imu_params.yaml']
                )
            ]
        ),

        # GNSS Node (delayed start)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='carla_vehicle_manager',
                    executable='gnss_node',
                    name='carla_gnss_node',
                    output='screen',
                    parameters=['config/gnss_params.yaml']
                )
            ]
        ),

        # NavSat Transform Node
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='perception_module',
                    executable='navsat_transform_node',
                    name='navsat_transform_node',
                    output='screen',
                    parameters=['/home/aditya/hydrakon_ws/src/perception_module/config/navsat_params.yaml']
                )
            ]
        ),

        # EKF Node (Extended Kalman Filter for localization)
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='robot_localization',
                    executable='ekf_node',
                    name='ekf_filter_node',
                    output='screen',
                    parameters=[{
                        'frequency': 30.0,
                        'sensor_timeout': 5.0,
                        'two_d_mode': True,
                        'publish_tf': True,
                        'publish_acceleration': True,
                        'print_diagnostics': True,
                        'debug': True,
                        
                        'map_frame': 'map',
                        'odom_frame': 'odom',
                        'base_link_frame': 'base_link',
                        'world_frame': 'odom',
                        
                        'imu0': '/carla/imu_sensor',
                        'imu0_config': [False, False, False,
                                       True, True, True,
                                       False, False, False,
                                       True, True, True,
                                       True, True, True],
                        'imu0_nodelay': True,
                        'imu0_differential': False,
                        'imu0_relative': True,
                        'imu0_queue_size': 10,
                        'imu0_remove_gravitational_acceleration': True,
                        
                        'odom0': '/odometry/gps',
                        'odom0_config': [True, True, False,
                                        False, False, True,
                                        True, True, False,
                                        False, False, False,
                                        False, False, False],
                        'odom0_nodelay': True,
                        'odom0_differential': False,
                        'odom0_relative': True,
                        'odom0_queue_size': 10,
                    }]
                )
            ]
        ),
        
        # LiDAR Handler
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='lidar_handler',
                    executable='lidar_node',
                    name='lidar_handler',
                    output='screen',
                    parameters=['config/lidar_params.yaml']
                )
            ]
        ),

        # LiDAR Cluster Node
        Node(
            package='lidar_cluster',
            executable='lidar_cluster_node',
            name='lidar_cluster_node',
            output='screen',
            parameters=['config/lidar_cluster.yaml']
        ),

        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='planning_module',
                    executable='planning_node',
                    name='planning_node',
                    output='screen',
                    parameters=[
                        PathJoinSubstitution([
                            FindPackageShare('planning_module'),
                            'config',
                            'planning_params.yaml'
                        ])
                    ]
                )
            ]
        ),

        # # CONTROL MODULE - Single Launch (handles all control nodes internally)
        # TimerAction(
        #     period=7.0,  # Start after planning is ready
        #     actions=[
        #         IncludeLaunchDescription(
        #             PythonLaunchDescriptionSource([
        #                 get_package_share_directory('control_module'),
        #                 '/launch/control_system.launch.py'
        #             ]),
        #             launch_arguments={
        #                 'use_carla': 'true',
        #                 'carla_host': 'localhost',
        #                 'carla_port': '2000'
        #             }.items()
        #         )
        #     ]
        # ),
    ])