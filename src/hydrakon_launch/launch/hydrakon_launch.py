from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        ExecuteProcess(
            cmd=['/usr/bin/python3', '/home/aditya/hydrakon_ws/scripts/carla_cleanup.py'],
            shell=False,
            output='screen'
        ),

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

        # Node(
        # package='tf2_ros',
        # executable='static_transform_publisher',
        # name='map_to_base_link_tf',
        # arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        # output='screen'
        # ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),

        Node(
            package='carla_vehicle_manager',
            executable='vehicle_node',
            name='carla_vehicle_manager',
            output='screen',
            parameters=['config/vehicle_params.yaml']
        ),

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
        # TimerAction(
        #     period=2.0,
        #     actions=[
        #         Node(
        #             package='zed2i_camera_sim',
        #             executable='zed_node',
        #             name='zed_camera_sim_node',
        #             output='screen',
        #             parameters=['config/zed_camera_params.yaml']
        #         )
        #     ]
        # ),

        TimerAction(
            period=6.0,  # Start after EKF and other systems are ready
            actions=[
                Node(
                    package='planning_module',
                    executable='planning_node',
                    name='planning_node',
                    output='screen',
                )
            ]
        ),

        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', 'config/setup_config.rviz'],
        #     output='screen'
        # )

        Node(
            package='lidar_cluster',
            executable='lidar_cluster_node',
            name='lidar_cluster_node',
            output='screen',
            parameters=['config/lidar_cluster.yaml']
        )
    ])
