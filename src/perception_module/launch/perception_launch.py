from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Use source directory paths directly
    src_dir = os.path.expanduser("~/hydrakon_ws/src/perception_module")
    config_dir = os.path.join(src_dir, "config")
    
    # Print confirmation
    lua_file = os.path.join(config_dir, 'cartographer.lua')
    # print(f"Using direct path to config directory: {config_dir}")
    # print(f"Lua file path: {lua_file}")
    # print(f"Lua file exists: {os.path.exists(lua_file)}")
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )
    
    # Cartographer node with direct path to source files
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', config_dir,
            '-configuration_basename', 'cartographer.lua'
        ],
        remappings=[
            ('points2', '/carla/lidar'),
            ('imu', '/carla/imu_sensor'),
            ('odom', '/odometry/filtered')
        ]
    )
    
    occupancy_grid_node = Node(
    package='cartographer_ros',
    executable='cartographer_occupancy_grid_node',
    name='cartographer_occupancy_grid_node',
    output='screen',
    parameters=[{
        'use_sim_time': use_sim_time,
        'resolution': 0.05,
        'publish_period_sec': 0.5,  # Faster publishing
        'include_frozen_submaps': True,
        'include_unfrozen_submaps': True,
        'track_unknown_space': False
    }]
    )
    # Add RViz to your launch file
    
    rviz_node = Node(
    package='rviz2',
    executable='rviz2',
    name='rviz2',
    arguments=['-d', os.path.join(config_dir, 'cartographer.rviz')],
    output='screen'
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        cartographer_node,
        occupancy_grid_node,
        rviz_node
    ])