from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for perception module including the new sensor fusion node.
    """
    # Launch arguments for fusion node
    fusion_rate_arg = DeclareLaunchArgument(
        'fusion_rate',
        default_value='20.0',
        description='Rate (Hz) at which fusion is performed'
    )
    
    cone_cluster_distance_arg = DeclareLaunchArgument(
        'cone_cluster_distance',
        default_value='0.5',
        description='Maximum distance between points to be considered the same cone (in meters)'
    )
    
    min_points_per_cone_arg = DeclareLaunchArgument(
        'min_points_per_cone',
        default_value='2',
        description='Minimum number of points to constitute a cone'
    )
    
    use_height_filtering_arg = DeclareLaunchArgument(
        'use_height_filtering',
        default_value='false',
        description='Whether to use height-based filtering (disabled by default)'
    )
    
    distance_threshold_arg = DeclareLaunchArgument(
        'distance_threshold',
        default_value='50.0',
        description='Maximum distance from vehicle to detect cones (meters)'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable more detailed logging'
    )
    
    # Original perception node
    perception_node = Node(
        package='perception_module',
        executable='peception_node',  # Keep the original spelling as in your file
        name='perception_module',
        output='screen',
        parameters=['config/perception_params.yaml']
    )
    
    # New fusion node
    fusion_node = Node(
        package='perception_module',
        executable='fusion_timer_node',
        name='fusion_timer_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'fusion_rate': LaunchConfiguration('fusion_rate'),
            'cone_cluster_distance': LaunchConfiguration('cone_cluster_distance'),
            'min_points_per_cone': LaunchConfiguration('min_points_per_cone'),
            'use_height_filtering': LaunchConfiguration('use_height_filtering'),
            'distance_threshold': LaunchConfiguration('distance_threshold'),
            'debug_mode': LaunchConfiguration('debug_mode'),
            # Default values for other parameters
            'gnss_weight': 0.3,
            'imu_weight': 0.7,
            'ground_threshold': -0.1,
            'max_height': 1.0,
            'use_ekf': True,
        }]
    )

    diagnostic_node = Node(
        package='perception_module',
        executable='diagnostic_fusion_node',
        name='diagnostic_fusion_node',
        output='screen',
        emulate_tty=True
    )
    
    return LaunchDescription([
        # Launch arguments
        fusion_rate_arg,
        cone_cluster_distance_arg,
        min_points_per_cone_arg,
        use_height_filtering_arg,
        distance_threshold_arg,
        debug_mode_arg,
        
        # Nodes
        perception_node,
        fusion_node,
        diagnostic_node
    ])