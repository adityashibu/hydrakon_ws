from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get control module directory
    control_pkg_dir = FindPackageShare('control_module')
    
    # Config file path - SINGLE SOURCE OF TRUTH
    pid_config = PathJoinSubstitution([control_pkg_dir, 'config', 'pid_params.yaml'])
    
    # Launch arguments - ONLY for runtime overrides
    use_carla_arg = DeclareLaunchArgument(
        'use_carla',
        default_value='true',
        description='Use CARLA simulation'
    )
    
    carla_host_arg = DeclareLaunchArgument(
        'carla_host',
        default_value='localhost',
        description='CARLA host address'
    )
    
    carla_port_arg = DeclareLaunchArgument(
        'carla_port',
        default_value='2000',
        description='CARLA port number'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug logging'
    )
    
    return LaunchDescription([
        # Launch arguments
        use_carla_arg,
        carla_host_arg,
        carla_port_arg,
        debug_mode_arg,
        
        # Speed Processor Node (IMU → Speed)
        Node(
            package='control_module',
            executable='speed_processor',
            name='speed_processor',
            output='screen',
            parameters=[pid_config]  # ALL parameters from YAML
        ),
        
        # PID Controller Node
        Node(
            package='control_module',
            executable='pid_controller',
            name='pid_controller',
            output='screen',
            parameters=[pid_config]  # ALL parameters from YAML
        ),
        
        # Vehicle Interface Node
        Node(
            package='control_module',
            executable='vehicle_interface',
            name='vehicle_interface',
            output='screen',
            parameters=[
                pid_config,  # PRIMARY parameter source
                {
                    # ONLY runtime overrides here
                    'use_carla': LaunchConfiguration('use_carla'),
                    'carla_host': LaunchConfiguration('carla_host'),
                    'carla_port': LaunchConfiguration('carla_port'),
                }
            ]
        ),
    ])