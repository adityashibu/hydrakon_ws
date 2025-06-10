from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    """
    Individual planning module launch - for testing planning module alone
    """
    planning_pkg_dir = FindPackageShare('planning_module')
    planning_config = PathJoinSubstitution([
        planning_pkg_dir, 'config', 'planning_params.yaml'
    ])
    
    return LaunchDescription([
        Node(
            package='planning_module',
            executable='planning_node',
            name='planning_node',
            output='screen',
            parameters=[planning_config]
        )
    ])
