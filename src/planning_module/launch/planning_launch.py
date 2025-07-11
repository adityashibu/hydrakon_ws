# planning_launch.py - Corrected launch file

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('planning_module')
    
    # Path to the parameter file
    config_file = os.path.join(pkg_dir, 'config', 'planning_params.yaml')
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/legion5/hydrakon_ws/src/planning_module/planning_module/best.pt',
        description='Path to YOLO model file'
    )
    
    # Launch the planning node
    planning_node = Node(
        package='planning_module',
        executable='planning_node',       # This should match your setup.py entry point
        name='planning_node',             # This should match parameter file top-level key
        parameters=[config_file],
        # Remove arguments since we handle model path internally now
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        model_path_arg,
        planning_node,
    ])