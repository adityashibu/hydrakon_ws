# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         # LIO-SAM Components
#         Node(
#             package='perception_module',
#             executable='lio_image_projection',
#             name='lio_image_projection',
#             output='screen',
#             parameters=['config/lio_sam_params.yaml'],
#         ),
#         Node(
#             package='perception_module',
#             executable='lio_imu_preintegration',
#             name='lio_imu_preintegration',
#             output='screen',
#             parameters=['config/lio_sam_params.yaml'],
#         ),
#         Node(
#             package='perception_module',
#             executable='lio_map_optimization',
#             name='lio_map_optimization',
#             output='screen',
#             parameters=['config/lio_sam_params.yaml'],
#         ),

#         # Optional: Navsat node if you're fusing GNSS later
#         # Node(
#         #     package='perception_module',
#         #     executable='navsat_transform_node',
#         #     name='navsat_transform_node',
#         #     output='screen',
#         #     parameters=['config/navsat_params.yaml'],
#         # )
#     ])
