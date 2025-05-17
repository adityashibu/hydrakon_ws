# Folder Structure
- **scripts:**
    - This has all utility scripts to use with carla, such as ``carla_cleanup.py`` which is used to cleanup the carla simulator when launching ros2 topics

- **src:**
    - **carla_vehicle_manager:**
        - Handles all the vehicle related components in Carla, including vehicle launch, vehicle control and the onboard IMU and GNSS sensors
        - **Sensors:**
            - [IMU Node](src/carla_vehicle_manager/carla_vehicle_manager/imu_node.py)
            - [GNSS Node](src/carla_vehicle_manager/carla_vehicle_manager/gnss_node.py)

    - **hydrakon_launch:**
        - Main workspace launcher, a unified [launch file](src/hydrakon_launch/launch/hydrakon_launch.py) under the launch folder handles and launches all nodes called from within that file

    - **lidar_cluster:**
        - Package to cluser all incoming lidar points to create clusters for potential cones. Uses ``DBScan``, and publishes the following ROS2 Topics:
        - **Topics:**
           - [/carla/lidar](#ros2-topics)
           - [/perception/lidar_cluster](#ros2-topics)
           - [/perception/cone_markers](#ros2-topics)

# ROS2 Topics
- **/carla/lidar:** Raw LiDAR data-points
    - Type: PointCloud2
- **/perception/lidar_cluster:** Filtered and clustered LiDAR data-points
    - Type: PointCloud2
- **/perception/cone_markers:** Visualize cone markers at the positions of the clustered data-points
    - Type: MarkerArray