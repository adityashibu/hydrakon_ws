# Folder Structure
- **scripts:**
    - This has all utility scripts to use with carla, such as ``carla_cleanup.py`` which is used to cleanup the carla simulator when launching ros2 topics

- **src:**
    - **carla_vehicle_manager:**
        - Package that handles all the vehicle related components in Carla, including vehicle launch, vehicle control and the onboard IMU and GNSS sensors
        - **Sensors:**
            - [IMU Node](src/carla_vehicle_manager/carla_vehicle_manager/imu_node.py)
            - [GNSS Node](src/carla_vehicle_manager/carla_vehicle_manager/gnss_node.py)
        - The package also publishes the following ROS2 Topics:
        - **Topics:**
            - [/carla/imu_sensor](#ros2-topics)
            - [/carla/gnss](#ros2-topics)
            - [/carla/vehicle/*](#ros2-topics)

    - **hydrakon_launch:**
        - Main workspace launcher, a unified [launch file](src/hydrakon_launch/launch/hydrakon_launch.py) under the launch folder handles and launches all nodes called from within that file

    - **lidar_cluster:**
        - Package to cluser all incoming lidar points to create clusters for potential cones. Uses ``DBScan``, and publishes the following ROS2 Topics:
        - **Topics:**
           - [/perception/lidar_cluster](#ros2-topics)
           - [/perception/cone_markers](#ros2-topics)

    - **lidar_handler:**
        - Package to setup the base LiDAR that simulates the Robosense Helios 16 LiDAR and publishes the following ROS2 Topics:
        - **Sensors:**
            - [LiDAR Node](src/lidar_handler/lidar_handler/lidar_node.py)
        - **Topics:**
           - [/carla/lidar](#ros2-topics)

# ROS2 Topics
- **/carla/imu_sensor:** IMU Data directly from the onboard IMU sensor
    - Type: IMU Data
- **/carla/gnss:** GNSS Data directly from the onboard GNSS sensor
    - Type: GNSS Data
- **/carla/vehicle/*:** ROS2 topics for controlling the car, does not to be manually changed, instead called from by running the [keyboard_control_node.py](/src/carla_vehicle_manager/carla_vehicle_manager/keyboard_control_node.py)
    - Type: Carla vehicle control data
- **/carla/lidar:** Raw LiDAR data-points
    - Type: PointCloud2
- **/perception/lidar_cluster:** Filtered and clustered LiDAR data-points
    - Type: PointCloud2
- **/perception/cone_markers:** Visualize cone markers at the positions of the clustered data-points
    - Type: MarkerArray