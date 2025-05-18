# Table of Contents
- [Folder Structure](#folder-structure)
- [ROS Topics](#ros2-topics)
- [Getting Started](#getting-started)

# Folder Structure

- **scripts/**
  - Utility scripts for working with CARLA.
  - Example: [`carla_cleanup.py`](scripts/carla_cleanup.py) — cleans up the CARLA simulator before launching ROS 2 nodes.

- **src/**
  - **carla_vehicle_manager/**
    - Manages vehicle control and sensor interfaces in CARLA.
    - **Sensors:**
      - [IMU Node](src/carla_vehicle_manager/carla_vehicle_manager/imu_node.py)
      - [GNSS Node](src/carla_vehicle_manager/carla_vehicle_manager/gnss_node.py)
    - **Topics:**
      - [`/carla/imu_sensor`](#ros2-topics)
      - [`/carla/gnss`](#ros2-topics)
      - [`/carla/vehicle/*`](#ros2-topics)

  - **hydrakon_launch/**
    - Unified [launch file](src/hydrakon_launch/launch/hydrakon_launch.py) that launches all core modules.
      - **Has EKF Parameters directly baked into the launch file**

  - **lidar_cluster/**
    - Processes incoming LiDAR point clouds to detect and cluster potential cones using DBSCAN.
    - **Topics:**
      - [`/perception/lidar_cluster`](#ros2-topics)
      - [`/perception/cone_markers`](#ros2-topics)

  - **lidar_handler/**
    - Simulates the Robosense Helios 16 LiDAR using a Carla sensor wrapper.
    - **Sensor Node:**
      - [LiDAR Node](src/lidar_handler/lidar_handler/lidar_node.py)
    - **Topics:**
      - [`/carla/lidar`](#ros2-topics)

  - **perception_module/**
    - Handles perception-related transformations, localization, and sensor fusion (e.g., navsat transform).
    - **Topics:**
      - [`/gps/filtered`](#ros2-topics)
      - [`/gps/pose`](#ros2-topics)
      - [`/odometry/gps`](#ros2-topics)
      - [`/odometry/filtered`](#ros2-topics)

  - **planning_module/**
    - (To be implemented) Will handle path planning algorithms such as Pure Pursuit and MPC.

  - **zed2i_camera_sim/**
    - Simulates the ZED 2i stereo camera in Carla.
    - **Topics:**
      - [`/zed2i/camera_info`](#ros2-topics)
      - [`/zed2i/depth/image`](#ros2-topics)
      - [`/zed2i/rgb/image`](#ros2-topics)

---

# ROS2 Topics

### 🛰 Sensors & Environment
- **`/carla/imu_sensor`**  
  IMU data from Carla’s onboard IMU  
  ‣ Type: `sensor_msgs/msg/Imu`

- **`/carla/gnss`**  
  GNSS data from Carla’s onboard GPS sensor  
  ‣ Type: `sensor_msgs/msg/NavSatFix`

- **`/carla/lidar`**  
  Raw LiDAR point cloud  
  ‣ Type: `sensor_msgs/msg/PointCloud2`

- **`/zed2i/camera_info`**  
  Intrinsic camera parameters from the ZED2i simulation  
  ‣ Type: `sensor_msgs/msg/CameraInfo`

- **`/zed2i/depth/image`**  
  Depth image from the ZED2i simulation  
  ‣ Type: `sensor_msgs/msg/Image`

- **`/zed2i/rgb/image`**  
  RGB image from the ZED2i simulation  
  ‣ Type: `sensor_msgs/msg/Image`

---

### 🚘 Vehicle Control
- **`/carla/vehicle/*`**  
  Set of vehicle control topics (e.g., throttle, brake, reverse)  
  ‣ Type: Carla-specific control interfaces  
  ‣ Driven via [`keyboard_control_node.py`](src/carla_vehicle_manager/carla_vehicle_manager/keyboard_control_node.py)

---

### 🧠 Perception & Localization
- **`/perception/lidar_cluster`**  
  Clustered cone candidates from LiDAR  
  ‣ Type: `sensor_msgs/msg/PointCloud2`

- **`/perception/cone_markers`**  
  RViz markers representing clustered cones  
  ‣ Type: `visualization_msgs/msg/MarkerArray`

- **`/gps/filtered`**  
  GPS data after initial fusion/filtering  
  ‣ Type: `sensor_msgs/msg/NavSatFix`

- **`/gps/pose`**  
  GPS fused pose as `geometry_msgs/PoseStamped`  
  ‣ Type: `geometry_msgs/msg/PoseStamped`

- **`/odometry/gps`**  
  Odometry based on raw GNSS + IMU  
  ‣ Type: `nav_msgs/msg/Odometry`

- **`/odometry/filtered`**  
  Contains odometry output from `robot_localization` EKF  
  ‣ Type: `nav_msgs/msg/Odometry`

---

### 🧭 Transforms & Logs
- **`/tf`**  
  Dynamic transform tree (e.g., map → base_link)  
  ‣ Type: `tf2_msgs/msg/TFMessage`

- **`/tf_static`**  
  Static transforms (e.g., base_link → imu_link)  
  ‣ Type: `tf2_msgs/msg/TFMessage`

- **`/rosout`**  
  Internal ROS 2 logging messages (for `rqt_console`, `ros2 log`)  
  ‣ Type: `rcl_interfaces/msg/Log`

# Getting Started
1. **✅ Prerequisites**
    - Ensure the following are installed:
        - Ubuntu 22.04 (Preferred) or later
        - ROS2 Humble
        - CARLA Simulator
        - Python 3.10 
        - ``colcon``, ``rosdep``, and other ROS dev tools
        - NVIDIA GPU with drivers for Carla rendering
2. **📁 Clone the Workspace**
    ```shell
    cd ~
    git clone https://github.com/<your-username>/hydrakon_ws.git
    cd hydrakon_ws
    ```
3. **📦 Build the Workspace**
    ```shell
    source /opt/ros/humble/setup.bash
    rosdep install --from-paths src --ignore-src -r -y
    colcon build --symlink-install
    ```
4. **🔧 Source the Setup**
    ```shell
    source install/setup.bash
    ```
5. **🧪 Run Carla Simulator**
    ```shell
    cd ~/<location-to-your-Carla-sim>
    cd Linux
    ./CarlaUnreal.sh 
    ```

    Use the `-quality-level=Low` flag if needed
    ```shell
    ./CarlaUnreal.sh -quality-level=Low
    ```
6. **🚗 Launch the Full System**
    ```shell
    ros2 launch hydrakon_launch hydrakon_launch.py
    ```
7. **📊 Visualize in RViz**
    ```
    rviz2
    ```
    Open this in a new terminal, and make sure `base_link` is selected as the correct fixed frame, then go ahead and choose your desired topics to visualize