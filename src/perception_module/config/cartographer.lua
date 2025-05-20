-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",  -- IMU link frame
  published_frame = "base_link",     -- Usually odom frame
  odom_frame = "odom",          -- Usually odom frame
  provide_odom_frame = false,   -- We already have odometry from EKF
  publish_frame_projected_to_2d = true, -- Formula Student is usually 2D
  use_odometry = true,          -- Use odometry from EKF
  use_nav_sat = false,          -- We handle GPS separately through navsat_transform
  use_landmarks = false,        -- No landmarks in Formula Student
  num_laser_scans = 0,          -- Not using laser scan
  num_multi_echo_laser_scans = 0, -- Not using multi-echo laser
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,         -- Using point cloud from LiDAR
  lookup_transform_timeout_sec = 1.0,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3, -- 200Hz for racing
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

-- Tuned for racing with a focus on real-time performance
MAP_BUILDER.use_trajectory_builder_2d = true
MAP_BUILDER.num_background_threads = 2  -- Use multiple threads for performance

-- Adjust for high-speed racing
TRAJECTORY_BUILDER_2D.min_range = 0.5  -- Reduced min range
TRAJECTORY_BUILDER_2D.max_range = 30.0  -- Increased max range
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0  -- Adjust based on track size
TRAJECTORY_BUILDER_2D.use_imu_data = true  -- Use IMU for better orientation
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true  -- Better scan matching at speed
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(2.0)  -- Less restrictive
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.2  -- Less restrictive
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.1  -- Tune for racing speeds
TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.05

-- Tune ceres scan matcher for racing
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight = 20.0  
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 1.0  -- Reduced
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Real-time loop closing
POSE_GRAPH.optimize_every_n_nodes = 40  -- Less frequent optimizations
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.optimization_problem.huber_scale = 1e1
POSE_GRAPH.constraint_builder.sampling_ratio = 0.2  -- Sample fewer constraints for performance
POSE_GRAPH.global_sampling_ratio = 0.05  -- Sample fewer nodes for global search
POSE_GRAPH.max_num_final_iterations = 10  -- More iterations for better quality

-- Tune based on your environment
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 100  -- More range data per submap
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05  -- 5cm resolution is good for cones


return options