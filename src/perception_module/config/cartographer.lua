include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = false,
  publish_frame_projected_to_2d = false, -- Important for 3D
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,
  lookup_transform_timeout_sec = 1.0,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

-- Important: Setting use_trajectory_builder_3d and disabling 2d
MAP_BUILDER.use_trajectory_builder_2d = false
MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 4

-- 3D specific settings
TRAJECTORY_BUILDER_3D.min_range = 0.5
TRAJECTORY_BUILDER_3D.max_range = 30.0
TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 2  -- Increased from 1 to accumulate more points
TRAJECTORY_BUILDER_3D.voxel_filter_size = 0.03  -- More detailed than 0.05

-- 3D Adaptive voxel filter settings (required for 3D)
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 2.0
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 150
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_length = 4.0
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 100

-- 3D scan matcher settings - made more forgiving
TRAJECTORY_BUILDER_3D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.linear_search_window = 0.3  -- Increased from 0.15
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.angular_search_window = math.rad(5.0)  -- Increased from 1.0
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 0.1  -- Reduced from 1.0
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1.0  -- Reduced from 1e3

-- Motion filter - made less restrictive
TRAJECTORY_BUILDER_3D.motion_filter.max_time_seconds = 1.0  -- Increased from 0.1
TRAJECTORY_BUILDER_3D.motion_filter.max_distance_meters = 0.5  -- Increased from 0.2
TRAJECTORY_BUILDER_3D.motion_filter.max_angle_radians = math.rad(5.0)  -- Increased from 2.0

-- Submaps
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.10
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.30
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 100
TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.hit_probability = 0.55
TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.miss_probability = 0.49

-- 3D Ceres scan matcher (made more forgiving)
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 1.0  -- Reduced from 5.0
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 1.0  -- Reduced from 10.0
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.only_optimize_yaw = false

-- IMU settings for 3D (critical for correct 3D mapping)
-- TRAJECTORY_BUILDER_3D.use_imu_data = true  -- Commented out as it's redundant with global options
TRAJECTORY_BUILDER_3D.imu_gravity_time_constant = 100.0  -- Increased from 10.0 for slower adaptation

-- Loop closure
POSE_GRAPH.optimize_every_n_nodes = 40
POSE_GRAPH.constraint_builder.sampling_ratio = 0.2
POSE_GRAPH.constraint_builder.max_constraint_distance = 15.0
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66
POSE_GRAPH.global_sampling_ratio = 0.05
POSE_GRAPH.optimization_problem.huber_scale = 1e1
POSE_GRAPH.max_num_final_iterations = 10

-- Added optimization parameters for better robustness
POSE_GRAPH.optimization_problem.rotation_weight = 3e5
POSE_GRAPH.optimization_problem.acceleration_weight = 1e3
-- Add this to your options table to extend the lookup_transform_timeout
-- options.lookup_transform_timeout_sec = 10.0  -- Increased from 1.0
-- options.publish_to_tf = true
-- options.publish_tracked_pose = true

-- Add these to make tracking more robust
options.trajectory_builder.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

return options