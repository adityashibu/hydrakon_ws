// File: src/image_projection.cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <deque>
#include <mutex>
#include <vector>

using std::placeholders::_1;
using PointType = pcl::PointXYZI;

// Custom point type matching LIO-SAM
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

using PointXYZIRT = VelodynePointXYZIRT;

// Custom CloudInfo message structure (simplified)
struct CloudInfo
{
    std_msgs::msg::Header header;
    
    std::vector<int> start_ring_index;
    std::vector<int> end_ring_index;
    std::vector<int> point_col_ind;
    std::vector<float> point_range;
    
    bool imu_available = false;
    bool odom_available = false;
    
    float imu_roll_init = 0.0;
    float imu_pitch_init = 0.0;
    float imu_yaw_init = 0.0;
    
    float initial_guess_x = 0.0;
    float initial_guess_y = 0.0;
    float initial_guess_z = 0.0;
    float initial_guess_roll = 0.0;
    float initial_guess_pitch = 0.0;
    float initial_guess_yaw = 0.0;
    
    sensor_msgs::msg::PointCloud2 cloud_deskewed;
};

class ImageProjectionNode : public rclcpp::Node
{
private:
    // Configuration parameters
    static const int N_SCAN = 16;           // Number of scan rings
    static const int Horizon_SCAN = 1800;   // Horizontal resolution
    static const int queueLength = 2000;
    
    const float lidarMinRange = 1.0;
    const float lidarMaxRange = 1000.0;
    const int downsampleRate = 1;
    
    // Mutexes for thread safety
    std::mutex imuLock;
    std::mutex odoLock;
    
    // ROS2 subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_extracted_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_full_cloud_;
    
    // Data queues
    std::deque<sensor_msgs::msg::Imu> imuQueue;
    std::deque<nav_msgs::msg::Odometry> odomQueue;
    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;
    
    // Current processing data
    sensor_msgs::msg::PointCloud2 currentCloudMsg;
    
    // IMU integration arrays
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];  
    double *imuRotZ = new double[queueLength];

    // Topic names (to be loaded from ROS params)
    std::string imu_topic_;
    std::string odom_topic_;
    std::string lidar_topic_;

    
    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;
    
    // Point clouds
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    
    // Processing flags and matrices
    int deskewFlag;
    cv::Mat rangeMat;
    
    // Odometry deskewing
    bool odomDeskewFlag;
    float odomIncreX, odomIncreY, odomIncreZ;
    
    // Cloud info and timing
    CloudInfo cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::msg::Header cloudHeader;

    // Utility functions
    double stamp2Sec(const builtin_interfaces::msg::Time& stamp)
    {
        return stamp.sec + stamp.nanosec * 1e-9;
    }
    
    float pointDistance(const PointType& p)
    {
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    }
    
    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu& imu_in)
    {
        // Simple pass-through for now - add coordinate transformation if needed
        return imu_in;
    }
    
   void imuRPY2rosRPY(const sensor_msgs::msg::Imu* imu, float* roll, float* pitch, float* yaw)
{
    tf2::Quaternion orientation;
    tf2::fromMsg(imu->orientation, orientation);
    double r, p, y;
    tf2::Matrix3x3(orientation).getRPY(r, p, y);
    *roll = static_cast<float>(r);
    *pitch = static_cast<float>(p);
    *yaw = static_cast<float>(y);
}

    
    void imuAngular2rosAngular(const sensor_msgs::msg::Imu* imu, double* x, double* y, double* z)
    {
        *x = imu->angular_velocity.x;
        *y = imu->angular_velocity.y;
        *z = imu->angular_velocity.z;
    }

public:
    ImageProjectionNode() : Node("lio_sam_image_projection"), deskewFlag(0)
{
    // Declare and get parameters
    this->declare_parameter<std::string>("imu_topic", "/imu/data");
    this->declare_parameter<std::string>("odom_topic", "/odometry/imu");
    this->declare_parameter<std::string>("lidar_topic", "/points_raw");

    this->get_parameter("imu_topic", imu_topic_);
    this->get_parameter("odom_topic", odom_topic_);
    this->get_parameter("lidar_topic", lidar_topic_);

    rclcpp::SensorDataQoS sensor_qos;

    // Initialize subscribers with remappable topics
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, sensor_qos, std::bind(&ImageProjectionNode::pointCloudCallback, this, _1));
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, sensor_qos, std::bind(&ImageProjectionNode::imuCallback, this, _1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(10), std::bind(&ImageProjectionNode::odomCallback, this, _1));

    // Initialize publishers
    pub_extracted_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "lio_sam/deskew/cloud_deskewed", 1);
    pub_full_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "lio_sam/deskew/cloud_full", 1);

    allocateMemory();
    resetParameters();

    RCLCPP_INFO(this->get_logger(), "ImageProjection node initialized.");
}

    
    ~ImageProjectionNode()
    {
        delete[] imuTime;
        delete[] imuRotX;
        delete[] imuRotY;
        delete[] imuRotZ;
    }

private:
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        
        fullCloud->points.resize(N_SCAN * Horizon_SCAN);
        
        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);
        cloudInfo.point_col_ind.assign(N_SCAN * Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN * Horizon_SCAN, 0);
        
        resetParameters();
    }
    
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        
        // Reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        
        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;
        
        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }
    
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg);
        
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }
    
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odomMsg);
    }
    
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (!cachePointCloud(msg))
            return;
            
        if (!deskewInfo())
            return;
            
        projectPointCloud();
        cloudExtraction();
        publishClouds();
        resetParameters();
    }
    
    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
    {
        // Cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;
            
        // Convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        
        // Convert to internal format (assuming CARLA provides basic PointXYZI)
        pcl::PointCloud<PointType>::Ptr tempCloud(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(currentCloudMsg, *tempCloud);
        
        // Convert to PointXYZIRT format (simulate ring and time)
        laserCloudIn->points.resize(tempCloud->size());
        laserCloudIn->is_dense = tempCloud->is_dense;
        
        for (size_t i = 0; i < tempCloud->size(); i++)
        {
            auto &src = tempCloud->points[i];
            auto &dst = laserCloudIn->points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            
            // Calculate ring based on vertical angle
            float verticalAngle = atan2(dst.z, sqrt(dst.x*dst.x + dst.y*dst.y)) * 180.0 / M_PI;
            dst.ring = std::max(0, std::min(N_SCAN-1, (int)((verticalAngle + 15.0) / 2.0)));
            
            // Simulate time (linear progression through scan)
            dst.time = (float)i / tempCloud->size() * 0.1; // Assume 100ms scan time
        }
        
        // Get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = stamp2Sec(cloudHeader.stamp);
        timeScanEnd = timeScanCur + 0.1; // Assume 100ms scan time
        
        // Remove NaN points manually to avoid template instantiation issues
        std::vector<int> validIndices;
        for (size_t i = 0; i < laserCloudIn->points.size(); ++i)
        {
            const auto& point = laserCloudIn->points[i];
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
            {
                validIndices.push_back(i);
            }
        }
        
        // Create new cloud with only valid points
        pcl::PointCloud<PointXYZIRT>::Ptr cleanCloud(new pcl::PointCloud<PointXYZIRT>());
        cleanCloud->points.reserve(validIndices.size());
        for (int idx : validIndices)
        {
            cleanCloud->points.push_back(laserCloudIn->points[idx]);
        }
        cleanCloud->width = cleanCloud->points.size();
        cleanCloud->height = 1;
        cleanCloud->is_dense = true;
        
        laserCloudIn = cleanCloud;
        
        // Check deskew capability
        if (deskewFlag == 0)
        {
            deskewFlag = 1; // Assume we have time information
            RCLCPP_INFO(this->get_logger(), "Deskew function enabled");
        }
        
        return true;
    }
    
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);
        
        // Make sure IMU data available for the scan
        if (imuQueue.empty() ||
            stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for IMU data ...");
            return false;
        }
        
        imuDeskewInfo();
        odomDeskewInfo();
        
        return true;
    }
    
    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;
        
        while (!imuQueue.empty())
        {
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }
        
        if (imuQueue.empty())
            return;
            
        imuPointerCur = 0;
        
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);
            
            // Get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            if (currentImuTime > timeScanEnd + 0.01)
                break;
                
            if (imuPointerCur == 0)
            {
                imuRotX[0] = 0; imuRotY[0] = 0; imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }
            
            // Get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);
            
            // Integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }
        
        --imuPointerCur;
        
        if (imuPointerCur <= 0)
            return;
            
        cloudInfo.imu_available = true;
    }
    
    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;
        
        if (odomQueue.empty())
            return;
            
        // Simple odometry deskewing implementation
        // You can expand this based on your needs
        auto odomMsg = odomQueue.back();
        
        cloudInfo.initial_guess_x = odomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = odomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = odomMsg.pose.pose.position.z;
        
        tf2::Quaternion orientation;
        tf2::fromMsg(odomMsg.pose.pose.orientation, orientation);
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;
        
        cloudInfo.odom_available = true;
    }
    
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        
        if (!cloudInfo.imu_available || imuPointerCur <= 0)
            return;
            
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        }
        else
        {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }
    
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imu_available == false)
            return *point;
            
        double pointTime = timeScanCur + relTime;
        
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(0, 0, 0, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }
        
        // Transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(0, 0, 0, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;
        
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;
        
        return newPoint;
    }
    
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        
        // Range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;
                
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
                
            if (rowIdn % downsampleRate != 0)
                continue;
                
            // Calculate column index
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;
                
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
                
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
                
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }
    
    void cloudExtraction()
    {
        int count = 0;
        
        // Extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // Mark the points' column index for marking occlusion later
                    cloudInfo.point_col_ind[count] = j;
                    // Save range info
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);
                    // Save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // Size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.end_ring_index[i] = count - 1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        
        // Publish extracted cloud
        sensor_msgs::msg::PointCloud2 extractedCloudMsg;
        pcl::toROSMsg(*extractedCloud, extractedCloudMsg);
        extractedCloudMsg.header = cloudHeader;
        extractedCloudMsg.header.frame_id = "base_link";
        pub_extracted_cloud_->publish(extractedCloudMsg);
        
        // Publish full cloud
        sensor_msgs::msg::PointCloud2 fullCloudMsg;
        pcl::toROSMsg(*fullCloud, fullCloudMsg);
        fullCloudMsg.header = cloudHeader;
        fullCloudMsg.header.frame_id = "base_link";
        pub_full_cloud_->publish(fullCloudMsg);
        
        RCLCPP_INFO(this->get_logger(), "Published clouds - extracted: %ld points, full: %ld points", 
                   extractedCloud->size(), fullCloud->size());
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProjectionNode>());
    rclcpp::shutdown();
    return 0;
}