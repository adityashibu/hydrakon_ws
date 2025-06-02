#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

struct smoothness_t {
    float value;
    size_t ind; 
};

struct by_value {
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer {
public:
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo;

    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction(const rclcpp::NodeOptions &options) :
        ParamServer("lio_sam_featureExtraction", options)
    {
        // Subscribe to output of image projection node
        subLaserCloudInfo = create_subscription<lio_sam::msg::CloudInfo>(
            "/lio_sam/imageProjection/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));

        // Publish for use in mapOptimization
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "/lio_sam/mapping/cloud_info", qos);
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/lio_sam/mapping/cloud_corner", 1);
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/lio_sam/mapping/cloud_surface", 1);

        initializationValue();
    }

    void initializationValue() {
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN * Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
        cloudLabel = new int[N_SCAN * Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn) {
        cloudInfo = *msgIn;
        cloudHeader = msgIn->header;
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud);

        calculateSmoothness();
        markOccludedPoints();
        extractFeatures();
        publishFeatureCloud();
    }

    void calculateSmoothness() {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) {
            float diffRange = cloudInfo.point_range[i - 5] + cloudInfo.point_range[i - 4] +
                              cloudInfo.point_range[i - 3] + cloudInfo.point_range[i - 2] +
                              cloudInfo.point_range[i - 1] - cloudInfo.point_range[i] * 10 +
                              cloudInfo.point_range[i + 1] + cloudInfo.point_range[i + 2] +
                              cloudInfo.point_range[i + 3] + cloudInfo.point_range[i + 4] +
                              cloudInfo.point_range[i + 5];

            cloudCurvature[i] = diffRange * diffRange;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints() {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 6; ++i) {
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i + 1];
            int columnDiff = std::abs(int(cloudInfo.point_col_ind[i + 1] - cloudInfo.point_col_ind[i]));

            if (columnDiff < 10) {
                if (depth1 - depth2 > 0.3) {
                    for (int j = -5; j <= 0; ++j) cloudNeighborPicked[i + j] = 1;
                } else if (depth2 - depth1 > 0.3) {
                    for (int j = 1; j <= 6; ++j) cloudNeighborPicked[i + j] = 1;
                }
            }

            float diff1 = std::abs(float(cloudInfo.point_range[i - 1] - cloudInfo.point_range[i]));
            float diff2 = std::abs(float(cloudInfo.point_range[i + 1] - cloudInfo.point_range[i]));
            if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.02 * cloudInfo.point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures() {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++) {
            surfaceCloudScan->clear();

            for (int j = 0; j < 6; j++) {
                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep) continue;

                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) {
                        largestPickedNum++;
                        if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else break;

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            if (std::abs(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]) > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            if (std::abs(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]) > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold) {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {
                            if (std::abs(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]) > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            if (std::abs(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]) > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0) {
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory() {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();
    }

    void publishFeatureCloud() {
        freeCloudInfoMemory();
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints, cornerCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;
    auto FE = std::make_shared<FeatureExtraction>(options);
    exec.add_node(FE);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
