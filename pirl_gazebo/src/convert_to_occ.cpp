#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <gridmap3D/Grid3D.h>

#define VIS_OCCUPIED_CELLS

class MappingServer {
protected:
    const std::string   FIXED_FRAME_ID  = "base_footprint";
    const double        RESOLUTION      = 0.02;

    const double        MAXIMUM_RANGE   = 3.0;
    const double        MAXIMUM_X  = 1.2;
    const double        MINIMUM_X  = 0.2;
    const double        MAXIMUM_Y  = 0.7;
    const double        MINIMUM_Y  = -0.7;
    const double        MAXIMUM_Z  = 1.2;
    const double        MINIMUM_Z  = 0.2;

public:
    // Constructor
    MappingServer() : nh() {
        pointcloud_subscriber = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/velodyne_points2", 1);
        tf_pointcloud_subscriber = new tf::MessageFilter<sensor_msgs::PointCloud2>(*pointcloud_subscriber, tf_listener, FIXED_FRAME_ID, 1);
        tf_pointcloud_subscriber->registerCallback(boost::bind(&MappingServer::update_occupancy_map, this, _1));
#ifdef VIS_OCCUPIED_CELLS
        occupied_cells_publisher = nh.advertise<sensor_msgs::PointCloud2>("/gridmap3d/occupied_cells", 1);
#endif

        occupancy_gridmap = new gridmap3D::Grid3D(RESOLUTION);
        occupancy_gridmap->setProbHit(0.5);
        occupancy_gridmap->setProbMiss(0.2);
        std::cout << occupancy_gridmap->getProbHit() << std::endl;
        std::cout << occupancy_gridmap->getProbMiss() << std::endl;
    }
    // Destructor
    ~MappingServer() {
        delete pointcloud_subscriber;
        delete tf_pointcloud_subscriber;

        delete occupancy_gridmap;
    }

    /*
     *
     */
    void update_occupancy_map(const sensor_msgs::PointCloud2ConstPtr& _src_pc) {
        // Pose of the sensor frame
        tf::StampedTransform sensorToWorldTf;
        try{
            tf_listener.lookupTransform(FIXED_FRAME_ID, _src_pc->header.frame_id, _src_pc->header.stamp, sensorToWorldTf);
        }
        catch(tf::TransformException& e){
            ROS_ERROR_STREAM("Transform error of sensor data: " << e.what() << ", quitting callback");
            return;
        }

        gridmap3D::point3d origin(sensorToWorldTf.getOrigin().x(), sensorToWorldTf.getOrigin().y(), sensorToWorldTf.getOrigin().z());
        gridmap3D::Pointcloud pointcloud;
        filter_pointcloud(*_src_pc, sensorToWorldTf, pointcloud);

        delete occupancy_gridmap;
        occupancy_gridmap = new gridmap3D::Grid3D(RESOLUTION);
        occupancy_gridmap->insertPointCloudRays(pointcloud, origin);

#ifdef VIS_OCCUPIED_CELLS
        if(occupied_cells_publisher.getNumSubscribers() > 0)
            publish_occupied_cells();
#endif
    }


protected:
    // Node handle
    ros::NodeHandle nh;
    // Pointcloud subscribers
    message_filters::Subscriber<sensor_msgs::PointCloud2>*  pointcloud_subscriber;
    tf::MessageFilter<sensor_msgs::PointCloud2>*            tf_pointcloud_subscriber;
    tf::TransformListener                                   tf_listener;
#ifdef VIS_OCCUPIED_CELLS
    // Map publisher
    ros::Publisher                                          occupied_cells_publisher;
#endif

    // Occupancy grid
    gridmap3D::Grid3D*  occupancy_gridmap;


    /*
     *
     */
    void filter_pointcloud(const sensor_msgs::PointCloud2& _src_pc, const tf::StampedTransform& _transform, gridmap3D::Pointcloud& _dst_pc) {
        // in sensor coordinate ========================================================================================
        pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
        pcl::fromROSMsg(_src_pc, pcl_pointcloud);

        pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud_in_sensor_coordinate;
        for(const auto& point : pcl_pointcloud) {
            // Remove NaN data
            if(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
                continue;

            // Remove out of sensing range
            double l = std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            if(l > MAXIMUM_RANGE)
                continue;

            pcl_pointcloud_in_sensor_coordinate.push_back(point);
        }

        // in world coordinate =========================================================================================
        Eigen::Matrix4f sensorToWorld;
        pcl_ros::transformAsMatrix(_transform, sensorToWorld);
        pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud_in_world_coordinate;
        pcl::transformPointCloud(pcl_pointcloud_in_sensor_coordinate, pcl_pointcloud_in_world_coordinate, sensorToWorld);

        for(const auto& point : pcl_pointcloud_in_world_coordinate) {
            // Remove ground
            if(point.x < MINIMUM_X)
                continue;

            // Remove floor
            if(point.x > MAXIMUM_X)
                continue;

            // Remove ground
            if(point.y < MINIMUM_Y)
                continue;

            // Remove floor
            if(point.y > MAXIMUM_Y)
                continue;

            // Remove ground
            if(point.z < MINIMUM_Z)
                continue;

            // Remove floor
            if(point.z > MAXIMUM_Z)
                continue;

            if(point.x < 0.3 && point.x > -0.3 && point.y < 0.3 && point.y > -0.3 && point.z < 1.2){
                continue;
            }

            _dst_pc.push_back(point.x, point.y, point.z);
        }
    }

    /*
     *
     */
    void publish_occupied_cells() {
        pcl::PointCloud<pcl::PointXYZ> pcl_pointcloud;
        for(auto it = occupancy_gridmap->getGrid()->begin(); it != occupancy_gridmap->getGrid()->end(); it++) {
            if(occupancy_gridmap->isNodeOccupied(it->second)) {
                gridmap3D::point3d center = occupancy_gridmap->keyToCoord(it->first);
                pcl_pointcloud.push_back(pcl::PointXYZ(center.x(), center.y(), center.z()));
            }
        }

        sensor_msgs::PointCloud2 msg_pointcloud;
        pcl::toROSMsg(pcl_pointcloud, msg_pointcloud);
        msg_pointcloud.header.frame_id = FIXED_FRAME_ID;
        msg_pointcloud.header.stamp = ros::Time::now();
        msg_pointcloud.header.seq = 0;

//         std::cout << pcl_pointcloud.size() << std::endl;

        occupied_cells_publisher.publish(msg_pointcloud);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "rcik_data_sensing");
    ros::NodeHandle nh;

    MappingServer mapping_server;

    try{
        ros::spin();
    }
    catch(std::runtime_error& e) {
        ROS_ERROR("Exception: %s", e.what());
        return -1;
    }

    return 0;
}