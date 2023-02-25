/* Author: Mincheul Kang */

#ifndef TORM_TORM_DEBUG_H
#define TORM_TORM_DEBUG_H

#include <ros/ros.h>
#include <eigen_conversions/eigen_kdl.h>

#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseArray.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit/planning_scene/planning_scene.h>

#include <torm/torm_ik_solver.h>

namespace torm
{
    class TormDebug
    {
    public:
        TormDebug(const planning_scene::PlanningSceneConstPtr& planning_scene,
                  std::string planning_group,
                  std::string frame_id,
                  std::string planning_base_link,
                  TormIKSolver& iksolver);
        virtual ~TormDebug(){};

        void visualizeConfiguration(std::vector<std::string> &indices, std::vector<double> &conf);
        void visualizeConfiguration(std::vector<std::string> &indices, KDL::JntArray &conf);
        void visualizeTrajectory(std::vector<std::string> &indices, std::vector<std::vector<double>> &traj);
        void visualizeTrajectory(std::vector<std::string> &indices, std::vector<KDL::JntArray> &traj);


        void publishEETrajectory(std::vector<std::vector<double>>& path, int target_idx);
        void publishEETrajectory(std::vector<KDL::Frame>& targetPoses, int target_idx);
        void make_EEpose_line(geometry_msgs::Point p1, geometry_msgs::Point p2, int color);
        void publish_EEpose_line_and_arrow();


        void show(std::vector<KDL::Frame>& targetPoses, Eigen::Matrix<double, -1, -1> optimizedJoints,
                  double sleep_time, int plot_idx=2, int n_middle_interpol=0);

        void addContactVector(double px, double py, double pz, double dx, double dy, double dz, bool normalize = false);
        void publishContactVectors();
        void clearContactVectors();

        void clear();
    private:

        int num_joints_;

        std::string planning_group_;
        std::string frame_id_;
        std::string planning_base_link_;
        KDL::Frame baseKDL_;

        torm::TormIKSolver& iksolver_;

        sensor_msgs::JointState js_;
        ros::Publisher joint_pub_;
        ros::Publisher display_pub_;
        ros::Publisher marker_pub_;
        ros::Publisher arrow1_pub_;
        ros::Publisher arrow2_pub_;
        ros::Publisher arrow3_pub_;
        ros::Publisher arrow4_pub_;
        ros::Publisher Markers_pub_;

        moveit_msgs::DisplayTrajectory display_trajectory_;
        moveit_msgs::RobotTrajectory robot_traj_;

        planning_scene::PlanningSceneConstPtr planning_scene_;

        visualization_msgs::Marker line_list_;
        geometry_msgs::PoseArray Arrow1_list_;
        geometry_msgs::PoseArray Arrow2_list_;
        geometry_msgs::PoseArray Arrow3_list_;
        geometry_msgs::PoseArray Arrow4_list_;
        visualization_msgs::MarkerArray Marker_list_;
        int marker_counter_{0};
    };

    typedef std::shared_ptr<TormDebug> TormDebugPtr;
}

#endif //TORM_TORM_DEBUG_H
