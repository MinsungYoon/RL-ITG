#ifndef CATKIN_WS_TORM_INTERPOLATOR_H
#define CATKIN_WS_TORM_INTERPOLATOR_H

#include <ros/ros.h>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <eigen3/Eigen/Core>
#include <torm/torm_ik_solver.h>
#include <random>
#include <math.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

class TormInterpolator {
public:
    TormInterpolator(int n_IKcand, KDL::JntArray& start_conf,
                     std::vector<KDL::Frame>& targetPoses, std::vector<int>& simplified_points,
                     torm::TormIKSolver& iksolver, planning_scene::PlanningScenePtr planning_scene);
    ~TormInterpolator();

    void setTormInitialTraj();
    void fillInLinearInterpolation(Eigen::MatrixXd& trajsegment_);
    double getEndPoseCost(Eigen::MatrixXd& trajsegment_, int start_idx);
    void refineContinuousJoint(KDL::JntArray& q);

    void setCollisionChecker();
    bool collision_check(KDL::JntArray& q);

    inline int getNumPoints() const {
        return num_total_points_;
    }
    inline Eigen::MatrixXd::RowXpr getTrajectoryPoint(int traj_point){
        return trajectory_.row(traj_point);
    }
    inline Eigen::MatrixXd getTrajectory(){
        return trajectory_;
    }
private:
    ros::NodeHandle nh_;
    std::vector<double> c_joints_;
    std::vector<double> ll_;
    std::vector<double> ul_;
    std::string planning_group_;
    planning_scene::PlanningScenePtr planning_scene_;

    collision_detection::CollisionRequest c_request_;
    collision_detection::CollisionResult c_result_;

    int n_IKcand_;

    int num_total_points_;
    int num_simple_points_;
    int num_joints_;

    KDL::JntArray& start_conf_;
    std::vector<KDL::Frame>& targetPoses_;
    std::vector<int>& simplified_points_;
    Eigen::MatrixXd trajectory_;

    torm::TormIKSolver& iksolver_;
};















#endif //CATKIN_WS_TORM_INTERPOLATOR_H
