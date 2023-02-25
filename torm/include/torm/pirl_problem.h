/* Author: Mincheul Kang */

#ifndef PIRL_PROBLEM_H
#define PIRL_PROBLEM_H

#include <ros/ros.h>
#include <ros/package.h>

#include <fstream>
#include <trac_ik/trac_ik.hpp>
#include <moveit_msgs/PlanningScene.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <torm/torm_utils.h>


class PirlProblem
{
public:
    PirlProblem(std::string robot_name, planning_scene::PlanningScenePtr& planning_scene);
    ~PirlProblem();

    void setDefaultJointPosition();

    void setCollisionObjects(std::string obs_file_name);
    void resetCollisionObjects();

    moveit_msgs::CollisionObject makeCollisionObject(std::string name, double x, double y, double z,
                                                     double roll, double pitch, double yaw,
                                                     double size_x, double size_y, double size_z);
    std::string getPlanningGroup();
    std::string getFixedFrame();
    std::string getBaseLink();
    std::string getTipLink();

private:
    ros::NodeHandle                                             nh_;

    std::string                                                 fixed_frame_;
    std::string                                                 planning_group_;
    std::string                                                 planning_base_link_;
    std::string                                                 planning_tip_link_;
    std::vector<std::string>                                    default_setting_joints_;
    std::vector<double>                                         default_setting_values_;

    std::vector<moveit_msgs::CollisionObject>                   collision_objects_;

    planning_scene::PlanningScenePtr&                           planning_scene_;
    moveit::planning_interface::PlanningSceneInterface          planning_scene_interface_;
};

#endif //PIRL_PROBLEM_H
