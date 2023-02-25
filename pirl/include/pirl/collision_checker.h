// MoveIt!
#include <ros/ros.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <pirl_msgs/collision.h>
#include <pirl_msgs/scene_set.h>
#include <pirl_msgs/scene_reset.h>

#include <fstream>
#include <iostream>
#include <string>

// Ex.
// [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]: self-collision
// [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]: joint limit violation

class CollisionCheckSrv {
public:
    CollisionCheckSrv(planning_scene::PlanningScenePtr planning_scene);
    ~CollisionCheckSrv();

    void setCollisionChecker();

    bool collisionCheck_srv(pirl_msgs::collision::Request  &req,
                            pirl_msgs::collision::Response &res);
    bool setCollisionObjects_srv(pirl_msgs::scene_set::Request  &req,
                                 pirl_msgs::scene_set::Response &res);
    bool resetCollisionObjects_srv(pirl_msgs::scene_reset::Request  &req,
                                   pirl_msgs::scene_reset::Response &res);

    moveit_msgs::CollisionObject makeCollisionObject(std::string name, double x, double y, double z,
                                                     double roll, double pitch, double yaw,
                                                     double size_x, double size_y, double size_z);
    moveit_msgs::ObjectColor makeColorObject(std::string name, std::string color);

private:

    ros::NodeHandle nh_;

    planning_scene::PlanningScenePtr planning_scene_;
    std::string planning_group_;
    std::string fixed_frame_;
    int n_dof_;

    ros::ServiceServer collision_service_;
    collision_detection::CollisionRequest c_request_;
    collision_detection::CollisionResult c_result_;

    ros::ServiceServer scene_set_service_;
    ros::ServiceServer scene_reset_service_;

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
    std::vector<moveit_msgs::CollisionObject> collision_objects_;
    std::vector<moveit_msgs::ObjectColor> color_objects_;
};