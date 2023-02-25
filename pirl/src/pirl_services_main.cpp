#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>

#include <pirl/collision_checker.h>
#include <pirl/kinematic_solver.h>

#include <sensor_msgs/JointState.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "pirl_services_main");
    ros::NodeHandle nh("~");

    ros::Duration(5).sleep(); // for giving loading time move_group (demo.launch)
    ROS_WARN("[===INFO===] =======> pirl_services_main");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));

    // set default joint values
    std::vector<std::string>  default_setting_joints_;
    std::vector<double>       default_setting_values_;
    nh.getParam("/problem/default_setting_joints", default_setting_joints_);
    nh.getParam("/problem/default_setting_values", default_setting_values_);
    robot_state::RobotState state = planning_scene->getCurrentState();
    for(uint i = 0; i < default_setting_joints_.size(); i++)
        state.setJointPositions(default_setting_joints_[i], &default_setting_values_[i]);
    planning_scene->setCurrentState(state);

    // set services
    CollisionCheckSrv cc_srv(planning_scene);
    KinematicsSolverSrv ks_srv(planning_scene);

    ROS_WARN("[===INFO===] <======= collision and kinematic checking module is initialized.");
    ros::spin();

    return 0;
}
