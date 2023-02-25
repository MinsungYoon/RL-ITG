#include <vector>
#include <iostream>
#include <fstream>
#include <eigen_conversions/eigen_kdl.h>

#include <ros/ros.h>
#include <torm/torm_debug.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

template<typename rtype>
static inline void split(std::string str, char delimiter, std::vector<rtype>& result) {
    std::stringstream ss(str);
    std::string temp;

    while (getline(ss, temp, delimiter)) {
        result.push_back((rtype)(std::atof(temp.c_str())));
    }
}

void path_load( std::string file_name,
                std::vector<std::vector<double>>& configs, std::vector<std::vector<double>>& targetpath, std::vector<int>& simplified_points,
                int& n_way_points, double& interval_length, int& target_total_n, double& target_total_len,
                std::vector<double>& cost_log, std::vector<double>& time_log){

    std::fstream fs;

    {
        std::string f_name;
        f_name = file_name;
        fs.open(f_name.append("_config.csv"), std::ios::in);
        std::string line;
        while (fs >> line) {
            std::vector<double> data;
            data.reserve(7);
            split<double>(line, ',', data);
            configs.push_back(data);
        }
        fs.close();
    }

    {
        std::string f_name;
        f_name = file_name;
        fs.open(f_name.append("_targetpose.csv"), std::ios::in);
        std::string line;
        while (fs >> line) {
            std::vector<double> data;
            data.reserve(7);
            split<double>(line, ',', data);
            targetpath.push_back(data);
        }
        fs.close();
    }

    {
        std::string f_name;
        f_name = file_name;
        fs.open(f_name.append("_targetpose_info.csv"), std::ios::in);
        std::string line;
        fs >> line;
        std::vector<double> data;
        data.reserve(4);
        split<double>(line, ',', data);
        n_way_points = data[0];
        interval_length = data[1];
        target_total_n = data[2];
        target_total_len = data[3];
        fs.close();
    }

    {
        std::string f_name;
        f_name = file_name;
        fs.open(f_name.append("_simplified_points.csv"), std::ios::in);
        std::string line;
        fs >> line;
        std::vector<int> data;
        split<int>(line, ',', data);
        simplified_points = data;
        fs.close();
    }

    {
        std::string f_name;
        f_name = file_name;
        fs.open(f_name.append("_cost_log.csv"), std::ios::in);
        std::string line;
        fs >> line;
        std::vector<double> data;
        split<double>(line, ',', data);
        cost_log = data;
        fs.close();
    }

    {
        std::string f_name;
        f_name = file_name;
        fs.open(f_name.append("_time_log.csv"), std::ios::in);
        std::string line;
        fs >> line;
        std::vector<double> data;
        split<double>(line, ',', data);
        time_log = data;
        fs.close();
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "pirl_data_check");
    ros::NodeHandle node_handle("~");

    std::string fixed_frame_ = "/base_link";
    std::string planning_group_ = "arm";
    std::string planning_base_link_ = "torso_lift_link";
    std::string planning_tip_link_ = "gripper_link";

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));

    const std::string PLANNING_GROUP = planning_group_;
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int)joint_bounds.size();

    torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, planning_base_link_, planning_tip_link_);
    torm::TormDebug debug(planning_scene, PLANNING_GROUP, fixed_frame_, planning_base_link_, iksolver);

    std::vector<std::vector<double>> configs;
    std::vector<std::vector<double>> targetpath;
    std::vector<int> simplified_points;

    int n_way_points;
    double interval_length;
    int target_total_n;
    double target_total_len;

    std::vector<double> cost_log;
    std::vector<double> time_log;

    //"/home/minsungyoon/nvme/torm_data/suc/4/565" weard
    //"/home/minsungyoon/nvme/torm_data/suc/5/424"
    path_load(  std::string("/data/torm_data/free/torm_solution/") + argv[1] + "/" + argv[2],
                configs, targetpath, simplified_points,
                n_way_points, interval_length, target_total_n, target_total_len,
                cost_log, time_log);

    std::cout << "file has been loaded." << std::endl;
    std::cout << "[Total points]: " << target_total_n << std::endl;
    std::cout << "[Total length]: " << target_total_len << std::endl;
    std::cout << "[Cost]: " << cost_log[cost_log.size()-1] << std::endl;
    std::cout << "[Time]: " << time_log[time_log.size()-1] << std::endl;

    debug.clear();

    // visualize target EE trajectory
    std::vector<KDL::Frame> targetPoses;
    targetPoses.reserve(targetpath.size());
    for (int i = 0; i < targetpath.size(); i++) {
        KDL::Frame f;
        f.p[0] = targetpath[i][0];
        f.p[1] = targetpath[i][1];
        f.p[2] = targetpath[i][2];
        f.M = KDL::Rotation::Quaternion(targetpath[i][3], targetpath[i][4], targetpath[i][5], targetpath[i][6]);
        targetPoses.push_back(f);
    }
    debug.publishEETrajectory(targetPoses, 0);
    ros::Duration(1).sleep();

    // visualize result EE trajectory
    std::vector<KDL::Frame> optimizedPoses;
    for (uint i = 0; i < configs.size(); i++) {
        KDL::Frame ee;
        KDL::JntArray conf(7);
        conf(0) = configs[i][0];
        conf(1) = configs[i][1];
        conf(2) = configs[i][2];
        conf(3) = configs[i][3];
        conf(4) = configs[i][4];
        conf(5) = configs[i][5];
        conf(6) = configs[i][6];
        iksolver.fkSolver(conf, ee);

        optimizedPoses.push_back(ee);
    }
    debug.publishEETrajectory(optimizedPoses, 2);
    ros::Duration(1).sleep();

    // visualize joint trajectory
    std::vector<std::string> indices_vis = indices;
    debug.visualizeTrajectory(indices_vis, configs);
    ros::Duration(1.0).sleep();


    return 0;
}
