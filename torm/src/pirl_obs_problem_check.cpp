#include <ros/ros.h>
#include <iostream>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

#include <interpolation/pirl_problem_generator.h>
#include <torm/pirl_problem.h>

#include <torm/torm_parameters.h>
#include <torm/torm_ik_solver.h>
#include <torm/torm_optimizer.h>
#include <torm/torm_trajectory.h>

template<typename rtype>
static inline void split(std::string str, char delimiter, std::vector<rtype>& result) {
    std::stringstream ss(str);
    std::string temp;

    while (getline(ss, temp, delimiter)) {
        result.push_back((rtype)(std::atof(temp.c_str())));
    }
}

void path_load( std::string file_name,
                std::vector<std::vector<double>>& targetpath) {
    std::fstream fs;
    fs.open(file_name.c_str(), std::ios::in);
    std::string line;
    while (fs >> line) {
        std::vector<double> data;
        data.reserve(7);
        split<double>(line, ',', data);
        targetpath.push_back(data);
    }
    fs.close();
}

void initParameters(torm::TormParameters &params_, int endPose_size){
    params_.planning_time_limit_ = 30.0;

    params_.smoothness_update_weight_ = 15.0/endPose_size;
    params_.obstacle_update_weight_ = 6.0;
    params_.jacobian_update_weight_ = 1.0;

    params_.learning_rate_ = 0.005;

    params_.pose_cost_weight_ = 1.0;
    params_.collision_cost_weight_ = 0.1;

    params_.rotation_scale_factor_ = 0.17;

    params_.smoothness_cost_velocity_ = 1.0;
    params_.smoothness_cost_acceleration_ = 0.0;
    params_.smoothness_cost_jerk_ = 0.0;
    params_.ridge_factor_ = 0.0;
    params_.use_pseudo_inverse_ = true;
    params_.pseudo_inverse_ridge_factor_ = 1e-4;
    params_.joint_update_limit_ = 0.3;
    params_.min_clearence_ = 0.3;
    params_.time_duration_ = 1.0;
    params_.use_stochastic_descent_ = true;
    params_.use_velocity_check_ = false;
    params_.use_singularity_check_ = false;
    params_.use_collision_check_ = true;

    params_.singularity_lower_bound_ = 0.005; // fetch arm
    params_.exploration_iter_ = 50; // 2 stage gradient iters
    params_.traj_generation_iter_ = 100; // # of Ik candidates
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "exter_obs_datacheck");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    robot_state::RobotState state = planning_scene->getCurrentState();

    PirlProblem prob("fetch", planning_scene);

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int) joint_bounds.size();

    torm::TormIKSolver iksolver(prob.getPlanningGroup(), planning_scene, prob.getBaseLink(), prob.getTipLink());
    torm::TormDebug debug(planning_scene, prob.getPlanningGroup(), prob.getFixedFrame(), prob.getBaseLink(), iksolver);
    std::vector<double> s_conf{1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0}; // fetch robot's home position (initial position)
    debug.visualizeConfiguration(indices, s_conf);
    debug.visualizeConfiguration(indices, s_conf);
    debug.visualizeConfiguration(indices, s_conf);
    debug.visualizeConfiguration(indices, s_conf);
    debug.visualizeConfiguration(indices, s_conf);
    debug.visualizeConfiguration(indices, s_conf);

    bool path_check = false;

    for(int s=0; s < 1000; s++) {
        // setting env
//        std::string obs_file =
//                std::string("/data/torm_data/obs/scene/scene_") + std::to_string(s) + ".txt";
        std::string obs_file =
                std::string("/data/pirl_data/eval_show/scene_with_box/scene_") + std::to_string(s) + ".txt";
        ROS_WARN_STREAM("load obs: " << obs_file);
        prob.setCollisionObjects(obs_file);
        ros::Duration(0.01).sleep();

        if (path_check) {
            // gen problems
            std::vector<std::vector<double>> path; // [(xyz),(xyzw)]
            std::vector<std::vector<double>> rpypath;
            for (int p = 0; p < 10; p++) {
                std::string file_name = std::string("/data/torm_data/obs/problem/") +
                                        std::to_string(s) +
                                        "/prob_" + std::to_string(p) + ".csv";
                std::vector<std::vector<double>> path;
                path_load(file_name, path);

                std::cout << p << ", " << path.size() << std::endl;

                std::vector<KDL::Frame> targetPoses;
                targetPoses.reserve(path.size());
                for (int i = 0; i < path.size(); i++) {
                    KDL::Frame f;
                    f.p[0] = path[i][0];
                    f.p[1] = path[i][1];
                    f.p[2] = path[i][2];
                    f.M = KDL::Rotation::Quaternion(path[i][3], path[i][4], path[i][5], path[i][6]);
                    targetPoses.push_back(f);
                }
                debug.publishEETrajectory(targetPoses, 0);
                ros::Duration(0.01).sleep();
                debug.clear();
            }
        }

        // removing env
        ROS_WARN_STREAM("loaded all paths.");
        ros::Duration(0.2).sleep();
        prob.resetCollisionObjects();
    }
}





