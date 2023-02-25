
#include <vector>
#include <iostream>
#include <eigen_conversions/eigen_kdl.h>
#include <random>

#include <interpolation/pirl_problem_generator.h>

#include <ros/ros.h>
#include <torm/torm_parameters.h>
#include <torm/torm_ik_solver.h>
#include <torm/torm_optimizer.h>
#include <torm/torm_trajectory.h>
#include <torm/pirl_interpolator.h>
#include <torm/pirl_problem.h>


// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

using namespace std;

void path_write(std::string file_name,
                Eigen::MatrixXd& outputMatrix, std::vector<std::vector<double>>& targetpath, std::vector<int>& simplified_points,
                int n_way_points, double interval_length, int target_total_n, double target_total_len,
                std::vector<double>& cost_log, std::vector<double>& time_log){

    std::string f_name;
    f_name = file_name;
    fstream fs;
    fs.open(f_name.append("_config.csv").c_str(), ios::out);
    if(fs){
        for(int i=0; i<outputMatrix.rows() ;i++){
            for(int j=0; j<outputMatrix.cols(); j++){
                fs << outputMatrix(i, j);
                if(j==outputMatrix.cols()-1){
                    fs << "\n";
                }else{
                    fs << ",";
                }
            }
        }
        fs.close();
    }else{
        cout << strerror(errno) << '\n';
        exit(0);
    }

    f_name = file_name;
    fs.open(f_name.append("_targetpose.csv").c_str(), ios::out);
    for(int i=0; i<targetpath.size() ;i++){
        fs << targetpath[i][0] << ",";
        fs << targetpath[i][1] << ",";
        fs << targetpath[i][2] << ",";
        fs << targetpath[i][3] << ",";
        fs << targetpath[i][4] << ",";
        fs << targetpath[i][5] << ",";
        fs << targetpath[i][6] << "\n";
    }
    fs.close();

    f_name = file_name;
    fs.open(f_name.append("_targetpose_info.csv").c_str(), ios::out);
    fs << n_way_points << "," << interval_length << "," << target_total_n << "," << target_total_len << "\n";
    fs.close();

    f_name = file_name;
    fs.open(f_name.append("_simplified_points.csv").c_str(), ios::out);
    for(int i=0; i<simplified_points.size() ;i++){
        if(i==simplified_points.size()-1){
            fs << simplified_points[i] << "\n";
        }else{
            fs << simplified_points[i] << ",";
        }
    }
    fs.close();

    f_name = file_name;
    fs.open(f_name.append("_cost_log.csv").c_str(), ios::out);
    for(int i=0; i<cost_log.size() ;i++){
        if(i==cost_log.size()-1){
            fs << cost_log[i] << "\n";
        }else{
            fs << cost_log[i] << ",";
        }
    }
    fs.close();

    f_name = file_name;
    fs.open(f_name.append("_time_log.csv").c_str(), ios::out);
    for(int i=0; i<time_log.size() ;i++){
        if(i==time_log.size()-1){
            fs << time_log[i] << "\n";
        }else{
            fs << time_log[i] << ",";
        }
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
    ros::init(argc, argv, "pirl_data_collection");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    PirlProblem prob("fetch", planning_scene);

    // initialize planning interface
    std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map = planning_scene_interface_.getObjects();
    std::vector<moveit_msgs::CollisionObject> v_co;
    for (auto &kv : collision_objects_map) {
        kv.second.operation = kv.second.REMOVE;
        v_co.push_back(kv.second);
    }
    planning_scene_interface_.applyCollisionObjects(v_co);
    ros::Duration(1.0).sleep();

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int) joint_bounds.size();

    robot_state::RobotState state = planning_scene->getCurrentState();

    torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, prob.getBaseLink(), prob.getTipLink());

    int gap = 6;
    double interval_length = 0.01;
    int n_candidate_ee_poses = 2500;
    int total_data_to_collect = 5000;
    for (int w = 5; w <= 10; w++) {
        PirlProblemGenerator dataGen(n_candidate_ee_poses, PLANNING_GROUP, planning_scene, iksolver);
        int n_suc = 0;
        int n_fail = 0;
        while (n_suc != total_data_to_collect) {
            std::cout << "----<<<<<<<<<<<<<< w: " << w << ", n_suc: " << n_suc << " >>>>>>>>>>>>>>----" << std::endl;

            std::vector<std::vector<double>> path; // [(xyz),(xyzw)]
            std::vector<std::vector<double>> rpypath;
            bool is_valid_target = dataGen.sampleTargetEEList(w, interval_length, path, rpypath);
            if (!is_valid_target) {
                std::cout << "-- generated target ee path (problem) isn't valid.....!" << std::endl;
                continue;
            }

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

            std::vector<int> simplified_points;
            for (int i = gap; i < targetPoses.size(); i += gap) {
                simplified_points.push_back(i);
            }
            if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
                simplified_points.push_back(targetPoses.size() - 1);
            } // simplified_points: not include initial ee pose but include end ee pose.

            torm::TormParameters params;
            initParameters(params, targetPoses.size());

            // setup initial config
            bool fix_start_config = false;
            KDL::JntArray q_start(num_dof);
            std::vector<double> s_conf;
            if(s_conf.size() == 0){
                if(!iksolver.ikSolverCollFree(targetPoses[0], q_start)){
                    ROS_INFO("No found a valid start configuration.");
                    return 0;
                }
                ROS_INFO("--- set random start configuration!");
            }
            else{
                ROS_INFO("--- set loaded start configuration from yaml file.");
                fix_start_config = true;
                for (uint j = 0; j < num_dof; j++) {
                    q_start(j) = s_conf[j];
                }
            }

            // No Pirl.
            PirlInterpolatorPtr PirlModel = nullptr;

            // generate trajectory
            torm::TormTrajectory trajectory(planning_scene->getRobotModel(), int(targetPoses.size()), params.time_duration_, PLANNING_GROUP);
            trajectory.getTrajectoryPoint(0) = q_start.data; // set initial configuration!
            std::cout<< "[INFO] trajectory initialized." <<std::endl;

            // trajectory optimization
            torm::TormOptimizer opt(&trajectory, planning_scene, PLANNING_GROUP, &params, state,
                                    targetPoses, simplified_points, iksolver, joint_bounds, fix_start_config, false, false, false, false, PirlModel);
            std::cout<< "[INFO] opt initialized." <<std::endl;

            bool result = opt.iterativeExploration();
            if (result) {
                ROS_INFO("************************ Succeed to find a valid trajectory.");
                Eigen::MatrixXd &optimizedConfigs = trajectory.getTrajectory();
                std::vector<double> cost_log = opt.getBestCostLog();
                std::vector<double> time_log = opt.getTimeLog();
                std::string file_name = "/data/torm_data/free/torm_solution/";
                file_name += std::to_string(w);
                file_name += "/";
                file_name += std::to_string(n_suc);
                path_write(file_name,
                           optimizedConfigs, path, simplified_points,
                           w, interval_length, dataGen.getTargetEEN(), dataGen.getTargetEELen(),
                           cost_log, time_log);
                std::cout << file_name << std::endl;
                n_suc++;
            } else {
                n_fail++;
            }
        }
    }
}