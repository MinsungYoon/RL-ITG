//* Author: Mincheul Kang */

#include <ros/ros.h>
#include <torm/torm_parameters.h>
#include <torm/torm_ik_solver.h>
#include <torm/torm_debug.h>
#include <torm/torm_problem.h>
#include <torm/torm_optimizer.h>
#include <torm/torm_trajectory.h>
#include <torm/torm_utils.h>
#include <torm/traj_evaluator.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <visualization_msgs/Marker.h>
#include <eigen_conversions/eigen_kdl.h>

#include <torm/pirl_interpolator.h>

void path_write(std::string file_name,
                Eigen::MatrixXd& outputMatrix){
    std::fstream fs;
    fs.open(file_name.append(".csv").c_str(), std::ios::out);
    if(fs.is_open()){
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
        std::cout << strerror(errno) << '\n';
        exit(0);
    }
}

void initParameters(torm::TormParameters &params_, int endPose_size, bool debug_visual){
    if (debug_visual){
        params_.planning_time_limit_ = 1.0;
    }else{
        params_.planning_time_limit_ = 1.0;
    }

    // === increment objective function weights ===
    params_.smoothness_update_weight_ = 30.0/endPose_size;
    params_.obstacle_update_weight_ = 1.0;
    params_.learning_rate_ = 0.01; // only affect feasible increments terms...

    params_.jacobian_update_weight_ = 1.0;

    // === cost weights === (pose err vs col err) when generate new traj of TORM
    params_.pose_cost_weight_ = 1.0;
    params_.collision_cost_weight_ = 0.05;

    params_.rotation_scale_factor_ = 0.17;

    params_.smoothness_cost_velocity_ = 1.0;
    params_.smoothness_cost_acceleration_ = 0.0;
    params_.smoothness_cost_jerk_ = 0.0;
    params_.ridge_factor_ = 0.0;
    params_.use_pseudo_inverse_ = true;
    params_.pseudo_inverse_ridge_factor_ = 1e-4;
    params_.joint_update_limit_ = 0.17; // 10 deg
    params_.min_clearence_ = 0.001;
    params_.singularity_lower_bound_ = 0.005; // fetch arm

    params_.use_stochastic_descent_ = true;
    params_.use_velocity_check_ = true;
    params_.use_singularity_check_ = false;
    params_.use_collision_check_ = true;

    if(debug_visual){
        params_.exploration_iter_ = 10; // 2 stage gradient iters
    }else{
        params_.exploration_iter_ = 100; // 2 stage gradient iters
    }
    params_.traj_generation_iter_ = 50; // # of Ik candidates
    params_.time_duration_ = 0.1;
}

// random_obs_63_1 pirl_rl_test fix 0 false false false 1 false

/// rosrun torm torm_main [0]
/// [problem] [1]
/// [torm|torm_test|pirl_{bc/rl}|pirl_{bc/rl}_test](run type) [2]
/// [start_conf_type] [3]
/// [start_conf_idx] [4]
/// [true/false] dubug_verbose [5]
/// [true/false] dubug_visual [6]
/// [true/false] dubug_visual_onlybest [7]
/// [0,1,2] lr_schedule mode [8]
/// [true/false] save_current_trajectory [9]

int main(int argc, char** argv) {
    ros::init(argc, argv, "torm");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    torm::TormProblem prob(argv[1], "fetch", planning_scene);

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int)joint_bounds.size();
    std::vector<double> vel_limit;
    vel_limit.reserve(num_dof);
    for(uint i = 0; i < joint_bounds.size(); i++){
        vel_limit.push_back(joint_bounds[i][0][0].max_velocity_);
    }
    torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, prob.getBaseLink(), prob.getTipLink());

    std::vector<KDL::Frame> targetPoses = prob.getTargetPoses(); // [0]: start pose (matching with start conf)
    std::vector<int> simplified_points;

    int gap;
    if(std::string(argv[2]).find("pirl") !=  std::string::npos){
        gap = 1;
    }else{
        gap = 10;
    }
    for(int i = gap; i < targetPoses.size(); i+=gap){
        simplified_points.push_back(i);
    }
    if(simplified_points[simplified_points.size()-1] != targetPoses.size()-1){
        simplified_points.push_back(targetPoses.size()-1);
    } // simplified_points: not include initial ee pose but include final ee pose.

    torm::TormParameters params;
    initParameters(params, targetPoses.size(), std::string(argv[6]) == "true");

    bool fix_start_config = (std::string(argv[3]).find("fix") !=  std::string::npos);
    // [setup initial config]
    KDL::JntArray q_start(num_dof);
    std::vector<double> s_conf;
    s_conf.reserve(num_dof);
    if (fix_start_config){
        prob.setStartConfig(atoi(argv[4]));
        s_conf = prob.getStartConfiguration();
        ROS_WARN("--- set loaded start configuration from yaml file.");
        for (uint j = 0; j < num_dof; j++) {
            q_start(j) = s_conf[j];
        }
    }else{
        if(!iksolver.ikSolverCollFree(targetPoses[0], q_start)){
            ROS_ERROR("[ERROR] No found a valid start configuration.");
            return 0;
        }else {
            for (int j = 0; j < num_dof; j++) {
                s_conf.push_back(q_start(j));
            }
            ROS_WARN("--- set random start configuration!!!!");
        }
    }

    std::cout << "==================================================================" << std::endl;
    std::cout << argv[1] << " & " << argv[2] << std::endl;
    std::cout << "==> Gap: " << gap << ", N: " << targetPoses.size() << ", S: " << simplified_points.size() << std::endl;

    // set current state
    robot_state::RobotState rs = planning_scene->getCurrentState();
    rs.setJointGroupPositions(PLANNING_GROUP, s_conf);
    rs.update();
    planning_scene->setCurrentState(rs);

    torm::TormDebugPtr debug = std::make_shared<torm::TormDebug>(planning_scene, prob.getPlanningGroup(), prob.getFixedFrame(), prob.getBaseLink(), iksolver);
    // visualize input poses
//    debug->publishEETrajectory(targetPoses, 0);
    ros::Duration(1.0).sleep();
    debug->visualizeConfiguration(indices, q_start);
    ros::Duration(1.0).sleep();

    // [onlyConf|Conf+LinkPoses|Conf+EE|onlyLinkPoses]
    // Pirl ===========================================================================================================
    std::cout << argv[2] << std::endl;
    bool use_PIRL = std::string(argv[2]).find("pirl") != std::string::npos;
    PirlInterpolatorPtr PirlModel = nullptr;
    if(use_PIRL) {
        if(std::string(argv[2]).find("pirl_bc") != std::string::npos){
            std::cout << "[INFO] load pirl_bc model..." << std::endl;
            const std::string model_path = "/data/pirl_network/BC/bc_basic_model.pt";
            int t = 6;
            std::string obsType = "Conf+LinkPoses";
            PirlModel = std::make_shared<PirlInterpolator>(model_path, t, obsType, targetPoses, simplified_points, iksolver);
        }
        else if(std::string(argv[2]).find("pirl_rl") != std::string::npos){
            std::cout << "[INFO] load pirl_rl model..." << std::endl;
            std::string model_path = "/data/pirl_network/RL/";
            std::string model;
            if (prob.getIsLoadScene()){
//                model = "rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task_Ent3_best.pt"; // best
              model = "rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task_Ent3_last.pt";
//              model = "rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task+Imitation_Ent3_best.pt";
//                model = "rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task+Imitation_Ent3_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task+Imitation_Ent7_best.pt";
//                model = "rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task+Imitation_Ent7_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_TABLEOBS_Task+Imitation_best.pt";
//                model = "rl_Conf+LinkPoses_T1_model_TABLEOBS_Task+Imitation_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_OBSALL_Task_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_OBSALL_Task_best.pt";
//                model = "rl_Conf+LinkPoses_T1_model_OBSALL_Task+Imitation_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_OBSALL_Task+Imitation_best.pt";
            }else{
                model = "rl_Conf+LinkPoses_T1_model_JACOFREE_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_JACOFREE_best.pt";
//                model = "rl_Conf+LinkPoses_T1_model_FREEALL_Task.pt";
//                model = "rl_Conf+LinkPoses_T1_model_FREEALL_Task_Refine_last.pt";
//                model = "rl_Conf+LinkPoses_T1_model_FREEALL_Task+Imitation.pt";
//                model = "rl_Conf+LinkPoses_T1_model_FREEALL_Imitation.pt";
//                model = "rl_Conf+LinkPoses_T1_model_W4W5.pt";
//                model = "rl_Conf+LinkPoses_T1_model_FREEW4W5W6W7.pt";
//                model = "rl_Conf+LinkPoses_T1_model_FREEW4W5W6W7best.pt";
//                model = "rl_Conf+LinkPoses_T6_model_FREEW4W5W6W7.pt";
            }
            int t = atoi(&(model[model.find('T')+1]));
            std::string obsType = model.substr(model.find("rl_")+3, (model.find("_T") - (model.find("rl_")+3)));
            int multi_action = 1;
            PirlModel = std::make_shared<PirlInterpolator>(model_path+model, t, obsType, targetPoses, simplified_points, iksolver, multi_action);
        }
        std::cout<< "[INFO] PIRL has been initialized, and will be used." <<std::endl;
    }
    // ================================================================================================================

    // generate trajectory
    torm::TormTrajectory trajectory(planning_scene->getRobotModel(), int(targetPoses.size()), params.time_duration_, PLANNING_GROUP);
    torm::TormTrajectory init_trajectory(planning_scene->getRobotModel(), int(targetPoses.size()), params.time_duration_, PLANNING_GROUP);
    trajectory.getTrajectoryPoint(0) = q_start.data; // set initial configuration!
    std::cout<< "[INFO] trajectory initialized." <<std::endl;

    // trajectory optimization
    torm::TormOptimizer opt(&trajectory, planning_scene, PLANNING_GROUP, &params, planning_scene->getCurrentState(),
                            targetPoses, simplified_points, iksolver, joint_bounds,
                            fix_start_config, false, std::string(argv[5]) == "true", std::string(argv[6]) == "true", std::string(argv[7]) == "true",
                            PirlModel,debug, atoi(argv[8]), 1e-5, &init_trajectory);
    std::cout<< "[INFO] opt initialized." <<std::endl;

    if((strcmp(argv[2],"pirl_bc")==0) or (strcmp(argv[2],"pirl_rl")==0) or (strcmp(argv[2],"torm")==0) or (strcmp(argv[2],"torm_jli")==0)){
        if (strcmp(argv[2],"torm_jli")==0){
            opt.set_interpolation_mode(0);
        }else if (strcmp(argv[2],"torm")==0){
            opt.set_interpolation_mode(1);
        }else if (strcmp(argv[2],"pirl_rl")==0 or strcmp(argv[2],"pirl_bc")==0){
            opt.set_interpolation_mode(2);
        }
        bool result = opt.iterativeExploration();
        if(!result) {
            ROS_INFO("No found a valid trajectory.");
            return 0;
        }
    }

    double interpolation_time = 0.0;
    if((strcmp(argv[2],"torm_jli_test")==0)){
        ros::WallTime wt = ros::WallTime::now();
        opt.getJointInterpolatedTrajectory();
        interpolation_time = (ros::WallTime::now() - wt).toSec();
    }

    if((strcmp(argv[2],"torm_test")==0)){
        ros::WallTime wt = ros::WallTime::now();
        opt.getNewTrajectory();
        interpolation_time = (ros::WallTime::now() - wt).toSec();
    }

    if((strcmp(argv[2],"pirl_bc_test")==0) or (strcmp(argv[2],"pirl_rl_test")==0)) {
        ros::WallTime wt = ros::WallTime::now();
        opt.setPIRLTrajectory(true);
//        opt.setPIRLTrajectory(false);
//        opt.setPIRLTrajectory(false);
        interpolation_time = (ros::WallTime::now() - wt).toSec();
//        ROS_INFO_STREAM("[PIRL] interpolation time: " << (ros::WallTime::now() - wt) << "(# waypoints: "
//                                                      << PirlModel->getNumPoints() << ")");
    }

    if(std::string(argv[9]) == "true"){
        std::string f_name = std::string("/data/ready_to_show/")
                             + argv[1] + "-" + argv[2] + "-" + argv[4];
        path_write(f_name, trajectory.getTrajectory());
    }


    std::vector<double> res;
    opt.calcInitialTrajQuality(res);

    std::cout << "From calcInitialTrajQuality function:" << std::endl;
    for(double re : res){
        std::cout << re << ", " << std::endl;
    }
    std::cout << "===========================================" << std::endl;

    // trajectory evaluator
    traj_evaluator evaluator(targetPoses, trajectory.getTrajectory(), {2, 4, 6}, 0.01, iksolver, vel_limit);
    double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
    evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);

    debug->show(targetPoses, trajectory.getTrajectory(), 1, 2, 1);

    opt.updateLocalGroupTrajectory();

    std::cout << "[" << argv[2] << "] interpolation time: " << interpolation_time << std::endl;
    std::cout << "[" << argv[2] << "] costs: " << cost_pose + params.rotation_scale_factor_ * cost_rot
    << " | " << cost_pose << ", " << cost_rot << " | " << cost_vel << ", " << cost_acc << ", " << cost_jerk << std::endl;
    std::cout << "[" << argv[2] << "] feasiblity check: vel= " << opt.checkJointVelocityLimit(params.time_duration_) <<
    ", col= " << opt.isCurrentTrajectoryCollisionFree() << ", singularity= " << opt.checkSingularity() << std::endl;

    int x;
    std::cout << "Enter to display init traj...";
    std::cin >> x;
    debug->show(targetPoses, init_trajectory.getTrajectory(), 1, 2, 1);


    return 0;
}