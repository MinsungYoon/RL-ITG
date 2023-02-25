/* Author: Mincheul Kang */

#include <ros/ros.h>
#include <torm/torm_parameters.h>
#include <torm/torm_ik_solver.h>
#include <torm/torm_debug.h>
#include <torm/torm_problem.h>
#include <torm/torm_optimizer.h>
#include <torm/torm_trajectory.h>
#include <torm/torm_utils.h>

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

#include <torm/torm_utils.h>
#include <torm/pirl_interpolator.h>

void initParameters(torm::TormParameters &params_, int endPose_size, bool only_first){
    if (only_first) {
        params_.planning_time_limit_ = 5.0;
    }else{
        params_.planning_time_limit_ = 50.0;
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
    params_.min_clearence_ = 1.0;
    params_.use_stochastic_descent_ = true;
    params_.use_velocity_check_ = true;
    params_.use_singularity_check_ = false;
    params_.use_collision_check_ = true;

    params_.singularity_lower_bound_ = 0.005; // fetch arm

    if (only_first){
        params_.exploration_iter_ = 100; // 2 stage gradient iters
    }else{
        params_.exploration_iter_ = 50; // 2 stage gradient iters
    }
    params_.traj_generation_iter_ = 100; // # of Ik candidates
    params_.time_duration_ = 0.2;
}

/// rosrun torm torm_evaluation(0)
/// [exp_name](1)
/// [algorithm](2)
/// [#iter](3)
/// [model_index](4)
/// [start_conf_type](5)
/// [start_conf_idx](6)
/// [True/False] Only first [7]
/// [0,1,2,3] LR scheduler method [8]

int main(int argc, char** argv) {
    ros::init(argc, argv, "torm_eval");
    ros::NodeHandle node_handle("~");

    std::cout << argv[0] << ", " << argv[1] << ", " << argv[2] << ", " << argv[3] << ", " << argv[4] << ", " << argv[5] << ", " << argv[6] << ", " << argv[7] << ", " << argv[8] <<std::endl;

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    // initialize planning interface
    std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map = planning_scene_interface_.getObjects();
    std::vector<moveit_msgs::CollisionObject> v_co;
    for(auto& kv : collision_objects_map){
        kv.second.operation = kv.second.REMOVE;
        v_co.push_back(kv.second);
    }
    planning_scene_interface_.applyCollisionObjects(v_co);
    ros::Duration(1.0).sleep();

    torm::TormProblem prob(std::string(argv[1]), "fetch", planning_scene);

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int)joint_bounds.size();

    torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, prob.getBaseLink(), prob.getTipLink());

    int gap;
    if(std::string(argv[2]).find("pirl") !=  std::string::npos){
        gap = 1;
    }else{
        gap = 10;
    }
    std::vector<KDL::Frame> targetPoses = prob.getTargetPoses();
    std::vector<int> simplified_points;
    for(int i = gap; i < targetPoses.size(); i+=gap){
        simplified_points.push_back(i);
    }
    if(simplified_points[simplified_points.size()-1] != targetPoses.size()-1){
        simplified_points.push_back(targetPoses.size()-1);
    } // simplified_points: not include initial ee pose but include final ee pose.

    bool fix_start_config = (std::string(argv[5]).find("fix") !=  std::string::npos);

    torm::TormParameters params;
    initParameters(params, targetPoses.size(), std::string(argv[7]) == "true");

    // [setup initial config]
    KDL::JntArray q_start(num_dof);
    std::vector<double> s_conf;
    if (fix_start_config){
        prob.setStartConfig(atoi(argv[6]));
        s_conf = prob.getStartConfiguration();
    }else{
        s_conf.reserve(num_dof);
    }
    if(s_conf.size() == 0){
        if(!iksolver.ikSolverCollFree(targetPoses[0], q_start)){
            ROS_INFO("No found a valid start configuration.");
            return 0;
        }
        for (int j=0; j<num_dof; j++){
            s_conf.push_back(q_start(j));
        }
        ROS_WARN("--- set random start configuration!!!!");
    }
    else{
        ROS_INFO("--- set loaded start configuration from yaml file.");
        for (uint j = 0; j < num_dof; j++) {
            q_start(j) = s_conf[j];
        }
    }

    // set current state
    robot_state::RobotState rs = planning_scene->getCurrentState();
    rs.setJointGroupPositions(PLANNING_GROUP, s_conf);
    rs.update();
    planning_scene->setCurrentState(rs);

    torm::TormDebugPtr debug = std::make_shared<torm::TormDebug>(planning_scene, prob.getPlanningGroup(), prob.getFixedFrame(), prob.getBaseLink(), iksolver);
    // visualize input poses
    debug->publishEETrajectory(targetPoses, 0);
    ros::Duration(1.0).sleep();

    for( int i=0 ; i<atoi(argv[3]) ; i++) {
        std::cout << argv[2] << std::endl;


        std::vector<std::string> models;
        if (prob.getIsLoadScene()) {
            models.emplace_back("rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task+Imitation_Ent3_best.pt"); // best
            models.emplace_back("rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task_Ent3_best.pt"); // best
            models.emplace_back("rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task_Ent3_last.pt"); // best
        }else{
            models.emplace_back("rl_Conf+LinkPoses_T1_model_W4W5.pt");
            models.emplace_back("rl_Conf+LinkPoses_T1_model_FREEALL_Task.pt");
            models.emplace_back("rl_Conf+LinkPoses_T1_model_FREEALL_Task+Imitation.pt");
            models.emplace_back("rl_Conf+LinkPoses_T1_model_FREEALL_Imitation.pt");
        }

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
                std::string model = models[atoi(argv[4])];
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
        trajectory.getTrajectoryPoint(0) = q_start.data; // set initial configuration!
        std::cout<< "[INFO] trajectory initialized." <<std::endl;

        // trajectory optimization
        torm::TormOptimizer opt(&trajectory, planning_scene, PLANNING_GROUP, &params, planning_scene->getCurrentState(),
                                targetPoses, simplified_points, iksolver, joint_bounds,
                                fix_start_config, true, false, false, false,
                                PirlModel, debug, atoi(argv[8]));
        std::cout<< "[INFO] opt initialized." <<std::endl;

        if (strcmp(argv[2],"torm_jli")==0){
            opt.set_interpolation_mode(0);
        }else if (strcmp(argv[2],"torm")==0){
            opt.set_interpolation_mode(1);
        }else if (strcmp(argv[2],"pirl_rl")==0 or strcmp(argv[2],"pirl_bc")==0){
            opt.set_interpolation_mode(2);
        }

        bool result = opt.iterativeExploration(std::string(argv[7]) == "true");
        if (!result) {
            ROS_INFO("No found a valid trajectory.");
        } else {
            std::vector<std::pair<std::string, std::vector<double>>> vals = {
                    {"BestCost", opt.getBestCostLog()},
                    {"PoseCost", opt.getPoseCostLog()},
                    {"RotCost",  opt.getRotCostLog()},
                    {"VelCost",  opt.getVelCostLog()},
                    {"AccCost",  opt.getAccCostLog()},
                    {"JerkCost", opt.getJerkCostLog()},
                    {"Time",     opt.getTimeLog()}
            };
            std::vector<std::pair<std::string, std::vector<double>>> first_vals = {
                    {"BestCost", opt.getFirstBestCostLog()},
                    {"PoseCost", opt.getFirstPoseCostLog()},
                    {"RotCost",  opt.getFirstRotCostLog()},
                    {"VelCost",  opt.getFirstVelCostLog()},
                    {"AccCost",  opt.getFirstAccCostLog()},
                    {"JerkCost", opt.getFirstJerkCostLog()},
                    {"Time",     opt.getFirstTimeLog()}
            };
            std::stringstream ss;
            ss << "/data/pirl_result/" << std::string(argv[5]) << "/" << argv[1] << "/" << argv[6] << "/" <<       argv[2] << "_M" << argv[4] << "_LR" << argv[8] << "_" << i << ".csv";
            torm::write_csv(std::string(ss.str()), vals);
            std::stringstream first_ss;
            first_ss << "/data/pirl_result/" << std::string(argv[5]) << "/" << argv[1] << "/" << argv[6] << "/" << argv[2] << "_M" << argv[4]  << "_LR" << argv[8] << "_" << i << "_first" << ".csv";
            torm::write_csv(std::string(first_ss.str()), first_vals);
            std::cout << "[INFO] saved log, file name: " << ss.str() << std::endl;
        }
    }

    return 0;
}