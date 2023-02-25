//* Author: Mincheul Kang */

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

#include <torm/pirl_interpolator.h>
#include <torm/traj_evaluator.h>

using namespace std;

void write_problem( std::string file_name,
                    std::vector<KDL::Frame> targetpath ){
    fstream fs;
    fs.open(file_name.c_str(), ios::out);
    for(int i=0; i<targetpath.size() ;i++){
        fs << targetpath[i].p.x() << ",";
        fs << targetpath[i].p.y() << ",";
        fs << targetpath[i].p.z() << ";";
        double x, y, z, w;
        targetpath[i].M.GetQuaternion(x, y, z, w);
        fs << w << ",";
        fs << x << ",";
        fs << y << ",";
        fs << z << "\n"; // [x y z][w x y z]
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
    params_.use_stochastic_descent_ = true;
    params_.use_velocity_check_ = false;
    params_.use_singularity_check_ = false;
    params_.use_collision_check_ = false;

    params_.singularity_lower_bound_ = 0.005; // fetch arm
    params_.exploration_iter_ = 50; // 2 stage gradient iters
    params_.traj_generation_iter_ = 100; // # of Ik candidates
    params_.time_duration_ = 0.2;
}

// rosrun torm torm_main [torm|torm_test|pirl_{bc/rl}|pirl_{bc/rl}_test](run type) [1...10](gap)

int main(int argc, char** argv) {
    ros::init(argc, argv, "torm");
    ros::NodeHandle node_handle("~");

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

    torm::TormProblem prob("zig", "fetch", planning_scene);

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
    torm::TormIKSolver iksolver(prob.getPlanningGroup(), planning_scene, prob.getBaseLink(), prob.getTipLink());
    torm::TormDebug debug(planning_scene, prob.getPlanningGroup(), prob.getFixedFrame(), prob.getBaseLink(), iksolver);

    std::vector<KDL::Frame> targetPoses = prob.getTargetPoses();
    std::vector<int> simplified_points;

    int gap = 1;
    for(int i = gap; i < targetPoses.size(); i+=gap){
        simplified_points.push_back(i);
    }
    if(simplified_points[simplified_points.size()-1] != targetPoses.size()-1){
        simplified_points.push_back(targetPoses.size()-1);
    } // simplified_points: not include initial ee pose but include final ee pose.

    torm::TormParameters params;
    initParameters(params, targetPoses.size());

    // setup initial config
    bool fix_start_config = false;
    KDL::JntArray q_start(num_dof);
    std::vector<double> s_conf = prob.getStartConfiguration();
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

    // [onlyConf|Conf+LinkPoses|Conf+EE|onlyLinkPoses]
    // Pirl ===========================================================================================================
    std::cout << "[INFO] load pirl_rl model..." << std::endl;

    std::vector<std::string> models;
    models.push_back("rl_Conf+LinkPoses_T1_model_FREEALL_Task.pt");
    models.push_back("rl_Conf+LinkPoses_T1_model_FREEALL_Task+Imitation.pt");
    models.push_back("rl_Conf+LinkPoses_T1_model_FREEALL_Imitation.pt");
//    models.push_back("rl_Conf+LinkPoses_T3_model_W4W5.pt");
//    models.push_back("rl_Conf+LinkPoses_T6_model_W4W5.pt");
//    models.push_back("rl_Conf+LinkPoses_T6_model_FREEW4W5W6W7.pt");
//    models.push_back("rl_Conf+LinkPoses_T6_model_FREEW4W5.pt");

    for(int m=0; m<models.size(); m++) {

        std::string model_path = "/data/pirl_network/RL/";
        int t = atoi(&(models[m][models[m].find('T')+1]));
        std::string obsType = models[m].substr(models[m].find("rl_")+3, (models[m].find("_T") - (models[m].find("rl_")+3)));
        int multi_action = 1;
        PirlInterpolatorPtr PirlModel = std::make_shared<PirlInterpolator>(model_path+models[m], t, obsType, targetPoses, simplified_points, iksolver, multi_action);

        debug.clear();

        PirlModel->interpolate(q_start, true);


        // evaluate
        traj_evaluator evaluator(targetPoses, PirlModel->getTrajectory(), {2, 4, 6}, 0.1, iksolver, vel_limit);
        double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
        evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
        std::cout << cost_pose + params.rotation_scale_factor_ * cost_rot << ": " <<
                  cost_pose << ", " << cost_rot << " | " << cost_vel << ", " << cost_acc << ", " << cost_jerk << std::endl;

        debug.show(targetPoses, PirlModel->getTrajectory(), 1.1);
    }

    return 0;
}