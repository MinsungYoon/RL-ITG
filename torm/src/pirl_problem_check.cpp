
#include <vector>
#include <iostream>
#include <fstream>
#include <eigen_conversions/eigen_kdl.h>

#include <interpolation/interpolator6D.h>
#include <interpolation/pirl_problem_generator.h>
#include <interpolation/eval_problem_generator.h>

#include <ros/ros.h>
#include <torm/torm_parameters.h>
#include <torm/torm_ik_solver.h>
#include <torm/torm_debug.h>
#include <torm/torm_problem.h>
#include <torm/torm_optimizer.h>
#include <torm/torm_trajectory.h>
#include <torm/torm_utils.h>

#include <torm/pirl_interpolator.h>
#include <torm/torm_interpolator.h>

#include <torm/pirl_problem.h>
#include <torm/traj_evaluator.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>

#include <random>

using namespace std;


int main(int argc, char** argv) {
    ros::init(argc, argv, "pirl_prob_check");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    PirlProblem prob("fetch", planning_scene);

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


    int n_candidate_ee_poses = 1000;
//    PirlProblemGenerator dataGen(n_candidate_ee_poses, prob.getPlanningGroup(), planning_scene, iksolver);
    EvalProblemGenerator dataGen(n_candidate_ee_poses, prob.getPlanningGroup(), planning_scene, iksolver);

//    // [Semantic task Generation] ///////////////////////////////////////////////////////////////////////////////////
//    for (int k = 0; k < 1000; k++) {
//        std::vector<std::vector<double>> wps;
//
//        std::string wps_name = "test";
//        std::string base_path = ros::package::getPath("torm");
//        std::string wps_file = base_path + "/launch/config/waypoints/" + wps_name + ".txt";
//        std::ifstream ifs(wps_file);
//        if (ifs.is_open()) {
//            std::string line;
//            while (getline(ifs, line)) {
//                std::vector<double> wps_node = torm::split_f(line, ' '); // [x y z][r p y]
//                KDL::Frame pose;
//                pose.p[0] = wps_node[0];
//                pose.p[1] = wps_node[1];
//                pose.p[2] = wps_node[2];
//                pose.M = KDL::Rotation::RPY(wps_node[3], wps_node[4], wps_node[5]);
//                double x, y, z, w;
//                pose.M.GetQuaternion(x, y, z, w);
//                wps.push_back(std::vector<double>{pose.p.x(), pose.p.y(), pose.p.z(), x, y, z, w}); // [x y z][x y z w]
//            }
//        }
//
//        std::vector<std::vector<double>> path; // [x y z][x y z w]
//        std::vector<std::vector<double>> configs;
//        bool is_valid_target = dataGen.interpolate_linear(wps, 0.01, path, configs);
////        bool is_valid_target = dataGen.interpolate_spline(wps, 0.01, path, configs);
//        if (!is_valid_target) {
//            std::cout << "-- cannot solve this problem with naive IK solver..." << std::endl;
//        }
//
//        std::vector<KDL::Frame> targetPoses;
//        targetPoses.reserve(path.size());
//        for (int i = 0; i < path.size(); i++) {
//            KDL::Frame f;
//            f.p[0] = path[i][0];
//            f.p[1] = path[i][1];
//            f.p[2] = path[i][2];
//            f.M = KDL::Rotation::Quaternion(path[i][3], path[i][4], path[i][5], path[i][6]);
//            targetPoses.push_back(f);
//        }
//        debug.publishEETrajectory(targetPoses, 0);
//
//        // evaluate
//        if(configs.size() > 4) { // need at least 4 points to calculate jerk.
//            traj_evaluator evaluator(targetPoses, configs, {2, 4, 6}, 0.1, iksolver, vel_limit);
//            double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
//            evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
//            std::cout << cost_pose << ", " << cost_rot << ", " << cost_vel << ", " << cost_acc << ", " << cost_jerk
//                      << std::endl;
//        }
//
//        // visualize joint trajectory
//        Eigen::MatrixXd trajectory_;
//        trajectory_.resize(configs.size(), num_dof);
//        for(int i=0; i<configs.size(); i++){
//            for(int j=0; j<num_dof; j++){
//                trajectory_.row(i)(j) = configs[i][j];
//            }
//        }
//        debug.show(targetPoses, trajectory_, 10, 3);
//        std::vector<std::string> indices_vis = indices;
//        debug.visualizeTrajectory(indices_vis, configs);
//
//        // TORM //////////////////////////////////////////////////////////////
//        bool show_torm_interpolation = true;
//        if (show_torm_interpolation){
//
//            ros::Duration(5).sleep();
//            int gap = 10;
//            std::vector<int> simplified_points;
//            for (int t = gap; t < targetPoses.size(); t += gap) {
//                simplified_points.push_back(t);
//            }
//            if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
//                simplified_points.push_back(targetPoses.size() - 1);
//            } // simplified_points: not include initial ee pose but include final ee pose.
//
//            KDL::JntArray q_start(num_dof);
//            for (uint j = 0; j < num_dof; j++) {
//                q_start(j) = configs[0][j];
//            }
//
//            // ===== TORM initial traj =====
//            TormInterpolator torm_interpolator(100, q_start, targetPoses, simplified_points, iksolver, planning_scene);
//
//            // evaluate
//            traj_evaluator evaluator(targetPoses, torm_interpolator.getTrajectory(), {2,4,6}, 0.1, iksolver, vel_limit);
//            double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
//            evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
//            std::cout << cost_pose << ", " << cost_rot << ", " << cost_vel << ", " << cost_acc << ", " << cost_jerk << std::endl;
//
//            debug.show(targetPoses, torm_interpolator.getTrajectory(), 10, 2);
//        }
//
//        // PIRL //////////////////////////////////////////////////////////////
//        bool show_nn_interpolation = true;
//        if(show_nn_interpolation)
//        {
//            ros::Duration(5).sleep();
//            int gap = 1;
//            std::vector<int> simplified_points;
//            for (int t = gap; t < targetPoses.size(); t += gap) {
//                simplified_points.push_back(t);
//            }
//            if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
//                simplified_points.push_back(targetPoses.size() - 1);
//            } // simplified_points: not include initial ee pose but include final ee pose.
//
//            KDL::JntArray q_start(num_dof);
//            for (uint j = 0; j < num_dof; j++) {
//                q_start(j) = configs[0][j];
//            }
//
//            // ===== PIRL initial traj =====
//            std::string model_path = "/data/pirl_network/RL/";
//            std::string model = "rl_Conf+LinkPoses_T1_model_W4W5.pt";
//            int t = atoi(&(model[model.find('T')+1]));
//            std::string obsType = model.substr(model.find("rl_")+3, (model.find("_T") - (model.find("rl_")+3)));
//            int multi_action = 1;
//            PirlInterpolatorPtr PirlModel = std::make_shared<PirlInterpolator>(model_path+model, t, obsType, targetPoses, simplified_points, iksolver, multi_action);
//
//            PirlModel->interpolate(q_start, true);
//
//            // evaluate
//            traj_evaluator evaluator(targetPoses, PirlModel->getTrajectory(), {2,4,6}, 0.1, iksolver, vel_limit);
//            double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
//            evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
//            std::cout << cost_pose << ", " << cost_rot << ", " << cost_vel << ", " << cost_acc << ", " << cost_jerk << std::endl;
//
//            debug.show(targetPoses, PirlModel->getTrajectory(), 10, 1);
//        }
//    }

    // [Random Problem Generation] ///////////////////////////////////////////////////////////////////////////////////
    std::vector<std::vector<KDL::Frame>> targetPoses_list;
    int w = 6; // spline interpolation needs al least 5 wps. (w>=5)
    while(targetPoses_list.size() != 100){
        std::vector<std::vector<double>> path; // [(xyz),(xyzw)]
        std::vector<std::vector<double>> configs;

        bool is_valid_target = dataGen.getRandomProblem(w, 0.005, path, configs);
        if (!is_valid_target) {
            std::cout << "/-- generated target ee path isn't valid.....!" << std::endl;
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
        targetPoses_list.push_back(targetPoses);
    }
    for(auto target_path: targetPoses_list){
        debug.publishEETrajectory(target_path, 0);
        ros::Duration(0.2).sleep();

        // visualize joint trajectory
//        std::vector<std::string> indices_vis = indices;
//        debug.visualizeTrajectory(indices_vis, configs);
//        ros::Duration(0.1).sleep();
        debug.clear();
    }

}
