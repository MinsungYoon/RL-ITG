
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


void write_problem( std::string file_name,
                    std::vector<std::vector<double>>& configs,
                    std::vector<std::vector<double>>& targetpath ){ // original frame:
    fstream fs;
    fs.open(file_name.c_str(), ios::out);
    for(int i=0; i<targetpath.size() ;i++){
        fs << targetpath[i][0] << ",";
        fs << targetpath[i][1] << ",";
        fs << targetpath[i][2] << ";";
        fs << targetpath[i][6] << ",";
        fs << targetpath[i][3] << ",";
        fs << targetpath[i][4] << ",";
        fs << targetpath[i][5] << "\n"; // [x y z][w x y z]
    }
    fs.close();

    fs.open(file_name.append("_config.csv").c_str(), ios::out);
    for(int i=0; i<configs.size() ;i++){
        fs << configs[i][0] << ",";
        fs << configs[i][1] << ",";
        fs << configs[i][2] << ",";
        fs << configs[i][3] << ",";
        fs << configs[i][4] << ",";
        fs << configs[i][5] << ",";
        fs << configs[i][6] << "\n";
    }
    fs.close();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pirl_prob_gen");
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
    debug.clear();

    std::vector<std::vector<double>> path; // [x y z][x y z w]
    std::vector<std::vector<double>> configs;

    std::string base_path = ros::package::getPath("torm");

    // [Random Problem Generation] ///////////////////////////////////////////////////////////////////////////////////
    bool random_task_gen = false;
    int n_prob = 100;
    int cur_n_prob = 0;
    if(random_task_gen) {
        EvalProblemGenerator dataGen(500, prob.getPlanningGroup(), planning_scene, iksolver);
        int w = 7; // spline interpolation needs al least 5 wps. (w>=5)
        while(cur_n_prob < n_prob) {
            path.clear();
            configs.clear();
            bool is_valid_target = dataGen.getRandomProblem(w, 0.005, path, configs);
            if (!is_valid_target) {
                std::cout << cur_n_prob << "-- cannot solve this problem with naive IK solver..." << std::endl;
                continue;
            }
            std::string save_name = base_path + "/launch/config/path/random_" + std::to_string(cur_n_prob);
            write_problem(save_name, configs, path);
            cur_n_prob++;

            bool visualize = true;
            if (visualize) {
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
                ros::Duration(0.5).sleep();

                // PIRL //////////////////////////////////////////////////////////////
                bool show_nn_interpolation = true;
                if (show_nn_interpolation) {
                    ros::Duration(5).sleep();
                    int gap = 1;
                    std::vector<int> simplified_points;
                    for (int t = gap; t < targetPoses.size(); t += gap) {
                        simplified_points.push_back(t);
                    }
                    if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
                        simplified_points.push_back(targetPoses.size() - 1);
                    } // simplified_points: not include initial ee pose but include final ee pose.

                    KDL::JntArray q_start(num_dof);
                    for (uint j = 0; j < num_dof; j++) {
                        q_start(j) = configs[0][j];
                    }

                    // ===== PIRL initial traj =====
                    std::string model_path = "/data/pirl_network/RL/";
                    std::string model = "rl_Conf+LinkPoses_T1_model_FREEW4W5W6W7best.pt";
                    int t = atoi(&(model[model.find('T') + 1]));
                    std::string obsType = model.substr(model.find("rl_") + 3,
                                                       (model.find("_T") - (model.find("rl_") + 3)));
                    int multi_action = 1;
                    PirlInterpolatorPtr PirlModel = std::make_shared<PirlInterpolator>(model_path + model, t, obsType,
                                                                                       targetPoses, simplified_points,
                                                                                       iksolver, multi_action);

                    PirlModel->interpolate(q_start, true);

                    // evaluate
                    traj_evaluator evaluator(targetPoses, PirlModel->getTrajectory(), {2, 4, 6}, 0.1, iksolver, vel_limit);
                    double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
                    evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
                    std::cout << cost_pose << ", " << cost_rot << ", " << cost_vel << ", " << cost_acc << ", "
                              << cost_jerk << std::endl;

                    debug.show(targetPoses, PirlModel->getTrajectory(), 1);
                }
                debug.clear();
            }
        }
    }// end

    // [Semantic task Generation] ///////////////////////////////////////////////////////////////////////////////////
    bool semantic_task_gen = true;
    if(semantic_task_gen) {
        EvalProblemGenerator dataGen(0, prob.getPlanningGroup(), planning_scene, iksolver);

        std::vector<std::vector<double>> wps;

        std::string wps_name = "test";
        std::string wps_file = base_path + "/launch/config/waypoints/" + wps_name + ".txt";
        std::ifstream ifs(wps_file);
        if (ifs.is_open()) {
            std::string line;
            while (getline(ifs, line)) {
                std::vector<double> wps_node = torm::split_f(line, ' '); // [x y z][r p y]
                KDL::Frame pose;
                pose.p[0] = wps_node[0];
                pose.p[1] = wps_node[1];
                pose.p[2] = wps_node[2];
                pose.M = KDL::Rotation::RPY(wps_node[3], wps_node[4], wps_node[5]);
                double x, y, z, w;
                pose.M.GetQuaternion(x, y, z, w);
                wps.push_back(std::vector<double>{pose.p.x(), pose.p.y(), pose.p.z(), x, y, z, w}); // [x y z][x y z w]
            }
        }

        bool is_valid_target = dataGen.interpolate_linear(wps, 0.01, path, configs);
//        bool is_valid_target = dataGen.interpolate_spline(wps, 0.01, path, configs);
        if (!is_valid_target) {
            std::cout << "-- cannot solve this problem with naive IK solver..." << std::endl;
        }
        std::string save_name = base_path + "/launch/config/path/" + wps_name;
        write_problem(save_name, configs, path);
    }// end

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
    ros::Duration(0.5).sleep();

    // visualize joint trajectory
    std::vector<std::string> indices_vis = indices;
    debug.visualizeTrajectory(indices_vis, configs);
    ros::Duration(0.1).sleep();

    // TORM //////////////////////////////////////////////////////////////
    bool show_torm_interpolation = true;
    if (show_torm_interpolation){

        ros::Duration(5).sleep();
        int gap = 10;
        std::vector<int> simplified_points;
        for (int t = gap; t < targetPoses.size(); t += gap) {
            simplified_points.push_back(t);
        }
        if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
            simplified_points.push_back(targetPoses.size() - 1);
        } // simplified_points: not include initial ee pose but include final ee pose.

        KDL::JntArray q_start(num_dof);
        for (uint j = 0; j < num_dof; j++) {
            q_start(j) = configs[0][j];
        }

        // ===== TORM initial traj =====
        TormInterpolator torm_interpolator(100, q_start, targetPoses, simplified_points, iksolver, planning_scene);

        // evaluate
        traj_evaluator evaluator(targetPoses, torm_interpolator.getTrajectory(), {2,4,6}, 0.1, iksolver, vel_limit);
        double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
        evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
        std::cout << cost_pose << ", " << cost_rot << ", " << cost_vel << ", " << cost_acc << ", " << cost_jerk << std::endl;

        debug.show(targetPoses, torm_interpolator.getTrajectory(), 1);
    }

    // PIRL //////////////////////////////////////////////////////////////
    bool show_nn_interpolation = true;
    if(show_nn_interpolation)
    {
        ros::Duration(5).sleep();
        int gap = 1;
        std::vector<int> simplified_points;
        for (int t = gap; t < targetPoses.size(); t += gap) {
            simplified_points.push_back(t);
        }
        if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
            simplified_points.push_back(targetPoses.size() - 1);
        } // simplified_points: not include initial ee pose but include final ee pose.

        KDL::JntArray q_start(num_dof);
        for (uint j = 0; j < num_dof; j++) {
            q_start(j) = configs[0][j];
        }

        // ===== PIRL initial traj =====
        std::string model_path = "/data/pirl_network/RL/";
        std::string model = "rl_Conf+LinkPoses_T1_model_FREEW4W5W6W7best.pt";
        int t = atoi(&(model[model.find('T')+1]));
        std::string obsType = model.substr(model.find("rl_")+3, (model.find("_T") - (model.find("rl_")+3)));
        int multi_action = 1;
        PirlInterpolatorPtr PirlModel = std::make_shared<PirlInterpolator>(model_path+model, t, obsType, targetPoses, simplified_points, iksolver, multi_action);

        PirlModel->interpolate(q_start, true);

        // evaluate
        traj_evaluator evaluator(targetPoses, PirlModel->getTrajectory(), {2,4,6}, 0.1, iksolver, vel_limit);
        double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
        evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
        std::cout << cost_pose << ", " << cost_rot << ", " << cost_vel << ", " << cost_acc << ", " << cost_jerk << std::endl;

        debug.show(targetPoses, PirlModel->getTrajectory(), 1);
    }
    debug.clear();



    return 0;
}
