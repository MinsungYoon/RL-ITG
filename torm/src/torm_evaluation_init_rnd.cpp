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

void res_write(std::string file_name,
               std::vector<double>& res){
    std::fstream fs;
    fs.open(file_name.append(".csv").c_str(), std::ios::out);
    if(fs.is_open()){
        for(int i=0; i<res.size() ;i++){
            fs << res[i];
            if(i!=res.size()-1){
                fs << ",";
            }
        }
        fs.close();
    }else{
        std::cout << strerror(errno) << '\n';
        exit(0);
    }
}

struct Mean_Std {
    double mean;
    double std;
};

Mean_Std standardDeviation(std::vector<double> v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double squareSum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double std = sqrt(squareSum / v.size() - mean * mean);

    Mean_Std return_struct;
    return_struct.mean = mean;
    return_struct.std = std;
    return return_struct;
}

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
        params_.planning_time_limit_ = 1000.0;
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
    params_.min_clearence_ = 0.001;
    params_.use_stochastic_descent_ = true;
    params_.use_velocity_check_ = true;
    params_.use_singularity_check_ = true;
    params_.use_collision_check_ = true;

    params_.singularity_lower_bound_ = 0.01; // fetch arm
    if (debug_visual){
        params_.exploration_iter_ = 50; // 2 stage gradient iters
    }else{
        params_.exploration_iter_ = 50; // 2 stage gradient iters
    }
    params_.traj_generation_iter_ = 100; // # of Ik candidates
    params_.time_duration_ = 0.2;
}

/// rosrun torm torm_evaluation_init

int main(int argc, char** argv) {
    ros::init(argc, argv, "torm");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    bool save_initial_trajectory = false;
    int n_regeneration = 0; // not including the first generated initial trajectory.

    // ************************************************************************************************** , "square", "s"
    std::vector<std::string> problem_list{"random_obs"};
    std::vector<std::string> method_list {"pirl_rl_test_0", "pirl_rl_test_1", "pirl_rl_test_2", "torm_test", "torm_jli_test"};

    for (auto method_name : method_list) {
        double final_pose_error = 0.0, final_smoothness = 0.0, final_time = 0.0, final_fv = 0.0, final_fc = 0.0, final_fs = 0.0, final_f = 0.0;
        std::vector<double> n_nodes;
        for (int problem_idx = 0; problem_idx < problem_list.size(); problem_idx++) {

            std::vector<double> res_pose, res_vel, res_time, res_fv, res_fc, res_fs, res_f;
            int n_scene, n_path, i_scene = 0, i_path = 0;
            if (problem_list[problem_idx].find("random_obs") != std::string::npos) {
                n_scene = 20; // 100
                n_path = 2; // 10
            } else if (problem_list[problem_idx].find("random") != std::string::npos) {
                n_scene = 1; // 1
                n_path = 100; // 100
            } else {
                n_scene = 1;
                n_path = 1;
            }
            res_pose.reserve(n_scene * n_path);
            res_vel.reserve(n_scene * n_path);
            res_time.reserve(n_scene * n_path);
            res_fv.reserve(n_scene * n_path);
            res_fc.reserve(n_scene * n_path);
            res_fs.reserve(n_scene * n_path);
            res_f.reserve(n_scene * n_path);

            while (true) {
                std::string problem_name = problem_list[problem_idx];
                if (problem_list[problem_idx].find("random_obs") != std::string::npos) {
                    problem_name += "_";
                    problem_name += std::to_string(i_scene);
                    problem_name += "_";
                    problem_name += std::to_string(i_path);
                } else if (problem_list[problem_idx].find("random") != std::string::npos) {
                    problem_name += "_";
                    problem_name += std::to_string(i_path);
                }
                ///////
                torm::TormProblem prob(problem_name, "fetch", planning_scene);
                const std::string PLANNING_GROUP = prob.getPlanningGroup();
                const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
                std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
                robot_model::JointBoundsVector joint_bounds;
                joint_bounds = joint_model_group->getActiveJointModelsBounds();
                unsigned int num_dof = (unsigned int) joint_bounds.size();
                std::vector<double> vel_limit;
                vel_limit.reserve(num_dof);
                for(uint i = 0; i < joint_bounds.size(); i++){
                    vel_limit.push_back(joint_bounds[i][0][0].max_velocity_);
                }
                torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, prob.getBaseLink(), prob.getTipLink());
                std::vector<KDL::Frame> targetPoses = prob.getTargetPoses(); // [0]: start pose (matching with start conf)

                prob.resetCollisionObjects();
                prob.setCollisionObjects();

                // **************************************************************************************************

                std::vector<int> simplified_points;
                int gap;
                if (method_name.find("pirl") != std::string::npos) {
                    gap = 1;
                } else {
                    gap = 10;
                }
                for (int i = gap; i < targetPoses.size(); i += gap) {
                    simplified_points.push_back(i);
                }
                if (simplified_points[simplified_points.size() - 1] != targetPoses.size() - 1) {
                    simplified_points.push_back(targetPoses.size() - 1);
                } // simplified_points: not include initial ee pose but include final ee pose.

                std::cout << "==================================================================" << std::endl;
                std::cout << problem_name << " & " << method_name << std::endl;
                std::cout << method_name << "==> Gap: " << gap << ", N: " << targetPoses.size() << ", S: "
                          << simplified_points.size() << std::endl;
                n_nodes.push_back(static_cast<double>(targetPoses.size()));

                torm::TormParameters params;
                initParameters(params, targetPoses.size(), false);

                // **************************************************************************************************
                int n_test_configs = 2;
                std::vector<double> time_list;
                std::vector<double> pose_list;
                std::vector<double> vel_list;
                std::vector<double> feasible_vel_list;
                std::vector<double> feasible_col_list;
                std::vector<double> feasible_singularity_list;
                time_list.reserve(n_test_configs);
                pose_list.reserve(n_test_configs);
                vel_list.reserve(n_test_configs);
                feasible_vel_list.reserve(n_test_configs);
                feasible_col_list.reserve(n_test_configs);
                feasible_singularity_list.reserve(n_test_configs);

                for (int start_config_idx = 0; start_config_idx < n_test_configs; start_config_idx++) {
                    bool fix_start_config = true;
                    // [setup initial config]
                    KDL::JntArray q_start(num_dof);
                    std::vector<double> s_conf;
                    s_conf.reserve(num_dof);
                    if (fix_start_config) {
                        prob.setStartConfig(start_config_idx);
                        s_conf = prob.getStartConfiguration();
                        for (uint j = 0; j < num_dof; j++) {
                            q_start(j) = s_conf[j];
                        }
                    } else {
                        if (!iksolver.ikSolverCollFree(targetPoses[0], q_start)) {
                            ROS_ERROR("[ERROR] No found a valid start configuration.");
                            return 0;
                        } else {
                            for (int j = 0; j < num_dof; j++) {
                                s_conf.push_back(q_start(j));
                            }
                            ROS_WARN("--- set random start configuration!!!!");
                        }
                    }

                    // set current state
                    robot_state::RobotState rs = planning_scene->getCurrentState();
                    rs.setJointGroupPositions(PLANNING_GROUP, s_conf);
                    rs.update();
                    planning_scene->setCurrentState(rs);

                    torm::TormDebugPtr debug = std::make_shared<torm::TormDebug>(planning_scene,
                                                                                 prob.getPlanningGroup(),
                                                                                 prob.getFixedFrame(),
                                                                                 prob.getBaseLink(), iksolver);
                    // visualize input poses
                    debug->publishEETrajectory(targetPoses, 0);
                    ros::Duration(1.0).sleep();
                    debug->visualizeConfiguration(indices, q_start);
                    ros::Duration(1.0).sleep();

                    // [onlyConf|Conf+LinkPoses|Conf+EE|onlyLinkPoses]
                    // Pirl ===========================================================================================================
                    bool use_PIRL = method_name.find("pirl") != std::string::npos;
                    PirlInterpolatorPtr PirlModel = nullptr;
                    if (use_PIRL) {
                        if (method_name.find("pirl_bc") != std::string::npos) {
                            const std::string model_path = "/data/pirl_network/BC/bc_basic_model.pt";
                            int t = 6;
                            std::string obsType = "Conf+LinkPoses";
                            PirlModel = std::make_shared<PirlInterpolator>(model_path, t, obsType, targetPoses,
                                                                           simplified_points, iksolver);
                        } else if (method_name.find("pirl_rl") != std::string::npos) {
                            std::string model_path = "/data/pirl_network/RL/";
                            std::string model;

                            std::vector<std::string> sp = torm::split(method_name, '_');
                            std::vector<std::string> models;
                            if (prob.getIsLoadScene()) {
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task+Imitation_Ent3_best.pt");
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task_Ent3_best.pt"); // best
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_TABLE2OBS_Task_Ent3_last.pt");
                            } else {
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_W4W5.pt");
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_FREEALL_Task.pt");
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_JACOFREE_best.pt");
                                models.emplace_back("rl_Conf+LinkPoses_T1_model_JACOFREE_last.pt");
                            }
                            model = models[std::atoi(sp[sp.size() - 1].c_str())];
                            int t = atoi(&(model[model.find('T') + 1]));
                            std::string obsType = model.substr(model.find("rl_") + 3,
                                                               (model.find("_T") - (model.find("rl_") + 3)));
                            int multi_action = 1;
                            PirlModel = std::make_shared<PirlInterpolator>(model_path + model, t, obsType, targetPoses,
                                                                           simplified_points, iksolver, multi_action);
                        }
                    }
                    // generate trajectory
                    torm::TormTrajectory trajectory(planning_scene->getRobotModel(), int(targetPoses.size()),
                                                    params.time_duration_, PLANNING_GROUP);
                    trajectory.getTrajectoryPoint(0) = q_start.data; // set initial configuration!

                    // trajectory optimization
                    torm::TormOptimizer opt(&trajectory, planning_scene, PLANNING_GROUP, &params,
                                            planning_scene->getCurrentState(),
                                            targetPoses, simplified_points, iksolver, joint_bounds,
                                            fix_start_config, true, false, false, false,
                                            PirlModel, debug, 1);

                    if (method_name.find("torm_jli_test") != std::string::npos) {
                        auto t1 = std::chrono::high_resolution_clock::now();
                        opt.getJointInterpolatedTrajectory();
                        auto t2 = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                        time_list.push_back(duration);
                    }
                    if (method_name.find("torm_test") != std::string::npos) {
                        auto t1 = std::chrono::high_resolution_clock::now();
                        opt.getNewTrajectory();
                        auto t2 = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                        time_list.push_back(duration);
                    }
                    if (method_name.find("pirl_bc_test") != std::string::npos or
                        method_name.find("pirl_rl_test") != std::string::npos) {
                        auto t1 = std::chrono::high_resolution_clock::now();
                        opt.setPIRLTrajectory(true);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                        time_list.push_back(duration);
                    }
                    debug->show(targetPoses, trajectory.getTrajectory(), 1);
                    if (save_initial_trajectory) {
                        std::string f_name = ros::package::getPath("torm") + "/launch/config/initial_traj/"
                                             + problem_name + "-" + method_name + "-" +
                                             std::to_string(start_config_idx) + "-" + std::to_string(0);
                        path_write(f_name, trajectory.getTrajectory());
                    } else {
                        debug->show(targetPoses, trajectory.getTrajectory(), 1);

                        std::vector<double> res;
                        opt.calcInitialTrajQuality(res);
                        std::string f_name = ros::package::getPath("torm") + "/result/"
                                             + problem_list[problem_idx] + "-" + method_name + "-" +
                                             std::to_string((i_scene*(n_path*n_test_configs) + i_path * n_test_configs + start_config_idx));
                        res_write(f_name, res);
                        std::cout << f_name << std::endl;

//                        opt.updateLocalGroupTrajectory();
//                        feasible_vel_list.push_back(opt.checkJointVelocityLimit(params.time_duration_));
//                        feasible_col_list.push_back(opt.isCurrentTrajectoryCollisionFree());
//                        feasible_singularity_list.push_back(opt.checkSingularity());
//
//                        traj_evaluator evaluator(targetPoses, trajectory.getTrajectory(), {2, 4, 6}, 0.01, iksolver, vel_limit);
//                        double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
//                        evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
//
//                        pose_list.push_back(cost_pose + params.rotation_scale_factor_ * cost_rot);
//                        vel_list.push_back(cost_vel);
                    }

                    if (save_initial_trajectory and n_regeneration >= 1) {
                        for (int i_p = 1; i_p <= n_regeneration; i_p++) {
                            if (method_name.find("torm_jli_test") != std::string::npos) {
                                opt.getJointInterpolatedTrajectory();
                            }
                            if (method_name.find("torm_test") != std::string::npos) {
                                opt.getNewTrajectory();
                            }
                            if (method_name.find("pirl_bc_test") != std::string::npos or
                                method_name.find("pirl_rl_test") != std::string::npos) {
                                opt.setPIRLTrajectory(false);
                            }
                            std::string f_name = ros::package::getPath("torm") + "/launch/config/initial_traj/"
                                                 + problem_name + "-" + method_name + "-" +
                                                 std::to_string(start_config_idx) + "-" + std::to_string(i_p);
                            path_write(f_name, trajectory.getTrajectory());
                        }
                    }
                }
//                if (!save_initial_trajectory) {
//                    std::vector<double> feasible_list;
//                    feasible_list.reserve(n_test_configs);
//                    for (int ttt = 0; ttt < n_test_configs; ttt++) {
//                        feasible_list.push_back(
//                                feasible_vel_list[ttt] * feasible_col_list[ttt] * feasible_singularity_list[ttt]);
//                    }
//
//                    Mean_Std pose_analysis = standardDeviation(pose_list);
//                    Mean_Std vel_analysis = standardDeviation(vel_list);
//                    Mean_Std time_analysis = standardDeviation(time_list);
//                    Mean_Std fv_analysis = standardDeviation(feasible_vel_list);
//                    Mean_Std fc_analysis = standardDeviation(feasible_col_list);
//                    Mean_Std fs_analysis = standardDeviation(feasible_singularity_list);
//                    Mean_Std f_analysis = standardDeviation(feasible_list);
//
////                    std::cout << "==================================================================" << std::endl;
////                    std::cout << problem_name << " & " << method_name << std::endl <<
////                              ",pose_analysis: " << pose_analysis.mean << ", " << pose_analysis.std << std::endl <<
////                              ",vel_analysis:" << vel_analysis.mean << ", " << vel_analysis.std << std::endl <<
////                              ",time_analysis:" << time_analysis.mean << ", " << time_analysis.std << std::endl <<
////                              ",fv_analysis:" << fv_analysis.mean << ", " << fv_analysis.std << std::endl <<
////                              ",fc_analysis:" << fc_analysis.mean << ", " << fc_analysis.std << std::endl <<
////                              ",fs_analysis:" << fs_analysis.mean << ", " << fs_analysis.std << std::endl <<
////                              ",f_analysis:" << f_analysis.mean << ", " << f_analysis.std << std::endl;
//
//                    res_pose.push_back(pose_analysis.mean);
//                    res_vel.push_back(vel_analysis.mean);
//                    res_time.push_back(time_analysis.mean);
//                    res_fv.push_back(fv_analysis.mean);
//                    res_fc.push_back(fc_analysis.mean);
//                    res_fs.push_back(fs_analysis.mean);
//                    res_f.push_back(f_analysis.mean);
//                }
                ///////
                i_path++;
                if (i_path == n_path) {
                    i_scene++;
                    if (i_scene == n_scene) {
                        break;
                    } else if (i_scene < n_scene) {
                        i_path = 0;
                    }
                }
            }
//            if (!save_initial_trajectory) {
//                Mean_Std pose_analysis = standardDeviation(res_pose);
//                Mean_Std vel_analysis = standardDeviation(res_vel);
//                Mean_Std time_analysis = standardDeviation(res_time);
//                Mean_Std fv_analysis = standardDeviation(res_fv);
//                Mean_Std fc_analysis = standardDeviation(res_fc);
//                Mean_Std fs_analysis = standardDeviation(res_fs);
//                Mean_Std f_analysis = standardDeviation(res_f);
//
//                std::cout << "=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=" << std::endl;
//                std::cout << "=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=" << std::endl;
//                std::cout << problem_list[problem_idx] << std::endl <<
//                          ",pose_analysis: " << pose_analysis.mean << ", " << pose_analysis.std << std::endl <<
//                          ",vel_analysis:" << vel_analysis.mean << ", " << vel_analysis.std << std::endl <<
//                          ",time_analysis:" << time_analysis.mean << ", " << time_analysis.std << std::endl <<
//                          ",fv_analysis:" << fv_analysis.mean << ", " << fv_analysis.std << std::endl <<
//                          ",fc_analysis:" << fc_analysis.mean << ", " << fc_analysis.std << std::endl <<
//                          ",fs_analysis:" << fs_analysis.mean << ", " << fs_analysis.std << std::endl <<
//                          ",f_analysis:" << f_analysis.mean << ", " << f_analysis.std << std::endl;
//                std::cout << "=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=" << std::endl;
//                std::cout << "=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=" << std::endl;
//                final_pose_error += pose_analysis.mean;
//                final_smoothness += vel_analysis.mean;
//                final_time += time_analysis.mean;
//                final_fv += fv_analysis.mean;
//                final_fc += fc_analysis.mean;
//                final_fs += fs_analysis.mean;
//                final_f += f_analysis.mean;
//            }
        }
//        if (!save_initial_trajectory) {
//            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
//            std::cout << final_pose_error / static_cast<double>(problem_list.size()) << ", "
//                      << final_smoothness / static_cast<double>(problem_list.size()) << ", "
//                      << final_time / static_cast<double>(problem_list.size()) << ", "
//                      << final_fv / static_cast<double>(problem_list.size()) << ", "
//                      << final_fc / static_cast<double>(problem_list.size()) << ", "
//                      << final_fs / static_cast<double>(problem_list.size()) << ", "
//                      << final_f / static_cast<double>(problem_list.size()) << std::endl;
//            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
//            Mean_Std n_nodes_analysis = standardDeviation(n_nodes);
//            std::cout << *max_element(n_nodes.begin(), n_nodes.end())
//                      << ", " << *min_element(n_nodes.begin(), n_nodes.end())
//                      << " | " << n_nodes_analysis.mean
//                      << ", " << n_nodes_analysis.std << std::endl;
//            std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
//        }
    }
    return 0;
}