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

void path_write(std::string file_name,
                Eigen::MatrixXd& outputMatrix,
                std::vector<double>& cost_log, std::vector<double>& time_log){

    std::string f_name;
    f_name = file_name;
    std::fstream fs;
    fs.open(f_name.append("_config.csv").c_str(), std::ios::out);
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

    f_name = file_name;
    fs.open(f_name.append("_cost_log.csv").c_str(), std::ios::out);
    for(int i=0; i<cost_log.size() ;i++){
        if(i==cost_log.size()-1){
            fs << cost_log[i] << "\n";
        }else{
            fs << cost_log[i] << ",";
        }
    }
    fs.close();

    f_name = file_name;
    fs.open(f_name.append("_time_log.csv").c_str(), std::ios::out);
    for(int i=0; i<time_log.size() ;i++){
        if(i==time_log.size()-1){
            fs << time_log[i] << "\n";
        }else{
            fs << time_log[i] << ",";
        }
    }
    fs.close();
}

void path_write(std::string file_name,
                std::vector<std::vector<double>>& targetpath){
    std::fstream fs;
    fs.open(file_name.c_str(), std::ios::out);
    for(int i=0; i<targetpath.size() ;i++){
        fs << targetpath[i][0] << ",";
        fs << targetpath[i][1] << ",";
        fs << targetpath[i][2] << ";";
        fs << targetpath[i][6] << ","; // w
        fs << targetpath[i][3] << ","; // x
        fs << targetpath[i][4] << ","; // y
        fs << targetpath[i][5] << "\n"; // z
    }
    fs.close();
}

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
    params_.planning_time_limit_ = 50.0;
    params_.smoothness_update_weight_ = 30.0/endPose_size;
    params_.obstacle_update_weight_ = 1.0;
    params_.learning_rate_ = 0.01; // only affect feasible increments terms...

    params_.jacobian_update_weight_ = 1.0;

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
    params_.min_clearence_ = 0.5;
    params_.use_stochastic_descent_ = true;
    params_.use_velocity_check_ = true;
    params_.use_singularity_check_ = false;
    params_.use_collision_check_ = true;

    params_.singularity_lower_bound_ = 0.005; // fetch arm
    params_.exploration_iter_ = 50; // 2 stage gradient iters
    params_.traj_generation_iter_ = 100; // # of Ik candidates
    params_.time_duration_ = 0.2;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "exter_obs_datagen");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    robot_state::RobotState state = planning_scene->getCurrentState();

    PirlProblem prob("fetch", planning_scene);
    prob.resetCollisionObjects();

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int) joint_bounds.size();

    torm::TormIKSolver iksolver(prob.getPlanningGroup(), planning_scene, prob.getBaseLink(), prob.getTipLink());
    torm::TormDebugPtr debug = std::make_shared<torm::TormDebug>(planning_scene, prob.getPlanningGroup(), prob.getFixedFrame(), prob.getBaseLink(), iksolver);

    // for valid problem (end-effector path)
    bool gen_problems = true;
    bool is_visualize = true;
    if(gen_problems){
        int n_total_problems_per_scene = 20;
        int n_candidate_ee_poses = 100;
        int w = 6;
        double interval_length = 0.01;

        int n_total_scene = 100;
        for(int s=0; s < n_total_scene; s++){
            // setting env
//            std::string obs_file = std::string("/data/pirl_data/scene/scene_")+std::to_string(s)+".txt";
            std::string obs_file = ros::package::getPath("torm") + "/launch/config/scene/scene_"+std::to_string(s)+".txt";
            ROS_WARN_STREAM("load obs: " << obs_file);
            prob.setCollisionObjects(obs_file);
            ros::Duration(0.1).sleep();
            PirlProblemGenerator dataGen(n_candidate_ee_poses, PLANNING_GROUP, planning_scene, iksolver);

            if (is_visualize){
                debug->clear();
            }

            // gen problems
            std::vector<std::vector<double>> path; // [(xyz),(xyzw)]
            int n_suc = 0;
            while(n_suc < n_total_problems_per_scene) {
                bool is_valid_target = dataGen.sampleTargetEEList(w, interval_length, path);
                if (is_valid_target) {
                    if (is_visualize) {
                        ROS_INFO("found valid problem.");
                        debug->publishEETrajectory(path, 0);
                    }
//                    std::string save_file_name = std::string("/data/pirl_data/problem/")+
//                            std::to_string(s)+
//                            "/prob_"+std::to_string(n_suc)+".csv";
                    std::string save_file_name = ros::package::getPath("torm") + "/launch/config/path/random_obs_" +
                            std::to_string(s) + "_" + std::to_string(n_suc);
                    path_write(save_file_name, path);
                    n_suc++;
                    ROS_WARN_STREAM("problem saved. scene: "<< s <<", n_path: "<< n_suc);
                }
                path.clear();
            }
            // removing env
            prob.resetCollisionObjects();
        }
    }

    // for demonstration set
    bool solve_problems = false;
    bool is_visualize_opt = false;
    if(solve_problems){
        int n_total_scene = 1000;
        int n_total_problems_per_scene = 100;
        for(int s=5; s < n_total_scene; s++) {
            // setting env
            std::string obs_file = std::string("/data/pirl_data/scene/scene_")+std::to_string(s)+".txt";
            ROS_WARN_STREAM("load obs: " << obs_file);
            prob.setCollisionObjects(obs_file);
            ros::Duration(0.1).sleep();

            for(int p=0; p < n_total_problems_per_scene; p++){
                std::string file_name = std::string("/data/pirl_data/problem/")+
                                        std::to_string(s)+
                                        "/prob_"+std::to_string(p)+".csv";
                std::vector<std::vector<double>> path;
                path_load(file_name, path);
                ROS_WARN_STREAM("load problem: " << file_name);

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

                int gap = 10;
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
                    if(!iksolver.ikSolverCollFree(targetPoses[0], q_start, 9999)){
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

                // generate trajectory
                torm::TormTrajectory trajectory(planning_scene->getRobotModel(), int(targetPoses.size()), params.time_duration_, PLANNING_GROUP);
                trajectory.getTrajectoryPoint(0) = q_start.data; // set initial configuration!
                std::cout<< "[INFO] trajectory initialized." <<std::endl;

                // trajectory optimization
                torm::TormOptimizer opt(&trajectory, planning_scene, PLANNING_GROUP, &params, state,
                                        targetPoses, simplified_points, iksolver, joint_bounds, fix_start_config, false, false, false, false,
                                        nullptr, debug, 1, 1e-3);
                std::cout<< "[INFO] opt initialized." <<std::endl;

                bool result = opt.iterativeExploration();
                if (result) {
                    ROS_INFO("[RESULT] *Succeed* to find a valid trajectory.");
                    Eigen::MatrixXd &optimizedConfigs = trajectory.getTrajectory();
                    std::vector<double> cost_log = opt.getBestCostLog();
                    std::vector<double> time_log = opt.getTimeLog();
                    std::string file_name = "/data/pirl_data/torm_solution/";
                    file_name += std::to_string(s);
                    file_name += "/";
                    file_name += std::to_string(p);

                    std::cout << file_name << std::endl;
                    path_write(file_name, optimizedConfigs, cost_log, time_log);

                    if (is_visualize_opt) {
                        debug->show(targetPoses, trajectory.getTrajectory(), 0.1);
                    }
                }
            }
            // removing env
            prob.resetCollisionObjects();
        }
    }


} // end main








//        std::vector<std::vector<double>> path; // [(xyz),(xyzw)]
//        std::vector<std::vector<double>> rpypath;
//        int n_suc = 0;
//        while(n_suc < n_total_problems_per_scene) {
//            bool is_valid_target = dataGen.sampleTargetEEList(w, interval_length, path, rpypath);
//            if (!is_valid_target) {
////                ROS_INFO("generated target ee path (problem) isn't valid.....!");
//            }else{
////                ROS_INFO("found valid problem.");
////                std::vector<KDL::Frame> targetPoses;
////                targetPoses.reserve(path.size());
////                for (int i = 0; i < path.size(); i++) {
////                    KDL::Frame f;
////                    f.p[0] = path[i][0];
////                    f.p[1] = path[i][1];
////                    f.p[2] = path[i][2];
////                    f.M = KDL::Rotation::Quaternion(path[i][3], path[i][4], path[i][5], path[i][6]);
////                    targetPoses.push_back(f);
////                }
////                debug->publishEETrajectory(targetPoses, 0);
////                ros::Duration(0.3).sleep();
//                path_write(file_name,
//                           optimizedConfigs, path, simplified_points,
//                           w, interval_length, dataGen.getTargetEEN(), dataGen.getTargetEELen(),
//                           cost_log, time_log);
//                n_suc++
//            }
//        }