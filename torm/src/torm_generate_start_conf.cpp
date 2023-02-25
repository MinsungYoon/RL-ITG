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

// argv[1]: scene name.

int main(int argc, char** argv) {
    ros::init(argc, argv, "torm_gen_start_conf");
    ros::NodeHandle node_handle("~");

    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

    std::cout << argv[1] << std::endl;
    torm::TormProblem prob(argv[1], "fetch", planning_scene);

    const std::string PLANNING_GROUP = prob.getPlanningGroup();
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);
    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds;
    joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = (unsigned int)joint_bounds.size();

    torm::TormIKSolver iksolver(PLANNING_GROUP, planning_scene, prob.getBaseLink(), prob.getTipLink());
    torm::TormDebugPtr debug = std::make_shared<torm::TormDebug>(planning_scene, prob.getPlanningGroup(), prob.getFixedFrame(), prob.getBaseLink(), iksolver);

    std::vector<KDL::Frame> targetPoses = prob.getTargetPoses(); // [0]: start pose (matching with start conf)
    debug->publishEETrajectory(targetPoses, 0);

    bool select_visually = false;

    int n_start_confs = 100;
    std::vector<KDL::JntArray> list_confs;
    list_confs.reserve(n_start_confs);

    KDL::JntArray q_init(num_dof);
    int n_suc = 0;
    while (true){
        if(iksolver.ikSolverCollFree(targetPoses[0], q_init, 9999)){
            if(select_visually){
                debug->visualizeConfiguration(indices, q_init);
                int in;
                std::cin >> in;
                if (in==1){
                    list_confs.push_back(q_init);
                    n_suc++;
                    std::cout << n_suc << std::endl;
                }else if (in==2){
                    break;
                }
            }else{
                list_confs.push_back(q_init);
                n_suc++;
                std::cout << n_suc << std::endl;
            }
            if (n_suc==n_start_confs){
                break;
            }
        }
    }

    debug->visualizeTrajectory(indices, list_confs);
    ros::Duration(0.1).sleep();

    std::string base_path = ros::package::getPath("torm");
    std::fstream fs;
    fs.open((base_path + "/launch/config/start_conf/" + argv[1] + "_config.csv").c_str(), std::ios::out);
    for (int k=0; k<n_suc; k++) {
        for (int i = 0; i < num_dof; i++) {
            if (i == num_dof - 1) {
                if (k == n_suc - 1){
                    fs << list_confs[k](i);
                }else{
                    fs << list_confs[k](i) << "\n";
                }
            } else {
                fs << list_confs[k](i) << ",";
            }
        }
    }
    fs.close();

//    ros::Duration(10).sleep();
    return 0;
}