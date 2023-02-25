#ifndef CATKIN_WS_EVAL_PROBLEM_GENERATOR_H
#define CATKIN_WS_EVAL_PROBLEM_GENERATOR_H

#include <vector>
#include <iostream>

#include <interpolation/interpolator6D.h>
#include <torm/torm_utils.h>
#include <torm/torm_ik_solver.h>

// MoveIt!
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/conversions.h>
#include <random>



class EvalProblemGenerator {
public:
    EvalProblemGenerator(int n_samples, std::string planning_group, planning_scene::PlanningScenePtr planning_scene, torm::TormIKSolver& iksolver);
    ~EvalProblemGenerator();

    void setCollisionChecker();
    bool collision_check(KDL::JntArray& rnd_q);

    void collectEEPoses();
    bool getRandomProblem(  int n_wps, double interval_length,
                            std::vector<std::vector<double>>& path,
                            std::vector<std::vector<double>>& configs);

    bool interpolate_linear(std::vector<std::vector<double>>& wps_list, double interval_length,
                            std::vector<std::vector<double>>& path,
                            std::vector<std::vector<double>>& configs);
    bool interpolate_spline(std::vector<std::vector<double>>& wps_list, double interval_length,
                            std::vector<std::vector<double>>& path,
                            std::vector<std::vector<double>>& configs);



    bool checkExistIKwSolution( std::vector<std::vector<double>>& path,
                                std::vector<std::vector<double>>& configs);
    bool checkExistIK(std::vector<std::vector<double>>& path,
                      std::vector<std::vector<double>>& configs);

    int getTargetEEN();
    double getTargetEELen();

private:
    ros::NodeHandle nh_;

    std::string planning_group_;
    planning_scene::PlanningScenePtr planning_scene_;
    torm::TormIKSolver& iksolver_;

    collision_detection::CollisionRequest c_request_;
    collision_detection::CollisionResult c_result_;

    int n_dof_;

    std::vector<double> fk_base_position_;
    std::vector<double> sample_range_;

    int n_samples_;
    std::vector<std::vector<double>> EEPoses_;
    std::vector<KDL::JntArray> Joints_;

    std::mt19937 mersenne_;
    std::uniform_real_distribution<double> p_add_rot_;
    std::uniform_real_distribution<double> rnd_x_;
    std::uniform_real_distribution<double> rnd_y_;
    std::uniform_real_distribution<double> rnd_z_;
    std::normal_distribution<double> rnd_yaw_;
    std::normal_distribution<double> rnd_pitch_;
    std::uniform_real_distribution<double> rnd_roll_;

    int path_total_n_{0};
    double path_total_len_{0};

};


















#endif //CATKIN_WS_EVAL_PROBLEM_GENERATOR_H
