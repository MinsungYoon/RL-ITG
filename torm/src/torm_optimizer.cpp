/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Modified by : Mincheul Kang */

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <torm/torm_utils.h>
#include <torm/torm_optimizer.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <eigen3/Eigen/Core>

namespace torm
{
    TormOptimizer::TormOptimizer(TormTrajectory* trajectory,
                                 planning_scene::PlanningScenePtr planning_scene,
                                 const std::string& planning_group, TormParameters* parameters,
                                 const moveit::core::RobotState& start_state,
                                 const std::vector<KDL::Frame> &end_poses,
                                 const std::vector<int> &simplified_points,
                                 TormIKSolver& iksolver,
                                 const robot_model::JointBoundsVector& bounds,
                                 bool fix_start_config, bool debug_log, bool debug_verbose, bool debug_visual, bool debug_visual_onlybest,
                                 PirlInterpolatorPtr PirlModel,
                                 TormDebugPtr debug,
                                 int lr_schedule_mode,
                                 double return_threshold,
                                 TormTrajectory* init_trajectory)
            : full_trajectory_(trajectory)
            , kmodel_(planning_scene->getRobotModel())
            , planning_group_(planning_group)
            , parameters_(parameters)
            , group_trajectory_(*full_trajectory_, planning_group_, DIFF_RULE_LENGTH)
            , planning_scene_(planning_scene)
            , state_(start_state)
            , endPoses_desired_(end_poses)
            , simplified_points_(simplified_points)
            , iksolver_(iksolver)
            , fix_start_config_(fix_start_config)
            , debug_log_(debug_log)
            , debug_verbose_(debug_verbose)
            , debug_visual_(debug_visual)
            , debug_visual_onlybest_(debug_visual_onlybest)
            , PirlModel_(PirlModel)
            , debug_(debug)
            , lr_schedule_mode_(lr_schedule_mode)
            , return_threshold_(return_threshold)
            , init_trajectory_(init_trajectory)
    {
        best_cost_log_.reserve(100);
        best_cost_pose_log_.reserve(100);
        best_cost_rot_log_.reserve(100);
        best_cost_vel_log_.reserve(100);
        best_cost_acc_log_.reserve(100);
        best_cost_jerk_log_.reserve(100);
        time_log_.reserve(100);

        first_best_cost_log_.reserve(100);
        first_best_cost_pose_log_.reserve(100);
        first_best_cost_rot_log_.reserve(100);
        first_best_cost_vel_log_.reserve(100);
        first_best_cost_acc_log_.reserve(100);
        first_best_cost_jerk_log_.reserve(100);
        first_time_log_.reserve(100);

        learning_rate_ = parameters_->learning_rate_;
        learning_rate_max_ = parameters_->learning_rate_ * 1000;
        learning_rate_min_ = parameters_->learning_rate_ / 100;
        alpha_ = 0.5;

        if (lr_schedule_mode_ == 0){
            parameters_->joint_update_limit_ = 0.1;
        }

        collision_point_ = -1;

        double size_x = 2.5, size_y = 2.5, size_z = 5.0;
        double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
        double resolution = 0.01;
        double max_propogation_distance = 1.0;

        ros::WallTime wt = ros::WallTime::now();
        const collision_detection::WorldPtr& w = (const collision_detection::WorldPtr &) planning_scene->getWorld();
        hy_world_ = new collision_detection::CollisionWorldHybrid(w, Eigen::Vector3d(size_x, size_y, size_z),
                                                                  Eigen::Vector3d(origin_x, origin_y, origin_z),
                                                                  true,
                                                                  resolution, 0.0, max_propogation_distance);
        if (!hy_world_)
        {
            ROS_WARN_STREAM("Could not initialize hybrid collision world from planning scene");
            return;
        }else{
            if(!debug_log_)
                ROS_INFO_STREAM("Generate world distance field: " << (ros::WallTime::now() - wt));
        }

        wt = ros::WallTime::now();
        std::map<std::string, std::vector<collision_detection::CollisionSphere>> link_body_decompositions;
        hy_robot_ = new collision_detection::CollisionRobotHybrid(kmodel_, link_body_decompositions,
                                                                  size_x, size_y, size_z,
                                                                  true,
                                                                  resolution, 0.0, max_propogation_distance);
        if (!hy_robot_)
        {
            ROS_WARN_STREAM("Could not initialize hybrid collision robot from planning scene");
            return;
        }else{
            if(!debug_log_)
                ROS_INFO_STREAM("Generate robot distance field: " << (ros::WallTime::now() - wt));
        }

        for(uint i = 0; i < bounds.size(); i++){
            vel_limit_.push_back(bounds[i][0][0].max_velocity_);
        }

        initialize();

        const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();
        for (int joint_i = 0; joint_i < joint_models.size(); joint_i++) {
            const moveit::core::JointModel *joint_model = joint_models[joint_i];

            if (joint_model->getType() == moveit::core::JointModel::REVOLUTE) {
                const moveit::core::RevoluteJointModel *revolute_joint =
                        dynamic_cast<const moveit::core::RevoluteJointModel *>(joint_model);
                if (revolute_joint->isContinuous()) {
                    joint_kinds_.push_back(true);
                    c_joints_.push_back(joint_i);
                }
                else{
                    joint_kinds_.push_back(false);
                }
            }
        }

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator_.seed(seed);

    }

    void TormOptimizer::set_interpolation_mode(int mode_idx){
        interpolation_mode_ = mode_idx;
    }

    void TormOptimizer::initialize()
    {
        // init some variables:
        num_vars_free_ = group_trajectory_.getNumFreePoints();
        num_vars_all_ = group_trajectory_.getNumPoints();
        num_joints_ = group_trajectory_.getNumJoints();

        free_vars_start_ = group_trajectory_.getStartIndex();
        free_vars_end_ = group_trajectory_.getEndIndex();

        delta_twist_.reserve(num_vars_free_);

        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;
        req.group_name = planning_group_;
        ros::WallTime wt = ros::WallTime::now();
        collision_detection::AllowedCollisionMatrix acm_ = planning_scene_->getAllowedCollisionMatrix();

        hy_world_->getCollisionGradients(req, res, *hy_robot_->getCollisionRobotDistanceField().get(), state_,
                                         &acm_, gsr_);

        if(!debug_log_)
            ROS_INFO_STREAM("First coll check took " << (ros::WallTime::now() - wt));
        num_collision_points_ = 0;
        start_collision_ = 3;
        end_collision_ = gsr_->gradients_.size(); // including (gripper_link) and (r_gripper, l_gripper)
        for (size_t i = start_collision_; i < end_collision_; i++)
        {
            num_collision_points_ += gsr_->gradients_[i].gradients.size();
        }

        // set up the joint costs:
        joint_costs_.reserve(num_joints_);

        double max_cost_scale = 1.0;

        joint_model_group_ = planning_scene_->getRobotModel()->getJointModelGroup(planning_group_);

        const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();
        for (int i = 0; i < joint_models.size(); i++)
        {
            const moveit::core::JointModel* model = joint_models[i];
            double joint_cost = 1.0;
            std::string joint_name = model->getName();
            // nh.param("joint_costs/" + joint_name, joint_cost, 1.0);
            std::vector<double> derivative_costs(3);
            derivative_costs[0] = joint_cost * parameters_->smoothness_cost_velocity_;
            derivative_costs[1] = joint_cost * parameters_->smoothness_cost_acceleration_;
            derivative_costs[2] = joint_cost * parameters_->smoothness_cost_jerk_;
            joint_costs_.emplace_back(group_trajectory_, i, derivative_costs, parameters_->ridge_factor_);
            double cost_scale = joint_costs_[i].getMaxQuadCostInvValue();
            if (max_cost_scale < cost_scale)
                max_cost_scale = cost_scale;
        }

        // scale the smoothness costs
        for (int i = 0; i < num_joints_; i++)
        {
            joint_costs_[i].scale(max_cost_scale); // 96.77624
        }

        // allocate memory for matrices:
        smoothness_increments_ = Eigen::MatrixXd::Zero(num_vars_free_, num_joints_);
        collision_increments_ = Eigen::MatrixXd::Zero(num_vars_free_, num_joints_);
        endPose_increments_ = Eigen::MatrixXd::Zero(num_vars_free_, num_joints_);
        final_increments_ = Eigen::MatrixXd::Zero(num_vars_free_, num_joints_);
        smoothness_derivative_ = Eigen::VectorXd::Zero(num_vars_all_);
        jacobian_ = Eigen::MatrixXd::Zero(3, num_joints_);
        jacobian_pseudo_inverse_ = Eigen::MatrixXd::Zero(num_joints_, 3);
        jacobian_jacobian_tranpose_ = Eigen::MatrixXd::Zero(3, 3);

        group_trajectory_backup_ = group_trajectory_.getTrajectory();
        local_group_trajectory_ = group_trajectory_.getTrajectory();
        best_trajectory_ = group_trajectory_.getTrajectory();

        collision_point_joint_names_.resize(num_vars_all_, std::vector<std::string>(num_collision_points_));
        collision_point_pos_eigen_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_collision_points_));
        collision_point_vel_eigen_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_collision_points_));
        collision_point_acc_eigen_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_collision_points_));
        joint_axes_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_joints_));
        joint_positions_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_joints_));

        collision_point_potential_.resize(num_vars_all_, std::vector<double>(num_collision_points_));
        collision_point_vel_mag_.resize(num_vars_all_, std::vector<double>(num_collision_points_));
        collision_point_potential_gradient_.resize(num_vars_all_, EigenSTL::vector_Vector3d(num_collision_points_));

        std::map<std::string, std::string> fixed_link_resolution_map;
        for (int i = 0; i < num_joints_; i++)
        {
            joint_names_.push_back(joint_model_group_->getActiveJointModels()[i]->getName());
            registerParents(joint_model_group_->getActiveJointModels()[i]);
            fixed_link_resolution_map[joint_names_[i]] = joint_names_[i];
        }

        for (const moveit::core::JointModel* jm : joint_model_group_->getFixedJointModels())
        {
            if (!jm->getParentLinkModel())  // root joint doesn't have a parent
                continue;

            fixed_link_resolution_map[jm->getName()] = jm->getParentLinkModel()->getParentJointModel()->getName();
        }

        for (size_t i = 0; i < joint_model_group_->getUpdatedLinkModels().size(); i++)
        {
            if (fixed_link_resolution_map.find(
                    joint_model_group_->getUpdatedLinkModels()[i]->getParentJointModel()->getName()) ==
                fixed_link_resolution_map.end())
            {
                const moveit::core::JointModel* parent_model = nullptr;
                bool found_root = false;

                while (!found_root)
                {
                    if (parent_model == nullptr)
                    {
                        parent_model = joint_model_group_->getUpdatedLinkModels()[i]->getParentJointModel();
                    }
                    else
                    {
                        parent_model = parent_model->getParentLinkModel()->getParentJointModel();
                        for (size_t j = 0; j < joint_names_.size(); j++)
                        {
                            if (parent_model->getName() == joint_names_[j])
                            {
                                found_root = true;
                            }
                        }
                    }
                }
                fixed_link_resolution_map[joint_model_group_->getUpdatedLinkModels()[i]->getParentJointModel()->getName()] =
                        parent_model->getName();
            }
        }

        int start = free_vars_start_;
        int end = free_vars_end_;
        for (int i = start; i <= end; ++i)
        {
            size_t j = 0;
            for (size_t g = start_collision_; g < end_collision_; g++)
            {
                collision_detection::GradientInfo& info = gsr_->gradients_[g];

                for (size_t k = 0; k < info.sphere_locations.size(); k++)
                {
                    if (fixed_link_resolution_map.find(info.joint_name) != fixed_link_resolution_map.end())
                    {
                        collision_point_joint_names_[i][j] = fixed_link_resolution_map[info.joint_name];
                    }
                    else
                    {
                        ROS_ERROR("Couldn't find joint %s!", info.joint_name.c_str());
                    }
                    j++;
                }
            }
        }
    }

    TormOptimizer::~TormOptimizer()
    {
        destroy();
    }

    void TormOptimizer::registerParents(const moveit::core::JointModel* model)
    {
        const moveit::core::JointModel* parent_model = nullptr;
        bool found_root = false;

        if (model == kmodel_->getRootJoint())
            return;

        while (!found_root)
        {
            if (parent_model == nullptr)
            {
                if (model->getParentLinkModel() == nullptr)
                {
                    ROS_ERROR_STREAM("Model " << model->getName() << " not root but has NULL link model parent");
                    return;
                }
                else if (model->getParentLinkModel()->getParentJointModel() == nullptr)
                {
                    ROS_ERROR_STREAM("Model " << model->getName() << " not root but has NULL joint model parent");
                    return;
                }
                parent_model = model->getParentLinkModel()->getParentJointModel();
            }
            else
            {
                if (parent_model == kmodel_->getRootJoint())
                {
                    found_root = true;
                }
                else
                {
                    parent_model = parent_model->getParentLinkModel()->getParentJointModel();
                }
            }
            joint_parent_map_[model->getName()][parent_model->getName()] = true;
        }
    }

    // =========================== localOptimizeTSGD ===========================
    bool TormOptimizer::localOptimizeTSGD(int maxiter, int eph){
        bool optimization_result = false;
        double cur_Cost = INFINITY, prev_Cost = INFINITY, prev_vel = INFINITY;
        bool infea_col = false, infea_vel = false;

        if(!debug_visual_onlybest_ && debug_visual_) {
            debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1, 3);
            std::cout << "[initial traj]" << std::endl;
        }
        if(debug_visual_onlybest_ && debug_visual_ && eph==0) {
            debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1, 3);
            std::cout << "[initial traj]" << std::endl;
        }

        bool feasible_stage = true; // 0: joint smoothness + obs avoidance | 1: end pose

        for (int iter = 0; iter < maxiter; iter++) {
            for (int stage = 0; stage < 2; stage++) {

                if (stage == 0){
                    if (lr_schedule_mode_ == 1 and learning_rate_ < 1e-2 and !infea_col and !infea_vel){
                        feasible_stage = false;
                    }else{
                        feasible_stage = true;
                    }
                }else{
                    if (lr_schedule_mode_ == 1 and learning_rate_ == learning_rate_max_ and infea_col){
                        feasible_stage = true;
                    }else{
                        feasible_stage = false;
                    }
                }

                if (debug_visual_) {
                    debug_->clearContactVectors();
                }
                if(feasible_stage){
                    performForwardKinematics(); // calc collision sphere grad, vel, acc, etc... for collision avoidance.
                    getEndPoseCost(false);
                    ros::WallTime st = ros::WallTime::now();
                    traj_evaluator evaluator(endPoses_desired_, full_trajectory_->getTrajectory(),
                                             c_joints_, parameters_->time_duration_, iksolver_, vel_limit_);
                    cur_Cost = evaluator.getWeightedPoseCost();
                    spent_eval_time_ += (ros::WallTime::now() - st).toSec();
                    if (debug_visual_) {
                        debug_->publishContactVectors();
                    }
                }
                else{
                    getEndPoseCost(true); // calc delta q for fitting.
                    ros::WallTime st = ros::WallTime::now();
                    traj_evaluator evaluator(endPoses_desired_, full_trajectory_->getTrajectory(),
                                             c_joints_, parameters_->time_duration_, iksolver_, vel_limit_);
                    cur_Cost = evaluator.getWeightedPoseCost();
                    spent_eval_time_ += (ros::WallTime::now() - st).toSec();
                }

                local_group_trajectory_ = group_trajectory_.getTrajectory(); // local_group_trajectory_ is for checking col, vel, and singularity
                if(isCurrentTrajectoryCollisionFree()){
                    infea_col = false;
                }else{
                    infea_col = true;
                }
                if(checkJointVelocityLimit(parameters_->time_duration_)){
                    infea_vel = false;
                }else{
                    infea_vel = true;
                }
                ROS_WARN_STREAM("[" << iter << "|" << stage << "] Feasible col: " << !infea_col << ", Feasible vel: " << !infea_vel);

                if (lr_schedule_mode_ == 1 || lr_schedule_mode_ == 2){
                    if (!infea_col && !infea_vel){
                        alpha_ = 0.5;
                    }else if (infea_col && !infea_vel){
                        if (alpha_ < 0.5){
                            alpha_ = 0.5;
                        }else{
                            double buf_alpha = alpha_ + 0.1;
                            alpha_ = std::min(buf_alpha, 1.0);
                        }
                    }else if (!infea_col && infea_vel){
                        if (alpha_ > 0.5){
                            alpha_ = 0.5;
                        }else{
                            double buf_alpha = alpha_ - 0.1;
                            alpha_ = std::max(buf_alpha, 0.0);
                        }
                    }else{
                        alpha_ = 0.5;
                    }
                }else if (lr_schedule_mode_ == 3){
                    alpha_ = 0.5;
                }

                if (lr_schedule_mode_ == 1) {
                    if (!infea_col && !infea_vel) {
                        if (learning_rate_ > parameters_->learning_rate_) {
                            learning_rate_ = parameters_->learning_rate_;
                        } else {
                            learning_rate_ = std::max(learning_rate_ / 2.0,
                                                      learning_rate_min_);
                        }
                    } else if (infea_col && !infea_vel) {
                        learning_rate_ = std::min(learning_rate_ * 2.0,
                                                  learning_rate_max_);
                    } else if (!infea_col && infea_vel) {
                        learning_rate_ = std::min(learning_rate_ * 2.0,
                                                  learning_rate_max_);
                    } else {
                        learning_rate_ = std::min(learning_rate_ * 2.0,
                                                  learning_rate_max_);
                    }
                }else if (lr_schedule_mode_ == 2){
                    int period = maxiter / 2;
                    double p = (double)(iter % period) / period;
                    learning_rate_ = learning_rate_min_ + 0.5 * (learning_rate_max_ - learning_rate_min_) * (1 + std::cos(p * M_PI));
                }else if (lr_schedule_mode_ == 3){
                    learning_rate_ = parameters_->learning_rate_;
                }

                if( debug_verbose_ ) {
                    ROS_WARN_STREAM("[" << iter << "/" << stage << "] infea_Col: " << infea_col << ", infea_Vel: " << infea_vel);
                    ROS_WARN_STREAM("learning_rate_: " << learning_rate_ << "/ alpha_: " << alpha_);
                }

                if(best_trajectory_backup_cost_ > cur_Cost){
                    if(!infea_col){
                        if(!infea_vel){
                            if(checkSingularity()){
                                best_trajectory_backup_cost_ = cur_Cost;
                                best_trajectory_ = group_trajectory_.getTrajectory(); // backup for last return
//                                if(debug_verbose_)
                                double time_log_stamp = (ros::WallTime::now() - start_time_).toSec() - spent_eval_time_;
                                std::cout << cur_Cost << " " << time_log_stamp << std::endl;
                                best_cost_log_.push_back(cur_Cost);
                                time_log_.push_back(time_log_stamp);
                                if (eph == 0){
                                    first_best_cost_log_.push_back(cur_Cost);
                                    first_time_log_.push_back(time_log_stamp);
                                }
                                if(debug_log_) {
                                    updateFullTrajectory();
                                    traj_evaluator evaluator(endPoses_desired_, full_trajectory_->getTrajectory(),
                                                             c_joints_, parameters_->time_duration_, iksolver_, vel_limit_);
                                    double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
                                    evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
                                    best_cost_pose_log_.push_back(cost_pose);
                                    best_cost_rot_log_.push_back(cost_rot);
                                    best_cost_vel_log_.push_back(cost_vel);
                                    best_cost_acc_log_.push_back(cost_acc);
                                    best_cost_jerk_log_.push_back(cost_jerk);
                                    if (eph == 0){
                                        first_best_cost_pose_log_.push_back(cost_pose);
                                        first_best_cost_rot_log_.push_back(cost_rot);
                                        first_best_cost_vel_log_.push_back(cost_vel);
                                        first_best_cost_acc_log_.push_back(cost_acc);
                                        first_best_cost_jerk_log_.push_back(cost_jerk);
                                    }
                                }
                                optimization_result = true;
                                if(debug_visual_onlybest_ && debug_visual_) {
                                    if (!feasible_stage){
                                        debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1, 2);
                                    }else{
                                        debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1, 1);
                                    }
                                }
                            }else{
                                if(debug_verbose_)
                                    ROS_WARN_STREAM("Got new best cost (" << cur_Cost << "), but fail...[singularity] violation!");
                            }
                        }else{
                            if(debug_verbose_)
                                ROS_WARN_STREAM("Got new best cost (" << cur_Cost << "), but fail...[joint velocity] violation!");
                        }
                    }else{
                        if(debug_verbose_)
                            ROS_WARN_STREAM("Got new best cost (" << cur_Cost << "), but fail...[collision] violation!");
                    }
                }

                if(feasible_stage){ // collision & smoothness
                    calculateSmoothnessIncrements();
                    calculateCollisionIncrements();
                    calculateFeasibleIncrements();
                }
                else{ // Ik problem
                    calculateCollisionIncrements();
                    calculateEndPoseIncrements();
                }

                addIncrementsToTrajectory();
                updateGoalConfiguration();
                handleJointLimits();

                updateFullTrajectory();

//                if(debug_verbose_) {
//                    std::cout << "[" << iter << " - " << stage << "]" << std::endl;
//                    traj_evaluator evaluator(endPoses_desired_, full_trajectory_->getTrajectory(),
//                                             c_joints_, parameters_->time_duration_, iksolver_);
//                    double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
//                    evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk);
//                    std::cout << cur_Cost << ", " << cost_vel << std::endl;
//                    std::cout << cur_Cost - prev_Cost << ", " << cost_vel - prev_vel << std::endl;
//                    prev_vel = cost_vel;
//                }
//                prev_Cost = cur_Cost;

                if(!debug_visual_onlybest_ && debug_visual_) {
                    if (feasible_stage) {
                        debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1, 1);
                    } else {
                        debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1, 2);
                    }
                }
            }
        }

        return optimization_result;
    }

    // =========================== iterativeExploration ===========================
    bool TormOptimizer::iterativeExploration(bool only_first){
        std::cout << "start optimization" << std::endl;
        ros::WallTime wt;
        iteration_ = 0;
        best_trajectory_backup_cost_ = DBL_MAX;

        start_time_ = ros::WallTime::now();
        iteration_++;

        int eph = 0;

        wt = ros::WallTime::now();
        // New initial traj.
        if(interpolation_mode_ == 0){
            getJointInterpolatedTrajectory();
        }else if(interpolation_mode_ == 1){
            getNewTrajectory();
        }else if(interpolation_mode_ == 2){
            setPIRLTrajectory(true);
        }
        std::cout << "----------------------------------------- first interpolation time: "
                  << (ros::WallTime::now() - wt) << " (# waypoints: " << num_vars_free_ << ")"
                  << std::endl;
        updateInitTrajectory();

        while(true){
            if(debug_verbose_) {
                wt = ros::WallTime::now();
            }
            // TSGD
            localOptimizeTSGD(parameters_->exploration_iter_, eph);
            if ((ros::WallTime::now() - start_time_).toSec() > parameters_->planning_time_limit_){ // break (time out)
                break;
            }
            if(debug_verbose_) {
                std::cout << "----------------------------------------- [1] TSGD         [time]: "
                          << (ros::WallTime::now() - wt) << std::endl;
                std::cout << "[iter]:" << eph << ", [current best cost]: " << best_trajectory_backup_cost_ << std::endl;
            }

            if (return_threshold_ != -1 and best_trajectory_backup_cost_ < return_threshold_){
                std::cout << "*** STOP early with best cost, " << best_trajectory_backup_cost_ << std::endl;
                break;
            }

            // New initial traj.
            if (!only_first) {
                if(debug_verbose_) {
                    wt = ros::WallTime::now();
                }
                if(interpolation_mode_ == 0){
                    getJointInterpolatedTrajectory();
                }else if(interpolation_mode_ == 1){
                    getNewTrajectory();
                }else if(interpolation_mode_ == 2){
                    setPIRLTrajectory(false);
                }
                if(debug_verbose_) {
                    std::cout << "----------------------------------------- [2] Gen New traj [time]: "
                              << (ros::WallTime::now() - wt) << std::endl;
                }
            }else{
                break;
            }
            if ((ros::WallTime::now() - start_time_).toSec() > parameters_->planning_time_limit_){
                break;
            }
            eph++;
        }

        group_trajectory_.getTrajectory() = best_trajectory_;
        updateFullTrajectory();

        if(best_trajectory_backup_cost_ != DBL_MAX){
            std::cout << ">> success! best_cost: " << best_cost_log_[best_cost_log_.size()-1] << std::endl;
            if(debug_) {
                debug_->show(endPoses_desired_, full_trajectory_->getTrajectory(), 0.1);
            }
            return true;
        }
        else{
            std::cout << ">> fail...." << std::endl;
            return false;
        }
    }

    void TormOptimizer::updateLocalGroupTrajectory(){
        local_group_trajectory_ = group_trajectory_.getTrajectory();
    }

    // Collision check /////////////////////////////////////////////////////////////////////////////////////////////
    bool TormOptimizer::isCurrentTrajectoryCollisionFree() {
        if(!parameters_->use_collision_check_)
            return true;

        moveit_msgs::RobotTrajectory traj;
        traj.joint_trajectory.joint_names = joint_names_;

        std::vector<double> conf(num_joints_);
        for (int i = free_vars_start_-1; i <= free_vars_end_; i++)
        {
            for (int j = 0; j < group_trajectory_.getNumJoints(); j++) {
                conf[j] = local_group_trajectory_(i, j);
            }
            if(!iksolver_.collisionChecking(conf)){
                collision_point_ = i;
                if (debug_verbose_)
                    ROS_ERROR_STREAM("[COLLISION] at " << i);
                return false;
            }
        }
        collision_point_ = -1;
        return true;
    }

    // Velocity limit check
    bool TormOptimizer::checkJointVelocityLimit(double dt){
        if(!parameters_->use_velocity_check_)
            return true;

        // calculate velocity
        for (uint i = free_vars_start_-1; i < free_vars_end_; i++){
            Eigen::VectorXd q_c = local_group_trajectory_.row(i);
            Eigen::VectorXd q_t = local_group_trajectory_.row(i+1);

            for (uint j = 0; j < num_joints_; j++){
                double diff;
                if(joint_kinds_[j]){
                    diff = shortestAngularDistance(q_c(j), q_t(j));
                }else{
                    diff = q_t(j) - q_c(j);
                }
                double required_vel = std::abs(diff)/dt;
                if(required_vel > vel_limit_[j]){
                    if (debug_verbose_)
                        ROS_ERROR_STREAM("[VELOCITY] Node_idx: " << i << ", J_idx:" << j << ", [" << diff << "], Required_vel: " << required_vel << ", Vel_limit_[j]: " << vel_limit_[j]);
                    return false;
                }
            }
        }
        return true;
    }

    // Singularity check
    bool TormOptimizer::checkSingularity(){
        if(!parameters_->use_singularity_check_)
            return true;

        KDL::JntArray q(num_joints_);
        KDL::Jacobian jac(num_joints_);

        for (int i = free_vars_start_; i <= free_vars_end_; i++) {
            q.data = local_group_trajectory_.row(i);
            iksolver_.getJacobian(q, jac);
            double yosh = std::abs(std::sqrt((jac.data * jac.data.transpose()).determinant()));
            if(yosh < parameters_->singularity_lower_bound_){
                return false;
            }
        }

        return true;
    }// check //////////////////////////////////////////////////////////////////////////////////////////////////

    void TormOptimizer::calculatePseudoInverse()
    {
        jacobian_jacobian_tranpose_ =
                jacobian_ * jacobian_.transpose() + Eigen::MatrixXd::Identity(3, 3) * parameters_->pseudo_inverse_ridge_factor_;
        jacobian_pseudo_inverse_ = jacobian_.transpose() * jacobian_jacobian_tranpose_.inverse();
    }

    void TormOptimizer::updateFullTrajectory()
    {
        full_trajectory_->updateFromGroupTrajectory(group_trajectory_);
    }

    void TormOptimizer::updateInitTrajectory()
    {
        init_trajectory_->updateFromGroupTrajectory(group_trajectory_);
    }

    void TormOptimizer::fillInLinearInterpolation(int start, int goal)
    {
        Eigen::VectorXd st = group_trajectory_.getTrajectoryPoint(start);
        Eigen::VectorXd gt = group_trajectory_.getTrajectoryPoint(goal);
        double num = goal-start;
        for (int i = 0; i < num_joints_; i++) {
            double diff = (gt(i) - st(i));
            if(joint_kinds_[i]) {
                if (std::abs(diff) > M_PI) {
                    if (diff < 0) {
                        diff = 2 * M_PI - std::abs(diff);
                    } else {
                        diff = -2 * M_PI + std::abs(diff);
                    }
                }
            }
            double theta = diff / num;

            int t = 1;
            for (int j = start+1; j <= goal; j++) {
                group_trajectory_(j, i) = st(i) + (t++) * theta;
            }
        }
    }

    void TormOptimizer::calcInitialTrajQuality(std::vector<double>& res) {
        int start_idx = group_trajectory_.getStartIndex();
        int end_idx = group_trajectory_.getEndIndex();
        int extra = group_trajectory_.getExtra();
        int start_point = start_idx-1;

        updateLocalGroupTrajectory();

        performForwardKinematics(start_idx, end_idx); // query for collision info
        double cCost = getCollisionCost(start_idx, end_idx);

        updateFullTrajectory();
        traj_evaluator evaluator(endPoses_desired_, full_trajectory_->getTrajectory(),
                                 c_joints_, parameters_->time_duration_, iksolver_, vel_limit_);
        double cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk;
        int n_vel_violated = evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk, true);

        res.push_back(cost_pose + parameters_->rotation_scale_factor_ * cost_rot);
        res.push_back(cost_pose);
        res.push_back(cost_rot);

        bool res_col = isCurrentTrajectoryCollisionFree();
        res.push_back(res_col);
        res.push_back(cCost);


        {
            res.push_back(cost_vel);

            bool res_vel = checkJointVelocityLimit(parameters_->time_duration_);
            res.push_back(res_vel);
            res.push_back(n_vel_violated);

            res.push_back(res_vel and res_col);
        }

        {
            evaluator.setNewDt(parameters_->time_duration_ - 0.005);
            n_vel_violated = evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk, true);

            res.push_back(cost_vel);

            bool res_vel = checkJointVelocityLimit(parameters_->time_duration_ - 0.005);
            res.push_back(res_vel);
            res.push_back(n_vel_violated);

            res.push_back(res_vel and res_col);
        }

        {
            evaluator.setNewDt(parameters_->time_duration_ - 0.01);
            n_vel_violated = evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk, true);

            res.push_back(cost_vel);

            bool res_vel = checkJointVelocityLimit(parameters_->time_duration_ - 0.01);
            res.push_back(res_vel);
            res.push_back(n_vel_violated);

            res.push_back(res_vel and res_col);
        }

        {
            evaluator.setNewDt(parameters_->time_duration_ - 0.015);
            n_vel_violated = evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk, true);

            res.push_back(cost_vel);

            bool res_vel = checkJointVelocityLimit(parameters_->time_duration_ - 0.015);
            res.push_back(res_vel);
            res.push_back(n_vel_violated);

            res.push_back(res_vel and res_col);
        }
        {
            evaluator.setNewDt(parameters_->time_duration_ - 0.02);
            n_vel_violated = evaluator.getCost(cost_pose, cost_rot, cost_vel, cost_acc, cost_jerk, true);

            res.push_back(cost_vel);

            bool res_vel = checkJointVelocityLimit(parameters_->time_duration_ - 0.02);
            res.push_back(res_vel);
            res.push_back(n_vel_violated);

            res.push_back(res_vel and res_col);
        }
    }

    // =========================== setPIRLTrajectory ===========================
    void TormOptimizer::setPIRLTrajectory(bool deterministic) {

        uint num_simplified_points = simplified_points_.size();
        int start_idx = group_trajectory_.getStartIndex(); // start_idx doesn't include init_configuration
        int end_idx = group_trajectory_.getEndIndex();

        int start_point = start_idx-1;
        int extra = group_trajectory_.getExtra();

        // reset start configuartion
        if(!fix_start_config_){
            KDL::JntArray q_rnd_start_conf(num_joints_);
            if(iksolver_.ikSolverCollFree(endPoses_desired_[0], q_rnd_start_conf, 9999)){
            }
            group_trajectory_.getTrajectoryPoint(start_point) = q_rnd_start_conf.data;
            PirlModel_->interpolate(q_rnd_start_conf, deterministic);
        }else{
            KDL::JntArray q_fix_start_config(num_joints_);
            q_fix_start_config.data = group_trajectory_.getTrajectoryPoint(start_point);
            PirlModel_->interpolate(q_fix_start_config, deterministic);
        }

        for (uint i = 0; i < num_simplified_points; i++){
            // update goal configuariton
            group_trajectory_.getTrajectoryPoint(simplified_points_[i]+extra) = PirlModel_->getTrajectoryPoint(i);
            // calculate initial trajectory
            fillInLinearInterpolation(start_point, simplified_points_[i]+extra);

            start_point = simplified_points_[i]+extra;
        }
        for (int j = 0; j < start_idx-1; j++){
            group_trajectory_.getTrajectoryPoint(j) = group_trajectory_.getTrajectoryPoint(start_idx-1);
        }
        for (int j = end_idx+1; j < num_vars_all_; j++){
            group_trajectory_.getTrajectoryPoint(j) = group_trajectory_.getTrajectoryPoint(end_idx);
        }
        handleJointLimits();
        updateFullTrajectory();
    }

    // =========================== getNewTrajectory ===========================
    void TormOptimizer::getNewTrajectory() {
        KDL::JntArray q_t(num_joints_);

        uint num_simplified_points = simplified_points_.size();
        int start_idx = group_trajectory_.getStartIndex();
        int end_idx = group_trajectory_.getEndIndex();
        int extra = group_trajectory_.getExtra();

        int start_point = start_idx-1;
        Eigen::MatrixXd xi = group_trajectory_.getTrajectory();

        // reset start configuartion
        if(!fix_start_config_){
            KDL::JntArray q_rnd_start_conf(num_joints_);
            if(iksolver_.ikSolverCollFree(endPoses_desired_[0], q_rnd_start_conf, 9999)){
            }
            group_trajectory_.getTrajectoryPoint(start_point) = q_rnd_start_conf.data;
        }

        for (uint i = 0; i < num_simplified_points; i++){
            double bestCost = INFINITY;

            // [1] collect ik candidates in random_confs
            std::vector<KDL::JntArray> random_confs;
            for (uint j = 0; j < parameters_->traj_generation_iter_; j++) {
                if(iksolver_.ikSolverCollFree(endPoses_desired_[simplified_points_[i]], q_t)){
                    random_confs.push_back(q_t);
                }
                else if(iksolver_.ikSolver(endPoses_desired_[simplified_points_[i]], q_t)){
                    random_confs.push_back(q_t);
                }
            }

            // [2] select best confiuration among random_confs
            for (uint j = 0; j < random_confs.size(); j++){
                // 2-1 update goal configuariton
                group_trajectory_.getTrajectoryPoint(simplified_points_[i]+extra) = random_confs[j].data;
                // 2-2 calculate initial trajectory
                fillInLinearInterpolation(start_point, simplified_points_[i]+extra);
                // calculate cost
                double eCost = getEndPoseCost(start_point, simplified_points_[i]+extra); // errs with target ee poses
                if(bestCost < eCost){
                    continue;
                } else {
                    performForwardKinematics(start_point, simplified_points_[i]+extra); // query for collision info
                    double cCost = getCollisionCost(start_point, simplified_points_[i]+extra); // collision cost: magnitude of col gradient * col_weight
                    if (debug_verbose_)
                        std::cout << "[getNewTraj of TORM] cost: " << eCost << ", " << cCost << std::endl;
                    double cost =  parameters_->pose_cost_weight_ * eCost + parameters_->collision_cost_weight_ * cCost;
                    if(j == 0){
                        bestCost = cost;
                        for (int k=start_point+1; k <= simplified_points_[i]+extra; k++){
                            xi.row(k) = group_trajectory_.getTrajectoryPoint(k);
                        }
                    }
                    else{
                        if(bestCost > cost){
                            bestCost = cost;
                            for (int k = start_point+1; k <= simplified_points_[i]+extra; k++){
                                xi.row(k) = group_trajectory_.getTrajectoryPoint(k);
                            }
                        }
                    }
                }
            }
            // update group_trajectory to best xi ( one segment )
            for (int k = start_point+1; k <= simplified_points_[i]+extra; k++){
                group_trajectory_.getTrajectoryPoint(k) = xi.row(k);
            }
            start_point = simplified_points_[i]+extra;
        }
        for (int j = 0; j < start_idx-1; j++){
            group_trajectory_.getTrajectoryPoint(j) = xi.row(start_idx-1);
        }
        for (int j = end_idx+1; j < num_vars_all_; j++){
            group_trajectory_.getTrajectoryPoint(j) = xi.row(end_idx);
        }
        handleJointLimits();
        updateFullTrajectory();
    }

    // =========================== getJointInterpolatedTrajectory ===========================
    void TormOptimizer::getJointInterpolatedTrajectory(bool use_shortest_goal_conf) {
        KDL::JntArray q_t(num_joints_);

        uint num_simplified_points = simplified_points_.size();
        int start_idx = group_trajectory_.getStartIndex();
        int end_idx = group_trajectory_.getEndIndex();
        int extra = group_trajectory_.getExtra();

        int start_point = start_idx-1;
        Eigen::MatrixXd xi = group_trajectory_.getTrajectory();

        // reset start configuartion
        if(!fix_start_config_){
            KDL::JntArray q_rnd_start_conf(num_joints_);
            if(iksolver_.ikSolverCollFree(endPoses_desired_[0], q_rnd_start_conf, 9999)){
            }
            group_trajectory_.getTrajectoryPoint(start_point) = q_rnd_start_conf.data;
        }


        // [1] collect ik candidates at the last end-effector pose
        std::vector<KDL::JntArray> random_confs;
        if (not use_shortest_goal_conf){
            for (uint j = 0; j < 30 ; j++) {
                if(iksolver_.ikSolverCollFree(endPoses_desired_[simplified_points_[simplified_points_.size()-1]], q_t)){
                    random_confs.push_back(q_t);
                }
                else if(iksolver_.ikSolver(endPoses_desired_[simplified_points_[simplified_points_.size()-1]], q_t)){
                    random_confs.push_back(q_t);
                }
            }
        }else{
            KDL::JntArray q_init(num_joints_);
            q_init.data = group_trajectory_.getTrajectoryPoint(start_point);
            if(iksolver_.ikSolverCollFree(q_init, endPoses_desired_[simplified_points_[simplified_points_.size()-1]], q_t)){
                random_confs.push_back(q_t);
            }
            else if(iksolver_.ikSolver(q_init, endPoses_desired_[simplified_points_[simplified_points_.size()-1]], q_t)){
                random_confs.push_back(q_t);
            }
        }


        double bestCost = INFINITY;
        // [2] select best interpolation among linear interpolations between start and candidiate configs.
        for (uint j = 0; j < random_confs.size(); j++){
            // 2-1 update goal configuariton
            group_trajectory_.getTrajectoryPoint(end_idx) = random_confs[j].data;
            // 2-2 calculate interpolated trajectory
            fillInLinearInterpolation(start_point, end_idx);
            // calculate cost
            double eCost = getEndPoseCost(start_point, end_idx); // errs with target ee poses
            if(bestCost < eCost){
                continue;
            } else {
                performForwardKinematics(start_point, end_idx); // query for collision info
                double cCost = getCollisionCost(start_point, end_idx); // collision cost: magnitude of col gradient * col_weight
                if (debug_verbose_)
                    std::cout << "[getNewTraj of TORM] j:" << j << ", cost: " << eCost << ", " << cCost << std::endl;
                double cost =  parameters_->pose_cost_weight_ * eCost + parameters_->collision_cost_weight_ * cCost;
                if(j == 0){
                    bestCost = cost;
                    for (int k=start_point+1; k <= end_idx; k++){
                        xi.row(k) = group_trajectory_.getTrajectoryPoint(k);
                    }
                }
                else{
                    if(bestCost > cost){
                        bestCost = cost;
                        for (int k = start_point+1; k <= end_idx; k++){
                            xi.row(k) = group_trajectory_.getTrajectoryPoint(k);
                        }
                    }
                }
            }
        }
        // update group_trajectory to best xi ( one segment )
        for (int k = start_point+1; k <= end_idx; k++){
            group_trajectory_.getTrajectoryPoint(k) = xi.row(k);
        }
        for (int j = 0; j < start_idx-1; j++){
            group_trajectory_.getTrajectoryPoint(j) = xi.row(start_idx-1);
        }
        for (int j = end_idx+1; j < num_vars_all_; j++){
            group_trajectory_.getTrajectoryPoint(j) = xi.row(end_idx);
        }
        handleJointLimits();
        updateFullTrajectory();
    }

    void TormOptimizer::updateStartConfiguration(){
        Eigen::MatrixXd::RowXpr xx = group_trajectory_.getTrajectoryPoint(free_vars_start_);
        for(uint i = 0; i < free_vars_start_; i++){
            group_trajectory_.getTrajectoryPoint(i) = xx;
        }
    }

    void TormOptimizer::updateGoalConfiguration(){
        Eigen::MatrixXd::RowXpr xx = group_trajectory_.getTrajectoryPoint(free_vars_end_);
        for(uint i = free_vars_end_+1; i < num_vars_all_; i++){
            group_trajectory_.getTrajectoryPoint(i) = xx;
        }
    }

    void TormOptimizer::computeJointProperties(int trajectory_point)
    {
        for (int j = 0; j < num_joints_; j++)
        {
            const moveit::core::JointModel* joint_model = state_.getJointModel(joint_names_[j]);
            const moveit::core::RevoluteJointModel* revolute_joint =
                    dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model);
            const moveit::core::PrismaticJointModel* prismatic_joint =
                    dynamic_cast<const moveit::core::PrismaticJointModel*>(joint_model);

            std::string parent_link_name = joint_model->getParentLinkModel()->getName();
            std::string child_link_name = joint_model->getChildLinkModel()->getName();
            Eigen::Affine3d joint_transform =
                    state_.getGlobalLinkTransform(parent_link_name) *
                    (kmodel_->getLinkModel(child_link_name)->getJointOriginTransform() * (state_.getJointTransform(joint_model)));

            // joint_transform = inverseWorldTransform * jointTransform;
            Eigen::Vector3d axis;

            if (revolute_joint != nullptr)
            {
                axis = revolute_joint->getAxis();
            }
            else if (prismatic_joint != nullptr)
            {
                axis = prismatic_joint->getAxis();
            }
            else
            {
                axis = Eigen::Vector3d::Identity();
            }

            axis = joint_transform * axis;

            joint_axes_[trajectory_point][j] = axis;
            joint_positions_[trajectory_point][j] = joint_transform.translation();
        }
    }

    template <typename Derived>
    void TormOptimizer::getJacobian(int trajectory_point, Eigen::Vector3d& collision_point_pos, std::string& joint_name,
                                    Eigen::MatrixBase<Derived>& jacobian) const
    {
        for (int j = 0; j < num_joints_; j++)
        {
            if (isParent(joint_name, joint_names_[j]))
            {
                Eigen::Vector3d column = joint_axes_[trajectory_point][j].cross(
                        Eigen::Vector3d(collision_point_pos(0), collision_point_pos(1), collision_point_pos(2)) -
                        joint_positions_[trajectory_point][j]);

                jacobian.col(j)[0] = column.x();
                jacobian.col(j)[1] = column.y();
                jacobian.col(j)[2] = column.z();
            }
            else
            {
                jacobian.col(j)[0] = 0.0;
                jacobian.col(j)[1] = 0.0;
                jacobian.col(j)[2] = 0.0;
            }
        }
    }

    void TormOptimizer::handleJointLimits()
    {
        const std::vector<const moveit::core::JointModel*> joint_models = joint_model_group_->getActiveJointModels();
        for (size_t joint_i = 0; joint_i < joint_models.size(); joint_i++)
        {
            const moveit::core::JointModel* joint_model = joint_models[joint_i];

            if (joint_model->getType() == moveit::core::JointModel::REVOLUTE)
            {
                const moveit::core::RevoluteJointModel* revolute_joint =
                        dynamic_cast<const moveit::core::RevoluteJointModel*>(joint_model);
                if (revolute_joint->isContinuous())
                {
                    continue;
                }
            }

            const moveit::core::JointModel::Bounds& bounds = joint_model->getVariableBounds();

            double joint_max = -DBL_MAX;
            double joint_min = DBL_MAX;

            for (moveit::core::JointModel::Bounds::const_iterator it = bounds.begin(); it != bounds.end(); it++)
            {
                if (it->min_position_ < joint_min)
                {
                    joint_min = it->min_position_;
                }

                if (it->max_position_ > joint_max)
                {
                    joint_max = it->max_position_;
                }
            }

            int count = 0;

            bool violation = false;
            do
            {
                double max_abs_violation = 1e-6;
                double max_violation = 0.0;
                int max_violation_index = 0;
                violation = false;
                for (int i = free_vars_start_; i <= free_vars_end_; i++)
                {
                    double amount = 0.0;
                    double absolute_amount = 0.0;
                    if (group_trajectory_(i, joint_i) > joint_max)
                    {
                        amount = joint_max - group_trajectory_(i, joint_i);
                        absolute_amount = fabs(amount);
                    }
                    else if (group_trajectory_(i, joint_i) < joint_min)
                    {
                        amount = joint_min - group_trajectory_(i, joint_i);
                        absolute_amount = fabs(amount);
                    }
                    if (absolute_amount > max_abs_violation)
                    {
                        max_abs_violation = absolute_amount;
                        max_violation = amount;
                        max_violation_index = i;
                        violation = true;
                    }
                }

                if (violation)
                {
                    int free_var_index = max_violation_index - free_vars_start_;
                    double multiplier =
                            max_violation / joint_costs_[joint_i].getQuadraticCostInverse()(free_var_index, free_var_index);
                    group_trajectory_.getFreeJointTrajectoryBlock(joint_i) +=
                            multiplier * joint_costs_[joint_i].getQuadraticCostInverse().col(free_var_index);
                }
                if (++count > 10)
                    break;
            } while (violation);
        }
    }

    //
    void TormOptimizer::performForwardKinematics(int start, int end)
    {
        // for each point in the trajectory
        for (int i = start; i <= end; ++i)
        {
            // Set Robot state from trajectory point...
            collision_detection::CollisionRequest req;
            collision_detection::CollisionResult res;
            req.group_name = planning_group_;
            setRobotStateFromPoint(group_trajectory_, i);
            hy_world_->getCollisionGradients(req, res, *hy_robot_->getCollisionRobotDistanceField().get(),
                                             state_, &acm_, gsr_);
            computeJointProperties(i);

            // Keep vars in scope
            {
                size_t j = 0;
                for (size_t g = start_collision_; g < end_collision_; g++)
                {
                    collision_detection::GradientInfo& info = gsr_->gradients_[g];

                    for (size_t k = 0; k < info.sphere_locations.size(); k++)
                    {
                        collision_point_potential_[i][j] = getPotential(info.distances[k], info.sphere_radii[k], parameters_->min_clearence_);
                        j++;
                    }
                }
            }
        }
    }

    void TormOptimizer::performForwardKinematics()
    {
        double inv_time = 1.0 / group_trajectory_.getDiscretization();
        double inv_time_sq = inv_time * inv_time;

        // calculate the forward kinematics for the fixed states only in the first iteration:
        int start = free_vars_start_;
        if (iteration_ == 0) {
            start = 0;
        }
        int end = num_vars_all_-1;

        // for each point in the trajectory
        for (int i = start; i <= end; ++i)
        {
            // Set Robot state from trajectory point...
            collision_detection::CollisionRequest req;
            collision_detection::CollisionResult res;
            req.group_name = planning_group_;
            setRobotStateFromPoint(group_trajectory_, i);

            hy_world_->getCollisionGradients(req, res, *hy_robot_->getCollisionRobotDistanceField().get(), state_, &acm_, gsr_);

            computeJointProperties(i);

            // Keep vars in scope
            {
                size_t j = 0;
                for (size_t g = start_collision_; g < end_collision_; g++)
                {
                    collision_detection::GradientInfo& info = gsr_->gradients_[g];

                    for (size_t k = 0; k < info.sphere_locations.size(); k++)
                    {
                        collision_point_pos_eigen_[i][j][0] = info.sphere_locations[k].x();
                        collision_point_pos_eigen_[i][j][1] = info.sphere_locations[k].y();
                        collision_point_pos_eigen_[i][j][2] = info.sphere_locations[k].z();

                        collision_point_potential_[i][j] =
                                getPotential(info.distances[k], info.sphere_radii[k], parameters_->min_clearence_);
                        collision_point_potential_gradient_[i][j][0] = info.gradients[k].x();
                        collision_point_potential_gradient_[i][j][1] = info.gradients[k].y();
                        collision_point_potential_gradient_[i][j][2] = info.gradients[k].z();
                        j++;
                        if (debug_visual_) {
//                            if (collision_point_potential_[i][j] > 0){
                            if ((info.distances[k] - info.sphere_radii[k]) < 0){
                                debug_->addContactVector(collision_point_pos_eigen_[i][j][0],
                                                         collision_point_pos_eigen_[i][j][1],
                                                         collision_point_pos_eigen_[i][j][2],
                                                         collision_point_potential_gradient_[i][j][0],
                                                         collision_point_potential_gradient_[i][j][1],
                                                         collision_point_potential_gradient_[i][j][2], false);
                            }

                        }
                    }
                }
            }
        }

        // now, get the vel and acc for each collision point (using finite differencing)
        for (int i = free_vars_start_; i <= free_vars_end_; i++)
        {
            for (int j = 0; j < num_collision_points_; j++)
            {
                collision_point_vel_eigen_[i][j] = Eigen::Vector3d(0, 0, 0);
                collision_point_acc_eigen_[i][j] = Eigen::Vector3d(0, 0, 0);
                for (int k = -DIFF_RULE_LENGTH / 2; k <= DIFF_RULE_LENGTH / 2; k++)
                {
                    collision_point_vel_eigen_[i][j] +=
                            (inv_time * DIFF_RULES[0][k + DIFF_RULE_LENGTH / 2]) * collision_point_pos_eigen_[i + k][j];
                    collision_point_acc_eigen_[i][j] +=
                            (inv_time_sq * DIFF_RULES[1][k + DIFF_RULE_LENGTH / 2]) * collision_point_pos_eigen_[i + k][j];
                }

                // get the norm of the velocity:
                collision_point_vel_mag_[i][j] = collision_point_vel_eigen_[i][j].norm();
            }
        }
    }

    void TormOptimizer::setRobotStateFromPoint(TormTrajectory& group_trajectory, int i) {
        const Eigen::MatrixXd::RowXpr& point = group_trajectory.getTrajectoryPoint(i);

        std::vector<double> joint_states;
        joint_states.reserve(num_joints_);
        for(int j=0; j<num_joints_; j++){
            joint_states.push_back(point(0, j));
        }

        state_.setJointGroupPositions(planning_group_, joint_states);
        state_.update();
        planning_scene_->setCurrentState(state_);
    }

    // cost functions and its gradient function
    double TormOptimizer::getEndPoseCost(bool grad) {
        double endPos_cost = 0.0;
        double endRot_cost = 0.0;

        // forward kinematics
        Eigen::MatrixXd xi = group_trajectory_.getTrajectory();
        KDL::JntArray q(num_joints_);
        KDL::JntArray delta_q(num_joints_);
        KDL::Frame endPoses_c;
        KDL::Twist delta_twist;

        for (int i = free_vars_start_; i <= free_vars_end_; i++) {
            q.data = xi.row(i);
            iksolver_.fkSolver(q, endPoses_c);
            delta_twist = diff(endPoses_c, endPoses_desired_[group_trajectory_.getFullTrajectoryIndex(i)]);
            if(grad){
                iksolver_.vikSolver(q, delta_twist, delta_q);
                endPose_increments_.row(group_trajectory_.getFullTrajectoryIndex(i)-1) = delta_q.data;
            }
            endPos_cost += std::sqrt(KDL::dot(delta_twist.vel, delta_twist.vel));
            endRot_cost += std::sqrt(KDL::dot(delta_twist.rot, delta_twist.rot));
        }
//        if(debug_verbose_)
//            std::cout << "---------------------------" << endPos_cost/num_vars_free_ << ", " << endRot_cost/num_vars_free_ << std::endl;

        return (endPos_cost + parameters_->rotation_scale_factor_ * endRot_cost);
    }

    double TormOptimizer::getEndPoseCost(int start_idx, int end_idx) {
        double endPose_cost = 0.0;

        // forward kinematics
        Eigen::MatrixXd xi = group_trajectory_.getTrajectory();
        KDL::JntArray q(num_joints_);
        KDL::JntArray delta_q(num_joints_);

        KDL::Frame ec;
        KDL::Twist dt;
        for (int i = start_idx+1; i <= end_idx; i++) {
            q.data = xi.row(i);
            iksolver_.fkSolver(q, ec);
            dt = diff(ec, endPoses_desired_[group_trajectory_.getFullTrajectoryIndex(i)]);
            endPose_cost += std::sqrt(KDL::dot(dt.vel, dt.vel)) +
                    parameters_->rotation_scale_factor_ * std::sqrt(KDL::dot(dt.rot, dt.rot));
        }

        return endPose_cost/(end_idx - start_idx);
    }

    double TormOptimizer::getCollisionCost(int start, int end)
    {
        double collision_cost = 0.0;

        for (int i = start; i <= end; i++)
        {
            double state_collision_cost = 0.0;
            for (int j = 0; j < num_collision_points_; j++)
            {
                state_collision_cost += collision_point_potential_[i][j];
            }
            collision_cost += state_collision_cost;
        }

        return collision_cost/(end - start + 1);
    }

    // calculate increments ////////////////////////////////////////////////////////////////////////////////////////
    void TormOptimizer::calculateSmoothnessIncrements()
    {
        for (int i = 0; i < num_joints_; i++)
        {
            joint_costs_[i].getDerivative(group_trajectory_.getJointTrajectory(i), smoothness_derivative_);
            smoothness_increments_.col(i) = -smoothness_derivative_.segment(group_trajectory_.getStartIndex(), num_vars_free_);
        }
    }

    void TormOptimizer::calculateEndPoseIncrements(){
        for (int i = 0; i < num_joints_; i++) {
            final_increments_.col(i) = parameters_->jacobian_update_weight_ * endPose_increments_.col(i);
        }
        if (debug_verbose_ )
            ROS_INFO_STREAM("[Pose] final_increments_inc: " << final_increments_.norm()
            << " [ " << final_increments_.maxCoeff() << " | " << final_increments_.minCoeff() << " ]");
    }

    void TormOptimizer::calculateCollisionIncrements()
    {
        double potential;
        double vel_mag_sq;
        double vel_mag;
        Eigen::Vector3d potential_gradient;
        Eigen::Vector3d normalized_velocity;
        Eigen::Matrix3d orthogonal_projector;
        Eigen::Vector3d curvature_vector;
        Eigen::Vector3d cartesian_gradient;

        collision_increments_.setZero(num_vars_free_, num_joints_);

        int startPoint = 0;
        int endPoint = free_vars_end_;

        // In stochastic descent, simply use a random_free point in the trajectory, rather than all the trajectory points.
        // This is faster and guaranteed to converge, but it may take more iterations in the worst case.
        if (parameters_->use_stochastic_descent_) {
            if (collision_point_ == -1) {
                startPoint = (int) (((double) random() / (double) RAND_MAX) * (free_vars_end_ - free_vars_start_) +
                                    free_vars_start_);
                if (startPoint < free_vars_start_)
                    startPoint = free_vars_start_;
                if (startPoint > free_vars_end_)
                    startPoint = free_vars_end_;
                endPoint = startPoint;
            }else{
                startPoint = collision_point_;
                endPoint = collision_point_;
            }
        }
        else {
            startPoint = free_vars_start_;
        }

        for (int i = startPoint; i <= endPoint; i++)
        {
            for (int j = 0; j < num_collision_points_; j++)
            {
                potential = collision_point_potential_[i][j];

                if (potential < 0.0001)
                    continue;

                potential_gradient = -collision_point_potential_gradient_[i][j];

                vel_mag = collision_point_vel_mag_[i][j];
                vel_mag_sq = vel_mag * vel_mag;

                normalized_velocity = collision_point_vel_eigen_[i][j] / vel_mag;
                orthogonal_projector = Eigen::Matrix3d::Identity() - (normalized_velocity * normalized_velocity.transpose());
                curvature_vector = (orthogonal_projector * collision_point_acc_eigen_[i][j]) / vel_mag_sq;
                cartesian_gradient = vel_mag * (orthogonal_projector * potential_gradient - potential * curvature_vector);

                // pass it through the jacobian transpose to get the increments
                getJacobian(i, collision_point_pos_eigen_[i][j], collision_point_joint_names_[i][j], jacobian_);

                if (parameters_->use_pseudo_inverse_)
                {
                    calculatePseudoInverse();
                    collision_increments_.row(i - free_vars_start_).transpose() -= jacobian_pseudo_inverse_ * cartesian_gradient;
                }
                else
                {
                    collision_increments_.row(i - free_vars_start_).transpose() -= jacobian_.transpose() * cartesian_gradient;
                }
            }
        }
    }

    void TormOptimizer::calculateFeasibleIncrements()
    {
        if (lr_schedule_mode_ !=0) {
            {
                double max_inc = smoothness_increments_.maxCoeff();
                double min_inc = smoothness_increments_.minCoeff();
                double scale = std::max(std::abs(max_inc), std::abs(min_inc));
                if (debug_verbose_ )
                    ROS_INFO_STREAM("[Feasible] smoothness_increments_inc: " << smoothness_increments_.norm()
                    << " [ " << smoothness_increments_.maxCoeff() << " | " << smoothness_increments_.minCoeff() << " ]");
                if (scale > 0) {
                    smoothness_increments_ /= scale;
                }
            }
            {
                double max_inc = collision_increments_.maxCoeff();
                double min_inc = collision_increments_.minCoeff();
                double scale = std::max(std::abs(max_inc), std::abs(min_inc));
                if (debug_verbose_ )
                    ROS_INFO_STREAM("[Feasible] collision_increments_inc: " << collision_increments_.norm()
                    << " [ " << collision_increments_.maxCoeff() << " | " << collision_increments_.minCoeff() << " ]");
                if (scale > 0) {
                    collision_increments_ /= scale;
                }
            }
            for (int i = 0; i < num_joints_; i++) {
                final_increments_.col(i) =
                        learning_rate_ *
                        (
                                joint_costs_[i].getQuadraticCostInverse() *
                                (
                                        alpha_ * collision_increments_.col(i) +
                                        (1 - alpha_) * smoothness_increments_.col(i)
                                )
                        );
            }
        }else{
            for (int i = 0; i < num_joints_; i++){
                final_increments_.col(i) =
                        parameters_->learning_rate_ * ((joint_costs_[i].getQuadraticCostInverse() *
                                                        (parameters_->smoothness_update_weight_ * smoothness_increments_.col(i)
                                                         + parameters_->obstacle_update_weight_ * collision_increments_.col(i))
                        ));
            }
            if(debug_verbose_ ) {
                ROS_INFO_STREAM("[Feasible] smoothness_increments_inc: " << smoothness_increments_.norm()
                << " [ " << smoothness_increments_.maxCoeff() << " | " << smoothness_increments_.minCoeff() << " ]");
                ROS_INFO_STREAM("[Feasible] collision_increments_inc: " << collision_increments_.norm()
                << " [ " << collision_increments_.maxCoeff() << " | " << collision_increments_.minCoeff() << " ]");
            }
        }
        if (debug_verbose_ )
            ROS_INFO_STREAM("[Feasible] final_increments_inc: " << final_increments_.norm()
            << " [ " << final_increments_.maxCoeff() << " | " << final_increments_.minCoeff() << " ]");
    }

    void TormOptimizer::addIncrementsToTrajectory() {
        if (lr_schedule_mode_ != 0) {
            {
                double max_inc = final_increments_.maxCoeff();
                double min_inc = final_increments_.minCoeff();
                double scale = std::max(std::abs(max_inc), std::abs(min_inc));
                if (scale > parameters_->joint_update_limit_ and scale > 0) {
                    final_increments_ = final_increments_ / scale * parameters_->joint_update_limit_;
                }
            }
            for (size_t i = 0; i < num_joints_; i++) {
                group_trajectory_.getFreeTrajectoryBlock().col(i) += final_increments_.col(i);
            }
        }else{
            for (size_t i = 0; i < num_joints_; i++)
            {
                double scale = 1.0;
                double max = final_increments_.col(i).maxCoeff();
                double min = final_increments_.col(i).minCoeff();
                double max_scale = parameters_->joint_update_limit_ / fabs(max);
                double min_scale = parameters_->joint_update_limit_ / fabs(min);
                if (max_scale < scale)
                    scale = max_scale;
                if (min_scale < scale)
                    scale = min_scale;
                group_trajectory_.getFreeTrajectoryBlock().col(i) += scale * final_increments_.col(i);
            }
        }
    }

}  // namespace torm

