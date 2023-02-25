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

#include <torm/torm_trajectory.h>
#include <iostream>

namespace torm
{
    TormTrajectory::TormTrajectory(const moveit::core::RobotModelConstPtr& robot_model,
                                   int num_points,
                                   double discretization,
                                   std::string group_name)
            : planning_group_name_(group_name)
            , num_points_(num_points)
            , discretization_(discretization)
            , duration_((num_points - 1) * discretization)
    {
        extra_ = 0;
        start_index_ = 1;
        end_index_ = num_points_ - 1;
        const moveit::core::JointModelGroup* model_group = robot_model->getJointModelGroup(planning_group_name_);
        num_joints_ = model_group->getActiveJointModels().size();
        trajectory_.resize(num_points_, num_joints_);
    }

    TormTrajectory::TormTrajectory(const TormTrajectory& source_traj,
                                   const std::string& planning_group,
                                   int diff_rule_length)
            : planning_group_name_(planning_group), discretization_(source_traj.discretization_)
    {
        num_joints_ = source_traj.getNumJoints();

        // figure out the num_points_:
        // we need diff_rule_length-2 extra points on either side:
        int start_extra = (diff_rule_length - 2);
        int end_extra = (diff_rule_length - 2);
        extra_ = (diff_rule_length - 2);

        num_points_ = source_traj.num_points_ + start_extra + end_extra;
        start_index_ = diff_rule_length - 1;

        end_index_ = (num_points_ - 1) - (diff_rule_length - 2);
        duration_ = (num_points_ - 1) * discretization_;

        // allocate the memory:
        trajectory_.resize(num_points_, num_joints_);

        full_trajectory_index_.resize(num_points_);

        // now copy the trajectories over:
        for (int i = 0; i < num_points_; i++)
        {
            int source_traj_point = i - start_extra;
            if (source_traj_point <= 0)
                source_traj_point = 0;
            if (source_traj_point >= source_traj.num_points_)
                source_traj_point = source_traj.num_points_ - 1;
            full_trajectory_index_[i] = source_traj_point;
            for (int j = 0; j < num_joints_; j++)
            {
                (*this)(i, j) = source_traj(source_traj_point, j);
            }
        }
    }

    TormTrajectory::~TormTrajectory()
    {
    }

    void TormTrajectory::updateFromGroupTrajectory(const TormTrajectory& group_trajectory)
    {
        int num_vars_free = end_index_ - start_index_ + 2;
        trajectory_.block(start_index_-1, 0, num_vars_free, num_joints_) =
                group_trajectory.trajectory_.block(group_trajectory.getStartIndex()-1, 0, num_vars_free, group_trajectory.getNumJoints());
    }
}  // namespace torm
