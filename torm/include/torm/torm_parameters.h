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

#ifndef TORM_PARAMETERS_H_
#define TORM_PARAMETERS_H_

#include <ros/ros.h>

namespace torm
{
    class TormParameters
    {
    public:
        TormParameters();
        virtual ~TormParameters();

        TormParameters getNonConstParams(TormParameters params);

    public:
        double planning_time_limit_;

        double smoothness_update_weight_;
        double obstacle_update_weight_;
        double jacobian_update_weight_;

        double learning_rate_;

        double pose_cost_weight_;
        double collision_cost_weight_;

        double rotation_scale_factor_;

        double smoothness_cost_velocity_;
        double smoothness_cost_acceleration_;
        double smoothness_cost_jerk_;

        bool use_stochastic_descent_;
        bool use_velocity_check_;
        bool use_singularity_check_;
        bool use_collision_check_;

        double ridge_factor_;
        bool use_pseudo_inverse_;
        double pseudo_inverse_ridge_factor_;

        double joint_update_limit_;
        double min_clearence_;

        int exploration_iter_;
        int traj_generation_iter_;
        double time_duration_;
        double singularity_lower_bound_;
    };

}  // namespace tomp

#endif /* TORM_PARAMETERS_H_ */
