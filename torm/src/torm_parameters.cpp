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

#include <torm/torm_parameters.h>

namespace torm
{
    TormParameters::TormParameters(){
        planning_time_limit_ = 6.0;
        smoothness_update_weight_ = 0.1;
        jacobian_update_weight_ = 1.0;
        obstacle_update_weight_ = 1.0;
        learning_rate_ = 0.01;

        smoothness_cost_velocity_ = 0.0;
        smoothness_cost_acceleration_ = 1.0;
        smoothness_cost_jerk_ = 0.0;
        ridge_factor_ = 0.0;
        use_pseudo_inverse_ = false;
        pseudo_inverse_ridge_factor_ = 1e-4;

        joint_update_limit_ = 0.15;
        min_clearence_ = 0.2;
        use_stochastic_descent_ = true;
        use_velocity_check_ = true;
        use_singularity_check_ = true;
        use_collision_check_ = true;
        singularity_lower_bound_ = 0.015;

        exploration_iter_ = 50;
        traj_generation_iter_ = 15;
        time_duration_ = 0.2;
    }

    TormParameters::~TormParameters(){
    }

    TormParameters TormParameters::getNonConstParams(TormParameters params) {
        TormParameters non_const;
        non_const = params;
        return non_const;
    }

}  // namespace torm
