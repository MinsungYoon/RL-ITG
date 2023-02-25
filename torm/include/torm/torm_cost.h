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

#ifndef TORM_COST_H_
#define TORM_COST_H_

#include <eigen3/Eigen/Core>
#include <torm/torm_trajectory.h>
#include <vector>

namespace torm
{
/**
 * \brief Represents the smoothness cost for TORM, for a single joint
 */
    class TormCost
    {
    public:
        TormCost(const TormTrajectory& trajectory, int joint_number, const std::vector<double>& derivative_costs,
                  double ridge_factor = 0.0);
        virtual ~TormCost();

        template <typename Derived>
        void getDerivative(Eigen::MatrixXd::ColXpr joint_trajectory, Eigen::MatrixBase<Derived>& derivative) const;

        const Eigen::MatrixXd& getQuadraticCostInverse() const;
        const Eigen::MatrixXd& getQuadraticCost() const;

        double getCost(Eigen::MatrixXd::ColXpr joint_trajectory) const;
        double getCost(Eigen::MatrixXd::ColXpr joint_trajectory, int start, int end) const;

        double getMaxQuadCostInvValue() const;

        void scale(double scale);

    private:
        Eigen::MatrixXd quad_cost_full_;
        Eigen::MatrixXd quad_cost_one_;
        Eigen::MatrixXd quad_cost_;
        // Eigen::VectorXd linear_cost_;
        Eigen::MatrixXd quad_cost_inv_;

        Eigen::MatrixXd getDiffMatrix(int size, const double* diff_rule) const;
    };

    template <typename Derived>
    void TormCost::getDerivative(Eigen::MatrixXd::ColXpr joint_trajectory, Eigen::MatrixBase<Derived>& derivative) const
    {
        derivative = (quad_cost_full_ * (2.0 * joint_trajectory));
    }

    inline const Eigen::MatrixXd& TormCost::getQuadraticCostInverse() const
    {
        return quad_cost_inv_;
    }

    inline const Eigen::MatrixXd& TormCost::getQuadraticCost() const
    {
        return quad_cost_;
    }

    inline double TormCost::getCost(Eigen::MatrixXd::ColXpr joint_trajectory) const
    {
        return joint_trajectory.dot(quad_cost_full_ * joint_trajectory);
    }

    inline double TormCost::getCost(Eigen::MatrixXd::ColXpr joint_trajectory, int start, int end) const
    {
        Eigen::VectorXd vx = Eigen::VectorXd::Zero(15);
        int t = 0;
        for(uint i = 0; i < 6; i++){
            vx(t++) = joint_trajectory(start);
        }
        for(uint i = start+1; i < end; i++){
            vx(t++) = joint_trajectory(i);
        }
        for(uint i = 0; i < 6; i++){
            vx(t++) = joint_trajectory(end);
        }
        return vx.dot(quad_cost_one_ * vx);
    }

}  // namespace torm

#endif /* TORM_COST_H_ */
