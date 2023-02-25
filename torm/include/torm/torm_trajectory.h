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

#ifndef TORM_TRAJECTORY_H_
#define TORM_TRAJECTORY_H_

#include <trajectory_msgs/JointTrajectory.h>
#include <moveit/robot_model/robot_model.h>
#include <torm/torm_utils.h>

#include <moveit_msgs/MotionPlanDetailedResponse.h>
#include <moveit_msgs/MotionPlanRequest.h>

#include <vector>
#include <eigen3/Eigen/Core>

namespace torm
{
/**
 * \brief Represents a discretized joint-space trajectory for TORM
 */
    class TormTrajectory
    {
    public:
        TormTrajectory(const moveit::core::RobotModelConstPtr& robot_model, int num_points, double discretization,
                        std::string groupName);
        TormTrajectory(const TormTrajectory& source_traj, const std::string& planning_group, int diff_rule_length);
        virtual ~TormTrajectory();

        double& operator()(int traj_point, int joint);

        double operator()(int traj_point, int joint) const;
        Eigen::MatrixXd::RowXpr getTrajectoryPoint(int traj_point);
        Eigen::MatrixXd::ColXpr getJointTrajectory(int joint);
        int getNumPoints() const;
        int getNumFreePoints() const;
        int getNumJoints() const;
        double getDiscretization() const;
        void setStartEndIndex(int start_index, int end_index);
        int getStartIndex() const;
        int getEndIndex() const;
        int getExtra() const;
        Eigen::MatrixXd& getTrajectory();
        Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic> getFreeTrajectoryBlock();
        Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic> getFreeJointTrajectoryBlock(int joint);

        void updateFromGroupTrajectory(const TormTrajectory& group_trajectory);

        int getFullTrajectoryIndex(int i) const;

        template <typename Derived>
        void getJointVelocities(int traj_point, Eigen::MatrixBase<Derived>& velocities);
        double getDuration() const;

    private:
        std::string planning_group_name_;
        int num_points_;
        int num_joints_;
        double discretization_;
        double duration_;
        Eigen::MatrixXd trajectory_;
        int extra_;
        int start_index_;
        int end_index_;
        std::vector<int> full_trajectory_index_;
    };

    inline double& TormTrajectory::operator()(int traj_point, int joint)
    {
        return trajectory_(traj_point, joint);
    }

    inline double TormTrajectory::operator()(int traj_point, int joint) const
    {
        return trajectory_(traj_point, joint);
    }

    inline Eigen::MatrixXd::RowXpr TormTrajectory::getTrajectoryPoint(int traj_point)
    {
        return trajectory_.row(traj_point);
    }

    inline Eigen::MatrixXd::ColXpr TormTrajectory::getJointTrajectory(int joint)
    {
        return trajectory_.col(joint);
    }

    inline int TormTrajectory::getExtra() const
    {
        return extra_;
    }

    inline int TormTrajectory::getNumPoints() const
    {
        return num_points_;
    }

    inline int TormTrajectory::getNumFreePoints() const
    {
        return (end_index_ - start_index_) + 1;
    }

    inline int TormTrajectory::getNumJoints() const
    {
        return num_joints_;
    }

    inline double TormTrajectory::getDiscretization() const
    {
        return discretization_;
    }

    inline void TormTrajectory::setStartEndIndex(int start_index, int end_index)
    {
        start_index_ = start_index;
        end_index_ = end_index;
    }

    inline int TormTrajectory::getStartIndex() const
    {
        return start_index_;
    }

    inline int TormTrajectory::getEndIndex() const
    {
        return end_index_;
    }

    inline Eigen::MatrixXd& TormTrajectory::getTrajectory()
    {
        return trajectory_;
    }

    inline Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic> TormTrajectory::getFreeTrajectoryBlock()
    {
        return trajectory_.block(start_index_, 0, getNumFreePoints(), getNumJoints());
    }

    inline Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>
    TormTrajectory::getFreeJointTrajectoryBlock(int joint)
    {
        return trajectory_.block(start_index_, joint, getNumFreePoints(), 1);
    }

    inline int TormTrajectory::getFullTrajectoryIndex(int i) const
    {
        return full_trajectory_index_[i];
    }

    template <typename Derived>
    void TormTrajectory::getJointVelocities(int traj_point, Eigen::MatrixBase<Derived>& velocities)
    {
        velocities.setZero();
        double invTime = 1.0 / discretization_;

        for (int k = -DIFF_RULE_LENGTH / 2; k <= DIFF_RULE_LENGTH / 2; k++)
        {
            velocities += (invTime * DIFF_RULES[0][k + DIFF_RULE_LENGTH / 2]) * trajectory_.row(traj_point + k).transpose();
        }
    }

    inline double TormTrajectory::getDuration() const
    {
        return duration_;
    }
}

#endif /* TORM_TRAJECTORY_H_ */
