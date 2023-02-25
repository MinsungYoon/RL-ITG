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

#ifndef TORM_UTILS_H_
#define TORM_UTILS_H_

#include <iostream>
#include <Eigen/Core>
#include <moveit/planning_scene/planning_scene.h>
#include <fstream>
#include <utility>
#include <string>

namespace torm
{
    static const int DIFF_RULE_LENGTH = 7;
    // VEL:  Derivative=1, Accuracy=6
    // ACC:  Derivative=2, Accuracy=6
    // JERK: Derivative=3, Accuracy=4
    const double DIFF_RULES[3][DIFF_RULE_LENGTH] = {
            { -1.0/60,    3.0/20,    -3.0/4,      0.0,        3.0/4,     -3.0/20,     1.0/60 },
            {  1.0/90,   -3.0/20,     3.0/2,     -49.0/18,    3.0/2,     -3.0/20,     1.0/90 },
            {  1.0/8,    -1.0,        13.0/8,     0.0,       -13.0/8,     1.0,       -1.0/8  }
    };

    static inline void jointStateToArray(const moveit::core::RobotModelConstPtr& kmodel,
                                         const sensor_msgs::JointState& joint_state, const std::string& planning_group_name,
                                         Eigen::MatrixXd::RowXpr joint_array)
    {
      const moveit::core::JointModelGroup* group = kmodel->getJointModelGroup(planning_group_name);
      std::vector<const moveit::core::JointModel*> models = group->getActiveJointModels();

      for (unsigned int i = 0; i < joint_state.position.size(); i++)
      {
        for (size_t j = 0; j < models.size(); j++)
        {
          if (models[j]->getName() == joint_state.name[i])
          {
            joint_array(0, j) = joint_state.position[i];
          }
        }
      }
    }

    // copied from geometry/angles/angles.h
    static inline double normalizeAnglePositive(double angle)
    {
      return fmod(fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
    }

    static inline double normalizeAngle(double angle)
    {
      double a = normalizeAnglePositive(angle);
      if (a > M_PI)
        a -= 2.0 * M_PI;
      return a;
    }

    static inline double shortestAngularDistance(double start, double end)
    {
      double res = normalizeAnglePositive(normalizeAnglePositive(end) - normalizeAnglePositive(start));
      if (res > M_PI)
      {
        res = -(2.0 * M_PI - res);
      }
      return normalizeAngle(res);
    }

    static inline std::vector<std::string> split(std::string str, char delimiter) {
        std::vector<std::string> internal;
        std::stringstream ss(str);
        std::string temp;

        while (getline(ss, temp, delimiter)) {
            internal.push_back(temp);
        }

        return internal;
    }

    static inline std::vector<double> split_f(std::string str, char delimiter) {
        std::vector<double> internal;
        std::stringstream ss(str);
        std::string temp;

        while (getline(ss, temp, delimiter)) {
            internal.push_back(std::atof(temp.c_str()));
        }

        return internal;
    }

    inline void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> dataset){
        // Make a CSV file with one or more columns of integer values
        // Each column of data is represented by the pair <column name, column data>
        //   as std::pair<std::string, std::vector<int>>
        // The dataset is represented as a vector of these columns
        // Note that all columns should be the same size

        // Create an output filestream object
        std::ofstream myFile(filename);

        // Send column names to the stream
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).first;
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";

        // Send data to the stream
        for(int i = 0; i < dataset.at(0).second.size(); ++i)
        {
            for(int j = 0; j < dataset.size(); ++j)
            {
                myFile << dataset.at(j).second.at(i);
                if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
            }
            myFile << "\n";
        }

        // Close the file
        myFile.close();
    }

    inline void write_optResult( std::string file_name, double result){ // 1: suc, -1: col, -2: vel, -3: col & vel
        std::fstream fs;
        fs.open(file_name.c_str(), std::ios::out);
        fs << result;
        fs.close();
    }

}  // namespace torm

#endif /* TORM_UTILS_H_ */
