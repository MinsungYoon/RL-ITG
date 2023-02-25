#ifndef CATKIN_WS_TRAJ_EVALUATOR_H
#define CATKIN_WS_TRAJ_EVALUATOR_H

#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <eigen3/Eigen/Core>
#include <Eigen/Dense>

#include <torm/torm_utils.h>
#include <torm/torm_ik_solver.h>

class traj_evaluator{
public:
    traj_evaluator(std::vector<KDL::Frame> targetPoses, std::vector<std::vector<double>>& trajectory,
                   std::vector<int> c_joint, double discretization, torm::TormIKSolver& iksolver, std::vector<double>& vel_limit);
    traj_evaluator(std::vector<KDL::Frame> targetPoses, Eigen::MatrixXd trajectory,
                   std::vector<int> c_joint, double discretization, torm::TormIKSolver& iksolver, std::vector<double>& vel_limit);
    void init();
    ~traj_evaluator();

    void setCostMatrix();
    Eigen::MatrixXd createDiffMatrix(int size, int diffIdx) const;

    void getPoseCost(double& cpose, double& crot);
    double getWeightedPoseCost();

    int getCost(double& cpose, double& crot, double& cvel, double& cacc, double& cjerk, bool flag_for_velviolation_measure=false);

    double testJointCost(int j_idx);

    void calcJointCostwithNumericalDifferenciation(int j_idx, double& cvel, double& cacc, double& cjerk);
    void calcJointCostwithFiniteDifferenciation(int j_idx, double& cvel, double& cacc, double& cjerk);

    void printMatrix(Eigen::MatrixXd m) const; // right 'const': cannot modify member variable value in this function

    void setInterTargetPoses();

    int calcNofVelViolation();

    void setNewDt(double dt);
private:

    std::vector<double> vel_limit_;

    std::vector<KDL::Frame> targetPoses_;
    std::vector<KDL::Frame> targetPoses_inter_;

    double discretization_;

    std::vector<int> c_joint_;

    Eigen::MatrixXd trajectory_;
    int num_points_;
    int num_joints_;

    Eigen::MatrixXd full_trajectory_;
    int num_full_points_;

    int free_start_index_;
    int free_end_index_;

    std::vector<Eigen::MatrixXd> quadraticFullCost_;
    std::vector<Eigen::MatrixXd> diffMatrixList;

    torm::TormIKSolver& iksolver_;
};


























#endif //CATKIN_WS_TRAJ_EVALUATOR_H
