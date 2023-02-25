#ifndef PIRL_INTERPOLATOR_H
#define PIRL_INTERPOLATOR_H

#include <ros/ros.h>
#include <ros/package.h>

#include <torch/script.h> // One-stop header.
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <eigen3/Eigen/Core>
#include <torm/torm_ik_solver.h>
#include <random>

#include <torm/torm_utils.h>

class PirlInterpolator{
public:
    PirlInterpolator(const std::string& model_path, int t, std::string obsType,
                     std::vector<KDL::Frame>& targetPoses, std::vector<int>& simplified_points,
                     torm::TormIKSolver& iksolver, int multi_action=1);
    ~PirlInterpolator();

    void setAction(KDL::JntArray& cur_conf, std::vector<float>& act, KDL::JntArray& o2);
    void getAct(std::vector<float>& obs, std::vector<float>& act, bool deterministic);
    void getObs(KDL::JntArray& cur_conf, uint& i, std::vector<float>& obs);
    void interpolate(KDL::JntArray& start_conf, bool deterministic);

    void append_pos_and_rot(KDL::Frame &source, std::vector<float>& obs);
    void append_pos_and_rot_error(KDL::Frame &target, KDL::Frame &source, std::vector<float>& err);

    inline int getNumPoints() const {
        return num_points_;
    }
    inline Eigen::MatrixXd::RowXpr getTrajectoryPoint(int traj_point){
        return trajectory_.row(traj_point);
    }
    inline Eigen::MatrixXd getTrajectory(){
        return trajectory_;
    }


private:
    ros::NodeHandle nh_;
    std::vector<double> c_joints_;
    std::vector<double> ll_;
    std::vector<double> ul_;

    int t_;
    std::string obsType_;
    int mode_{0};

    int multi_action_{1};

    int n_segement_{0};
    int obs_dim_{0};
    int act_dim_{0};

    std::string problem_name_;
    bool load_scene_;
    std::vector<double> vae_latent_;

    int num_total_points_;
    int num_points_;
    int num_joints_;

    torch::jit::script::Module module_;
    std::vector<torch::jit::IValue> inputs_;

    std::vector<KDL::Frame>& targetPoses_;
    std::vector<int>& simplified_points_;
//    std::vector<Eigen::MatrixXd> trajectories_;
    Eigen::MatrixXd trajectory_;

    torm::TormIKSolver& iksolver_;

    std::mt19937 mersenne_;
    std::uniform_real_distribution<double> p_deterministic_action;
};

typedef std::shared_ptr<PirlInterpolator> PirlInterpolatorPtr;






#endif