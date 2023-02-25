#include <string>
#include <iostream>
#include <torch/script.h> // One-stop header.
#include <torm/pirl_interpolator.h>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <torm/torm_utils.h>

void KdlFrametoVector(KDL::Frame& in, std::vector<double>& out){
    out[0] = in.p.x();
    out[1] = in.p.y();
    out[2] = in.p.z();
    double x, y, z ,w;
    in.M.GetQuaternion(x, y, z ,w);
    out[3] = x;
    out[4] = y;
    out[5] = z;
    out[6] = w;
}

PirlInterpolator::PirlInterpolator(const std::string& model_path, int t, std::string obsType,
                                   std::vector<KDL::Frame>& targetPoses, std::vector<int>& simplified_points,
                                   torm::TormIKSolver& iksolver, int multi_action)
                                   :t_(t),
                                   obsType_(obsType),
                                   targetPoses_(targetPoses),
                                   simplified_points_(simplified_points),
                                   iksolver_(iksolver),
                                   multi_action_(multi_action){

    nh_.getParam("/robot/n_segement", n_segement_);
    nh_.getParam("/robot/act_dim", act_dim_);
    nh_.getParam("/robot/n_dof", num_joints_);
    nh_.getParam("/robot/ll", ll_);
    nh_.getParam("/robot/ul", ul_);
    nh_.getParam("/robot/continuous_joints", c_joints_);

    nh_.getParam("/problem/problem_name", problem_name_);
    nh_.getParam("/problem/load_scene", load_scene_);
    if (load_scene_){
        std::string path = ros::package::getPath("torm");
        std::string obs_file_name;
        if(problem_name_.find("random_obs") == std::string::npos) { // semantic problem
            obs_file_name = path + "/launch/config/scene/vae_" + problem_name_ + ".txt";
        } else {
            std::vector<std::string> sp = torm::split(problem_name_,'_');
            obs_file_name = path + "/launch/config/scene/vae_" + sp[2] + ".txt";
        }
        std::ifstream ifs(obs_file_name);
        if(ifs.is_open()){
            std::string line;
            getline(ifs, line);
            vae_latent_ = torm::split_f(line, ' ');
            ifs.close();
        }else{
            ROS_ERROR_STREAM("Cannot load latent vector..!");
            exit(0);
        }
    }

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module_ = torch::jit::load(model_path);
        inputs_.clear();
        inputs_.reserve(2);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        exit(0);
    }

    num_total_points_ = targetPoses.size();
    num_points_ = simplified_points.size() * multi_action_;

    trajectory_.resize(num_points_, num_joints_);
    trajectory_ = Eigen::MatrixXd(num_points_, num_joints_);

    if(std::strcmp(obsType_.c_str(),"onlyConf")==0){
        obs_dim_ = 7 + t_*9;
        mode_ = 1;
    }else if(std::strcmp(obsType_.c_str(),"Conf+LinkPoses")==0){
        obs_dim_ = 7 + 9 * 8 + t_*9;
        mode_ = 2;
    }else if(std::strcmp(obsType_.c_str(),"Conf+EE")==0){
        obs_dim_ = 7 + 9 + t_*9;
        mode_ = 3;
    }else if(std::strcmp(obsType_.c_str(),"onlyLinkPoses")==0){
        obs_dim_ = 9 * 8 + t_*9;
        mode_ = 4;
    }
    if(load_scene_){
        obs_dim_ += vae_latent_.size();
    }

    std::random_device rd;
    mersenne_ = std::mt19937(rd());
    p_deterministic_action = std::uniform_real_distribution<double>(0.0, 1.0);
}

PirlInterpolator::~PirlInterpolator(){
}

void PirlInterpolator::interpolate(KDL::JntArray& start_conf, bool deterministic){
    KDL::JntArray cur_conf(num_joints_);
    for(uint j=0; j<num_joints_; j++){
        cur_conf(j) = start_conf(j);
    }

    for(uint i=0; i<num_points_ ; i++){

        uint target_idx = floor(i / multi_action_);
        std::vector<float> obs;
        obs.reserve(obs_dim_);
        getObs(cur_conf, target_idx, obs);
        if (load_scene_){
            for(double l : vae_latent_){
                obs.push_back((float)l);
            }
        }

        std::vector<float> act;
        act.reserve(act_dim_);
        if(!deterministic){
            if(p_deterministic_action(mersenne_) > 0.1){
                getAct(obs, act, true);
            }else{
                getAct(obs, act, false);
            }
        }else{
            getAct(obs, act, true);
        }

        KDL::JntArray next_conf(num_joints_);
        setAction(cur_conf, act, next_conf);

        for(uint j=0; j<num_joints_; j++){
            cur_conf(j) = next_conf(j);
            trajectory_(i, j) = next_conf(j);
        }
    }
}

void PirlInterpolator::setAction(KDL::JntArray& cur_conf, std::vector<float>& act, KDL::JntArray& next_conf){
    for(uint j=0; j<num_joints_; j++){
        next_conf(j) = cur_conf(j) + act[j];
    }
    for(auto c : c_joints_){
        if(next_conf(c) > M_PI){
            next_conf(c) -= 2*M_PI;
        }
        else if(next_conf(c) < -M_PI){
            next_conf(c) += 2*M_PI;
        }
    }
    for(uint j=0; j<num_joints_; j++){
        if(next_conf(j) > ul_[j]){
            next_conf(j) = ul_[j];
        }
        else if(next_conf(j) < ll_[j]){
            next_conf(j) = ll_[j];
        }
    }
}

void PirlInterpolator::getAct(std::vector<float>& obs, std::vector<float>& act, bool deterministic){
    inputs_.clear();
    inputs_.push_back(torch::tensor(obs));
    inputs_.push_back(deterministic);
    auto output = module_.forward(inputs_).toTensor();
    auto dpt = output.data_ptr<float>();
    for(uint j=0; j<num_joints_; j++){
        act.push_back(dpt[j]);
    }
    // act = std::vector<float>(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
}

void PirlInterpolator::getObs(KDL::JntArray& cur_conf, uint& i, std::vector<float>& obs){
    if(mode_ == 1){ // onlyConf
        KDL::Frame EEpose;
        iksolver_.fkSolver(cur_conf, EEpose);

        for(uint j=0; j<num_joints_; j++){
            obs.push_back((float)cur_conf(j));
        }

        for(uint t = 0; t < t_; t++){
            uint t_idx = i + t;
            if(t_idx >= num_points_){
                t_idx = num_points_-1;
            }
            KDL::Frame& t_pose = targetPoses_[simplified_points_[t_idx]];
            append_pos_and_rot_error(t_pose, EEpose, obs);
        }

    }else if(mode_ == 2){ // Conf+LinkPoses
        std::vector<KDL::Frame> curlinks(n_segement_, KDL::Frame());
        iksolver_.allfkSolver(cur_conf, curlinks, n_segement_);

        for(uint j=0; j<num_joints_; j++){
            obs.push_back((float)cur_conf(j));
        }
        for(uint l=0; l<n_segement_; l++){
            append_pos_and_rot(curlinks[l], obs);
        }

        for(uint t=0; t<t_; t++){
            uint t_idx = i + t;
            if(t_idx >= num_points_){
                t_idx = num_points_-1;
            }
            KDL::Frame& t_pose = targetPoses_[simplified_points_[t_idx]];
            KDL::Frame& cur_ee_pose = curlinks[n_segement_-1];
            append_pos_and_rot_error(t_pose, cur_ee_pose, obs);
        }
    }else if(mode_ == 3){ // Conf+EE
        KDL::Frame EEpose;
        iksolver_.fkSolver(cur_conf, EEpose);

        for(uint j=0; j<num_joints_; j++){
            obs.push_back((float)cur_conf(j));
        }
        append_pos_and_rot(EEpose, obs);
        for(uint t = 0; t < t_; t++){
            uint t_idx = i + t;
            if(t_idx >= num_points_){
                t_idx = num_points_-1;
            }
            KDL::Frame& t_pose = targetPoses_[simplified_points_[t_idx]];
            append_pos_and_rot_error(t_pose, EEpose, obs);
        }
    }else if(mode_ == 4){ // onlyLinkPoses
        std::vector<KDL::Frame> curlinks(n_segement_, KDL::Frame());
        iksolver_.allfkSolver(cur_conf, curlinks, n_segement_);

        for(uint l=0; l<n_segement_; l++){
            append_pos_and_rot(curlinks[l], obs);
        }

        for(uint t=0; t<t_; t++){
            uint t_idx = i + t;
            if(t_idx >= num_points_){
                t_idx = num_points_-1;
            }
            KDL::Frame& t_pose = targetPoses_[simplified_points_[t_idx]];
            KDL::Frame& cur_ee_pose = curlinks[n_segement_-1];
            append_pos_and_rot_error(t_pose, cur_ee_pose, obs);
        }
    }
}

void PirlInterpolator::append_pos_and_rot_error(KDL::Frame &target, KDL::Frame &source, std::vector<float>& obs){
    obs.push_back((float)(target.p.x() - source.p.x()));
    obs.push_back((float)(target.p.y() - source.p.y()));
    obs.push_back((float)(target.p.z() - source.p.z()));
    KDL::Rotation R_err = target.M * source.M.Inverse();
    obs.push_back((float)R_err(0,0));
    obs.push_back((float)R_err(1,0));
    obs.push_back((float)R_err(2,0));
    obs.push_back((float)R_err(0,1));
    obs.push_back((float)R_err(1,1));
    obs.push_back((float)R_err(2,1));
}

void PirlInterpolator::append_pos_and_rot(KDL::Frame &source, std::vector<float>& obs){
    obs.push_back((float)source.p.x());
    obs.push_back((float)source.p.y());
    obs.push_back((float)source.p.z());
    obs.push_back((float)source.M(0,0));
    obs.push_back((float)source.M(1,0));
    obs.push_back((float)source.M(2,0));
    obs.push_back((float)source.M(0,1));
    obs.push_back((float)source.M(1,1));
    obs.push_back((float)source.M(2,1));

}


