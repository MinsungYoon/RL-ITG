/* Author: Mincheul Kang */

#include <torm/torm_problem.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Matrix3x3.h>

namespace torm {
    TormProblem::TormProblem(std::string problem_name, std::string robot_name, planning_scene::PlanningScenePtr&  planning_scene)
    :problem_name_(problem_name), robot_name_(robot_name), planning_scene_(planning_scene)
    {
        std::string path = ros::package::getPath("torm");
        system(std::string("rosparam load " + path + "/launch/config/robot/" + robot_name + ".yaml").c_str());

        if(problem_name.find("random") == std::string::npos) { // semantic problem
            system(std::string("rosparam load " + path + "/launch/config/problem/" + problem_name + ".yaml").c_str());
        } else {
            if(problem_name.find("random_obs") == std::string::npos){ // random_obs problem
                system(std::string("rosparam load " + path + "/launch/config/problem/" + problem_name.substr(0, 6) + ".yaml").c_str());
            }else{ // random problem
                system(std::string("rosparam load " + path + "/launch/config/problem/" + problem_name.substr(0, 10) + ".yaml").c_str());
            }
        }

        nh_.setParam("/problem/robot_name", robot_name);
        nh_.setParam("/problem/problem_name", problem_name);

        nh_.getParam("/robot/planning_group", planning_group_);
        nh_.getParam("/robot/fixed_frame", fixed_frame_);
        nh_.getParam("/robot/planning_base_link", planning_base_link_);
        nh_.getParam("/robot/planning_tip_link", planning_tip_link_);

        config_file_path_ = ros::package::getPath("torm")
                + "/launch/config/start_conf/" + problem_name_ + "_config.csv";

        nh_.getParam("/problem/default_setting_joints", default_setting_joints_);
        nh_.getParam("/problem/default_setting_values", default_setting_values_);
        nh_.getParam("/problem/load_path", load_path_);
        nh_.getParam("/problem/load_scene", load_scene_);

        resetCollisionObjects();

        setDefaultJointPosition();

        setStartConfig(0);

        if(load_scene_){
            setCollisionObjects();
        }
        if(load_path_){
            setPlanningPath();
        }

    }

    TormProblem::~TormProblem() {
    }

    void TormProblem::setStartConfig(int idx){
        std::ifstream ifs(config_file_path_);
        if (ifs.is_open()){
            std::string line;
            for (int i=0; i<=idx; i++){
                getline(ifs, line);
            }
            start_config_ = split_f(line, ',');
            nh_.setParam("/problem/start_config", start_config_);
        }
    }

    void TormProblem::setDefaultJointPosition(){
        robot_state::RobotState state = planning_scene_->getCurrentState();
        for(uint i = 0; i < default_setting_joints_.size(); i++)
        state.setJointPositions(default_setting_joints_[i], &default_setting_values_[i]);
        planning_scene_->setCurrentState(state);
    }

    void TormProblem::setCollisionObjects(){
        std::string path = ros::package::getPath("torm");
        std::string obs_file_name;
        if(problem_name_.find("random_obs") == std::string::npos) { // semantic problem
            obs_file_name = path + "/launch/config/scene/" + problem_name_ + ".txt";
        } else {
            std::vector<std::string> sp = split(problem_name_,'_');
            obs_file_name = path + "/launch/config/scene/scene_" + sp[2] + ".txt";
        }
        std::ifstream ifs(obs_file_name);
        if(ifs.is_open()){
            std::string line;
            int count_obs = 0;
            while(getline(ifs, line)){
                std::vector<double> obs_spec = split_f(line, ' ');
                std::string obs_name = "obs" + std::to_string(count_obs);
                collision_objects_.push_back(
                        makeCollisionObject(obs_name, obs_spec[0], obs_spec[1], obs_spec[2],
                                            obs_spec[3], obs_spec[4], obs_spec[5],
                                            obs_spec[6], obs_spec[7], obs_spec[8]
                        )
                );
                count_obs++;
            }
            ifs.close();
        }
        planning_scene_interface_.applyCollisionObjects(collision_objects_);
        for(auto& kv : collision_objects_){
            planning_scene_->processCollisionObjectMsg(kv);
        }
    }

    void TormProblem::resetCollisionObjects(){
        std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map = planning_scene_interface_.getObjects();
        std::vector<moveit_msgs::CollisionObject> v_co;
        for(auto& kv : collision_objects_map){
            kv.second.operation = kv.second.REMOVE;
            planning_scene_->processCollisionObjectMsg(kv.second);
            v_co.push_back(kv.second);
        }
        planning_scene_interface_.applyCollisionObjects(v_co);
        collision_objects_.clear();
    }

    moveit_msgs::CollisionObject TormProblem::makeCollisionObject(std::string name, double x, double y, double z,
                                                                  double roll, double pitch, double yaw,
                                                                  double size_x, double size_y, double size_z){
        moveit_msgs::CollisionObject co;

        co.id = name;
        co.header.frame_id = fixed_frame_;

        co.primitives.resize(1);
        co.primitives[0].type = co.primitives[0].BOX;
        co.primitives[0].dimensions.resize(3);
        co.primitives[0].dimensions[0] = size_x;
        co.primitives[0].dimensions[1] = size_y;
        co.primitives[0].dimensions[2] = size_z;

        co.primitive_poses.resize(1);
        co.primitive_poses[0].position.x = x;
        co.primitive_poses[0].position.y = y;
        co.primitive_poses[0].position.z = z;

        tf::Quaternion q;
        q.setRPY(roll, pitch, yaw);

        co.primitive_poses[0].orientation.w = q.w();
        co.primitive_poses[0].orientation.x = q.x();
        co.primitive_poses[0].orientation.y = q.y();
        co.primitive_poses[0].orientation.z = q.z();

        co.operation = co.ADD;

        return co;
    }

    void TormProblem::setPlanningPath(){
        std::string str;
        std::string path = ros::package::getPath("torm");
        std::ifstream inFile(path + "/launch/config/path/" + problem_name_);
        while(std::getline(inFile, str, '\n')){
            std::vector<std::string> tokens;
            KDL::Frame p;
            tokens = split(str, ';');
            if (tokens.size() == 1){
                std::vector<double> buf = split_f(tokens[0], ',');
                p.p.data[0] = buf[0];
                p.p.data[1] = buf[1];
                p.p.data[2] = buf[2];
                p.M = KDL::Rotation::Quaternion(buf[4], buf[5], buf[6], buf[3]);
            }else{
                std::vector<double> buf_p = split_f(tokens[0], ',');
                std::vector<double> buf_r = split_f(tokens[1], ',');
                p.p.data[0] = buf_p[0];
                p.p.data[1] = buf_p[1];
                p.p.data[2] = buf_p[2];
                p.M = KDL::Rotation::Quaternion(buf_r[1], buf_r[2], buf_r[3], buf_r[0]);
            }
            target_poses_.push_back(p);
        }
        inFile.close();
    }

    std::string TormProblem::getProblemName(){
        return problem_name_;
    }

    bool TormProblem::getIsLoadScene(){
        return load_scene_;
    };

    bool TormProblem::getIsLoadPath(){
        return load_path_;
    };

    std::string TormProblem::getPlanningGroup(){
        return planning_group_;
    }

    std::string TormProblem::getFixedFrame(){
        return fixed_frame_;
    }

    std::string TormProblem::getBaseLink(){
        return planning_base_link_;
    }

    std::string TormProblem::getTipLink(){
        return planning_tip_link_;
    }

    std::vector<KDL::Frame> TormProblem::getTargetPoses(){
        return target_poses_;
    }

    std::vector<std::string> TormProblem::getDefaultSettingJoints(){
        return default_setting_joints_;
    }

    std::vector<double> TormProblem::getDefaultSettingValues(){
        return default_setting_values_;
    }

    std::vector<double> TormProblem::getStartConfiguration(){
        return start_config_;
    }
}