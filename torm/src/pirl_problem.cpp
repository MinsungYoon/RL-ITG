/* Author: Mincheul Kang */

#include <torm/pirl_problem.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Matrix3x3.h>

#include <fstream>

PirlProblem::PirlProblem(std::string robot_name, planning_scene::PlanningScenePtr& planning_scene):planning_scene_{planning_scene}{

    std::string path = ros::package::getPath("torm");
    system(std::string("rosparam load " + path + "/launch/config/robot/" + robot_name + ".yaml").c_str());

    nh_.getParam("/robot/planning_group", planning_group_);
    nh_.getParam("/robot/fixed_frame", fixed_frame_);
    nh_.getParam("/robot/planning_base_link", planning_base_link_);
    nh_.getParam("/robot/planning_tip_link", planning_tip_link_);
    nh_.getParam("/problem/default_setting_joints", default_setting_joints_);
    nh_.getParam("/problem/default_setting_values", default_setting_values_);

    resetCollisionObjects();
    setDefaultJointPosition();
}

PirlProblem::~PirlProblem() {
}

void PirlProblem::setDefaultJointPosition(){
    robot_state::RobotState state = planning_scene_->getCurrentState();
    for(uint i = 0; i < default_setting_joints_.size(); i++)
        state.setJointPositions(default_setting_joints_[i], &default_setting_values_[i]);
    planning_scene_->setCurrentState(state);
}

void PirlProblem::setCollisionObjects(std::string obs_file_name){
    std::ifstream ifs(obs_file_name);
    if(ifs.is_open()){
        std::string line;
        int count_obs = 0;
        while(getline(ifs, line)){
            std::vector<double> obs_spec = torm::split_f(line, ' ');

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

void PirlProblem::resetCollisionObjects(){
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

moveit_msgs::CollisionObject PirlProblem::makeCollisionObject(std::string name, double x, double y, double z,
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

std::string PirlProblem::getPlanningGroup(){
    return planning_group_;
}

std::string PirlProblem::getFixedFrame(){
    return fixed_frame_;
}

std::string PirlProblem::getBaseLink(){
    return planning_base_link_;
}

std::string PirlProblem::getTipLink(){
    return planning_tip_link_;
}
