#include <pirl/collision_checker.h>
#include <time.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/LinearMath/Matrix3x3.h>

static inline std::vector<double> split_f(std::string str, char delimiter) {
    std::vector<double> internal;
    std::stringstream ss(str);
    std::string temp;

    while (getline(ss, temp, delimiter)) {
        internal.push_back(std::atof(temp.c_str()));
    }

    return internal;
}

CollisionCheckSrv::CollisionCheckSrv(planning_scene::PlanningScenePtr planning_scene)
: planning_scene_(planning_scene)
{
    nh_.getParam("/robot/planning_group", planning_group_);
    nh_.getParam("/robot/fixed_frame", fixed_frame_);
    nh_.getParam("/robot/n_dof", n_dof_);

    setCollisionChecker();
    collision_service_ = nh_.advertiseService("collision_check", &CollisionCheckSrv::collisionCheck_srv, this);
    scene_set_service_ = nh_.advertiseService("scene_set", &CollisionCheckSrv::setCollisionObjects_srv, this);
    scene_reset_service_ = nh_.advertiseService("scene_reset", &CollisionCheckSrv::resetCollisionObjects_srv, this);

}

CollisionCheckSrv::~CollisionCheckSrv(){
}

void CollisionCheckSrv::setCollisionChecker() {
    c_request_.group_name = planning_group_;
}

bool CollisionCheckSrv::collisionCheck_srv(pirl_msgs::collision::Request  &req,
                                           pirl_msgs::collision::Response &res)
{
    moveit::core::RobotState& current_state = planning_scene_->getCurrentStateNonConst();

    std::vector<double> joint_values(n_dof_, 0.0);
    for(int i = 0; i < n_dof_; i++) {
        joint_values[i] = req.query_config[i];
    }
    current_state.setJointGroupPositions(planning_group_, joint_values);
    if(!current_state.getJointModelGroup(planning_group_)->satisfiesPositionBounds(joint_values.data())){
        ROS_ERROR_STREAM("[CC] Joint_limit_violation. this shouldn't be called.");
        res.collision_result = true;
        res.result = true;
        return true;
    }

    c_result_.clear();
    planning_scene_->checkCollision(c_request_, c_result_, current_state);
    res.collision_result = c_result_.collision;
    res.result = true;
    return true;
}

bool CollisionCheckSrv::setCollisionObjects_srv(pirl_msgs::scene_set::Request  &req,
                                                pirl_msgs::scene_set::Response &res){
    std::string obs_file_name = "/data/torm_data/obs/scene/scene_" + std::to_string(req.scene_num) + ".txt";
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
    }else{
        ROS_ERROR_STREAM("[CC] Cannot open scene file.");
    }
    planning_scene_interface_.applyCollisionObjects(collision_objects_);
    for(auto& kv : collision_objects_){
        planning_scene_->processCollisionObjectMsg(kv);
    }

    res.result = true;
    return true;
}

bool CollisionCheckSrv::resetCollisionObjects_srv(pirl_msgs::scene_reset::Request  &req,
                                                  pirl_msgs::scene_reset::Response &res){
    std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map = planning_scene_interface_.getObjects();
    std::vector<moveit_msgs::CollisionObject> v_co;
    for(auto& kv : collision_objects_map){
        kv.second.operation = kv.second.REMOVE;
        planning_scene_->processCollisionObjectMsg(kv.second);
        v_co.push_back(kv.second);
    }
    planning_scene_interface_.applyCollisionObjects(v_co);
    collision_objects_.clear();

    res.result = true;
    return true;
}

moveit_msgs::CollisionObject CollisionCheckSrv::makeCollisionObject(  std::string name, double x, double y, double z,
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

moveit_msgs::ObjectColor CollisionCheckSrv::makeColorObject(std::string name, std::string color){

    moveit_msgs::ObjectColor oc;
    oc.id = name;

    std_msgs::ColorRGBA cl;
    if(color=="SILVER"){ // SILVER
        cl.r = 218.0/255;
        cl.g = 218.0/255;
        cl.b = 218.0/255;
        cl.a = 1.0;
    }
    else if(color=="WHITE"){ // WHITE
        cl.r = 200.0/255;
        cl.g = 200.0/255;
        cl.b = 200.0/255;
        cl.a = 1.0;
    }
    else if(color=="BLACK"){ // BLACK
        cl.r = 0.0/255;
        cl.g = 0.0/255;
        cl.b = 0.0/255;
        cl.a = 1.0;
    }
    oc.color = cl;

    return oc;
}

//    clock_t start, end;
//    start = clock();
//    end = clock();
//    double t_res = (double)((end - start)/CLOCKS_PER_SEC);
//    ROS_ERROR_STREAM("col processing time (ms): "<<t_res);


//void CollisionCheckSrv::setCollisionObjects(){
//    XmlRpc::XmlRpcValue obs;
//
//    nh_.getParam("/problem/obstacles", obs);
//    for(uint i = 0; i < obs.size(); i++){
//        std::string obs_name = "obs" + std::to_string(i);
//
//        collision_objects_.push_back(makeCollisionObject(obs_name, double(obs[i][0]["x"]), double(obs[i][1]["y"]), double(obs[i][2]["z"]),
//                                                         double(obs[i][3]["roll"]), double(obs[i][4]["pitch"]), double(obs[i][5]["yaw"]),
//                                                         double(obs[i][6]["size_x"]), double(obs[i][7]["size_y"]), double(obs[i][8]["size_z"])));
//        color_objects_.push_back( makeColorObject(obs_name, obs[i][9]["color"]) );
//        }
//}

//if (color == "BLACK"){cl.r = 0; cl.g = 0; cl.b = 0;}
//else if(color == "WHITE"){cl.r = 255; cl.g = 255; cl.b = 255;}
//else if(color == "RED"){cl.r = 255; cl.g = 0; cl.b = 0;}
//else if(color == "LIME"){cl.r = 0; cl.g = 255; cl.b = 0;}
//else if(color == "BLUE"){cl.r = 0; cl.g = 0; cl.b = 255;}
//else if(color == "YELLOW"){cl.r = 255; cl.g = 255; cl.b = 0;}
//else if(color == "CYAN"){cl.r = 0; cl.g = 255; cl.b = 255;}
//else if(color == "GRAY"){cl.r = 128; cl.g = 128; cl.b = 128;}
//else if(color == "DARK_SLATE_GRAY"){cl.r = 47; cl.g = 79; cl.b = 79;}
//else if(color == "BROWN"){cl.r = 165; cl.g = 42; cl.b = 42;}
//else if(color == "MAROON"){cl.r = 128; cl.g = 0; cl.b = 0;}
//else if(color == "OLIVE"){cl.r = 128; cl.g = 128; cl.b = 0;}
//else if(color == "NAVY"){cl.r = 0; cl.g = 0; cl.b = 128;}
//else if(color == "SILVER"){cl.r = 192; cl.g = 192; cl.b = 192;}
//else if(color == "GOLDEN_ROD"){cl.r = 218; cl.g = 165; cl.b = 32;}