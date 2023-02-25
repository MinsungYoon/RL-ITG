#include <pirl/kinematic_solver.h>
#include <ros/ros.h>
#include <time.h>

#include <memory>

void printMatrix(KDL::Jacobian& J_cur){
    for(int i=0; i<J_cur.rows(); i++){
        std::cout << "[ ";
        for (int j = 0; j < J_cur.columns(); j++)
            std::cout << std::setw(11) << J_cur.data(i,j) << ", ";
        std::cout << " ]" << std::endl;
    }
}

void printMatrix(Eigen::MatrixXd & J_cur){
    for(int i=0; i<J_cur.rows(); i++){
        std::cout << "[ ";
        for (int j = 0; j < J_cur.cols(); j++)
            std::cout << std::setw(11) << J_cur(i, j) << ", ";
        std::cout << " ]" << std::endl;
    }
}

//    clock_t start, end;
//    start = clock();
//    end = clock();
//    double t_res = (double)((end - start)/CLOCKS_PER_SEC);
//    ROS_ERROR_STREAM("ik processing time (ms): "<<t_res);
//    std::cout<<req.query_config[0]<<", "<<req.query_config[1]<<", "<<req.query_config[2]<<", "<<req.query_config[3]<<", "<<req.query_config[4]<<", "<<req.query_config[5]<<", "<<req.query_config[6]<<std::endl;

KinematicsSolverSrv::KinematicsSolverSrv(planning_scene::PlanningScenePtr planning_scene):planning_scene_(planning_scene){

    nh_.getParam("/robot/planning_group", planning_group_);
    nh_.getParam("/robot/planning_base_link", base_link_);
    nh_.getParam("/robot/planning_tip_link", tip_link_);

    tracik_solver_.reset(new TRAC_IK::TRAC_IK(base_link_, tip_link_, "/robot_description", 0.001, 1e-5));

    setCollisionChecker();

    bool valid = tracik_solver_->getKDLChain(chain_);
    valid = tracik_solver_->getKDLLimits(ll_, ul_);

    n_dof_ = chain_.getNrOfJoints();
    n_seg_ = chain_.getNrOfSegments();

    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    vik_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_); // PseudoInverse vel solver
    jac_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(chain_);

    max_tried_ = 30;

    fksolver_service_ = nh_.advertiseService("fk_solver", &KinematicsSolverSrv::fkSolver_srv, this);
    allLinkfksolver_service_ = nh_.advertiseService("allLinkfk_solver", &KinematicsSolverSrv::allLinkfkSolver_srv, this);
    iksolver_service_ = nh_.advertiseService("ik_solver", &KinematicsSolverSrv::ikSolver_srv, this);
    rndsg_service_ = nh_.advertiseService("rndsg", &KinematicsSolverSrv::rndStartAndGoal_srv, this);
    rndvalconf_service_ = nh_.advertiseService("rndvalconf", &KinematicsSolverSrv::rndValidSample_srv, this);

    jaco_service_ = nh_.advertiseService("jaco_reward", &KinematicsSolverSrv::jaco_srv, this);

    printJointLimitInfo();
}

void KinematicsSolverSrv::printJointLimitInfo() {
    ROS_WARN("[===INFO===] Joint limit (ll, ul): ");
    for(uint j = 0; j < n_dof_; j++){
        ROS_WARN_STREAM(ll_(j)<<", "<<ul_(j));
    }
}

void KinematicsSolverSrv::setCollisionChecker() {
    c_request_.group_name = planning_group_;
}

void KinematicsSolverSrv::getRandomConfiguration(KDL::JntArray& q){
    for(uint j = 0; j < n_dof_; j++){
        q(j) = fRand(ll_(j), ul_(j));
    }
}

double KinematicsSolverSrv::fRand(double min, double max) const {
    double f = (double)rand() / RAND_MAX;
    if(max > 2*M_PI){
        return -M_PI + f * (2*M_PI);
    }
    return min + f * (max - min);
}

double KinematicsSolverSrv::fRand(int i) const {
    double f = (double)rand() / RAND_MAX;
    if(ul_(i) > 2*M_PI){
        return -M_PI + f * (2*M_PI);
    }
    return ll_(i) + f * (ul_(i) - ll_(i));
}

bool KinematicsSolverSrv::collisionChecking(std::vector<double> values) {
    c_result_.clear();
    robot_state::RobotState state = planning_scene_->getCurrentState();
    state.setJointGroupPositions(planning_group_, values);
    planning_scene_->checkCollision(c_request_, c_result_, state);
    return c_result_.collision;
}

uint KinematicsSolverSrv::getDoF(){
    return n_dof_;
}

void KinematicsSolverSrv::fkSolver(const KDL::JntArray& q_init, KDL::Frame& p_in) {
    fk_solver_->JntToCart(q_init, p_in);
}

void KinematicsSolverSrv::allLinkfkSolver(const KDL::JntArray& q_init, std::vector<KDL::Frame>& p_in) {
    fk_solver_->JntToCart(q_init, p_in);
//    std::cout << chain_.getNrOfSegments() << ", " << chain_.getNrOfJoints() << std::endl;
//    std::cout << chain_.getSegment(0).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(1).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(2).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(3).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(4).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(5).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(6).getName() << ", " << std::endl;
//    std::cout << chain_.getSegment(7).getName() << ", " << std::endl;
}


bool KinematicsSolverSrv::ikSolver(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out) {
    int rc = tracik_solver_->CartToJnt(q_init, p_in, q_out);
    if(rc < 0){
        return false;
    }
    return true;
}

bool KinematicsSolverSrv::ikSolver(const KDL::Frame& p_in, KDL::JntArray& q_out) {
    int rc = -1;
    KDL::JntArray q_c(n_dof_);

    int tried = 0;
    while(rc < 0){
        for(uint j = 0; j < n_dof_; j++){
            q_c(j) = fRand(ll_(j), ul_(j));
        }
        rc = tracik_solver_->CartToJnt(q_c, p_in, q_out);
        tried++;
        if(tried >= max_tried_ && rc < 0){
            return false;
        }
    }
    return true;
}

bool KinematicsSolverSrv::ikSolverCollFree(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out) {
    int rc = tracik_solver_->CartToJnt(q_init, p_in, q_out);
    if(rc < 0){
        return false;
    }
    std::vector<double> conf(n_dof_);
    for(uint j = 0; j < n_dof_; j++){
        conf[j] = q_out(j);
    }
    if(collisionChecking(conf)){
        return false;
    }
    else{
        return true;
    }
}

bool KinematicsSolverSrv::ikSolverCollFree(const KDL::Frame& p_in, KDL::JntArray& q_out) {
    int rc = -1;
    int tried = 0;

    KDL::JntArray q_c(n_dof_);
    std::vector<double> conf(n_dof_);
    while(rc < 0){
        for(uint j = 0; j < n_dof_; j++){
            q_c(j) = fRand(ll_(j), ul_(j));
        }
        rc = tracik_solver_->CartToJnt(q_c, p_in, q_out);
        if(rc >= 0){
            for(uint j = 0; j < n_dof_; j++){
                conf[j] = q_out(j);
            }
            if(collisionChecking(conf)){
                rc = -1;
            }
        }
        tried++;
        if(tried >= max_tried_ && rc < 0){
            return false;
        }
    }
    return true;
}

void KinematicsSolverSrv::vikSolver(const KDL::JntArray& q, const KDL::Twist& delta_twist, KDL::JntArray& delta_q){
    vik_solver_->CartToJnt(q, delta_twist, delta_q);
}

void KinematicsSolverSrv::getJacobian(const KDL::JntArray& q, KDL::Jacobian& jac){
    jac_solver_->JntToJac(q, jac);
}

bool KinematicsSolverSrv::allLinkfkSolver_srv(pirl_msgs::allfk::Request  &req,  // q_(1...N)
                                              pirl_msgs::allfk::Response &res){ // all link positions [px,py,pz,qx,qy,qz,qw]

    KDL::JntArray q_c(n_dof_);
    for(uint j = 0; j < n_dof_; j++) {
        q_c(j) = req.query_config[j];
    }
    std::vector<KDL::Frame> poses;
    poses.reserve(n_seg_);
    for(int i=0; i<n_seg_; i++){
        poses.push_back(KDL::Frame());
    }
    allLinkfkSolver(q_c, poses);

    res.allfk_result.reserve(7*n_seg_);
    for(int i=0; i<n_seg_; i++){
        res.allfk_result.push_back(poses[i].p.x());
        res.allfk_result.push_back(poses[i].p.y());
        res.allfk_result.push_back(poses[i].p.z());
        double x,y,z,w;
        poses[i].M.GetQuaternion(x,y,z,w);
        res.allfk_result.push_back(x);
        res.allfk_result.push_back(y);
        res.allfk_result.push_back(z);
        res.allfk_result.push_back(w);
    }
    res.result = true;
    return true;
}
//std::cout << v10*v21-v20*v11 << ", " << -(v00*v21-v20*v01) << ", " << v00*v11-v10*v01 << std::endl;

bool KinematicsSolverSrv::fkSolver_srv(pirl_msgs::fk::Request  &req,  // q_(1...N)
                                       pirl_msgs::fk::Response &res){ // x, y, z, quat(x,y,z,w)
    KDL::JntArray q_init(n_dof_);
    KDL::Frame p_in;
    for(uint j = 0; j < n_dof_; j++) {
        q_init(j) = req.query_config[j];
    }
    fkSolver(q_init, p_in);

    double x, y, z, w;
    p_in.M.GetQuaternion(x, y, z, w);

    res.fk_result.resize(7, 0.0);
    res.fk_result[0] = p_in.p[0];
    res.fk_result[1] = p_in.p[1];
    res.fk_result[2] = p_in.p[2];
    res.fk_result[3] = x;
    res.fk_result[4] = y;
    res.fk_result[5] = z;
    res.fk_result[6] = w;

    res.result = true;

    return true;
}

bool KinematicsSolverSrv::ikSolver_srv(pirl_msgs::ik::Request  &req,  // x, y, z, quat(x,y,z,w)
                                       pirl_msgs::ik::Response &res){ // q_(1...N)
    KDL::Frame p_in;
    KDL::Vector p(req.query_config[0], req.query_config[1], req.query_config[2]);
    KDL::Rotation M = KDL::Rotation::Quaternion(req.query_config[3],req.query_config[4],req.query_config[5],req.query_config[6]);
    p_in.p = p;
    p_in.M = M;

    KDL::JntArray q_out(n_dof_);
    bool result = ikSolverCollFree(p_in, q_out);

    if(result){
        res.ik_result.resize(n_dof_, 0.0);
        for(uint j = 0; j < n_dof_; j++){
            res.ik_result[j] = q_out(j);
        }
        res.result = true;
        return true;
    }else{
        res.result = false;
        return true;
    }
}

// TODO
//bool KinematicsSolverSrv::best_ikSolver_srv(pirl_msgs::bestik::Request  &req,  // x, y, z, quat(x,y,z,w) and q_prev(1...N)
//                                            pirl_msgs::bestik::Response &res){ // q_(1...N)
//    KDL::Frame p_in;
//    KDL::Vector p(req.query_config[0], req.query_config[1], req.query_config[2]);
//    KDL::Rotation M = KDL::Rotation::Quaternion(req.query_config[3],req.query_config[4],req.query_config[5],req.query_config[6]);
//    p_in.p = p;
//    p_in.M = M;
//
//    std::vector<double> q_prev(n_dof_);
//    for(uint j = 0; j < n_dof_; j++) {
//        q_prev[j] = req.q_prev[j];
//    }
//    std::vector<double> q_best(n_dof_);
//    bool is_found_solution = false;
//
//    for(uint k=0; k<100; k++){
//        KDL::JntArray q_out(n_dof_);
//        bool result = ikSolverCollFree(p_in, q_out);
//
//        if(result){
//            if(!is_found_solution){
//                is_found_solution = true;
//                for(uint j = 0; j < n_dof_; j++){
//                    q_best[j] = q_out(j);
//                }
//            }else{
//
//            }
//        }
//    }
//    res.ik_result.resize(n_dof_, 0.0);
//    for(uint j = 0; j < n_dof_; j++){
//        res.ik_result[j] = q_out(j);
//    }
//
//    if(is_found_solution){
//        res.result = true;
//        return true;
//    }else{
//        res.result = false;
//        return true;
//    }
//}

bool KinematicsSolverSrv::jaco_srv( pirl_msgs::jaco::Request  &req,
                                    pirl_msgs::jaco::Response &res){
    KDL::JntArray q_diff(n_dof_);
    KDL::JntArray q_cur(n_dof_);
    for(uint j = 0; j < n_dof_; j++) {
        q_diff(j) = req.err_config[j];
        q_cur(j) = req.cur_config[j];
    }
    KDL::Jacobian J_cur(n_dof_);
    int result = jac_solver_->JntToJac(q_cur, J_cur);
    printMatrix(J_cur);

    Eigen::MatrixXd M_Identity(n_dof_, n_dof_);
    M_Identity.setIdentity();

    Eigen::MatrixXd null_space_proj_M = M_Identity - J_cur.data.transpose() * (J_cur.data * J_cur.data.transpose()).inverse() * J_cur.data;
    auto q_null_diff = null_space_proj_M * q_diff.data;
    res.objective_value = q_null_diff.norm();
    return true;
}


bool KinematicsSolverSrv::rndValidSample_srv( pirl_msgs::rndvalconf::Request  &req,
                                              pirl_msgs::rndvalconf::Response &res){
    double min_x, max_x, min_y, max_y, min_z, max_z, min_roll, max_roll, min_pitch, max_pitch, min_yaw, max_yaw;
    min_x = req.sample_range[0];
    max_x = req.sample_range[1];
    min_y = req.sample_range[2];
    max_y = req.sample_range[3];
    min_z = req.sample_range[4];
    max_z = req.sample_range[5];
    min_roll = req.sample_range[6];
    max_roll = req.sample_range[7];
    min_pitch = req.sample_range[8];
    max_pitch = req.sample_range[9];
    min_yaw = req.sample_range[10];
    max_yaw = req.sample_range[11];

    KDL::Frame goal_ee;
    while(true){
        goal_ee.p[0] = min_x + ((double)rand()/RAND_MAX)*(max_x - min_x);
        goal_ee.p[1] = min_y + ((double)rand()/RAND_MAX)*(max_y - min_y);
        goal_ee.p[2] = min_z + ((double)rand()/RAND_MAX)*(max_z - min_z);
        double roll = min_roll + ((double)rand()/RAND_MAX)*(max_roll - min_roll);
        double pitch = min_pitch + ((double)rand()/RAND_MAX)*(max_pitch - min_pitch);
        double yaw = min_yaw + ((double)rand()/RAND_MAX)*(max_yaw - min_yaw);
        goal_ee.M = KDL::Rotation::RPY(roll, pitch, yaw);
        KDL::JntArray conf_goal(n_dof_);
        if(ikSolverCollFree(goal_ee, conf_goal)) {
            res.configuration.resize(n_dof_, 0.0);
            for(uint j = 0; j < n_dof_; j++){
                res.configuration[j] = conf_goal(j);
            }
            res.ee_pose.resize(7,0.0);
            res.ee_pose[0] = goal_ee.p[0];
            res.ee_pose[1] = goal_ee.p[1];
            res.ee_pose[2] = goal_ee.p[2];
            double x, y, z, w;
            goal_ee.M.GetQuaternion(x, y, z, w);
            res.ee_pose[3] = x;
            res.ee_pose[4] = y;
            res.ee_pose[5] = z;
            res.ee_pose[6] = w;
            res.result = true;
            return true;
        }
    }
}


bool KinematicsSolverSrv::rndStartAndGoal_srv( pirl_msgs::rndsg::Request  &req,
                                               pirl_msgs::rndsg::Response &res){
    double min_x, max_x, min_y, max_y, min_z, max_z, min_angle, max_angle;
    double range;
    double ang_range;
    min_x = req.sample_range[0];
    max_x = req.sample_range[1];
    min_y = req.sample_range[2];
    max_y = req.sample_range[3];
    min_z = req.sample_range[4];
    max_z = req.sample_range[5];
    min_angle = req.sample_range[6];
    max_angle = req.sample_range[7];
    range = req.sample_range[8];
    ang_range = req.sample_range[9];

    while(true){
        KDL::JntArray q_init(n_dof_);
        getRandomConfiguration(q_init);
        std::vector<double> conf(n_dof_,0.0);
        for(uint j = 0; j < n_dof_; j++){
            conf[j] = q_init(j);
        }
        if(!collisionChecking(conf)){
            KDL::Frame p_ee;
            fkSolver(q_init, p_ee);

            if( min_x > p_ee.p[0] || max_x < p_ee.p[0] ||
                min_y > p_ee.p[1] || max_y < p_ee.p[1] ||
                min_z > p_ee.p[2] || max_z < p_ee.p[2] ){
                continue;
            }
            double roll, pitch, yaw;
            p_ee.M.GetRPY(roll, pitch, yaw);
            if( pitch < min_angle || pitch > max_angle ||
                yaw < min_angle || yaw > max_angle ){
                continue;
            }

            KDL::Frame p_goal;
            for(uint k=0; k<20; k++){
                p_goal.p[0] = p_ee.p[0] + (((double)rand()/RAND_MAX)*2-1) * range;
                p_goal.p[1] = p_ee.p[1] + (((double)rand()/RAND_MAX)*2-1) * range;
                p_goal.p[2] = p_ee.p[2] + (((double)rand()/RAND_MAX)*2-1) * range;
                p_goal.M = KDL::Rotation::RPY(roll + (((double)rand()/RAND_MAX)*2-1) * ang_range,
                                              pitch + (((double)rand()/RAND_MAX)*2-1) * ang_range,
                                              yaw + (((double)rand()/RAND_MAX)*2-1) * ang_range);
                KDL::JntArray q_goal(n_dof_);
                if(ikSolverCollFree(p_goal, q_goal)){
                    res.start_conf.resize(n_dof_, 0.0);
                    for(uint j = 0; j < n_dof_; j++){
                        res.start_conf[j] = q_init(j);
                    }
                    res.goal_pos.resize(7,0.0);
                    res.goal_pos[0] = p_goal.p[0];
                    res.goal_pos[1] = p_goal.p[1];
                    res.goal_pos[2] = p_goal.p[2];
                    double x, y, z, w;
                    p_goal.M.GetQuaternion(x, y, z, w);
                    res.goal_pos[3] = x;
                    res.goal_pos[4] = y;
                    res.goal_pos[5] = z;
                    res.goal_pos[6] = w;
                    res.result = true;
                    return true;
                }
            }
        }
    }
}
