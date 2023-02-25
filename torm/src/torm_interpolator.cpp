#include <torm/torm_interpolator.h>


TormInterpolator::TormInterpolator(int n_IKcand, KDL::JntArray& start_conf,
                                   std::vector<KDL::Frame>& targetPoses, std::vector<int>& simplified_points,
                                   torm::TormIKSolver& iksolver, planning_scene::PlanningScenePtr planning_scene)
                                   :n_IKcand_(n_IKcand),
                                   start_conf_(start_conf),
                                   targetPoses_(targetPoses), simplified_points_(simplified_points),
                                   iksolver_(iksolver),
                                   planning_scene_(planning_scene){

    nh_.getParam("/robot/planning_group", planning_group_);
    nh_.getParam("/robot/ll", ll_);
    nh_.getParam("/robot/ul", ul_);
    nh_.getParam("/robot/continuous_joints", c_joints_);

    num_total_points_ = targetPoses.size();
    num_simple_points_ = simplified_points.size();
    num_joints_ = start_conf.data.size();

    trajectory_.resize(num_total_points_, num_joints_); // include start conf
    trajectory_.row(0) = start_conf.data;

    setTormInitialTraj();
}

TormInterpolator::~TormInterpolator(){
}

void TormInterpolator::setTormInitialTraj(){
    KDL::JntArray q(num_joints_);

    int start_idx = 0;
    for (int i = 0; i < num_simple_points_; i++){
        double bestCost = INFINITY;
        std::cout << "===== i: " << i << "=====" << std::endl;
        // ======= [1] collect ik candidates in rnd_confs =======
        std::vector<KDL::JntArray> rnd_confs;
        for (int j = 0; j < n_IKcand_; j++) {
            if(iksolver_.ikSolverCollFree(targetPoses_[simplified_points_[i]], q)){
                rnd_confs.push_back(q);
            }
            else if(iksolver_.ikSolver(targetPoses_[simplified_points_[i]], q)){
                rnd_confs.push_back(q);
            }else{
                ROS_ERROR("[TormInterpolator] Cannot find IK candidate.");
            }
        }

        // ======= [2] select best confiuration among random_confs =======
        for (int j = 0; j < rnd_confs.size(); j++){
            Eigen::MatrixXd trajsegment_;
            int n_seg_points = simplified_points_[i]-start_idx+1;
            trajsegment_.resize(n_seg_points, num_joints_);
            trajsegment_.row(0) = trajectory_.row(start_idx);
            trajsegment_.row(n_seg_points-1) = rnd_confs[j].data;

            fillInLinearInterpolation(trajsegment_);
            double rndCost = getEndPoseCost(trajsegment_, start_idx);
            if(bestCost < rndCost){
                continue;
            }else{
                // TODO: consider collision cost additionally.
                bestCost = rndCost;
                trajectory_.block(start_idx, 0, n_seg_points, num_joints_) = trajsegment_;
                std::cout << "cost: " << rndCost << std::endl;
            }
        }
        start_idx = simplified_points_[i];
    }
}


double TormInterpolator::getEndPoseCost(Eigen::MatrixXd& trajsegment_, int start_idx) {
    double endPose_cost = 0.0;
    KDL::JntArray q(num_joints_);
    KDL::Frame cur_pose;
    KDL::Twist delta_pose;

    int num_seg = trajsegment_.rows();

    for (int i = 1; i < num_seg; i++) {
        q.data = trajsegment_.row(i);
        iksolver_.fkSolver(q, cur_pose);
        delta_pose = diff(cur_pose, targetPoses_[start_idx+i], 1);
        endPose_cost += std::sqrt(KDL::dot(delta_pose.vel, delta_pose.vel)) +
                        std::sqrt(KDL::dot(delta_pose.rot, delta_pose.rot));
    }
    return endPose_cost;
}


void TormInterpolator::fillInLinearInterpolation(Eigen::MatrixXd& trajsegment_)
{
    double num_seg = trajsegment_.rows();
    Eigen::VectorXd st = trajsegment_.row(0);
    Eigen::VectorXd gt = trajsegment_.row(num_seg-1);

    for (int j_idx = 0; j_idx < num_joints_; j_idx++) {
        double diff = (gt(j_idx) - st(j_idx));
        if(std::find(c_joints_.begin(), c_joints_.end(), j_idx) != c_joints_.end()) { // continuous joint
            if (std::abs(diff) > M_PI) {
                if (diff < 0) {
                    diff = 2 * M_PI - std::abs(diff);
                } else {
                    diff = -2 * M_PI + std::abs(diff);
                }
            }
        }
        double delta_theta = diff / (num_seg-1);
        for (int i = 1; i < num_seg; i++) {
            trajsegment_(i, j_idx) = st(j_idx) + i * delta_theta;
        }
    }
    for (int i = 1; i < num_seg; i++) {
        KDL::JntArray buf_q(num_joints_);
        buf_q.data = trajsegment_.row(i);
        refineContinuousJoint(buf_q);
        trajsegment_.row(i) = buf_q.data;
    }
}

void TormInterpolator::refineContinuousJoint(KDL::JntArray& q){
    for(auto i : c_joints_){
        auto buf = std::fmod(q(i), 2.0 * M_PI);
        if ( buf > M_PI ){
            buf -= 2.0 * M_PI;
        }
        else if ( buf < -M_PI ){
            buf += 2.0 * M_PI;
        }
        q(i) = buf;
    }
}

void TormInterpolator::setCollisionChecker() {
    c_request_.group_name = planning_group_;
    c_request_.contacts = true;
    c_request_.max_contacts = 100;
    c_request_.max_contacts_per_pair = 1;
    c_request_.verbose = true;
    c_request_.distance = true;
    c_request_.cost = true;
}

bool TormInterpolator::collision_check(KDL::JntArray& q)
{
    std::vector<double> joint_values;
    joint_values.reserve(num_joints_);
    for(int i = 0; i < num_joints_; i++) {
        joint_values.push_back(q(i));
    }

    moveit::core::RobotState& current_state = planning_scene_->getCurrentStateNonConst();
    const moveit::core::JointModelGroup* joint_model_group = current_state.getJointModelGroup(planning_group_);
    current_state.setJointGroupPositions(joint_model_group, joint_values);
    if(!current_state.satisfiesBounds(joint_model_group)){
        return true; // out of joint limit
    }

    c_result_.clear();
    planning_scene_->checkCollision(c_request_, c_result_);
    return c_result_.collision; // col
}