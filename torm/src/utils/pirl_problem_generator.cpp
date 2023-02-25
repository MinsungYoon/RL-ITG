#include <interpolation/pirl_problem_generator.h>

template <typename t>
t clamp2(t x, t min, t max)
{
    if (x < min) x = min;
    if (x > max) x = max;
    return x;
}

PirlProblemGenerator::PirlProblemGenerator(int n_samples, std::string planning_group, planning_scene::PlanningScenePtr planning_scene, torm::TormIKSolver& iksolver)
:n_samples_(n_samples), planning_group_(planning_group), planning_scene_(planning_scene), iksolver_(iksolver){

    EEPoses_.reserve(n_samples);
    Joints_.reserve(n_samples);

    setCollisionChecker();

    nh_.getParam("/robot/n_dof", n_dof_);
    nh_.getParam("/robot/fk_base_position", fk_base_position_);
    nh_.getParam("/robot/sample_range", sample_range_);

    std::random_device rd;
    mersenne_ = std::mt19937(rd());

    p_add_rot_ = std::uniform_real_distribution<double>(0.0, 1.0);

    rnd_x_ = std::uniform_real_distribution<double>(0.6, 1.0);
    rnd_y_ = std::uniform_real_distribution<double>(-0.7, 0.7);
    rnd_z_ = std::uniform_real_distribution<double>(0.4, 1.1);
    rnd_yaw_ = std::normal_distribution<double>(0, 1.3);
    rnd_pitch_ = std::normal_distribution<double>(0.298, 0.572); // m: 40 deg, std: 50 deg
    rnd_roll_ = std::uniform_real_distribution<double>(-3.1415926535897, 3.1415926535897);


    collectEEPoses();
}

PirlProblemGenerator::~PirlProblemGenerator(){
}

void PirlProblemGenerator::setCollisionChecker() {
    c_request_.group_name = planning_group_;
    c_request_.contacts = true;
    c_request_.max_contacts = 100;
    c_request_.max_contacts_per_pair = 1;
    c_request_.verbose = false;
    c_request_.distance = false;
    c_request_.cost = false;
}

bool PirlProblemGenerator::collision_check(KDL::JntArray& rnd_q)
{
    std::vector<double> joint_values;
    joint_values.reserve(n_dof_);
    for(int i = 0; i < n_dof_; i++) {
        joint_values.push_back(rnd_q(i));
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

void PirlProblemGenerator::collectEEPoses(){
    while(EEPoses_.size()<n_samples_) {

//        if(EEPoses_.size()<n_samples_/3){ // forward sample (joint->ee)
        if(EEPoses_.size()<0){ // forward sample (joint->ee)
            KDL::JntArray rnd_q(n_dof_);
            iksolver_.getRandomConfiguration(rnd_q);
            if (!collision_check(rnd_q)) {
                Joints_.push_back(rnd_q);
                KDL::Frame ee;
                iksolver_.fkSolver(rnd_q, ee);

                std::vector<double> p;
                p.reserve(7);
                p.push_back(ee.p.x());
                p.push_back(ee.p.y());
                p.push_back(ee.p.z());
                double x, y, z, w;
                ee.M.GetQuaternion(x, y, z, w);
                p.push_back(x);
                p.push_back(y);
                p.push_back(z);
                p.push_back(w);

                EEPoses_.push_back(p);
            }
        }else{ // inverse sample (ee->joint)
            KDL::Frame goal_ee;
            while(true){
                double sample_x = rnd_x_(mersenne_);
                double sample_y = rnd_y_(mersenne_);
                double sample_z = rnd_z_(mersenne_);
                goal_ee.p[0] = clamp2<double>(sample_x, sample_range_[0], sample_range_[1]) - fk_base_position_[0];
                goal_ee.p[1] = clamp2<double>(sample_y, sample_range_[2], sample_range_[3]) - fk_base_position_[1];
                goal_ee.p[2] = clamp2<double>(sample_z, sample_range_[4], sample_range_[5]) - fk_base_position_[2];
                double roll = clamp2<double>(rnd_roll_(mersenne_), sample_range_[6], sample_range_[7]);
                double pitch = clamp2<double>(rnd_pitch_(mersenne_), sample_range_[8], sample_range_[9]);
                double yaw = clamp2<double>(rnd_yaw_(mersenne_), sample_range_[10], sample_range_[11]);
                goal_ee.M = KDL::Rotation::RPY(roll, pitch, yaw);
                KDL::JntArray conf_goal(n_dof_);
                if(iksolver_.ikSolverCollFree(goal_ee, conf_goal)){
                    Joints_.push_back(conf_goal);

                    std::vector<double> p;
                    p.reserve(7);
                    p.push_back(goal_ee.p[0]);
                    p.push_back(goal_ee.p[1]);
                    p.push_back(goal_ee.p[2]);
                    double x, y, z, w;
                    goal_ee.M.GetQuaternion(x, y, z, w);
                    p.push_back(x);
                    p.push_back(y);
                    p.push_back(z);
                    p.push_back(w);

                    EEPoses_.push_back(p);

                    if(  p_add_rot_(mersenne_) < 0.3 && EEPoses_.size() < n_samples_){
                        KDL::Frame goal_ee2;
                        goal_ee2.p[0] = goal_ee.p[0];
                        goal_ee2.p[1] = goal_ee.p[1];
                        goal_ee2.p[2] = goal_ee.p[2];
                        double roll2 = clamp2<double>(rnd_roll_(mersenne_), sample_range_[6], sample_range_[7]);
                        double pitch2 = clamp2<double>(rnd_pitch_(mersenne_), sample_range_[8], sample_range_[9]);
                        double yaw2 = clamp2<double>(rnd_yaw_(mersenne_), sample_range_[10], sample_range_[11]);
                        goal_ee2.M = KDL::Rotation::RPY(roll2, pitch2, yaw2);
                        KDL::JntArray conf_goal2(n_dof_);
                        if(iksolver_.ikSolverCollFree(goal_ee2, conf_goal2)){
                            Joints_.push_back(conf_goal2);

                            std::vector<double> p2;
                            p2.reserve(7);
                            p2.push_back(goal_ee2.p[0]);
                            p2.push_back(goal_ee2.p[1]);
                            p2.push_back(goal_ee2.p[2]);
                            double x2, y2, z2, w2;
                            goal_ee2.M.GetQuaternion(x2, y2, z2, w2);
                            p2.push_back(x2);
                            p2.push_back(y2);
                            p2.push_back(z2);
                            p2.push_back(w2);

                            EEPoses_.push_back(p2);
                        }
                    }

                    break;
                }

            }
        }
        std::cout<< "EE size: " << EEPoses_.size() << " / " << n_samples_ << std::endl;
    }
}

std::vector<KDL::JntArray>& PirlProblemGenerator::get_Joints() {
    return Joints_;
}
std::vector<std::vector<double>>& PirlProblemGenerator::get_EEPoses(){
    return EEPoses_;
}

int PirlProblemGenerator::getTargetEEN(){
    return path_total_n_;
}

double PirlProblemGenerator::getTargetEELen(){
    return path_total_len_;
}

bool PirlProblemGenerator::sampleTargetEEList(int n_wps, double interval_length,
                                              std::vector<std::vector<double>>& path,
                                              std::vector<std::vector<double>>& rpypath){
    std::vector<std::vector<double>> test_wps; // require at least 5 waypoints.
    test_wps.reserve(n_wps);
    for(int i=0; i<n_wps; i++){
        test_wps.push_back(EEPoses_[rand()%EEPoses_.size()]);
    }
    interpolator6D ITP6D(test_wps, interval_length);
    path_total_n_ = ITP6D.get_total_n();
    path_total_len_ = ITP6D.get_total_len();
//    std::cout << "[Total] # node: " << path_total_n_ << ", len: " << path_total_len_ << std::endl;
    path = ITP6D.get_result();
    rpypath = ITP6D.get_rpy_result();
//    ITP6D.print();
//    ITP6D.rpy_print();
    return checkExistIK(path);
}

bool PirlProblemGenerator::sampleTargetEEList(int n_wps, double interval_length,
                                              std::vector<std::vector<double>>& path){
    std::vector<std::vector<double>> test_wps; // require at least 5 waypoints.
    test_wps.reserve(n_wps);
    for(int i=0; i<n_wps; i++){
        test_wps.push_back(EEPoses_[rand()%EEPoses_.size()]);
    }
    interpolator6D ITP6D(test_wps, interval_length);
    path_total_n_ = ITP6D.get_total_n();
    path_total_len_ = ITP6D.get_total_len();
    path = ITP6D.get_result();
    return checkExistIK(path);
}

bool PirlProblemGenerator::checkExistIK(std::vector<std::vector<double>>& path){
    for(int i=0; i<path.size(); i++){
        KDL::JntArray q(n_dof_);
        KDL::Frame f;
        f.p[0] = path[i][0];
        f.p[1] = path[i][1];
        f.p[2] = path[i][2];
        f.M = KDL::Rotation::Quaternion(path[i][3], path[i][4], path[i][5], path[i][6]);

        if(!iksolver_.ikSolverCollFree(f, q, 100)){
//            ROS_WARN_STREAM("[PirlProblemGenerator::checkExistIK] cannot find colfree ik. fail at " << i << "/" << path.size());
            return false;
        }
    }
    return true;
}
