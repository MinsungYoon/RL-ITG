#include <torm/traj_evaluator.h>

using namespace torm;

traj_evaluator::traj_evaluator(std::vector<KDL::Frame> targetPoses, std::vector<std::vector<double>>& trajectory,
                               std::vector<int> c_joint, double discretization,
                               torm::TormIKSolver& iksolver, std::vector<double>& vel_limit)
:targetPoses_(targetPoses), c_joint_(c_joint), discretization_(discretization), iksolver_(iksolver), vel_limit_(vel_limit){
    int num_points = trajectory.size();
    int num_joints = trajectory[0].size();

    Eigen::MatrixXd traj;
    traj.resize(num_points, num_joints);
    for(int i=0; i<num_points; i++){
        for(int j=0; j<num_joints; j++){
            traj(i,j) = trajectory[i][j];
        }
    }
    trajectory_ = traj;

    init();
}

traj_evaluator::traj_evaluator(std::vector<KDL::Frame> targetPoses, Eigen::MatrixXd trajectory,
                               std::vector<int> c_joint, double discretization,
                               torm::TormIKSolver& iksolver, std::vector<double>& vel_limit)
:targetPoses_(targetPoses), trajectory_(trajectory), c_joint_(c_joint), discretization_(discretization), iksolver_(iksolver), vel_limit_(vel_limit){
    init();
}

void traj_evaluator::init(){
    num_points_ = trajectory_.rows();
    num_joints_ = trajectory_.cols();

    num_full_points_ = num_points_ + (int)(DIFF_RULE_LENGTH/2) * 2;
    full_trajectory_.resize(num_full_points_, num_joints_);

    free_start_index_ = (int)(DIFF_RULE_LENGTH/2);
    free_end_index_ = num_full_points_ - (int)(DIFF_RULE_LENGTH/2) -1;

    for(int i=0; i<free_start_index_; i++){
        full_trajectory_.row(i) = trajectory_.row(0);
    }
    for(int i=free_end_index_+1; i<num_full_points_; i++){
        full_trajectory_.row(i) = trajectory_.row(num_points_-1);
    }
    full_trajectory_.block(free_start_index_, 0, num_points_, num_joints_) = trajectory_;

    quadraticFullCost_.reserve(3); // vel, acc, jerk;
    diffMatrixList.reserve(3);

    setCostMatrix();
    setInterTargetPoses();
}

traj_evaluator::~traj_evaluator(){
}

double traj_evaluator::getWeightedPoseCost(){
    double cpos = 0, crot = 0;
    this->getPoseCost(cpos, crot);
    return cpos + 0.17 * crot;
}

void traj_evaluator::getPoseCost(double& cpose, double& crot){
    KDL::JntArray q(num_joints_);
    KDL::Frame cur_pose;
    KDL::Twist delta_pose;
    for(int i=1; i<num_points_; i++){
        q.data = trajectory_.row(i);
        iksolver_.fkSolver(q, cur_pose);
        delta_pose = diff(cur_pose, targetPoses_[i], 1);
        cpose += std::sqrt(KDL::dot(delta_pose.vel, delta_pose.vel));
        crot += std::sqrt(KDL::dot(delta_pose.rot, delta_pose.rot));
    }
    for(int i=0; i<num_points_-1; i++){ // inter target points
        q.data = (trajectory_.row(i+1) + trajectory_.row(i))/2;
        iksolver_.fkSolver(q, cur_pose);
        delta_pose = diff(cur_pose, targetPoses_inter_[i], 1);
        cpose += std::sqrt(KDL::dot(delta_pose.vel, delta_pose.vel));
        crot += std::sqrt(KDL::dot(delta_pose.rot, delta_pose.rot));
    }
    cpose /= (num_points_-1)*2;
    crot /= (num_points_-1)*2;
}
    // Quaternion difference ("how much rotate" at certain rotation axis) == std::sqrt(KDL::dot(delta_pose.rot, delta_pose.rot))
//        double x ,y ,z ,w;
//        cur_pose.M.GetQuaternion(x ,y ,z ,w);
//        double x2 ,y2 ,z2 ,w2;
//        targetPoses_[i+1].M.GetQuaternion(x2 ,y2 ,z2 ,w2);
//        double cosHalfTheta = w*w2 + x*x2 + y*y2 + z*z2;
//        if (cosHalfTheta < 0) {
//            cosHalfTheta = -cosHalfTheta;
//            std::cout << cosHalfTheta << std::endl;
//        }
//        if (cosHalfTheta > 1.0){
//            cosHalfTheta = 1.0;
//        }
//        double theta = 2*acos(cosHalfTheta);
//        std::cout << theta << std::endl;
//        std::cout << "------------------------------" << std::endl;


int traj_evaluator::getCost(double& cpose, double& crot, double& cvel, double& cacc, double& cjerk, bool flag_for_velviolation_measure){
    cpose = 0;
    crot = 0;
    cvel = 0;
    cacc = 0;
    cjerk = 0;
    for(int j_idx=0; j_idx<num_joints_; j_idx++){
        double vel, acc, jerk;
        if(std::find(c_joint_.begin(), c_joint_.end(), j_idx) != c_joint_.end()){
            calcJointCostwithNumericalDifferenciation(j_idx, vel, acc, jerk); // continuous joint
        }else{
            calcJointCostwithFiniteDifferenciation(j_idx, vel, acc, jerk);
        }
        cvel += vel;
        cacc += acc;
        cjerk += jerk;
    }
    cvel /= num_joints_;
    cacc /= num_joints_;
    cjerk /= num_joints_;

    getPoseCost(cpose, crot);

    if(flag_for_velviolation_measure){
        return calcNofVelViolation();
    }else{
        return 0;
    }
}

//void traj_evaluator::getJointCost(std::vector<double>& cvel, std::vector<double>& cacc, std::vector<double>& cjerk){
//    cvel.reserve(num_joints_);
//    cacc.reserve(num_joints_);
//    cjerk.reserve(num_joints_);
//    for(int j_idx=0; j_idx<num_joints_; j_idx++){
//        double vel, acc, jerk;
//        if(std::find(c_joint_.begin(), c_joint_.end(), j_idx) != c_joint_.end()){
//            calcJointCostwithFiniteDifferenciation(j_idx, vel, acc, jerk);
//        }else{
//            calcJointCostwithNumericalDifferenciation(j_idx, vel, acc, jerk);
//        }
//        cvel.push_back(vel);
//        cacc.push_back(acc);
//        cjerk.push_back(jerk);
//    }
//}

double traj_evaluator::testJointCost(int j_idx){
    double ND_vel, ND_acc, ND_jerk;
    calcJointCostwithNumericalDifferenciation(j_idx, ND_vel, ND_acc, ND_jerk);

    double FD_vel, FD_acc, FD_jerk;
    calcJointCostwithFiniteDifferenciation(j_idx, FD_vel, FD_acc, FD_jerk);

    std::cout << ND_vel << ", " << ND_acc << ", " << ND_jerk << std::endl;
    std::cout << FD_vel << ", " << FD_acc << ", " << FD_jerk << std::endl;
    std::cout << ND_vel-FD_vel << ", " << ND_acc-FD_acc << ", " << ND_jerk-FD_jerk << std::endl;
    std::cout << "=============================================" << std::endl;
}

void traj_evaluator::calcJointCostwithNumericalDifferenciation(int j_idx, double& cvel, double& cacc, double& cjerk){
    Eigen::MatrixXd vel, acc, jerk;
    vel.resize(num_points_-1, 1);
    acc.resize(num_points_-2, 1);
    jerk.resize(num_points_-3, 1);
    for(int i=0; i<num_points_-1; i++){
        vel(i, 0) = shortestAngularDistance(trajectory_(i, j_idx), trajectory_(i+1, j_idx))/discretization_;
    }
    for(int i=0; i<num_points_-2; i++){
        acc(i, 0) = (vel(i+1,0) - vel(i,0))/discretization_;
    }
    for(int i=0; i<num_points_-3; i++){
        jerk(i, 0) = (acc(i+1,0) - acc(i,0))/discretization_;
    }
    cvel = vel.array().abs().mean();
    cacc = acc.array().abs().mean();
    cjerk = jerk.array().abs().mean();
}
//void traj_evaluator::calcJointCostwithNumericalDifferenciation(int j_idx, double& cvel, double& cacc, double& cjerk){
//    Eigen::MatrixXd vel, acc, jerk;
//    vel.resize(num_points_-1, 1);
//    acc.resize(num_points_-2, 1);
//    jerk.resize(num_points_-3, 1);
//    for(int i=0; i<num_points_-1; i++){
//        vel(i, 0) = shortestAngularDistance(trajectory_(i, j_idx), trajectory_(i+1, j_idx))/discretization_;
//    }
//    for(int i=0; i<num_points_-2; i++){
//        acc(i, 0) = (vel(i+1,0) - vel(i,0))/discretization_;
//    }
//    for(int i=0; i<num_points_-3; i++){
//        jerk(i, 0) = (acc(i+1,0) - acc(i,0))/discretization_;
//    }
//    cvel = vel.array().abs().mean();
//    cacc = acc.array().abs().mean();
//    cjerk = jerk.array().abs().mean();
//}

void traj_evaluator::calcJointCostwithFiniteDifferenciation(int j_idx, double& cvel, double& cacc, double& cjerk){
    cvel = (diffMatrixList[0] * full_trajectory_.col(j_idx)).block(free_start_index_, 0, num_points_, 1).array().abs().mean();
    cacc = (diffMatrixList[1] * full_trajectory_.col(j_idx)).block(free_start_index_, 0, num_points_, 1).array().abs().mean();
    cjerk = (diffMatrixList[2] * full_trajectory_.col(j_idx)).block(free_start_index_, 0, num_points_, 1).array().abs().mean();
}


void traj_evaluator::setCostMatrix() {
    for(int i = 0; i < 3; i++) {
        Eigen::MatrixXd diff = Eigen::MatrixXd::Zero(num_full_points_, num_full_points_);
        diff = createDiffMatrix(num_full_points_, i);
        diffMatrixList.push_back(diff/ pow(discretization_, i + 1));
        quadraticFullCost_.push_back( (diff.transpose() * diff) / pow(discretization_, (i + 1) * 2) );
    }
}

Eigen::MatrixXd traj_evaluator::createDiffMatrix(int size, int diffIdx) const
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);
    for (int i = 0; i < size; i++)
    {
        for (int j = -(DIFF_RULE_LENGTH/2); j <= (DIFF_RULE_LENGTH/2); j++)
        {
            int index = i + j;
            if (index < 0)
                continue;
            if (index >= size)
                continue;
            matrix(i, index) = DIFF_RULES[diffIdx][j + (DIFF_RULE_LENGTH/2)];
        }
    }
    return matrix;
}

void traj_evaluator::printMatrix(Eigen::MatrixXd m) const {
    for(int i=0 ; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            std::cout << m(i,j) << ", ";
        }
        std::cout << std::endl;
    }
}

void traj_evaluator::setInterTargetPoses(){
    targetPoses_inter_.reserve(targetPoses_.size()-1);
    for(int i = 0; i < targetPoses_.size()-1; i++){
        KDL::Frame inter_f;
        inter_f.p[0] = (targetPoses_[i+1].p[0] + targetPoses_[i].p[0])/2;
        inter_f.p[1] = (targetPoses_[i+1].p[1] + targetPoses_[i].p[1])/2;
        inter_f.p[2] = (targetPoses_[i+1].p[2] + targetPoses_[i].p[2])/2;

        KDL::Vector rot_axis;
        double angle = (targetPoses_[i].M.Inverse() * targetPoses_[i+1].M).GetRotAngle(rot_axis);
        inter_f.M = targetPoses_[i].M * KDL::Rotation::Rot2(rot_axis, angle/2);
        targetPoses_inter_.push_back(inter_f);
    }
}


// Velocity limit check
int traj_evaluator::calcNofVelViolation(){
    int n_violate = 0;
    for(int i=0; i<num_points_-1; i++){
        bool violate_flag = false;
        for(int j_idx=0; j_idx<num_joints_; j_idx++){
            if(std::find(c_joint_.begin(), c_joint_.end(), j_idx) != c_joint_.end()){ // continuous joint
                if(std::abs(shortestAngularDistance(trajectory_(i, j_idx), trajectory_(i+1, j_idx)))/discretization_ > vel_limit_[j_idx]){
                    violate_flag = true;
                    break;
                }
            }else {
                if(std::abs(trajectory_(i+1, j_idx) - trajectory_(i, j_idx))/discretization_ > vel_limit_[j_idx]){
                    violate_flag = true;
                    break;
                }
            }
        }
        n_violate += violate_flag;
    }
    return n_violate;
}

void traj_evaluator::setNewDt(double dt){
    discretization_ = dt;
}
