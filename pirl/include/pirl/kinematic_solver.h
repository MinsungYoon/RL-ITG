#include <ros/ros.h>
#include <trac_ik/trac_ik.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <pirl_msgs/allfk.h>
#include <pirl_msgs/fk.h>
#include <pirl_msgs/ik.h>
#include <pirl_msgs/rndsg.h>
#include <pirl_msgs/rndvalconf.h>
#include <pirl_msgs/jaco.h>

class KinematicsSolverSrv{
public:
    KinematicsSolverSrv(planning_scene::PlanningScenePtr planning_scene);
    virtual ~KinematicsSolverSrv(){};

    void printJointLimitInfo();
    double fRand(double min, double max) const;
    double fRand(int i) const;
    void getRandomConfiguration(KDL::JntArray& q);
    void setCollisionChecker();
    bool collisionChecking(std::vector<double> values);

    bool ikSolver(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out);
    bool ikSolver(const KDL::Frame& p_in, KDL::JntArray& q_out);
    bool ikSolverCollFree(const KDL::Frame& p_in, KDL::JntArray& q_out);
    bool ikSolverCollFree(const KDL::JntArray& q_init, const KDL::Frame& p_in, KDL::JntArray& q_out);
    void fkSolver(const KDL::JntArray& q_init, KDL::Frame& p_in);
    void allLinkfkSolver(const KDL::JntArray& q_init, std::vector<KDL::Frame>& p_in);
    void vikSolver(const KDL::JntArray& q, const KDL::Twist& delta_twist, KDL::JntArray& delta_q);
    void getJacobian(const KDL::JntArray& q, KDL::Jacobian& jac);
    unsigned int getDoF();

    bool allLinkfkSolver_srv(pirl_msgs::allfk::Request  &req, pirl_msgs::allfk::Response &res);
    bool fkSolver_srv(pirl_msgs::fk::Request  &req, pirl_msgs::fk::Response &res);
    bool ikSolver_srv(pirl_msgs::ik::Request  &req, pirl_msgs::ik::Response &res);

    bool rndStartAndGoal_srv(pirl_msgs::rndsg::Request  &req, pirl_msgs::rndsg::Response &res);
    bool rndValidSample_srv(pirl_msgs::rndvalconf::Request  &req, pirl_msgs::rndvalconf::Response &res);

    bool jaco_srv( pirl_msgs::jaco::Request  &req, pirl_msgs::jaco::Response &res);

private:
    ros::NodeHandle nh_;
    planning_scene::PlanningScenePtr planning_scene_;
    std::string planning_group_;
    std::string base_link_, tip_link_;

    KDL::Chain chain_;
    KDL::JntArray ll_, ul_; //lower joint limits, upper joint limits
    uint n_dof_;
    uint n_seg_;

    std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jac_solver_;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv> vik_solver_;
    uint max_tried_;

    collision_detection::CollisionRequest c_request_;
    collision_detection::CollisionResult c_result_;

    ros::ServiceServer fksolver_service_;
    ros::ServiceServer allLinkfksolver_service_;
    ros::ServiceServer iksolver_service_;
    ros::ServiceServer rndsg_service_;
    ros::ServiceServer rndvalconf_service_;
    ros::ServiceServer jaco_service_;
};

typedef std::shared_ptr<KinematicsSolverSrv> KinematicsSolverSrvPtr;
