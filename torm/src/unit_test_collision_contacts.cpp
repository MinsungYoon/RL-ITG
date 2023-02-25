#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf/tf.h>

#include <moveit_msgs/DisplayTrajectory.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/MarkerArray.h>

#include <chrono>

#include <cmath>
#include <vector>
#include <numeric> // for accumulate

#include <torm/torm_ik_solver.h>
#include <torm/torm_problem.h>

#include <moveit/collision_distance_field/collision_robot_hybrid.h>
#include <moveit/collision_distance_field/collision_world_hybrid.h>
#include <moveit/collision_distance_field/collision_common_distance_field.h>

using namespace std;


struct Mean_Std {
    double mean;
    double std;
};
Mean_Std calcMeanStd(std::vector<double> v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double squareSum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double std = sqrt(squareSum / v.size() - mean * mean);
    Mean_Std return_struct;
    return_struct.mean = mean;
    return_struct.std = std;
    return return_struct;
}

moveit_msgs::CollisionObject makeCollisionObject(std::string name,
                                                 double x, double y, double z,
                                                 double roll, double pitch, double yaw,
                                                 double size_x, double size_y, double size_z){
    moveit_msgs::CollisionObject co;

    co.id = name;
    co.header.frame_id = "base_link";

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
// obs_spec: [x,y,z, roll,pitch,yaw, size_x,size_y,size_z]
void setCollisionObjects(std::vector<moveit_msgs::CollisionObject>& collision_objects, std::string obs_name, std::vector<double> obs_spec){

    collision_objects.push_back(makeCollisionObject(obs_name, obs_spec[0], obs_spec[1], obs_spec[2]
            , obs_spec[3], obs_spec[4], obs_spec[5]
            , obs_spec[6], obs_spec[7], obs_spec[8]));
}



void appendContactPoint(Eigen::Vector3d pos, Eigen::Vector3d normal, geometry_msgs::PoseArray& Arrow_list_){
    geometry_msgs::Pose pose;
    pose.position.x = pos[0];
    pose.position.y = pos[1];
    pose.position.z = pos[2];
    pose.orientation.x = normal[0]/normal.norm();
    pose.orientation.y = normal[1]/normal.norm();
    pose.orientation.z = normal[2]/normal.norm();
    pose.orientation.w = 0;
    Arrow_list_.poses.push_back(pose);
}

void appendSpherePoint(double x, double y, double z, double radii, double r, double g, double b, int id, visualization_msgs::MarkerArray& Marker_list_){
    visualization_msgs::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = ros::Time();
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = id;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = radii * 2; //diameter in x-direction
    marker.scale.y = radii * 2;
    marker.scale.z = radii * 2;
    marker.color.a = 0.3; // Don't forget to set the alpha!
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    Marker_list_.markers.push_back(marker);
}

void appendDistanceSpherePoint(double x, double y, double z, double d_value, int id, visualization_msgs::MarkerArray& Marker_list_){
    visualization_msgs::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = ros::Time();
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = id;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05 * 2; //diameter in x-direction
    marker.scale.y = 0.05 * 2;
    marker.scale.z = 0.05 * 2;
    if(d_value<0){
        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 1.0;
        marker.color.g = 0;
        marker.color.b = 0;
    }else if(d_value>=0.5){
        marker.color.a = 0.05; // Don't forget to set the alpha!
        marker.color.r = 0.1;
        marker.color.g = 0.2;
        marker.color.b = 1.0;
    }else{
        marker.color.a = 0.3; // Don't forget to set the alpha!
        marker.color.r = 0;
        marker.color.g = d_value/1;
        marker.color.b = 0;
    }
    Marker_list_.markers.push_back(marker);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "collision_checking_debug");
    ros::NodeHandle node_handle("~");

    // Setting related to the Robot state
    const std::string PLANNING_GROUP = "arm";
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    const robot_state::JointModelGroup *joint_model_group = robot_state->getJointModelGroup(PLANNING_GROUP);

    std::vector<std::string> indices = joint_model_group->getActiveJointModelNames();
    robot_model::JointBoundsVector joint_bounds = joint_model_group->getActiveJointModelsBounds();
    unsigned int num_dof = joint_model_group->getVariableCount();

    // Initialize planning interface
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);

    auto RobotModel = *planning_scene->getRobotModel();
    auto a0 = RobotModel.getName();
    auto a1 = RobotModel.getJointModelGroupNames();
    auto a2 = RobotModel.getJointModelNames();
    auto a3 = RobotModel.getJointModelGroups();
    auto a4 = *RobotModel.getJointModelGroup(PLANNING_GROUP);
    auto a5 = a4.getJointModelNames();
    auto a6 = a4.getName();
    torm::TormProblem prob("free", "fetch", planning_scene);

    // [Obj Enrollment Test] Apply collision objects //////////////////////////////////////////////////////////////////
    // [1] Just for visualization
    std::vector<moveit_msgs::CollisionObject> collision_objects;
    setCollisionObjects(collision_objects, "box1", {1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    planning_scene_interface_.applyCollisionObjects(collision_objects);
    planning_scene->printKnownObjects(cout);

    // [2] Enroll objects to planning_scene for collision detection (important!)
    std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map = planning_scene_interface_.getObjects();
    for(auto& kv : collision_objects_map){
        planning_scene->processCollisionObjectMsg(kv.second);
    }
    planning_scene->printKnownObjects(cout);

//    // obj is registered, but not shown in rviz visually.
//    std::vector<moveit_msgs::CollisionObject> collision_objects;
//    setCollisionObjects(collision_objects, "box1", {1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
//    for(auto& cobj : collision_objects){
//        planning_scene->processCollisionObjectMsg(cobj);
//    }
//    planning_scene->printKnownObjects(cout);


//    // TEST: Setting object color
////    const std_msgs::ColorRGBA& box_color = planning_scene->getObjectColor("box1");
//    std_msgs::ColorRGBA box_color;
//    box_color.r = 255;
//    box_color.g = 0;
//    box_color.b = 0;
//    planning_scene->setObjectColor("box1", box_color);
//
//    std::vector<moveit_msgs::ObjectColor> object_colors;
//    planning_scene->getObjectColorMsgs(object_colors);

    // MAIN Part
    ros::Publisher display_pub_;
    display_pub_ = node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);

    ros::Publisher arrow_pub_;
    arrow_pub_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow4", 10);
    geometry_msgs::PoseArray Arrow_list_;
    Arrow_list_.header.frame_id = "base_link";
    Arrow_list_.header.stamp = ros::Time::now();

    ros::Publisher marker_pub_;
    marker_pub_ = node_handle.advertise<visualization_msgs::MarkerArray>("/visualization_marker_array", 10);
    visualization_msgs::MarkerArray Marker_list_;

    ros::Publisher arrow_pub2_;
    arrow_pub2_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow5", 10);
    geometry_msgs::PoseArray Arrow_list2_;
    Arrow_list2_.header.frame_id = "base_link";
    Arrow_list2_.header.stamp = ros::Time::now();

    ros::Publisher arrow_pub3_;
    arrow_pub3_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow6", 10);
    geometry_msgs::PoseArray Arrow_list3_;
    Arrow_list3_.header.frame_id = "base_link";
    Arrow_list3_.header.stamp = ros::Time::now();

    ros::Publisher joint_pub = node_handle.advertise<sensor_msgs::JointState>("/move_group/fake_controller_joint_states", 1);
    sensor_msgs::JointState joint_state;

    // collision checker
    ROS_INFO("Start collision checker");
    collision_detection::CollisionRequest collision_request;
    collision_request.group_name = PLANNING_GROUP;
    collision_request.distance = true;
    collision_request.cost = true;
    collision_request.contacts = true;
    collision_request.max_contacts = 100;
    collision_request.max_contacts_per_pair = 1;
    collision_request.max_cost_sources = 1;
    collision_request.min_cost_density = 0.2;
    collision_request.verbose = false;
    ROS_INFO("End collision checker");

    collision_detection::CollisionResult collision_result;
    robot_state::RobotState rs = planning_scene->getCurrentStateNonConst();
    std::cout << "[1] print robot state" << std::endl;
    rs.printStatePositions(std::cout);

    std::vector<double> config = {0.3, -0.5, 0.5, 0.8, -0.3, 0.1, 0.5};
//    std::vector<double> config = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    rs.setJointGroupPositions(PLANNING_GROUP, config);
    planning_scene->setCurrentState(rs);
    std::cout << "---------------" << std::endl;
    rs.printTransforms(std::cout);
    std::cout << "---------------" << std::endl;
    rs.printStatePositions(std::cout);
    std::cout << "---------------" << std::endl;
    auto d1p = rs.dirty();
    rs.printDirtyInfo(std::cout);
    std::cout << "---------------" << std::endl;

    rs.update();

    rs.printTransforms(std::cout);
    std::cout << "---------------" << std::endl;
    rs.printStatePositions(std::cout);
    std::cout << "---------------" << std::endl;
    rs.printStatePositionsWithJointLimits(rs.getJointModelGroup(PLANNING_GROUP), std::cout);
    std::cout << "---------------" << std::endl;
    rs.printDirtyInfo(std::cout);
    std::cout << "---------------" << std::endl;
    std::vector<double> cur_j;
    rs.copyJointGroupPositions(PLANNING_GROUP, cur_j);

    auto aa = move_group.getVariableCount();
    auto aa1 = move_group.getCurrentState();
    std::vector<std::string> aa2 = move_group.getJointModelGroupNames();
    std::vector<std::string> aa3 = move_group.getActiveJoints();
    std::vector<std::string> aa4 = move_group.getLinkNames();


    auto gripper_pos = rs.getGlobalLinkTransform("gripper_link");
    std::cout << gripper_pos.translation().x() << ", "
              << gripper_pos.translation().y() << ", "
              << gripper_pos.translation().z() << ", " << std::endl;

    // inv jaco test //////////////////////////////////////////////////////////////////////////////////////
    for (uint ci = 0; ci < 10; ci++) {
        joint_state.header.stamp = ros::Time::now();
        joint_state.name = indices;
        joint_state.position = config;
        joint_pub.publish(joint_state);
        ros::Duration(0.05).sleep();
    }

    double size_x = 2.5, size_y = 2.5, size_z = 5.0;
    double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
    double resolution = 0.1;
    double max_propogation_distance = 0.5;
    collision_detection::CollisionWorldHybrid *hy_world_;
    collision_detection::CollisionRobotHybrid *hy_robot_;
    hy_world_ = new collision_detection::CollisionWorldHybrid(planning_scene->getWorldNonConst(),
                                                              Eigen::Vector3d(size_x, size_y, size_z),
                                                              Eigen::Vector3d(origin_x, origin_y, origin_z),
                                                              true,
                                                              resolution, 0.0, max_propogation_distance);
    std::map<std::string, std::vector<collision_detection::CollisionSphere>> link_body_decompositions;
    hy_robot_ = new collision_detection::CollisionRobotHybrid(planning_scene->getRobotModel(),
                                                              link_body_decompositions,
                                                              size_x, size_y, size_z,
                                                              true,
                                                              resolution, 0.0, max_propogation_distance);
    collision_detection::CollisionResult res;
    collision_detection::AllowedCollisionMatrix acm_ = planning_scene->getAllowedCollisionMatrix();
    for(int c=0; c<1000; c++) {
        collision_detection::GroupStateRepresentationPtr gsr_;
        hy_world_->getCollisionGradients(collision_request, res, *(hy_robot_->getCollisionRobotDistanceField()),
                                         planning_scene->getCurrentStateNonConst(),
                                         &acm_, gsr_);

        auto distField = hy_world_->getCollisionWorldDistanceField()->getDistanceField();
//        auto distField = gsr_.get()->dfce_.get()->distance_field_.get();
//        auto distField = hy_robot_->getCollisionRobotDistanceField()->getLastDistanceFieldEntry()->distance_field_;
        std::cout << distField->getXNumCells() << ", " << distField->getYNumCells() << ", " << distField->getZNumCells() << std::endl;
        std::cout << distField->getOriginX() << ", " << distField->getOriginY() << ", " << distField->getOriginZ() << std::endl;
        std::cout << distField->getSizeX() << ", " << distField->getSizeY() << ", " << distField->getSizeZ() << std::endl;
        double w_x, w_y, w_z;
//        distField->gridToWorld(200,0,400,w_x, w_y, w_z);
//        std::cout << w_x << ", " << w_y << ", " << w_z << std::endl;
//        std::cout << distField->getDistance(200,125,400) << std::endl;

        int n_d = 0;
        for(int xx=0; xx<distField->getXNumCells(); xx++){
            for(int yy=0; yy<distField->getYNumCells(); yy++){
                for(int zz=0; zz<distField->getZNumCells(); zz++){
                    double w_x, w_y, w_z;
                    distField->gridToWorld(xx,yy,zz,w_x, w_y, w_z);
                    double dd = distField->getDistance(xx,yy,zz);
                    appendDistanceSpherePoint(w_x, w_y, w_z, dd, n_d, Marker_list_);
                    n_d++;
                    std::cout << dd << std::endl;
//                    std::cout << w_x << ", " << w_y << ", " << w_z << ": " << dd << std::endl;
                }
            }
        }
        for (uint i = 0; i < 100; i++) {
            marker_pub_.publish(Marker_list_);
            ros::Duration(0.05).sleep();
        }

        int num_sphere =0;
        Arrow_list3_.poses.clear();
        Marker_list_.markers.clear();
        std::vector<double> cur_jj;
        rs.copyJointGroupPositions(PLANNING_GROUP, cur_jj);
        for (size_t i = 2; i < gsr_->gradients_.size(); i++) // each link
        {
            for (size_t j = 0; j < gsr_->gradients_[i].sphere_locations.size(); j++) { // each sphere
                auto sphere_center = gsr_->gradients_[i].sphere_locations[j];
                auto sphere_radii = gsr_->gradients_[i].sphere_radii[j];
                std::cout << gsr_->gradients_[i].joint_name << ": " << gsr_->gradients_[i].collision << ": " <<
                          gsr_->gradients_[i].closest_distance << " / " << gsr_->gradients_[i].distances[j] << " | "
                          << sphere_radii
                          << " = " << (gsr_->gradients_[i].distances[j] - sphere_radii) << std::endl;

                double g_norm = gsr_->gradients_[i].gradients[j].norm();
                double g_x = gsr_->gradients_[i].gradients[j].x();
                double g_y = gsr_->gradients_[i].gradients[j].y();
                double g_z = gsr_->gradients_[i].gradients[j].z();

//                if ((gsr_->gradients_[i].distances[j] - sphere_radii) <= 0 && gsr_->gradients_[i].gradients[j].norm() > 0 && gsr_->gradients_[i].gradients[j].norm() <= 1.0) {
                if (gsr_->gradients_[i].gradients[j].norm() > 0) {
                    appendContactPoint(Eigen::Vector3d{sphere_center[0], sphere_center[1], sphere_center[2]},
                                       Eigen::Vector3d{g_x, g_y, g_z},
                                       Arrow_list3_);

                    Eigen::MatrixXd err;
                    err.resize(6, 1);
                    err(0, 0) = 1 * g_x/g_norm/10;
                    err(1, 0) = 1 * g_y/g_norm/10;
                    err(2, 0) = 1 * g_z/g_norm/10;
//                    std::cout << "======================================================= " <<
//                                gsr_->gradients_[i].gradients[j].x()<< ", " <<
//                                gsr_->gradients_[i].gradients[j].y()
//                              << ", " << gsr_->gradients_[i].gradients[j].z() << " :norm: " << gsr_->gradients_[i].gradients[j].norm() << std::endl;
                    auto aan = rs.getGlobalLinkTransform(rs.getJointModel(gsr_->gradients_[i].joint_name)->getChildLinkModel());
//                    std::cout << aan.translation().x() << ", " << aan.translation().y() << ", " << aan.translation().z() << std::endl;
                    Eigen::MatrixXd jaco;
                    rs.getJacobian(rs.getJointModelGroup(PLANNING_GROUP), rs.getJointModel(gsr_->gradients_[i].joint_name)->getChildLinkModel(),
                                   Eigen::Vector3d(sphere_center[0] - aan.translation().x(), sphere_center[1] - aan.translation().y(), sphere_center[2] - aan.translation().z()),
                                   jaco, false);
                    Eigen::MatrixXd dj = jaco.transpose() * (jaco * jaco.transpose()).inverse() * err;
                    for (int cj = 0; cj < num_dof; cj++) {
                        cur_jj[cj] += dj(cj, 0)/30.0;
                        std::cout << dj(cj,0) << std::endl;
                    }
                    appendSpherePoint(sphere_center[0], sphere_center[1], sphere_center[2], sphere_radii, 1, 0, 0,
                                      num_sphere, Marker_list_);
                } else {
                    appendSpherePoint(sphere_center[0], sphere_center[1], sphere_center[2], sphere_radii, 0, 1, 0,
                                      num_sphere, Marker_list_);
                }
                num_sphere++;

//                std::cout << "======================================================= " << g_x << ", " << g_y
//                          << ", " << g_z << " :norm: " << g_norm << std::endl;
            }
        }
        rs.setJointGroupPositions(PLANNING_GROUP, cur_jj);
        rs.enforceBounds();
        rs.copyJointGroupPositions(PLANNING_GROUP, cur_jj);
        planning_scene->setCurrentState(rs);
        for (uint ci = 0; ci < 10; ci++) {
            joint_state.header.stamp = ros::Time::now();
            joint_state.name = indices;
            joint_state.position = cur_jj;
            joint_pub.publish(joint_state);
            ros::Duration(0.05).sleep();
        }
        for (uint i = 0; i < 110; i++) {
            marker_pub_.publish(Marker_list_);
            arrow_pub3_.publish(Arrow_list3_);
            ros::Duration(0.05).sleep();
        }
    }

    for(int tt=0; tt<100 ; tt++) {

        Eigen::MatrixXd jaco;
        rs.getJacobian(rs.getJointModelGroup(PLANNING_GROUP), rs.getLinkModel("gripper_link"),
                       Eigen::Vector3d(0.0, 0.0, 0.0),
                       jaco, false);
//        for (int i = 0; i < jaco.rows(); i++) {
//            for (int j = 0; j < jaco.cols(); j++) {
//                std::cout << jaco(i, j) << ", ";
//            }
//            std::cout << std::endl;
//        }

        Eigen::MatrixXd err;
        err.resize(6, 1);
        err(0, 0) = -0.01;
        err(1, 0) = 0.01;
        err(2, 0) = 0;
        err(3, 0) = 0;
        err(4, 0) = 0;
        err(5, 0) = 0;
        ros::WallTime wt = ros::WallTime::now();
        Eigen::MatrixXd dj = jaco.transpose() * (jaco * jaco.transpose()).inverse() * err;
        ROS_WARN_STREAM("Jaco: " << (ros::WallTime::now() - wt));
        std::vector<double> cur_jj;
        rs.copyJointGroupPositions(PLANNING_GROUP, cur_jj);
        for (int j = 0; j < num_dof; j++) {
            cur_jj[j] += dj(j, 0);
        }
        rs.setJointGroupPositions(PLANNING_GROUP, cur_jj);
        wt = ros::WallTime::now();
        rs.update();
        ROS_WARN_STREAM("rs update: " << (ros::WallTime::now() - wt));
        for (uint i = 0; i < 10; i++) {
            joint_state.header.stamp = ros::Time::now();
            joint_state.name = indices;
            joint_state.position = cur_jj;
            joint_pub.publish(joint_state);
            ros::Duration(0.05).sleep();
        }
        wt = ros::WallTime::now();
        gripper_pos = rs.getGlobalLinkTransform("gripper_link");
        ROS_WARN_STREAM("getGlobalLinkTransform: " << (ros::WallTime::now() - wt));
        std::cout << gripper_pos.translation().x() << ", "
                  << gripper_pos.translation().y() << ", "
                  << gripper_pos.translation().z() << ", " << std::endl;
    }
    auto q1 = rs.getVariableCount();
    auto q2 = rs.getVariableNames();
    auto d1 = rs.dirty();
    auto gripper_pos2 = rs.getGlobalLinkTransform("gripper_link");
    std::cout << gripper_pos2.translation().x() << ", "
            << gripper_pos2.translation().y() << ", "
            << gripper_pos2.translation().z() << ", " << std::endl;


    planning_scene->setCurrentState(rs);
    std::vector<double> jj;
    planning_scene->getCurrentState().copyJointGroupPositions(PLANNING_GROUP, jj);
    for (uint i = 0; i < 10; i++) {
        joint_state.header.stamp = ros::Time::now();
        joint_state.name = indices;
        joint_state.position = jj;
        joint_pub.publish(joint_state);
        ros::Duration(0.05).sleep();
    }


//    std::vector<double> config = move_group.getRandomJointValues();
////    std::vector<double> config = {0.7, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0};
////    std::vector<double> config = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//    rs.setJointGroupPositions(PLANNING_GROUP, config);
//    planning_scene->setCurrentState(rs);
//    planning_scene->getCurrentState().printStatePositions(cout);
//    for (uint i = 0; i < 100; i++) {
//        //update joint_state
//        std::vector<double> jj;
//        planning_scene->getCurrentState().copyJointGroupPositions(PLANNING_GROUP, jj);
//        joint_state.header.stamp = ros::Time::now();
//        joint_state.name = indices;
//        joint_state.position = jj;
//        joint_pub.publish(joint_state);
//    }

    // TEST FCL distance distribution
    double best_distance = -1000000.0;
    int n_test = 5000;
    std::vector<double> distance_list;
    std::vector<double> time_list;
    distance_list.reserve(n_test);
    time_list.reserve(n_test);
    for(int t=0; t<n_test; t++){
//        std::vector<double> config = move_group.getRandomJointValues();
        std::vector<double> config = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        rs.setJointGroupPositions(PLANNING_GROUP, config);
        planning_scene->setCurrentState(rs);
        collision_result.clear();
        ros::WallTime wt = ros::WallTime::now();
        planning_scene->checkCollision(collision_request, collision_result);
        time_list.push_back((ros::WallTime::now() - wt).nsec * 1e-6);
        distance_list.push_back(collision_result.distance);
//        ROS_WARN_STREAM("===== collision: " << collision_result.collision << "  ,distance: " << collision_result.distance);
        for (auto cLinkPair: collision_result.contacts) {
            for (auto cContact: cLinkPair.second) {
//                ROS_WARN_STREAM("["<<cLinkPair.first.first<<":"<<cLinkPair.first.second<<"]: "<< "depth: " << cContact.depth << ", norm: " <<  cContact.normal.norm());
                if(best_distance < cContact.depth && collision_result.distance == 0.0){
                    best_distance = cContact.depth;
                    std::cout << best_distance << std::endl;
                    for (uint i = 0; i < 100; i++) {
                        std::vector<double> jj;
                        planning_scene->getCurrentState().copyJointGroupPositions(PLANNING_GROUP, jj);
                        joint_state.header.stamp = ros::Time::now();
                        joint_state.name = indices;
                        joint_state.position = jj;
                        joint_pub.publish(joint_state);
                    }
                }
            }
        }
//        if(best_distance < collision_result.distance){
//            best_distance = collision_result.distance;
//            for (uint i = 0; i < 100; i++) {
//                std::vector<double> jj;
//                planning_scene->getCurrentState().copyJointGroupPositions(PLANNING_GROUP, jj);
//                joint_state.header.stamp = ros::Time::now();
//                joint_state.name = indices;
//                joint_state.position = jj;
//                joint_pub.publish(joint_state);
//            }
//        }
    }
    Mean_Std calc_res;
    calc_res = calcMeanStd(distance_list);
    std::cout << calc_res.mean << ", " << calc_res.std << " | " << *max_element(distance_list.begin(), distance_list.end()) << ", " << *min_element(distance_list.begin(), distance_list.end()) << std::endl;
    calc_res = calcMeanStd(time_list);
    std::cout << calc_res.mean << ", " << calc_res.std << std::endl;



    // TEST: checkCollision FCL collision detection.
    for(int t=0; t<100; t++) {
        collision_result.clear();
        ros::WallTime wt = ros::WallTime::now();
        planning_scene->checkCollision(collision_request, collision_result);
        ROS_WARN_STREAM("FCL checkCollision: " << (ros::WallTime::now() - wt));
        ROS_WARN_STREAM("collision: " << collision_result.collision << "  ,distance: " << collision_result.distance); // distance: Closest distance between "two bodies"
        collision_result.print();
        int n_col = 0;
        for (auto cLinkPair: collision_result.contacts) {
            std::cout<<"["<<cLinkPair.first.first<<":"<<cLinkPair.first.second<<"]:"<<cLinkPair.second.size()<<std::endl;
            for (auto cContact: cLinkPair.second) {
                std::cout<< "depth: " << cContact.depth <<std::endl; // depth: penetration between "bodies"
                Eigen::Vector3d contact_pos = cContact.pos;
                Eigen::Vector3d contact_normal = cContact.normal;
                appendContactPoint(contact_pos, contact_normal, Arrow_list_);
                n_col++;
                std::cout << "norm: " << cContact.normal.norm() << std::endl;
            }
        }
        std::cout << "n_col: " << n_col << std::endl;
        std::cout << "---n_col: " << collision_result.contact_count << std::endl;
        for (uint i = 0; i < 100; i++) {
            arrow_pub_.publish(Arrow_list_);
        }
    }
    std::cout << "==============================================================================" << std::endl;
    {
        double size_x = 2.0, size_y = 2.5, size_z = 3.0;
        double origin_x = 0.5, origin_y = 0.0, origin_z = 0.5;
        double resolution = 0.01;
        collision_detection::CollisionWorldHybrid *hy_world_;
        collision_detection::CollisionRobotHybrid *hy_robot_;

        ros::WallTime wt = ros::WallTime::now();
        hy_world_ = new collision_detection::CollisionWorldHybrid(planning_scene->getWorldNonConst(),
                                                                  Eigen::Vector3d(size_x, size_y, size_z),
                                                                  Eigen::Vector3d(origin_x, origin_y, origin_z),
                                                                  true,
                                                                  resolution, 0.0, 0.3);
        if (!hy_world_) {
            ROS_WARN_STREAM("Could not initialize hybrid collision world from planning scene");
        } else {
            ROS_WARN_STREAM("Creat hy_world_: " << (ros::WallTime::now() - wt));
        }

//        hy_world_->getCollisionWorldDistanceField().;

        std::map<std::string, std::vector<collision_detection::CollisionSphere>> link_body_decompositions;
        wt = ros::WallTime::now();
        hy_robot_ = new collision_detection::CollisionRobotHybrid(planning_scene->getRobotModel(),
                                                                  link_body_decompositions,
                                                                  size_x, size_y, size_z,
                                                                  true,
                                                                  resolution, 0.0, 0.3);
        if (!hy_robot_) {
            ROS_WARN_STREAM("Could not initialize hybrid collision robot from planning scene");
        } else {
            ROS_WARN_STREAM("Creat hy_robot_: " << (ros::WallTime::now() - wt));
        }
        planning_scene->getCurrentStateNonConst().printStatePositions(std::cout);

        collision_detection::CollisionResult res;
        collision_detection::AllowedCollisionMatrix acm_ = planning_scene->getAllowedCollisionMatrix();
        collision_detection::GroupStateRepresentationPtr gsr_;



        { // Distance field TEST: getAllCollisions: (CollisionRobotDistanceField) getSelfCollisions, getIntraGroupCollisions (CollisionWorldDistanceField) getEnvironmentCollisions
            for(int t=0; t<100; t++) {
                ros::WallTime wt = ros::WallTime::now();
                hy_world_->getAllCollisions(collision_request, res, *(hy_robot_->getCollisionRobotDistanceField()),
                                            planning_scene->getCurrentStateNonConst(),
                                            &acm_, gsr_);
                ROS_WARN_STREAM("call getAllCollisions: " << (ros::WallTime::now() - wt));
            }
//            auto bb = *hy_robot_->getCollisionRobotDistanceField();
//            auto cc = bb.getLastDistanceFieldEntry();
//            auto dd = cc->distance_field_;

            std::cout << "distance: " << res.distance << std::endl; // distance: Closest distance between "two bodies"
            res.print();
            int n_col = 0;
            for (auto cLinkPair: res.contacts) {
                std::cout << "[" << cLinkPair.first.first << ":" << cLinkPair.first.second << "]:"
                          << cLinkPair.second.size() << std::endl;
                for (auto cContact: cLinkPair.second) {
                    if (cContact.depth > 1e-10) { // important!
                        std::cout << "depth: " << cContact.depth << std::endl; // depth: penetration between "bodies"
                        Eigen::Vector3d contact_pos = cContact.pos;
                        Eigen::Vector3d contact_normal = cContact.normal;
                        appendContactPoint(contact_pos, contact_normal, Arrow_list2_);
                        n_col++;
                        std::cout << "norm: " << cContact.normal.norm() << std::endl;
                    }
                }
            }
            std::cout << "n_col: " << n_col << std::endl;
            std::cout << "---n_col: " << res.contact_count << std::endl;
        }

        { // Distance field TEST: getCollisionGradients: (robot) getSelfProximityGradients, getIntraGroupProximityGradients, (world) getEnvironmentProximityGradients
            for(int t=0; t<100; t++) {
                ros::WallTime wt = ros::WallTime::now();
                hy_world_->getCollisionGradients(collision_request, res, *(hy_robot_->getCollisionRobotDistanceField()),
                                                 planning_scene->getCurrentStateNonConst(),
                                                 &acm_, gsr_);
                ROS_WARN_STREAM("call getCollisionGradients: " << (ros::WallTime::now() - wt));
            }
            int num_sphere = 0;
            for (size_t i = 0; i < gsr_->gradients_.size(); i++) // each link
            {
                for (size_t j = 0; j < gsr_->gradients_[i].sphere_locations.size(); j++) { // each sphere
                    auto sphere_center = gsr_->gradients_[i].sphere_locations[j];
                    auto sphere_radii = gsr_->gradients_[i].sphere_radii[j];
                    if ( (gsr_->gradients_[i].distances[j] - sphere_radii) <= -resolution) {
                        std::cout << "col!!" << std::endl;
                        appendSpherePoint(sphere_center[0], sphere_center[1], sphere_center[2], sphere_radii, 1, 0, 0,
                                          num_sphere, Marker_list_);
                    } else {
                        appendSpherePoint(sphere_center[0], sphere_center[1], sphere_center[2], sphere_radii, 0, 1, 0,
                                          num_sphere, Marker_list_);
                    }
                    num_sphere++;
                    std::cout << gsr_->gradients_[i].joint_name << ": " << gsr_->gradients_[i].collision << ": " <<
                    gsr_->gradients_[i].closest_distance << " / " << gsr_->gradients_[i].distances[j] << " | " << sphere_radii
                    << " = " << (gsr_->gradients_[i].distances[j] - sphere_radii) << std::endl;


                    double g_norm = gsr_->gradients_[i].gradients[j].norm();
                    double g_x = gsr_->gradients_[i].gradients[j].x() / g_norm;
                    double g_y = gsr_->gradients_[i].gradients[j].y() / g_norm;
                    double g_z = gsr_->gradients_[i].gradients[j].z() / g_norm;
                    std::cout << "======================================================= " << g_x << ", " << g_y
                              << ", " << g_z << " :norm: " << g_norm << std::endl;
                    if (g_norm > 0) {
                        appendContactPoint(Eigen::Vector3d{sphere_center[0], sphere_center[1], sphere_center[2]},
                                           Eigen::Vector3d{g_x, g_y, g_z},
                                           Arrow_list3_);
                    }
                }
            }
            std::cout << "num_sphere: " << num_sphere << std::endl;
        }
        for (uint i = 0; i < 100; i++) {
            marker_pub_.publish(Marker_list_);
            arrow_pub2_.publish(Arrow_list2_);
            arrow_pub3_.publish(Arrow_list3_);
        }
    }
    cout << "[MAIN_DEBUG] Debugging has been done." << endl;
    return 0;
}