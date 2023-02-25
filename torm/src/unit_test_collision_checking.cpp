#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf/tf.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <chrono>

#include <cmath>
#include <vector>
#include <numeric> // for accumulate

#include <torm/torm_problem.h>

using namespace std;

struct Mean_Std {
    double mean;
    double std;
};

Mean_Std standardDeviation(std::vector<double> v) {
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
    unsigned int num_dof = (unsigned int)joint_bounds.size();

    // Initialize planning interface
    planning_scene::PlanningScenePtr planning_scene(new planning_scene::PlanningScene(robot_model));
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

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

    // [Obj Removing Test] Remove collision object ////////////////////////////////////////////////////////////////////
//    std::map<std::string, moveit_msgs::CollisionObject> collision_objects_map2 = planning_scene_interface_.getObjects();
//    std::vector<moveit_msgs::CollisionObject> v_co2;
//    for(auto& kv : collision_objects_map2){ // moveit_msgs::CollisionObject kv
//        kv.second.operation = kv.second.REMOVE;
//        v_co2.push_back(kv.second);
//        planning_scene->processCollisionObjectMsg(kv.second); // [1] first function: remove internally (take effect), still remain obj visually.
//    }
//    planning_scene_interface_.applyCollisionObjects(v_co2); // [2] second function: remove only visually...
//    planning_scene->printKnownObjects(cout);
//    ros::Duration(0.5).sleep(); // In summary: need to call both [1] and [2]!

    // MAIN Part
    ros::Publisher display_pub_;
    display_pub_ = node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);

    // collision checker
    ROS_INFO("Start collision checker");
    collision_detection::CollisionRequest collision_request;
//    collision_request.group_name = PLANNING_GROUP;
//    collision_request.distance = true;
//    collision_request.cost = true;
//    collision_request.contacts = true;
//    collision_request.max_contacts = 5;
//    collision_request.max_contacts_per_pair = 1;
//    collision_request.max_cost_sources = 1;
//    collision_request.min_cost_density = 0.2;
//    collision_request.verbose = true;

    collision_request.group_name = PLANNING_GROUP;
    collision_request.contacts = false;
    collision_request.verbose = false;
    collision_request.distance = false;
    collision_request.cost = false;
    ROS_INFO("End collision checker");

    collision_detection::CollisionResult collision_result;
    robot_state::RobotState rs = planning_scene->getCurrentStateNonConst();

    moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);

    int n_sample = 100;
    std::vector<double> time_holder;
    time_holder.resize(n_sample);

    std::vector<std::vector<double> > trajectory_points_vis;
    while(trajectory_points_vis.size()<n_sample){
        std::vector<double> random_config = move_group.getRandomJointValues();

        collision_result.clear();
        rs.setJointGroupPositions(PLANNING_GROUP, random_config);

        if(!rs.getJointModelGroup(PLANNING_GROUP)->satisfiesPositionBounds(random_config.data())){
            ROS_ERROR("Joint violation!!!");
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        planning_scene->checkCollision(collision_request, collision_result, rs);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
//        std::cout << duration << std::endl;
        time_holder.push_back(duration);

        if(collision_result.collision){
            collision_result.print();
            trajectory_points_vis.push_back(random_config);
        }
    }
    cout << "[MAIN] collected size: " << trajectory_points_vis.size() << endl;
    Mean_Std result = standardDeviation(time_holder);
    std::cout << "[MAIN] Collision checking avg. time (microseconds): " << result.mean << " (" << result.std << ") " << std::endl;

    moveit_msgs::DisplayTrajectory display_trajectory_;
    moveit_msgs::RobotTrajectory robot_traj_;
    display_trajectory_.trajectory.clear();
    robot_traj_.joint_trajectory.joint_names = indices;
    robot_traj_.joint_trajectory.header.stamp = ros::Time::now();
    robot_traj_.joint_trajectory.points.clear();
    robot_traj_.joint_trajectory.points.resize(trajectory_points_vis.size());
    for (uint i = 0; i < robot_traj_.joint_trajectory.points.size(); i++) {
        robot_traj_.joint_trajectory.points[i].positions.resize(indices.size());
        for(uint j = 0; j < indices.size(); j++){
            robot_traj_.joint_trajectory.points[i].positions[j] = trajectory_points_vis[i][j];
        }
    }
    display_trajectory_.trajectory.push_back(robot_traj_);
    display_pub_.publish(display_trajectory_);
    ros::Duration(1.0).sleep();


    cout << "[MAIN_DEBUG] Debugging has been done." << endl;
    return 0;
}