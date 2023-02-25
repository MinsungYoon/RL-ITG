/* Author: Mincheul Kang */

#include <torm/torm_debug.h>

namespace torm {
    TormDebug::TormDebug(const planning_scene::PlanningSceneConstPtr& planning_scene,
                         std::string planning_group,
                         std::string frame_id,
                         std::string planning_base_link,
                         TormIKSolver& iksolver):
            planning_scene_(planning_scene),
            planning_group_(planning_group),
            frame_id_(frame_id),
            planning_base_link_(planning_base_link),
            iksolver_(iksolver){
        ros::NodeHandle node_handle("~");
        joint_pub_ = node_handle.advertise<sensor_msgs::JointState>("/move_group/fake_controller_joint_states", 10);
        display_pub_ = node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
        marker_pub_ = node_handle.advertise<visualization_msgs::Marker>("/visualization_marker", 10);
        arrow1_pub_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow1", 10);
        arrow2_pub_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow2", 10);
        arrow3_pub_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow3", 10);
        arrow4_pub_ = node_handle.advertise<geometry_msgs::PoseArray>("/visualization_arrow4", 10);

        const Eigen::Isometry3d &baseT = planning_scene->getFrameTransform(planning_base_link);
        tf::transformEigenToKDL(baseT, baseKDL_);

        line_list_.header.frame_id = frame_id;
        line_list_.header.stamp = ros::Time::now();
        line_list_.ns = "torm";
        line_list_.action = visualization_msgs::Marker::ADD;
        line_list_.pose.orientation.w = 1.0;
        line_list_.id = 2;
        line_list_.type = visualization_msgs::Marker::LINE_LIST;
        line_list_.scale.x = 0.01;
        line_list_.scale.y = 0.01;
        line_list_.color.r = 1.0;
        line_list_.color.a = 0.5;
        line_list_.colors.clear();

        Arrow1_list_.header.frame_id = frame_id;
        Arrow1_list_.header.stamp = ros::Time::now();
        Arrow2_list_.header.frame_id = frame_id;
        Arrow2_list_.header.stamp = ros::Time::now();
        Arrow3_list_.header.frame_id = frame_id;
        Arrow3_list_.header.stamp = ros::Time::now();
        Arrow4_list_.header.frame_id = frame_id;
        Arrow4_list_.header.stamp = ros::Time::now();

        Markers_pub_ = node_handle.advertise<visualization_msgs::MarkerArray>("/visualization_contact_vectors", 10);
        Marker_list_.markers.clear();

        this->clear();
    }

    void TormDebug::visualizeConfiguration(std::vector<std::string> &indices, std::vector<double> &conf) {
        js_.position.clear();
        js_.name.clear();
        for(int j = 0; j < indices.size(); j++){
            js_.name.push_back(indices[j]);
            js_.position.push_back(conf[j]);
        }
        js_.header.stamp = ros::Time::now();
        joint_pub_.publish(js_);
        ros::Duration(0.1).sleep();
    }
    void TormDebug::visualizeConfiguration(std::vector<std::string> &indices, KDL::JntArray &conf) {
        js_.position.clear();
        js_.name.clear();
        for(int j = 0; j < indices.size(); j++){
            js_.name.push_back(indices[j]);
            js_.position.push_back(conf(j));
        }
        js_.header.stamp = ros::Time::now();
        joint_pub_.publish(js_);
    }

    void TormDebug::visualizeTrajectory(std::vector<std::string> &indices, std::vector<std::vector<double>>& traj) {
        display_trajectory_.trajectory.clear();

        robot_traj_.joint_trajectory.joint_names = indices;
        robot_traj_.joint_trajectory.header.stamp = ros::Time::now();
        robot_traj_.joint_trajectory.points.clear();
        robot_traj_.joint_trajectory.points.resize(traj.size());

        for (uint i = 0; i < robot_traj_.joint_trajectory.points.size(); i++) {
            robot_traj_.joint_trajectory.points[i].positions.resize(indices.size());
            for(uint j = 0; j < indices.size(); j++){
                robot_traj_.joint_trajectory.points[i].positions[j] = traj[i][j];
            }
        }
        display_trajectory_.trajectory.push_back(robot_traj_);
        display_pub_.publish(display_trajectory_);
        ros::Duration(0.1).sleep();
    }
    void TormDebug::visualizeTrajectory(std::vector<std::string> &indices, std::vector<KDL::JntArray>& traj) {
        display_trajectory_.trajectory.clear();

        robot_traj_.joint_trajectory.joint_names = indices;
        robot_traj_.joint_trajectory.header.stamp = ros::Time::now();
        robot_traj_.joint_trajectory.points.clear();
        robot_traj_.joint_trajectory.points.resize(traj.size());

        for (uint i = 0; i < robot_traj_.joint_trajectory.points.size(); i++) {
            robot_traj_.joint_trajectory.points[i].positions.resize(indices.size());
            for(uint j = 0; j < indices.size(); j++){
                robot_traj_.joint_trajectory.points[i].positions[j] = traj[i](j);
            }
        }
        display_trajectory_.trajectory.push_back(robot_traj_);
        display_pub_.publish(display_trajectory_);
        ros::Duration(0.1).sleep();
    }

    void TormDebug::make_EEpose_line(geometry_msgs::Point p1, geometry_msgs::Point p2, int color) {
        // 0: red, 1: green, 2: blue
        std_msgs::ColorRGBA lineColor;
        lineColor.a = 0.7;

        if(color == 0){
            lineColor.r = 1.0;
        }
        else if(color == 1){
            lineColor.g = 1.0;
        }
        else if(color == 2){
            lineColor.b = 1.0;
        }
        else if(color == 3){
            lineColor.r = 0.0;
            lineColor.g = 0.0;
            lineColor.b = 0.0;
        }
        else{
            lineColor.r = 1.0;
            lineColor.g = 1.0;
            lineColor.b = 1.0;
        }
        line_list_.points.push_back(p1);
        line_list_.colors.push_back(lineColor);

        line_list_.points.push_back(p2);
        line_list_.colors.push_back(lineColor);
    }

    void TormDebug::publish_EEpose_line_and_arrow(){
        for(uint i = 0; i < 5; i++){
            marker_pub_.publish(line_list_);
            arrow1_pub_.publish(Arrow1_list_);
            arrow2_pub_.publish(Arrow2_list_);
            arrow3_pub_.publish(Arrow3_list_);
            arrow4_pub_.publish(Arrow4_list_);
            ros::Duration(0.1).sleep();
        }
    }

    void TormDebug::clear(){
        line_list_.action = visualization_msgs::Marker::DELETEALL;
        for(uint i = 0; i < 10; i++){
            marker_pub_.publish(line_list_);
        }
        line_list_.action = visualization_msgs::Marker::ADD;
        line_list_.points.clear();
        line_list_.colors.clear();
        Arrow1_list_.poses.clear();
        Arrow2_list_.poses.clear();
        Arrow3_list_.poses.clear();
        Arrow4_list_.poses.clear();
    }


    void TormDebug::publishEETrajectory(std::vector<std::vector<double>>& path, int target_idx) {
        std::vector<KDL::Frame> targetPoses;
        targetPoses.reserve(path.size());
        for (int i = 0; i < path.size(); i++) {
            KDL::Frame f;
            f.p[0] = path[i][0];
            f.p[1] = path[i][1];
            f.p[2] = path[i][2];
            f.M = KDL::Rotation::Quaternion(path[i][3], path[i][4], path[i][5], path[i][6]);
            targetPoses.push_back(f);
        }
        this->publishEETrajectory(targetPoses, target_idx);
    }


    void TormDebug::publishEETrajectory(std::vector<KDL::Frame>& targetPoses, int target_idx){
        for (uint i = 0; i < targetPoses.size() - 1; i++) {
            geometry_msgs::Point p1;
            p1.x = targetPoses[i].p.data[0] + baseKDL_.p.data[0];
            p1.y = targetPoses[i].p.data[1] + baseKDL_.p.data[1];
            p1.z = targetPoses[i].p.data[2] + baseKDL_.p.data[2];

            geometry_msgs::Point p2;
            p2.x = targetPoses[i + 1].p.data[0] + baseKDL_.p.data[0];
            p2.y = targetPoses[i + 1].p.data[1] + baseKDL_.p.data[1];
            p2.z = targetPoses[i + 1].p.data[2] + baseKDL_.p.data[2];

            make_EEpose_line(p1, p2, target_idx);

            geometry_msgs::Pose pose;
            pose.position.x = targetPoses[i].p.data[0] + baseKDL_.p.data[0];
            pose.position.y = targetPoses[i].p.data[1] + baseKDL_.p.data[1];
            pose.position.z = targetPoses[i].p.data[2] + baseKDL_.p.data[2];
            double x, y, z ,w;
            targetPoses[i].M.GetQuaternion(x, y, z ,w);
            pose.orientation.x = x;
            pose.orientation.y = y;
            pose.orientation.z = z;
            pose.orientation.w = w;

            if(target_idx==0){
                Arrow1_list_.poses.push_back(pose);
            }else if(target_idx==1){
                Arrow2_list_.poses.push_back(pose);
            }else if(target_idx==2){
                Arrow3_list_.poses.push_back(pose);
            }else if(target_idx==3){
                Arrow4_list_.poses.push_back(pose);
            }
        }
        geometry_msgs::Pose pose;
        pose.position.x = targetPoses[targetPoses.size()-1].p.data[0] + baseKDL_.p.data[0];
        pose.position.y = targetPoses[targetPoses.size()-1].p.data[1] + baseKDL_.p.data[1];
        pose.position.z = targetPoses[targetPoses.size()-1].p.data[2] + baseKDL_.p.data[2];
        double x, y, z ,w;
        targetPoses[targetPoses.size()-1].M.GetQuaternion(x, y, z ,w);
        pose.orientation.x = x;
        pose.orientation.y = y;
        pose.orientation.z = z;
        pose.orientation.w = w;

        if(target_idx==0){
            Arrow1_list_.poses.push_back(pose);
        }else if(target_idx==1){
            Arrow2_list_.poses.push_back(pose);
        }else if(target_idx==2){
            Arrow3_list_.poses.push_back(pose);
        }else if(target_idx==3){
            Arrow4_list_.poses.push_back(pose);
        }
        publish_EEpose_line_and_arrow();
    }

    void TormDebug::show(std::vector<KDL::Frame>& targetPoses, Eigen::Matrix<double, -1, -1> optimizedJoints,
                         double sleep_time, int plot_idx, int n_middle_interpol){
        this->clear();

        std::vector<KDL::Frame> targetPoses_buf_;
        Eigen::MatrixXd trajectory_buf_;

        if(n_middle_interpol>0){
            int n_interval = n_middle_interpol + 1;

            targetPoses_buf_.reserve((n_middle_interpol + 1)*targetPoses.size()-n_middle_interpol);
            for(int i = 0; i < targetPoses.size()-1; i++){
                targetPoses_buf_.push_back(targetPoses[i]);
                for(int j = 1; j <= n_middle_interpol; j++){
                    KDL::Frame inter_f;
                    inter_f.p[0] = j*(targetPoses[i+1].p[0] - targetPoses[i].p[0])/(n_interval) + targetPoses[i].p[0];
                    inter_f.p[1] = j*(targetPoses[i+1].p[1] - targetPoses[i].p[1])/(n_interval) + targetPoses[i].p[1];
                    inter_f.p[2] = j*(targetPoses[i+1].p[2] - targetPoses[i].p[2])/(n_interval) + targetPoses[i].p[2];

                    KDL::Vector rot_axis;
                    double angle = (targetPoses[i].M.Inverse() * targetPoses[i+1].M).GetRotAngle(rot_axis);
                    inter_f.M = targetPoses[i].M * KDL::Rotation::Rot2(rot_axis, j*angle/n_interval);
                    targetPoses_buf_.push_back(inter_f);
                }
            }
            targetPoses_buf_.push_back(targetPoses[targetPoses.size()-1]);

            trajectory_buf_.resize((n_interval)*optimizedJoints.rows()-n_middle_interpol, optimizedJoints.cols());
            for(int i = 0; i < optimizedJoints.rows()-1; i++){
                for(int j = 0; j <= n_middle_interpol; j++) {
                    trajectory_buf_.row(n_interval * i + j) =
                            (j*optimizedJoints.row(i + 1) + (n_interval-j)*optimizedJoints.row(i))/(n_interval);
                }
            }
            trajectory_buf_.row((n_interval)*optimizedJoints.rows()-n_middle_interpol-1) =
                    optimizedJoints.row(optimizedJoints.rows()-1);
        }else{
            targetPoses_buf_.reserve(targetPoses.size());
            trajectory_buf_.resize(optimizedJoints.rows(), optimizedJoints.cols());

            targetPoses_buf_ = targetPoses;
            trajectory_buf_ = optimizedJoints;
        }


        this->publishEETrajectory(targetPoses_buf_, 0);
        ros::Duration(0.2).sleep();

        KDL::JntArray q_c(trajectory_buf_.cols());
        std::vector<KDL::JntArray> trajectory_points; // for FK
        std::vector<std::vector<double> > trajectory_points_vis; // for pub joints
        std::vector<double> q(trajectory_buf_.cols());
        for (int i = 0; i < trajectory_buf_.rows(); i++) {
            for (int j = 0; j < trajectory_buf_.cols(); j++) {
                q_c(j) = trajectory_buf_.row(i)(j);
                q[j] = trajectory_buf_.row(i)(j);
            }
            trajectory_points.push_back(q_c);
            trajectory_points_vis.push_back(q);
        }
        std::vector<KDL::Frame> optimizedPoses;
        for (uint i = 0; i < trajectory_buf_.rows(); i++) {
            KDL::Frame ee;
            iksolver_.fkSolver(trajectory_points[i], ee);
            optimizedPoses.push_back(ee);
        }
        this->publishEETrajectory(optimizedPoses, plot_idx);
        ros::Duration(0.2).sleep();

        // visualize joint trajectory
        std::vector<std::string> j_names = planning_scene_->getRobotModel()->getJointModelGroup(planning_group_)->getActiveJointModelNames();
        this->visualizeTrajectory(j_names, trajectory_points_vis);
        ros::Duration(sleep_time).sleep();
    }

    void TormDebug::addContactVector(double px, double py, double pz, double dx, double dy, double dz, bool normalize){

        visualization_msgs::Marker m;
        m.header.frame_id = "base_link";
        m.header.stamp = ros::Time::now();
        m.id = marker_counter_++;
        m.type = visualization_msgs::Marker::ARROW;
        m.action = visualization_msgs::Marker::ADD;
//        m.lifetime = ros::Duration(1.0);
        // vector spec
        geometry_msgs::Point p_start;
        p_start.x = px;
        p_start.y = py;
        p_start.z = pz;
        geometry_msgs::Point p_end;
        if (!normalize){
            p_end.x = px + dx;
            p_end.y = py + dy;
            p_end.z = pz + dz;
        }else{
            p_end.x = px + dx/std::sqrt(dx*dx + dy*dy + dz*dz);
            p_end.y = py + dy/std::sqrt(dx*dx + dy*dy + dz*dz);
            p_end.z = pz + dz/std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        m.points.push_back(p_start);
        m.points.push_back(p_end);

        m.scale.x = 0.005; // shaft diameter
        m.scale.y = 0.01; // head diameter
        m.scale.z = 0.005; // head length.

        // color spec
        m.color.a = 1.0;
        m.color.r = 1.0;
        m.color.g = 1.0;
        m.color.b = 0.5;

        Marker_list_.markers.push_back(m);
    }
    void TormDebug::publishContactVectors(){
        for(uint i = 0; i < 5; i++){
            Markers_pub_.publish(Marker_list_);
            ros::Duration(0.1).sleep();
        }
    }
    void TormDebug::clearContactVectors(){
        marker_counter_ = 0;
        Marker_list_.markers.clear();
        visualization_msgs::Marker m;
        m.action = visualization_msgs::Marker::DELETEALL;
        Marker_list_.markers.push_back(m);
        for(uint i = 0; i < 5; i++){
            Markers_pub_.publish(Marker_list_);
            ros::Duration(0.1).sleep();
        }
        Marker_list_.markers.clear();
    }
}
