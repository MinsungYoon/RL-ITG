import numpy as np
from numpy import random as RD
import rospy

# class Box(object):
#     def __init__(self, low, high, dtype=np.float32):
#         self.dtype = np.dtype(dtype)
#         self.shape = low.shape
#         self.low = np.full(self.shape, low)
#         self.high = np.full(self.shape, high)
#         self.low = self.low.astype(self.dtype)
#         self.high = self.high.astype(self.dtype)
#     def sample(self):
#         sample = np.empty(self.shape)
#         for i in range(self.shape[0]):
#             sample[i] = RD.uniform(self.low[i], self.high[i])
#         return sample

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class PathVisualizer(object):
    def __init__(self, is_test=False, is_second_path=False):
        self.display_pub_ = rospy.Publisher('/move_group/display_planned_path' + ('' if not is_test else '_test'),
                                            DisplayTrajectory, queue_size=100)
        self.eepath_pub_ = rospy.Publisher('/visualization_ee_path' + ('' if not is_test else '_test'), Marker,
                                           queue_size=100)
        if is_second_path:
            self.eepath_pub2_ = rospy.Publisher('/visualization_ee_path2' + ('' if not is_test else '_test'), Marker,
                                           queue_size=100)
        self.marker_goal_pub_ = rospy.Publisher('/visualization_goal_marker' + ('' if not is_test else '_test'), Marker,
                                                queue_size=100)
        self.marker_start_pub_ = rospy.Publisher('/visualization_start_marker' + ('' if not is_test else '_test'),
                                                 Marker, queue_size=100)
        self.arrows_pub_ = rospy.Publisher("/visualization_arrows" + ('' if not is_test else '_test'), MarkerArray,
                                           queue_size=100)
        if is_second_path:
            self.arrows_pub2_ = rospy.Publisher("/visualization_arrows2" + ('' if not is_test else '_test'), MarkerArray,
                                           queue_size=100)
        print("[Visualizer] wait for publisher to have connections --->")
        while not (self.display_pub_.get_num_connections() and self.eepath_pub_.get_num_connections() and
                   self.marker_goal_pub_.get_num_connections() and self.marker_start_pub_.get_num_connections() and
                   self.arrows_pub_.get_num_connections()):
            pass
        if is_second_path:
            while not (self.eepath_pub2_.get_num_connections() and self.arrows_pub2_.get_num_connections()):
                pass
        print("[Visualizer] <--- connections to subscribers have been set-up.")

        self.base_frame = rospy.get_param("/robot/planning_base_link")
        self.joint_names = rospy.get_param("/robot/joint_names")
        self.n_dof = rospy.get_param("/robot/n_dof")

    def pub_path(self, path): # input: config path
        display_trajectory_ = DisplayTrajectory()
        robot_traj_ = RobotTrajectory()
        robot_traj_.joint_trajectory.joint_names = self.joint_names

        for conf in path:
            joint_traj_point_ = JointTrajectoryPoint()
            for j in conf:
                joint_traj_point_.positions.append(j)
            robot_traj_.joint_trajectory.points.append(joint_traj_point_)

        display_trajectory_.trajectory.append(robot_traj_)
        self.display_pub_.publish(display_trajectory_)


    def pub_eepath_strip(self, eepath, second_path=False): # input: SE(3) path
        marker = Marker(
            header=Header(frame_id=self.base_frame),
            action=Marker.ADD,
            id=2,
            type=Marker.LINE_STRIP,
        )
        marker.scale.x = 0.01
        for pos in eepath:
            marker.points.append(Point(pos[0], pos[1], pos[2]))
            if not second_path:
                marker.colors.append(ColorRGBA(1.0, 0.0, 0.0, 1.0))
            else:
                marker.colors.append(ColorRGBA(0.0, 0.0, 1.0, 1.0))
        if not second_path:
            self.eepath_pub_.publish(marker)
        else:
            self.eepath_pub2_.publish(marker)

    def pub_eepath_arrow(self, eepath, second_path=False):
        self.pub_arrows(eepath, second_path)

    def pub_arrow(self, pos, color='b'): # for start and goal visualization
        marker = Marker(
            header=Header(frame_id=self.base_frame),
            id=(0 if color == 'r' else 1),
            type=Marker.ARROW,
            pose=Pose(Point(pos[0], pos[1], pos[2]), Quaternion(pos[3], pos[4], pos[5], pos[6])),
            scale=Vector3(0.15, 0.03, 0.03),
            color=(ColorRGBA(1.0, 0.0, 0.0, 1.0) if color == 'r' else ColorRGBA(0.0, 0.0, 1.0, 1.0)),
        )
        if color == 'r':
            self.marker_goal_pub_.publish(marker)
        else:
            self.marker_start_pub_.publish(marker)

    def pub_arrows(self, poses, second_path): # input: SE(3) path
        markerArray = MarkerArray()
        for i, pos in enumerate(poses):
            marker = Marker(
                header=Header(frame_id=self.base_frame),
                action=Marker.ADD,
                id=(i + 3 if not second_path else i + 500),
                type=Marker.ARROW,
                pose=Pose(Point(pos[0], pos[1], pos[2]), Quaternion(pos[3], pos[4], pos[5], pos[6])),
                scale=Vector3(0.09, 0.005, 0.002),
                color=(ColorRGBA(1.0, 0.2, 0.0, 1.0) if not second_path else ColorRGBA(0.0, 0.2, 1.0, 1.0)),
            )
            markerArray.markers.append(marker)
        if not second_path:
            for _ in range(10):
                self.arrows_pub_.publish(markerArray)
        else:
            for _ in range(10):
                self.arrows_pub2_.publish(markerArray)

    def clear(self):
        markerArray = MarkerArray()
        marker = Marker(
            header=Header(frame_id=self.base_frame),
            action=Marker.DELETEALL,
        )
        markerArray.markers.append(marker)
        for _ in range(10):
            self.arrows_pub_.publish(markerArray)
            self.arrows_pub2_.publish(markerArray)
            self.eepath_pub_.publish(marker)
            self.eepath_pub2_.publish(marker)
