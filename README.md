## Learning-based Initialization of Trajectory Optimization for Path-following Problems of Redundant Manipulators (ICRA23)
The contents of each folder are as follows.
- **pirl** (path-wise IK using reinforcement learning): code related to reinforcement learning and behavior cloning to learn learning-based initializers (RL-ITG and BC-ITG)
- **pirl_gazebo**: Creates a 3-D occupancy grid map by randomly configuring the external environment
- **pirl_msgs**: consists of messages or services on ROS
- **pirl_vae**: variational autoencoder learning code for learning scene_encoding_vector
- **torm** (cloned from [link](https://github.com/cheulkang/TORM)): TORM planner and Greedy and Linear initializer baselines
- Velodyne Simulator (cloned from [link](https://github.com/florianshkurti/velodyne_simulator)): lidar is used to obtain a point cloud for the external environment in the gazebo
- fetch_description (cloned from [link](https://github.com/ZebraDevs/fetch_ros)): To use the fetch robot model
- fetch_moveit_config (cloned from [link](https://github.com/ZebraDevs/fetch_ros)): To use the fetch robot model






















