for s in s
do
  for o in {0..4}
  do
    rosrun gazebo_ros spawn_model -file /data/pirl_data/eval/scene_sdf/$s/obs_$o.sdf -sdf -model obs_$o
  done

  python ../script/save_pointcloud.py False $s

  rosservice call gazebo/delete_model 'model_name: obs_0'
  rosservice call gazebo/delete_model 'model_name: obs_1'
  rosservice call gazebo/delete_model 'model_name: obs_2'
  rosservice call gazebo/delete_model 'model_name: obs_3'
  rosservice call gazebo/delete_model 'model_name: obs_4'
done