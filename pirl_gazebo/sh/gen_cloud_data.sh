for s in {0..99}
do
  rosservice call gazebo/delete_model 'model_name: obs_0'
  rosservice call gazebo/delete_model 'model_name: obs_1'
  rosservice call gazebo/delete_model 'model_name: obs_2'
  rosservice call gazebo/delete_model 'model_name: obs_3'
  rosservice call gazebo/delete_model 'model_name: obs_4'

  for o in {0..4}
  do
    rosrun gazebo_ros spawn_model -file /data/pirl_data/eval_show/scene_with_box_sdf/$s/obs_$o.sdf -sdf -model obs_$o
  done
  python ../script/save_pointcloud.py True $s
done