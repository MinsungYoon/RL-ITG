for scene in {0..99}
do
  for path in {0..19}
  do
    rosrun torm torm_generate_start_conf random_obs_$scene\_$path
  done
done
#for prob in rotation
#do
#  rosrun torm torm_generate_start_conf $prob
#done