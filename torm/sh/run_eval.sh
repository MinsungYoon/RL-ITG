N_RUN=1
START_CONF_TYPE=fix
ONLY_FIRST=true

#for PROB in hello rotation zig # zig square hello s sgvr kaist
#do
#  for START_CONF_IDX in {0..99}
#  do
#    for LR_METHOD in {0..1}
#    do
##      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB pirl_bc $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
##      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB pirl_rl $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
##      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB pirl_rl $N_RUN 1 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
##      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB torm $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB torm_jli $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#    done
#  done
#done

#for RND_PROB in {0..99}
#do
#  for START_CONF_IDX in {0..9}
#  do
#    for LR_METHOD in {0..1}
#    do
##      /root/catkin_ws/devel/lib/torm/torm_evaluation random_$RND_PROB pirl_bc $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
##      /root/catkin_ws/devel/lib/torm/torm_evaluation random_$RND_PROB pirl_rl $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
##      /root/catkin_ws/devel/lib/torm/torm_evaluation random_$RND_PROB pirl_rl $N_RUN 1 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
##      /root/catkin_ws/devel/lib/torm/torm_evaluation random_$RND_PROB torm $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#      /root/catkin_ws/devel/lib/torm/torm_evaluation random_$RND_PROB torm_jli $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#    done
#  done
#done

#for PROB in square
#do
#  for START_CONF_IDX in {10..49}
#  do
#    for LR_METHOD in 1
#    do
#      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB pirl_rl $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB pirl_rl $N_RUN 1 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB torm $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#      /root/catkin_ws/devel/lib/torm/torm_evaluation $PROB torm_jli $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#    done
#  done
#done

#for SCENE in {0..99}
#do
#  for RND_PROB in {0..9}
#  do
#    for START_CONF_IDX in {0..1}
#    do
#      for LR_METHOD in 1
#      do
#        /root/catkin_ws/devel/lib/torm/torm_evaluation random_obs_$SCENE\_$RND_PROB pirl_rl $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#        /root/catkin_ws/devel/lib/torm/torm_evaluation random_obs_$SCENE\_$RND_PROB pirl_rl $N_RUN 1 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#        /root/catkin_ws/devel/lib/torm/torm_evaluation random_obs_$SCENE\_$RND_PROB torm $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#        /root/catkin_ws/devel/lib/torm/torm_evaluation random_obs_$SCENE\_$RND_PROB torm_jli $N_RUN 0 $START_CONF_TYPE $START_CONF_IDX $ONLY_FIRST $LR_METHOD
#      done
#    done
#  done
#done


#/// rosrun torm torm_evaluation(0)
#/// [exp_name](1)
#/// [algorithm](2)
#/// [#iter](3)
#/// [model_index](4)
#/// [start_conf_type](5)
#/// [start_conf_idx](6)
#/// [0,1,2] lr_schedule mode [7]
#for start_conf_idx in {0..99}
#do
#done