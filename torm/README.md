# TORM

You can see information about this code through this webpage. (We uploaded our paper and video.) </br>
https://sglab.kaist.ac.kr/TORM/</br>

1. Install the dependencies.

<pre> MoveIt!, Trac-ik </pre>

2. Setup your robot.

<pre> roslaunch moveit_setup_assistant setup_assistant.launch </pre>
  Make your PLANNING_GROUP

3. Execute the code.
  - Run the "move_group.launch" for your robot.
  - Run the problem using a launch file. (You can change a scene through the launch file.)
    <pre> roslaunch torm fetch.launch </pre>
  - Run the planning code.
    <pre> rosrun torm main </pre>
    - You should change the "main.cpp" according to your settings.
