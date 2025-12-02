to execute this program: 

1. start 2 terminals
2. colcon build within the /finalProject directory 
3. in the first terminal, type these commands:
   source /opt/ros/jazzy/setup.bash THEN 
   source YOUR MRTP DIRECTORY/MRTP/MRTP/install/setup.bash THEN 
   ros2 launch gazeboenvs tb4_warehouse.launch.py use_rviz:=true

4. in the second terminal, type these commands:
   source /opt/ros/jazzy/setup.bash THEN
   source YOUR MRTP DIRECTORY/MRTP/MRTP/install/setup.bash THEN 
   source WHEREVER YOU UNZIPPED THE FINAL PROJECT/finalProject/install/setup.bash
   python script: ros2 launch human_detection human_detection_launch.py






   

   
   

   
