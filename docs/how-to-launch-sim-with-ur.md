

```bash
colcon build --symlink-install && source install/setup.bash
```
! Temporarily

```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/ws/install/robotiq_description/share
```

Launch:

```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py description_file:=src/my_ur_robot_description/urdf/my_ur_description.urdf.xacro world_file:=src/world.sdf 
```

How to move gripper:

```bash
ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.2, max_effort: 50.0}}"
```

