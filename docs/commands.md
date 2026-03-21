## Uruchomienie świata kalibracyjnego
```bash
ros2 launch swiz_binpicking calibration_world.launch.py
```

## Uruchomienie świata z manipulatorem
```bash
ros2 launch swiz_binpicking ur_world.launch.py
```

### Podgląd video z kamery

```bash
ros2 run rqt_image_view rqt_image_view
```

### Ruch grippera

```bash
ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.2, max_effort: 50.0}}"
```