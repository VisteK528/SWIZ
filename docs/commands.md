```bash
ros2 run ros_gz_bridge parameter_bridge "front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" "front_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image" "front_camera/image@sensor_msgs/msg/Image[gz.msgs.Image"
```

```bash
ros2 run ros_gz_bridge parameter_bridge "front_camera@sensor_msgs/msg/Image[gz.msgs.Image"
```

```bash
ros2 run rqt_image_view rqt_image_view
```