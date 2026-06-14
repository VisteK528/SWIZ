import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def launch_setup(context: LaunchContext, *args, **kwargs):
    robot_description_package = "swiz_binpicking"
    package_name = "swiz_binpicking"

    worlds_directory = os.path.join(get_package_share_directory(package_name), "worlds")
    avaliable_worlds = {world for world in next(os.walk(worlds_directory))[1]}

    if len(avaliable_worlds) == 0:
        raise ValueError("No world available in specified directory")

    world_name = "ur"

    gazebo_models_share = os.path.join(
        os.path.dirname(get_package_share_directory("swiz_binpicking")),
        "swiz_binpicking",
        "models",
    )

    world_file = None
    world_dir = ""
    media_dir = ""
    if world_name in avaliable_worlds:
        world_dir = os.path.join(worlds_directory, world_name)
        gazebo_models_share = os.path.join(world_dir, "models")
        world_file = os.path.join(world_dir, world_name) + ".world"

        media_dir = os.path.join(world_dir, "media")
    else:
        raise ValueError(f"World named {world_name} doesn't exist")

    desc_share = os.path.dirname(get_package_share_directory("swiz_binpicking"))

    gripper_desc = os.path.dirname(get_package_share_directory("robotiq_description"))

    bridge_params = os.path.join(
        get_package_share_directory(robot_description_package),
        "config",
        "gz_bridge.yaml",
    )

    ur_controllers_config = os.path.join(
        get_package_share_directory(robot_description_package),
        "config",
        "ur_controllers.yaml",
    )

    xacro_file = os.path.join(
        get_package_share_directory(robot_description_package),
        "urdf",
        "ur_with_gripper.urdf.xacro",
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "--ros-args",
            "-p",
            f"config_file:={bridge_params}",
        ],
        output="screen",
        emulate_tty=True,
    )

    cylinder_ground_truth_pub = Node(
        package="swiz_detection",
        executable="ground_truth_publisher",
        output="screen",
        emulate_tty=True,
    )

    gz_resource = f"{gripper_desc}:{desc_share}:{world_dir}:{gazebo_models_share}"

    gz_environment = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", value=gz_resource
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "-c", "/controller_manager"],
    )

    custom_moveit_launch = os.path.join(
        get_package_share_directory("swiz_binpicking"),
        "launch",
        "moveit.launch.py",
    )

    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("ur_simulation_gz"),
                    "launch",
                    "ur_sim_moveit.launch.py",
                )
            ]
        ),
        launch_arguments={
            "world_file": world_file,
            "description_file": xacro_file,
            "controllers_file": ur_controllers_config,
            "moveit_launch_file": custom_moveit_launch,
        }.items(),
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        # use_sim_time=true prevents "jumped back in time" TF warnings
        # that occur when wall-clock timestamps mix with Gazebo sim-time TF data
        parameters=[{"use_sim_time": True}],
        arguments=[
            "0",
            "0.6",
            "1.2",
            "0",
            "1.5707",
            "0",
            "world",
            "rgbd_camera_link_optical",
        ],
    )

    return [
        gz_environment,
        ur_launch,
        gripper_controller_spawner,
        ros_gz_bridge,
        cylinder_ground_truth_pub,
        static_tf,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            OpaqueFunction(function=launch_setup),
        ]
    )
