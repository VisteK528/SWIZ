import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type").perform(context)
    use_sim_time = (
        LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    )

    urdf_xacro = os.path.join(
        get_package_share_directory("swiz_binpicking"),
        "urdf",
        "ur_with_gripper.urdf.xacro",
    )
    srdf_xacro = os.path.join(
        get_package_share_directory("swiz_binpicking"),
        "urdf",
        "ur_with_gripper.srdf.xacro",
    )

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description(
            file_path=urdf_xacro,
            mappings={"name": "ur", "ur_type": ur_type},
        )
        .robot_description_semantic(file_path=srdf_xacro)
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    return [move_group_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur5e"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument("launch_servo", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
