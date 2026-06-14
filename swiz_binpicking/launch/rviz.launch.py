import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction, TimerAction
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def make_rviz_node(context, *args, **kwargs):
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
            mappings={"name": "ur", "ur_type": "ur5e"},
        )
        .robot_description_semantic(file_path=srdf_xacro)
        .to_moveit_configs()
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"]
    )

    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_moveit",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
                {"use_sim_time": False},
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            TimerAction(
                period=2.0,
                actions=[OpaqueFunction(function=make_rviz_node)],
            )
        ]
    )
