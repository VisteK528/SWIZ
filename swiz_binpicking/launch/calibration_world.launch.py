from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context: LaunchContext, *args, **kwargs):
    robot_description_package = 'swiz_binpicking'
    package_name = 'swiz_binpicking'

    worlds_directory = os.path.join(
        get_package_share_directory(package_name),
        'worlds'
    )
    avaliable_worlds = {world for world in next(os.walk(worlds_directory))[1]}

    if len(avaliable_worlds) == 0:
        raise ValueError(f"No world available in specified directory")

    world_name = LaunchConfiguration('world').perform(context)    

    gazebo_models_share = os.path.join(
        os.path.dirname(get_package_share_directory('swiz_binpicking')),
        'swiz_binpicking', 'models'
    )

    world_file = None
    world_dir = ''
    media_dir = ''
    if world_name in avaliable_worlds:
        world_dir = os.path.join(
            worlds_directory,
            world_name
        )
        gazebo_models_share = os.path.join(
            world_dir,
            'models'
        )
        world_file = os.path.join(
            world_dir,
            world_name
        ) + '.world'

        media_dir = os.path.join(world_dir, 'media')
    else:
        raise ValueError(f"World named {world_name} doesn't exist")

    desc_share = os.path.dirname(
        get_package_share_directory('swiz_binpicking'))

    bridge_params = os.path.join(get_package_share_directory(
        robot_description_package), 'config', 'calibration_camera_gz_bridge.yaml')

    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': [
            '-r -v4 ', world_file], 'on_exit_shutdown': 'true', "emulate_tty": 'true'}.items()
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen',
        emulate_tty=True
    )

    gz_resource = f"{desc_share}:{world_dir}:{gazebo_models_share}"

    gz_environment = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gz_resource
    )

    return [
        gz_environment,
        gazebo,
        ros_gz_bridge
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='empty',
            description='World to load'
        ),

        OpaqueFunction(function=launch_setup),
    ])