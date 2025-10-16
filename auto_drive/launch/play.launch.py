from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('auto_drive')

    # Получаем параметры launch
    scene = LaunchConfiguration('scene').perform(context)
    record_name = LaunchConfiguration('record_name').perform(context)

    # Пути к bag
    scene_bag = os.path.join(pkg_share, '..', '..', '..', '..', 'scene_records', f'NuScenes-v1.0-mini-scene-{scene}')
    inference_bag = os.path.join(pkg_share, '..', '..', '..', '..', 'inference_records', record_name)

    return [
        # Воспроизведение сцены
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', scene_bag,
                '--clock'
            ],
            cwd=pkg_share,
        ),

        # Воспроизведение инференс-записи
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', inference_bag,
                '--clock'
            ],
            cwd=pkg_share,
        ),
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'scene',
            default_value='0103',
            description='Which scene bag to play'
        ),
        DeclareLaunchArgument(
            'record_name',
            default_value='0103-record-20251016-110000',
            description='Which inference record to play'
        ),

        OpaqueFunction(function=launch_setup)
    ])
