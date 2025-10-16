from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import ExecuteProcess

from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime

def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('auto_drive')

    # Получаем конфигурации launch
    scene = LaunchConfiguration('scene').perform(context)
    model_path = LaunchConfiguration('model_path').perform(context)

    # Пути для bag
    play_bag_path = os.path.join(pkg_share, '..', '..', '..', '..', 'scene_records', f'NuScenes-v1.0-mini-scene-{scene}')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    record_bag_path = os.path.join(pkg_share, '..', '..', '..', '..', 'inference_records', f'{scene}-record-{timestamp}')

    return [
        # Запуск perception_node
        Node(
            package='auto_drive',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'use_sensor_time': True,
                'onnx_path': model_path
            }]
        ),

        # Воспроизведение bag
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', play_bag_path,
                '--clock'
            ],
            cwd=pkg_share,
        ),

        # Запись топика /bevfusion/detections
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-s', 'mcap',
                '-o', record_bag_path,
                '/bevfusion/detections',
                '--use-sim-time'
            ],
            cwd=pkg_share,
        )
    ]

def generate_launch_description():
    # Параметры launch
    default_model_path = os.path.abspath(
        os.path.join(
            get_package_share_directory('auto_drive'),
            '..', '..', '..', '..', 'models', 'bevfusion_lidar_cam_s.onnx'
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'scene',
            default_value='0103',
            description='Which scene to play'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='Path to the ONNX model for perception_node'
        ),

        OpaqueFunction(function=launch_setup)
    ])