from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition

from ament_index_python.packages import get_package_share_directory
import os
from datetime import datetime

def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('auto_drive')

    # Получаем конфигурации launch
    scene = LaunchConfiguration('scene').perform(context)
    model_name = LaunchConfiguration('model_name').perform(context)
    use_cuda = LaunchConfiguration('use_cuda').perform(context)
    
    # Автоматически определяем путь к модели
    model_path = os.path.join(
        pkg_share, '..', '..', '..', '..', 'models', f'{model_name}.onnx'
    )
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Пути для bag
    play_bag_path = os.path.join(
        pkg_share, '..', '..', '..', '..', 'scene_records', 
        f'NuScenes-v1.0-mini-scene-{scene}'
    )

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    record_bag_path = os.path.join(
        pkg_share, '..', '..', '..', '..', 'inference_records', 
        f'{scene}-{model_name}-record-{timestamp}'
    )

    # Определяем дополнительные аргументы для ros2 bag play в зависимости от use_sim_time

    return [
        # Запуск perception_node
        Node(
            package='auto_drive',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'use_sensor_time': True,
                'onnx_path': model_path,
                'use_cuda': True if use_cuda=='true' else False,
            }],
        ),

        # Воспроизведение bag
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', play_bag_path, '--clock'
            ],
            cwd=pkg_share,
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-s', 'mcap',
                '-o', record_bag_path,
                '/bevfusion/detections',
            ],
            cwd=pkg_share,
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        # Основные параметры
        DeclareLaunchArgument(
            'scene',
            default_value='0553',
            description='Which scene to play (e.g., 0103, 0104, etc.)'
        ),
        DeclareLaunchArgument(
            'model_name',
            default_value='bevfusion_lidar_cam_s',
            description='Name of the ONNX model without extension (e.g., bevfusion_lidar_cam_s, bevfusion_lidar_cam_m)'
        ),
        DeclareLaunchArgument(
            'use_cuda',
            default_value='false',
            description='Whether to use CUDA for inference (true/false)'
        ),

        OpaqueFunction(function=launch_setup)
    ])
