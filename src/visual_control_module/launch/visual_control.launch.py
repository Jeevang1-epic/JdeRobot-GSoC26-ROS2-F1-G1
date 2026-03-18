from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='visual_control_module',
            executable='visual_node',
            name='visual_node',
            output='screen'
        )
    ])