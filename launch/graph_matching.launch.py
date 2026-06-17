import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    graph_structure_topic_arg = DeclareLaunchArgument(
        "graph_structure_topic",
        default_value="/s_graphs/graph_structure",
        description="Topic for graph structure",
    )

    config = os.path.join(
        get_package_share_directory('graph_matching'),
        'config',
        'params.yaml'
    )

    graph_matching_node = Node(
        package='graph_matching',
        executable='graph_matching',
        parameters=[config],
        remappings=[
            ('graph_matching/graphs', LaunchConfiguration("graph_structure_topic")),
        ]
    )

    return LaunchDescription([
        graph_structure_topic_arg,
        graph_matching_node,
    ])