# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy, os
import time
import copy
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt

from rclpy.node import Node
from .utils import *
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros.buffer_interface import BufferInterface
import ament_index_python
from ament_index_python.packages import get_package_share_directory
import tf2_geometry_msgs 
from visualization_msgs.msg import Marker as MarkerMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import Vector3 as Vector3Msg
from geometry_msgs.msg import PointStamped as PointStampedMsg
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Transform as TransformMsg
from geometry_msgs.msg import TransformStamped as TransformStampedMsg
from std_msgs.msg import ColorRGBA as ColorRGBSMsg
from std_msgs.msg import Header as HeaderMsg
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.parameter import Parameter
from rclpy.parameter import ParameterType

from situational_graphs_reasoning_msgs.srv import SubgraphMatch as SubgraphMatchSrv
from situational_graphs_reasoning_msgs.msg import Graph as GraphMsg
from situational_graphs_reasoning_msgs.msg import Match as MatchMsg
from situational_graphs_reasoning_msgs.msg import Node as NodeMsg
from situational_graphs_reasoning_msgs.msg import Edge as EdgeMsg
from situational_graphs_reasoning_msgs.msg import Attribute as AttributeMsg
from situational_graphs_reasoning_msgs.msg import AmbiguousRoom as AmbiguousRoomMsg
from situational_graphs_reasoning_msgs.msg import AmbiguousRoomMatch as AmbiguousRoomMatchMsg
from situational_graphs_wrapper.GraphWrapper import GraphWrapper


from .GraphMatcher import GraphMatcher
from .utils import plane_4_params_to_6_params
class GraphMatchingNode(Node):

    def __init__(self):
        super().__init__('graph_matching', allow_undeclared_parameters = True, automatically_declare_parameters_from_overrides = True)
        self.gm = GraphMatcher(self.get_logger())    
        self.set_interface()
        self.get_json_parameters_()
        # self.get_logger().info(f"{self.params}")

    def get_json_parameters_(self):
        matching_package_path = ament_index_python.get_package_share_directory("graph_matching")
        json_file_path = os.path.join(matching_package_path, "config/syntheticDS_params.json")
        with open(json_file_path) as json_file:
            self.params = json.load(json_file)
            self.get_logger().info('flag self.params invariants {}'.format(self.params["invariants"]))
            self.get_logger().info('flag self.params thresholds {}'.format(self.params["thresholds"]))

        
    def set_interface(self):
        self.graph_subscription = self.create_subscription(GraphMsg,'graph_matching/graphs', self.graph_callback, 0)
        # Room matches chosen by the operator for an ambiguous case (origin = prior room, target = online room)
        self.operator_match_subscription = self.create_subscription(MatchMsg, 'graph_matching/operator_match', self.operator_match_callback, 10)
        self.unique_match_publisher = self.create_publisher(MatchMsg, 'graph_matching/unique_match', 10)
        # self.best_match_publisher = self.create_publisher(MatchMsg, 'graph_matching/best_match', 10)
        self.unique_match_visualization_inc_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/incremental', 10)
        self.unique_match_visualization_full_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/full', 10)
        self.unique_match_visualization_dev_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/unique_match_visualization/dev', 10)
        self.ambiguous_match_publisher = self.create_publisher(AmbiguousRoomMatchMsg, 'graph_matching/ambigous_match', 10)
        self._published_ambiguous_sigs = set()  # signatures of ambiguous cases already published, to avoid resending the same one
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_1_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_2_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_3_visualization', 10)
        # self.symmetry_match_1_visualization_publisher = self.create_publisher(MarkerArrayMsg, 'graph_matching/symmetry_match_4_visualization', 10)
        self.subgraph_match_srv = self.create_service(SubgraphMatchSrv, 'graph_matching/subgraph_match', self.subgraph_match_srv_callback)


    def graph_callback(self, msg):
        self.get_logger().info('Incoming graph with name {}'.format(msg.name))
        graph = {"name" : msg.name}
        self.gm.set_parameters(self.params)
        nodes = []
        for node_msg in msg.nodes:
            node_id = str(node_msg.id)
            node = [node_id, {}]
            attributes = {}
            for attrib_msg in node_msg.attributes:
                if attrib_msg.str_value:
                    attributes[attrib_msg.name] = attrib_msg.str_value
                elif attrib_msg.fl_value:
                    attributes[attrib_msg.name] = np.array(attrib_msg.fl_value)
                else:
                    print("Bad definition of attribute {}".format(attrib_msg.name))

                if node_msg.type == "Plane" and attrib_msg.name == "Geometric_info" and len(attributes[attrib_msg.name]) == 4:
                    attributes[attrib_msg.name] = plane_4_params_to_6_params(attributes[attrib_msg.name])

            if node_msg.type == "Plane":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            elif node_msg.type == "Finite Room":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            elif node_msg.type == "floor":
                attributes["draw_pos"] = attributes["Geometric_info"][:2]
            else:
                self.get_logger().info('Received unknown node type: {}'.format(node_msg.type))
            
            node[1] = attributes
            node[1]["type"] = node_msg.type
            nodes.append(node)
            
        graph["nodes"] = nodes

        edges = []
        for edge_msg in msg.edges:
            edge = (str(edge_msg.origin_node), str(edge_msg.target_node))
            edges.append(edge)
        
        graph["edges"] = edges
        self.gm.set_graph_from_dict(graph, graph["name"])
        dbg_graph = GraphWrapper(graph_def=graph)
        accapted_node_types = ["Finite Room", "Plane"]
        self.gm.graphs[graph["name"]] = self.gm.graphs[graph["name"]].filter_graph_by_node_types(accapted_node_types)
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}

        # self.gm.graphs[graph["name"]].draw(None, options, True)


        # ### Save dictionary of graphs
        # self.get_logger().info(f"FLAG type(graph) {graph}")
        # json_object = json.dumps(graph)
        # with open(f"/home/adminpc/reasoning_ws/src/graph_matching/graph_dicts/{graph['name']}.json", "w") as outfile:
        #     outfile.write(json_object)


        ### Filtering unparented nodes
        # self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        # options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        # self.gm.graphs[graph["name"]].draw(graph["name"], options, True)


        # ### Match
        room_ids = list(self.gm.graphs[graph["name"]].filter_graph_by_node_types("Finite Room").get_nodes_ids())
        self.get_logger().info(f"Number of rooms: {len(room_ids)}, IDs: {room_ids}")
        if graph["name"] == "Online" and len(room_ids)>=2:
            self.get_logger().info(f"Starting match!")
            # prior_room_nodes = list(self.gm.graphs['Prior'].filter_graph_by_node_attributes({'type': 'Finite Room'}).get_nodes_ids())
            # self.gm.graphs["Prior"].remove_nodes(["58", "57", "56", "55", "54", "53", "52"])
            ### ROOM NODES IN A-GRAPH: 51, 52, 53, 54, 55, 56, 57, 58
            success, matches, matches_full, matches_dev = self.gm.match("Prior", "Online", add_deviations=True)
            # self.get_logger().info(f"DBG success: {success}")
            # self.get_logger().info(f"DBG matches: {matches}")
            # self.get_logger().info(f"DBG matches_full: {matches_full}")
            # self.get_logger().info(f"DBG matches_dev: {matches_dev}")
            for match in matches:
                self.get_logger().info(f"New consistent match!")
                for i in match:
                    self.get_logger().info(f"flag {i['origin_node_attrs']['type']}. nodes {i['origin_node']} - {i['target_node']}. score {i['score']}")
                self.get_logger().info(f" ")

            if success and len(matches) > 1:
                for i, match in enumerate(matches):
                    symmetry_match_msg = self.generate_match_msg(match)
                    symmetry_match_publisher = self.create_publisher(MatchMsg, f'graph_matching/symmetry_match_{i+1}', 10)
                    symmetry_match_publisher.publish(symmetry_match_msg)
                    symmetry_match_visualization_msg = self.generate_match_visualization_msg(matches[i])
                    symmetry_match_visualization_publisher = self.create_publisher(MarkerArrayMsg, f'graph_matching/symmetry_match_{i+1}_visualization', 10)
                    symmetry_match_visualization_publisher.publish(symmetry_match_visualization_msg)

                ambiguous_match_msg = self.generate_ambiguous_match_msg(matches)
                if ambiguous_match_msg.ambiguous_rooms:
                    # signature of this ambiguous case: each online room with its set of prior candidates
                    ambiguous_sig = frozenset(
                        (r.online_room_id, frozenset(r.prior_candidate_ids))
                        for r in ambiguous_match_msg.ambiguous_rooms
                    )
                    if ambiguous_sig in self._published_ambiguous_sigs:
                        self.get_logger().info("AMBIGUOUS ROOM MATCH: already published this case, skipping")
                    else:
                        self._published_ambiguous_sigs.add(ambiguous_sig)
                        self.get_logger().info(
                            "AMBIGUOUS ROOM MATCH: {}".format(
                                [(r.online_room_id, list(r.prior_candidate_ids)) for r in ambiguous_match_msg.ambiguous_rooms]
                            )
                        )
                        self.ambiguous_match_publisher.publish(ambiguous_match_msg)

            if not success or len(matches) == 0:
                self.get_logger().info("MATCH FAILED: no consistent match found for this graph")

            if success and len(matches) == 1:
                unique_match_msg = self.generate_match_msg(matches[0])
                self.get_logger().info(
                    "UNIQUE MATCH: publishing {} pairs on graph_matching/unique_match".format(len(matches[0]))
                )
                self.unique_match_publisher.publish(unique_match_msg)
                unique_match_visualization_inc_msg = self.generate_match_visualization_msg(matches[0])
                self.unique_match_visualization_inc_publisher.publish(unique_match_visualization_inc_msg)
                unique_match_visualization_full_msg = self.generate_match_visualization_msg(matches_full[0])
                self.unique_match_visualization_full_publisher.publish(unique_match_visualization_full_msg)
                unique_match_visualization_dev_msg = self.generate_match_visualization_msg(matches_dev[0], match_type="deviations")
                self.unique_match_visualization_dev_publisher.publish(unique_match_visualization_dev_msg)
                # time.sleep(999)


    def operator_match_callback(self, msg):
        """The rooms chosen by the operator are taken as a unique match: the matcher associates their
        walls, stores it and publishes it on graph_matching/unique_match like one of its own."""
        pairs = [(edge.origin_node, edge.target_node) for edge in msg.edges]  # non-room pairs are skipped by the matcher
        self.get_logger().info(f"OPERATOR MATCH received (prior, online): {pairs}")
        if not pairs or "Prior" not in self.gm.graphs or "Online" not in self.gm.graphs:
            self.get_logger().warn("OPERATOR MATCH: no rooms or graphs not received yet, ignored")
            return
        self.gm.set_parameters(self.params)
        match, match_full = self.gm.add_operator_match("Prior", "Online", pairs)
        if not match:
            self.get_logger().warn("OPERATOR MATCH: nothing new to publish")
            return
        n_walls = sum(1 for pair in match if pair["origin_node_attrs"]["type"] == "Plane")
        self.get_logger().info(
            f"UNIQUE MATCH (operator): publishing {len(match) - n_walls} rooms and {n_walls} walls on graph_matching/unique_match")
        self.unique_match_publisher.publish(self.generate_match_msg(match))
        self.unique_match_visualization_inc_publisher.publish(self.generate_match_visualization_msg(match))
        if match_full:
            self.unique_match_visualization_full_publisher.publish(self.generate_match_visualization_msg(match_full[0]))

    def dump_prior(self, graph):
        """Save the Prior graph as received (Doorway nodes included) to
        <dump_prior_dir>/prior_<date>.json, to replay bags offline. Empty dir = off.
        A prior identical to the last one dumped is not saved again."""
        out_dir = os.path.expanduser(self.params.get("dump_prior_dir", ""))
        if not out_dir:
            return
        to_list = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        data = {"name": graph["name"],
                "nodes": [[n, {k: to_list(v) for k, v in a.items()}] for n, a in graph["nodes"]],
                "edges": [list(e) for e in graph["edges"]]}
        text = json.dumps(data)
        if text == self._last_prior_dump:
            return
        self._last_prior_dump = text
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, time.strftime("prior_%Y%m%d_%H%M%S.json"))
        with open(path, "w") as f:
            f.write(text)
        self.get_logger().info(f"Prior dumped to {path}")

    def check_prior_symmetry(self, graph):
        """On a new prior: build the confusion table of blocks of adjacent rooms, growing k until
        no block is ambiguous (that k is self.prior_k_max: from there on a unique match is trusted).
        self.prior_confusion = {k: {block: [(twin, rot, mapping)]}} for k < k_max. Logs the per-room
        view and warns if the prior is ambiguous with k rooms (k = prior_symmetry.k)."""
        cfg = self.params.get("prior_symmetry", {})
        eps, k, k_cap = cfg.get("eps", 0.4), cfg.get("k", 2), cfg.get("k_cap", 12)
        rooms, doors = rooms_from_graph_def(graph)
        sig = (frozenset(rooms), frozenset(doors))
        if sig == self._prior_symmetry_sig:
            return      # same prior already checked
        self._prior_symmetry_sig = sig
        t0 = time.perf_counter()
        self.prior_confusion, self.prior_k_max = confusion_table(rooms, doors, eps, k_cap)
        dt = time.perf_counter() - t0
        k_max_txt = self.prior_k_max if self.prior_k_max else f">{k_cap} (k_cap reached)"
        self.get_logger().info(f"Prior confusion table: {len(rooms)} rooms, {len(doors)} doors, eps={eps} m, "
                               f"k_max={k_max_txt} ({dt*1e3:.0f} ms)")
        for kk, conf in room_confusion(self.prior_confusion).items():
            txt = ", ".join(f"{r}->{sorted(c)}" for r, c in sorted(conf.items()))
            self.get_logger().info(f"  k={kk}: {txt}")
        amb = self.prior_confusion.get(k, {})
        if amb:
            pairs = [f"{list(A)}->{list(B)} rot {deg}" for A, tw in amb.items() for B, deg, _ in tw if A <= B]
            more = f" (+{len(pairs) - 10} more)" if len(pairs) > 10 else ""
            self.get_logger().warn(f"AMBIGUOUS ENVIRONMENT: {len(amb)} ambiguous blocks with k={k} rooms: "
                                   f"{', '.join(pairs[:10])}{more}")
        else:
            self.get_logger().info(f"Prior not ambiguous for k={k}")

    def subgraph_match_srv_callback(self, request, response):
        self.get_logger().info('Graph Matching: Received match request from {} to {}'.format(request.base_graph, request.target_graph))

        def match_fn(request, response):
            if request.base_graph not in self.gm.graphs.keys() or request.target_graph not in self.gm.graphs.keys() or \
                self.gm.graphs[request.base_graph].is_empty() or self.gm.graphs[request.target_graph].is_empty():
                response.success = 3
            else:
                success, matches = self.gm.match(request.base_graph, request.target_graph)
                
                if success:
                    matches_msg = [self.generate_match_msg(match) for match in matches]
                    matches_visualization_msg = [self.generate_match_visualization_msg(match) for match in matches]
                    self.get_logger().warn('{} successful match(es) found!'.format(len(matches_msg)))
                    response.success = 0 if len(matches_msg) == 1 else 1

                else:
                    response.success = 2
                    self.get_logger().warn('Graph Matching: no good matches found!')

                if response.success == 0:
                    # self.unique_match_publisher.publish(matches_msg[0])
                    self.unique_match_visualization_inc_publisher.publish(matches_visualization_msg[0])
                if response.success == 0 or response.success == 1:
                    self.best_match_publisher.publish(matches_msg[0])
                    self.best_match_visualization_publisher.publish(matches_visualization_msg[0])

            for match in matches:
                self.get_logger().info(f"flag new consistent match")
                for i in match:
                    self.get_logger().info(f"flag {i['origin_node_attrs']['type']} {i['score']}")
                self.get_logger().info(f" ")

            return response
        
        response = match_fn(request, response)
        return response
        # try:
        #     while rclpy.ok():
        #         match_fn(request, response)
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     pass
        # return


    def generate_match_msg(self, match):
        match_msg = MatchMsg()
        for edge in match:
            ### Edge
            edge_msg = EdgeMsg()
            edge_msg.origin_node = edge["origin_node"]
            edge_msg.target_node = edge["target_node"]
            attrib_msg = AttributeMsg()
            attrib_msg.name = "score"
            attrib_msg.fl_value = [edge["score"]]
            edge_msg.attributes = [attrib_msg]
            match_msg.edges.append(edge_msg)
            # graph_msg.name = str(score)

            ### Origin node
            origin_node_msg = NodeMsg()
            origin_node_msg.id = edge["origin_node"]
            origin_node_msg.type = edge["origin_node_attrs"]["type"]
            origin_node_msg.attributes = self.dict_to_attr_msg_list(edge["origin_node_attrs"])
            match_msg.basis_nodes.append(origin_node_msg)


            ### Target node
            target_node_msg = NodeMsg()
            target_node_msg.id = edge["target_node"]
            target_node_msg.type = edge["target_node_attrs"]["type"]
            target_node_msg.attributes = self.dict_to_attr_msg_list(edge["target_node_attrs"])
            match_msg.target_nodes.append(target_node_msg)

        return match_msg


    def generate_ambiguous_match_msg(self, matches):
        # online room (observed) -> set of prior rooms it could be, across all symmetries.
        # Grouping by the online room collects EVERY prior candidate for that room, so a
        # prior that maps to a single online room is no longer dropped (origin=prior, target=online).
        online_priors = {}
        for match in matches:
            for pair in match:
                if pair["origin_node_attrs"]["type"] == "Finite Room":
                    online_priors.setdefault(pair["target_node"], set()).add(pair["origin_node"])

        ambiguous_match_msg = AmbiguousRoomMatchMsg()
        ambiguous_match_msg.header.stamp = self.get_clock().now().to_msg()
        ambiguous_match_msg.header.frame_id = "map"
        for online_room_id, prior_candidate_ids in online_priors.items():
            if len(prior_candidate_ids) > 1:
                ambiguous_room_msg = AmbiguousRoomMsg()
                ambiguous_room_msg.online_room_id = online_room_id
                ambiguous_room_msg.prior_candidate_ids = sorted(prior_candidate_ids)
                ambiguous_match_msg.ambiguous_rooms.append(ambiguous_room_msg)

        return ambiguous_match_msg


    def dict_to_attr_msg_list(self, attr_dict):
        attr_list = []
        for attr_name in attr_dict.keys():
            attr_msg = AttributeMsg()
            attr_msg.name = attr_name
            if isinstance(attr_dict[attr_name], str): 
                attr_msg.str_value = attr_dict[attr_name]
            elif isinstance(attr_dict[attr_name], np.ndarray):
                attr_msg.fl_value = list(attr_dict[attr_name].astype(float))
            
            attr_list.append(attr_msg)

        return attr_list


    def test_with_prior_graph(self, graph_old):

        ### Translate old graph
        no_tra = np.array([0,0,0])
        tra = - np.array([0,4,0])
        no_rot = rotation_matrix_from_euler_degrees(0,0,0)
        rot = rotation_matrix_from_euler_degrees(0,0,90)

        graph = copy.deepcopy(graph_old)
        graph["name"] = "ONLINE"

        nodes = []
        for node in graph["nodes"]:
            attrs = node[1]
            geom_info = attrs["Geometric_info"]
            if attrs["type"] == "Plane":
                # trans_geom_info = transform_plane_definition([geom_info], no_tra, rot)[0]
                # trans_geom_info = transform_plane_definition([trans_geom_info], tra, no_rot)[0]
                trans_geom_info = transform_plane_definition([geom_info], tra, rot)[0]

            elif attrs["type"] == "Finite Room":
                trans_geom_info = transform_point([geom_info], tra, no_rot)[0]
                trans_geom_info = transform_point([trans_geom_info], no_tra, rot)[0]
            attrs["Geometric_info"] = trans_geom_info
            attrs["draw_pos"] = attrs["Geometric_info"][:2]
            node[1] = attrs
            nodes.append(node)
        graph["nodes"] = nodes
        
        self.gm.set_graph_from_dict(graph, graph["name"])
        self.gm.graphs[graph["name"]].filterout_unparented_nodes()
        options = {'node_color': self.gm.graphs[graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
        # self.gm.graphs[graph["name"]].draw(graph["name"], options, True)

        ### create room point centered graph
        room_node = "56"
        wall_nodes = ["7","32", "18", "12"]
        room_node = "55"
        wall_nodes = ["29","4", "30", "5"]

        for graph_basic in [graph_old, graph]:
            for node in graph_basic["nodes"]:

                if node[0] == room_node:
                    tra = - node[1]["Geometric_info"]
            rot = rotation_matrix_from_euler_degrees(0,0,0)


            local_graph = {'nodes' : [], 'edges' : []}
            local_graph["name"] = "room centered {}".format(graph_basic["name"])

            nodes = []
            for node in graph_basic["nodes"]:
                if node[0] == room_node or node[0] in wall_nodes:
                    attrs = node[1]
                    geom_info = attrs["Geometric_info"]
                    if attrs["type"] == "Plane":
                        trans_geom_info = transform_plane_definition([geom_info], tra, rot)[0]

                    elif attrs["type"] == "Finite Room":
                        trans_geom_info = transform_point([geom_info], tra, rot)[0]
                    attrs["Geometric_info"] = trans_geom_info
                    attrs["draw_pos"] = attrs["Geometric_info"][:2]
                    node[1] = attrs
                    nodes.append(node)

            local_graph["nodes"] = nodes
            self.gm.set_graph_from_dict(local_graph, local_graph["name"])
            options = {'node_color': self.gm.graphs[local_graph["name"]].define_draw_color_option_by_node_type(), 'node_size': 50, 'width': 2, 'with_labels' : True}
            self.gm.graphs[local_graph["name"]].draw(local_graph["name"], options, True)
        
        success, matches = self.gm.match("Prior", "ONLINE")


    def generate_match_visualization_msg(self, match, match_type = "normal"):
        source_frame = "map"
        target_frame = "prior_map"
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, self)
        
        # can_transform = False
        # while not can_transform:
        #     can_transform = tf_buffer.can_transform(target_frame,source_frame,rclpy.time.Time(seconds=0.0)) #-rclpy.time.Duration(seconds=5.0)
        #     self.get_logger().info('can_transform {}'.format(can_transform))

        # transform = tf_buffer.lookup_transform(target_frame,source_frame,rclpy.time.Time(seconds=0.0), rclpy.time.Duration(seconds=2.0))
        # self.get_logger().info('transform{}'.format(transform))

        transform = TransformMsg()
        transform.translation.x, transform.translation.y, transform.translation.z = 10., 0., 0.
        transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w  = 0., 0., 0., 1.
        transform_stamped = TransformStampedMsg()
        transform_stamped.transform = transform
        
        marker_array = []
        for i, edge in enumerate(match):
            origin_point_original = edge["origin_node_attrs"]["Geometric_info"][:3]
            point_msg = PointStampedMsg()
            point_msg.point.x, point_msg.point.y, point_msg.point.z = origin_point_original[0], origin_point_original[1], origin_point_original[2]
            point_msg.header.frame_id = source_frame
            # origin_point_translated_msg = BufferInterface().transform(point_msg, target_frame, rclpy.time.Time(seconds=0.0))

            origin_point_translated_msg = tf2_geometry_msgs.do_transform_point(point_msg, transform_stamped)
            origin_point = [origin_point_translated_msg.point.x, origin_point_translated_msg.point.y, origin_point_translated_msg.point.z]
            target_point = edge["target_node_attrs"]["Geometric_info"][:3]
            if edge["origin_node_attrs"]["type"] == "Finite Room":
                origin_point[2] = 22.
                target_point[2] = 22.
            elif edge["origin_node_attrs"]["type"] == "Plane":
                origin_point[2] = 16.
                target_point[2] = 16.

            add_noise = True
            if add_noise:
                noise_scale = [.5,.5,1]
                noise = (np.random.rand(3) - [.5, .5, .5]) * noise_scale
                origin_point += noise
                target_point += noise
            
            marker_msg = MarkerMsg()
            header_msg = HeaderMsg()
            header_msg.frame_id = source_frame
            header_msg.stamp = self.get_clock().now().to_msg()
            marker_msg.header = header_msg
            marker_msg.ns = "match"
            marker_msg.id = i
            marker_msg.type = 4
            marker_msg.action = 0
            # marker_msg.pose = PoseMsg()
            scale_msg = Vector3Msg()
            if match_type == "normal":
                scale = .2
            elif match_type == "deviations":
                scale = .4
            scale_msg.x, scale_msg.y, scale_msg.z = scale, scale, scale
            marker_msg.scale = scale_msg
            color_msg = ColorRGBSMsg()
            if match_type == "normal":
                color_msg.r, color_msg.g, color_msg.b = np.random.uniform(0.5,1), np.random.uniform(0.5,1), np.random.uniform(0.5,1)
            elif match_type == "deviations":
                color_msg.r, color_msg.g, color_msg.b = 0., 0., 0.
            color_msg.a = 1.
            marker_msg.color = color_msg
            marker_msg.lifetime = DurationMsg()
            marker_msg.frame_locked = True
            origin_point_msg = PointMsg()
            origin_point_msg.x, origin_point_msg.y, origin_point_msg.z = origin_point[0], origin_point[1], origin_point[2]
            target_point_msg = PointMsg()
            target_point_msg.x, target_point_msg.y, target_point_msg.z = target_point[0], target_point[1], target_point[2]
            points = [origin_point_msg,target_point_msg]
            marker_msg.points = points
            marker_array.append(marker_msg)
        
        marker_array_msg = MarkerArrayMsg()
        marker_array_msg.markers = marker_array

        return marker_array_msg


def main(args=None):
    rclpy.init(args=args)
    graph_matching_node = GraphMatchingNode()

    rclpy.spin(graph_matching_node)
    rclpy.get_logger().warn('Destroying node!')
    graph_matching_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
