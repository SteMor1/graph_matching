"""Rooms chosen by the operator are taken as a unique match: walls associated through the rotation
given by the matched rooms, stored, and excluded from the following matches."""
import contextlib
import io

import gm_scenarios as G
from graph_matching.GraphMatcher import GraphMatcher

OFF = G.ONLINE_ID_OFFSET


def make(env, params, block, T):
    rooms, walls = G.load_prior(env)
    gm = GraphMatcher(G.QuietLogger())
    gm.set_parameters(G.load_params(params))
    for name, ids, off in (("Prior", sorted(rooms), 0), ("Online", block, OFF)):
        gm.set_graph_from_dict(G.build_graph_def(name, rooms, walls, ids, off, T if off else (0.0, 0.0, 0.0)), name)
        gm.graphs[name] = gm.graphs[name].filter_graph_by_node_types(["Finite Room", "Plane"])
    return gm, rooms, walls


def pairs(match, node_type):
    return {(p["origin_node"], p["target_node"]) for p in match if p["origin_node_attrs"]["type"] == node_type}


def test_two_rooms_walls_associated_and_excluded_afterwards():
    gm, rooms, walls = make("generic_env_18_rooms", "installed", [5000, 6000], (-1.2, -8.0, 5.0))
    match, full = gm.add_operator_match("Prior", "Online", [(5000, OFF + 5000), (6000, OFF + 6000)])
    assert pairs(match, "Finite Room") == {(5000, OFF + 5000), (6000, OFF + 6000)}
    expected_walls = {(w, w + OFF) for r in (5000, 6000) for w in rooms[r]["walls"]}
    assert pairs(match, "Plane") == expected_walls
    assert len(full) == 1 and len(full[0]) == len(match)

    # sent again: nothing new
    assert gm.add_operator_match("Prior", "Online", [(5000, OFF + 5000)]) == ([], [])

    # the following match does not put the operator rooms back in play
    gm.set_graph_from_dict(G.build_graph_def("Online", rooms, walls, [5000, 6000, 7000], OFF, (-1.2, -8.0, 5.0)), "Online")
    gm.graphs["Online"] = gm.graphs["Online"].filter_graph_by_node_types(["Finite Room", "Plane"])
    with contextlib.redirect_stdout(io.StringIO()):
        _, matches, _, _ = gm.match("Prior", "Online")
    online_rooms = {o for m in matches for _, o in pairs(m, "Finite Room")}
    assert online_rooms <= {OFF + 7000}


def test_single_room_without_rotation_matches_only_the_room():
    gm, _, _ = make("generic_env_18_rooms", "installed", [6000], (0.7, 3.0, 4.0))
    match, _ = gm.add_operator_match("Prior", "Online", [(6000, OFF + 6000)])
    assert pairs(match, "Finite Room") == {(6000, OFF + 6000)} and not pairs(match, "Plane")


def test_conflicting_choice_is_ignored():
    gm, _, _ = make("generic_env_18_rooms", "installed", [5000, 6000], (-1.2, -8.0, 5.0))
    gm.add_operator_match("Prior", "Online", [(5000, OFF + 5000), (6000, OFF + 6000)])
    assert gm.add_operator_match("Prior", "Online", [(7000, OFF + 5000)]) == ([], [])
