"""Utilities for tracking"""

import logging
from itertools import product

import networkx as nx
import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from acia.base import Contour, Instance, Overlay
from acia.tracking.output import CTCTrackingHelper


def life_cycle_lineage(tr_graph: nx.DiGraph) -> nx.DiGraph:
    """Compresses populated lineage to life cycle lineage (one node per cell cycle)

    Args:
        tr_graph (nx.DiGraph): populated tracking graph

    Returns:
        nx.DiGraph: Life cycle lineage with cell cylces as nodes
    """

    # compute the life-cycles of individual cells
    life_cycles = CTCTrackingHelper.compute_life_cycles(tr_graph)
    # create lookup (cont id --> life cycle index)
    life_cycle_lookup = CTCTrackingHelper.create_life_cycle_lookup(life_cycles)
    # contour_lookup = {cont.id: cont for cont in overlay}

    lc_graph = nx.DiGraph()

    # add all the nodes
    lc_graph.add_nodes_from(range(len(life_cycles)))

    # set the "cycle" property to contain the populated life cycle nodes
    for i, life_cycle in enumerate(life_cycles):
        lc_graph.nodes[i]["cycle"] = life_cycle

    # iterate over life_cycles
    for lc_id, lc in enumerate(life_cycles):
        start = lc[0]

        # extract parents from populated tracking
        parents = tr_graph.predecessors(start)

        for parent in parents:
            # get the parent life_cycle
            parent_lc_id = life_cycle_lookup[parent]

            # establish an edge between parent and child
            lc_graph.add_edge(parent_lc_id, lc_id)

    # set "start_frame" and "end_frame" for every node in the life cycle graph
    for node in lc_graph:
        lc = lc_graph.nodes[node]["cycle"]

        lc_graph.nodes[node]["start_frame"] = tr_graph.nodes[lc[0]]["frame"]
        lc_graph.nodes[node]["end_frame"] = tr_graph.nodes[lc[-1]]["frame"]

    return lc_graph


def delete_nodes(graph: nx.DiGraph, nodes_to_delete: list) -> nx.DiGraph:
    """Delete nodes while maintaining the connectivity

    Args:
        graph (nx.DiGraph): _description_
        nodes_to_delete (list): _description_

    Returns:
        nx.DiGraph: _description_
    """
    for node in nodes_to_delete:
        preds = list(graph.predecessors(node))
        succs = list(graph.successors(node))

        for p, s in product(preds, succs):
            graph.add_edge(p, s)

        graph.remove_node(node)

    return graph


def subsample_lineage(lineage: nx.DiGraph, subsampling_factor: int) -> nx.DiGraph:
    """Subsample lineage by only takeing nodes in every n-th (subsampling_factor) frame. Connectivity is maintained.

    Args:
        lineage (nx.DiGraph): lineage graph (needs the frame attributes)
        subsampling_factor (int): n-th frame will taken into account

    Returns:
        nx.DiGraph: Returns the pruned lineage only containing nodes of every n-th frame
    """

    # copy the lineage
    lineage = lineage.copy()

    # get all the frames in the lineage
    frames = list(sorted(np.unique([lineage.nodes[n]["frame"] for n in lineage.nodes])))

    # compute what frames to keep (every n-th)
    keep_frames = set(frames[::subsampling_factor])

    # create a list of nodes that are not inside the selected frames
    del_nodes = [
        n for n in lineage.nodes if lineage.nodes[n]["frame"] not in keep_frames
    ]

    # delete nodes (maintain connectivity)
    new_lineage = delete_nodes(lineage, del_nodes)

    # return the new lineage
    return new_lineage


def tracklet_to_tracking(ov: Overlay, tracklet_graph: nx.DiGraph) -> nx.DiGraph:
    """Compute a tracking graph based on the tracklet graph and an overlay

    Args:
        ov (Overlay): the current overlay
        tracklet_graph (nx.DiGraph): the corresponding tracklet_graph

    Returns:
        nx.DiGraph: the resulting tracking graph
    """
    tracking_graph = nx.DiGraph()

    label_lookup = {}

    for cont in ov:
        if cont.label in tracklet_graph.nodes:
            tracking_graph.add_node(cont.id, frame=cont.frame)

            label_lookup[cont.label] = label_lookup.get(cont.label, []) + [cont]

    for label in tracklet_graph.nodes:
        # get all the contours with this label
        contours = sorted(label_lookup[label], key=lambda c: c.frame)

        # add them sequentially
        for a, b in zip(contours, contours[1:]):
            tracking_graph.add_edge(a.id, b.id)

    for label in tracklet_graph.nodes:
        for succ in tracklet_graph.successors(label):
            # make the division edges
            tracking_graph.add_edge(
                label_lookup[label][-1].id, label_lookup[succ][0].id
            )

    return tracking_graph


def merge_incosistent_segmentation(
    sub_ov: Overlay, tracklet_graph: nx.DiGraph, num_timesteps=3
) -> tuple[Overlay, nx.DiGraph]:
    """Merges incosistent segmentation using the tracking information

    Args:
        sub_ov (Overlay): overlay containing segmentation information
        tracklet_graph (nx.DiGraph): the tracklet graph
        num_timesteps (int, optional): minimal number of timesteps that a segmentation must exist. Defaults to 3.

    Returns:
        tuple[Overlay, nx.DiGraph]: new overlay and tracklet_graph
    """

    def cond(n, graph, num_nodes=num_timesteps):
        """returns true if this is an event where the tracking is inconsistent and should be joined"""
        if graph.out_degree(n) != 2:
            return False

        children = sorted(graph.successors(n), key=graph.out_degree)

        # check whether we have a dead end and a continous cell
        if not (
            graph.out_degree(children[0]) == 0 and graph.out_degree(children[1]) >= 1
        ):
            return False

        # check that the dead end durtion is not too long
        dur = (
            graph.nodes[children[0]]["end_frame"]
            - graph.nodes[children[0]]["start_frame"]
        )
        if dur > num_nodes:
            return False

        return True

    # collect all the siblsings that should be joined
    to_join = []
    for n in tracklet_graph.nodes:
        # check the join condition
        if cond(n, tracklet_graph):
            children = sorted(
                tracklet_graph.successors(n), key=tracklet_graph.out_degree
            )
            to_join.append(children)

    new_ov = Overlay([])

    remove_labels = {join_set[0] for join_set in to_join}

    # create the new overay where masks are joined
    for i, ov in enumerate(sub_ov.timeIterator()):
        frame_label_set = {it.label for it in ov}

        to_add = []
        to_remove = []

        for join_set in to_join:
            if set(join_set).issubset(frame_label_set):
                # print(f"Frame: {i} -> Need to change overlay")

                def label_lookup(label, ov):
                    return [cont for cont in ov if cont.label == label][0]

                # print(ov.cont_lookup)
                polys = [
                    label_lookup(join_set[0], ov).polygon.buffer(2),
                    label_lookup(join_set[1], ov).polygon.buffer(5),
                ]

                res_poly = unary_union(polys)
                res_poly = res_poly.buffer(-5)

                if isinstance(res_poly, MultiPolygon):
                    area_before = res_poly.area
                    max_size_index = np.argmax([g.area for g in res_poly.geoms])
                    res_poly = res_poly.geoms[max_size_index]
                    logging.warning(
                        "Need to fix multipolygon. Area from %.2f to %.2f",
                        area_before,
                        res_poly.area,
                    )

                # this polygon needs to be added
                cont = Contour(
                    np.stack(res_poly.exterior.xy, axis=-1),
                    -1,
                    frame=i,
                    id=label_lookup(join_set[1], ov).id,
                    label=join_set[1],
                )
                # print(cont.coordinates)
                to_add.append(cont)
                to_remove.append(join_set[1])

        all_remove = remove_labels.union(set(to_remove))
        new_ov.add_contours(
            [cont for cont in ov if cont.label not in all_remove] + to_add
        )

    # remove joined labels from the tracklet graph
    remove_labels = {join_set[0] for join_set in to_join}
    for n in remove_labels:
        tracklet_graph.remove_node(n)

    # join tracklets (we have remove wrong divisions but still need to join the tracklets)
    tracklets_to_join = []

    for n in list(nx.dfs_preorder_nodes(tracklet_graph)):
        if tracklet_graph.out_degree(n) == 1:
            tracklets_to_join.append((n, list(tracklet_graph.successors(n))[0]))

    relabel_actions = {n: n for n in tracklet_graph.nodes}

    for a, b in tracklets_to_join:
        relabel_actions[b] = relabel_actions[a]

    # actually join the tracklets
    for b, a in relabel_actions.items():
        # join the two
        b_children = tracklet_graph.successors(b)

        # ensure connectivity
        for b_child in b_children:
            tracklet_graph.add_edge(a, b_child)

        # update end frame
        tracklet_graph.nodes[a]["end_frame"] = np.max(
            [tracklet_graph.nodes[a]["end_frame"], tracklet_graph.nodes[b]["end_frame"]]
        )

    # remove nodes
    tracklet_graph.remove_nodes_from(
        set(tracklet_graph.nodes).difference(relabel_actions.values())
    )

    for cont in new_ov:
        if isinstance(cont, Instance) and cont.label != relabel_actions[cont.label]:
            cont.mask = (cont.mask == cont.label) * relabel_actions[cont.label]
        cont.label = relabel_actions[cont.label]

    return new_ov, tracklet_graph


def compute_trace(lineage: nx.DiGraph) -> dict:
    """Computes the trace (cell heritage in a sequence) for every node in the lineage

    Args:
        lineage (nx.DiGraph): the lineage. Node ids will be concatenated to trace

    Returns:
        dict: Dictionary that contains the trace (str) for every node (type of lineage node)
    """

    traces = {}

    for n in nx.dfs_preorder_nodes(lineage):
        # traverse nodes in dfs
        parent = list(lineage.predecessors(n))
        if len(parent) == 0:
            parent = None
        elif len(parent) == 1:
            parent = parent[0]
        else:
            raise ValueError("More than one parent! I cannot handle that!")

        # some nodes may not have parents others have. Make sure that the "." is only placed between node ids.
        if parent is not None:
            parent_trace = traces.get(parent)

            if len(parent_trace) > 0:
                parent_trace += "."
        else:
            parent_trace = ""

        # make trace: (trace of parent.)(my own value)
        trace = f"{parent_trace}{n}"

        traces[n] = trace

    # return all traces
    return traces
