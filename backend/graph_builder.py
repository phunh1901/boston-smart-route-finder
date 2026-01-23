"""
backend/graph_builder.py — OpenStreetMap Parser & Graph Construction
====================================================================
Transforms raw OSM .pbf files into NetworkX MultiDiGraph.
Falls back to a simulated Boston landmark network if pyrosm is unavailable.
"""

import os
import pickle
import logging
from typing import Tuple
import numpy as np
import networkx as nx

try:
    from backend.config import GRAPH_PATH, PBF_PATH
except ImportError:
    from config import GRAPH_PATH, PBF_PATH

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points on Earth in meters."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def parse_osm_network(pbf_path: str):
    """
    Parses .pbf file and extracts driving road network.
    Returns (nodes GeoDataFrame, edges GeoDataFrame).
    """
    try:
        import pyrosm
    except ImportError:
        raise ImportError("Library pyrosm is not installed. Run: pip install pyrosm")

    logger.info("Reading OSM PBF file: %s", pbf_path)
    osm = pyrosm.OSM(pbf_path)

    logger.info("Extracting driving network...")
    nodes, edges = osm.get_network(network_type="driving", nodes=True)

    if nodes is None or edges is None or len(nodes) == 0:
        raise ValueError("Could not extract road network from PBF file.")

    logger.info("Extracted raw network: %d nodes, %d edges", len(nodes), len(edges))
    return nodes, edges


def _safe_float(val, default=0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) and v >= 0 else default
    except (TypeError, ValueError):
        return default


def _is_oneway(val) -> bool:
    return val in (True, 1, "yes", "true", "1", "-1", "Yes")


def build_graph(nodes, edges) -> nx.MultiDiGraph:
    """
    Converts GeoDataFrames to NetworkX MultiDiGraph.
    Edge weight = segment length (Haversine distance in meters).
    """
    G = nx.MultiDiGraph()

    # Add Nodes
    for _, row in nodes.iterrows():
        nid = int(row["id"])
        if "lon" in row and "lat" in row:
            x, y = float(row["lon"]), float(row["lat"])
        else:
            x, y = row["geometry"].x, row["geometry"].y
        G.add_node(nid, x=x, y=y)

    node_set = set(G.nodes())

    # Add Edges
    for _, row in edges.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        if u not in node_set or v not in node_set:
            continue

        if "length" in row and row["length"] is not None:
            length = _safe_float(row["length"])
        else:
            ud, vd = G.nodes[u], G.nodes[v]
            length = haversine_distance(ud["y"], ud["x"], vd["y"], vd["x"])

        if length <= 0:
            ud, vd = G.nodes[u], G.nodes[v]
            length = haversine_distance(ud["y"], ud["x"], vd["y"], vd["x"])

        oneway = _is_oneway(row.get("oneway", False))
        hw = row.get("highway", "unknown")
        if isinstance(hw, list):
            hw = hw[0] if hw else "unknown"
        highway = str(hw)

        attrs = dict(length=length, highway=highway, oneway=oneway)
        G.add_edge(u, v, **attrs)
        if not oneway:
            G.add_edge(v, u, **attrs)

    # Clean isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    logger.info("Removed %d isolated nodes", len(isolated))

    # Keep only largest weakly connected component
    if G.number_of_nodes() > 0:
        wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(wcc).copy()

    logger.info("Clean graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


