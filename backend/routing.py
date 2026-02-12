"""
backend/routing.py — Shortest Path Algorithms (Dijkstra & A*)
============================================================
Provides Dijkstra and A* pathfinding implementations with dynamic road blocking
and detailed geometric coordinates reconstruction for map rendering.
"""

import heapq
import time
import math
from typing import List, Optional, Tuple, Dict, Set
import networkx as nx


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """High-speed Haversine distance using math standard library (meters)."""
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    a = max(0.0, min(1.0, a))
    return 2 * R * math.asin(math.sqrt(a))


def _min_edge_length(edge_data) -> float:
    """Extracts minimum valid length from edge data (supports DiGraph & MultiDiGraph)."""
    if edge_data is None:
        return 1.0

    # DiGraph case
    if "length" in edge_data:
        try:
            return float(edge_data["length"])
        except (TypeError, ValueError):
            return 1.0

    # MultiDiGraph case (dictionary of parallel edge attributes)
    if isinstance(edge_data, dict):
        lengths = []
        for d in edge_data.values():
            if isinstance(d, dict) and "length" in d:
                try:
                    lengths.append(float(d["length"]))
                except (TypeError, ValueError):
                    continue
        if lengths:
            return min(lengths)

    return 1.0


def dijkstra(
    G: nx.MultiDiGraph,
    source: int,
    target: int,
    blocked_edges: Set[Tuple[int, int]] = None
) -> Tuple[Optional[List[int]], float, float]:
    """
    Dijkstra shortest path algorithm.
    Skips edges present in blocked_edges.
    Returns: (path_node_ids, total_distance_m, execution_time_s)
    """
    t0 = time.perf_counter()
    if blocked_edges is None:
        blocked_edges = set()

    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    visited: Set[int] = set()
    pq: List[Tuple[float, int]] = [(0.0, source)]

    while pq:
        cost, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v in G.neighbors(u):
            if (u, v) in blocked_edges or (v, u) in blocked_edges:
                continue

            w = _min_edge_length(G.get_edge_data(u, v))
            new_cost = cost + w
            if new_cost < dist.get(v, float("inf")):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    exec_time = time.perf_counter() - t0

    if target not in dist:
        return None, float("inf"), exec_time

    path = _reconstruct_path(prev, target)
    return path, dist[target], exec_time


