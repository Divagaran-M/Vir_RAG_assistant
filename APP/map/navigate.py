"""
College indoor-navigation graph + shortest path finder.

Usage as a script:
    python navigate.py G09 F17
    python navigate.py "Canteen" "ECE - IV Year"

Usage as an LLM tool (e.g. import into your Campus Assistant / RAG pipeline):
    from navigate import find_path, resolve_room
    result = find_path("G09", "F17")
    # result = {"path": [...], "distance": N, "directions": "..."}

Designed so an LLM can call `find_path(src, dst)` with either a room ID
("G09") or a room name/partial name ("canteen"), and get back a
deterministic shortest path plus a human-readable direction string it
can relay to the student.
"""

import json
import heapq
import sys
from pathlib import Path

GRAPH_PATH = Path(__file__).parent / "college_graph.json"
if not GRAPH_PATH.exists():
    GRAPH_PATH = Path(__file__).parent / "college_grap.json"


def load_graph():
    with open(GRAPH_PATH, "r") as f:
        data = json.load(f)
    nodes = {n["id"]: n for n in data["nodes"]}
    adj = {node_id: [] for node_id in nodes}
    for e in data["edges"]:
        w = e["weight"]
        adj[e["from"]].append((e["to"], w))
        adj[e["to"]].append((e["from"], w))  # corridors are bidirectional
    return nodes, adj


NODES, ADJ = load_graph()

FLOOR_NAMES = {0: "Ground Floor", 1: "First Floor", 2: "Second Floor"}


def resolve_room(query: str) -> str:
    """
    Resolve a user-provided string (room ID or partial name) to a node ID.
    Raises ValueError with suggestions if ambiguous or not found.
    """
    query_clean = query.strip().upper()

    # Exact ID match
    if query_clean in NODES:
        return query_clean

    # Exact/substring name match (case-insensitive)
    q_lower = query.strip().lower()
    matches = [
        nid for nid, n in NODES.items()
        if q_lower in n["name"].lower()
    ]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Prefer an actual room over a stairwell landing that merely
        # mentions the room name in its description, if that disambiguates.
        non_circulation = [m for m in matches if NODES[m]["category"] != "circulation"]
        if len(non_circulation) == 1:
            return non_circulation[0]
        options = ", ".join(f"{m} ({NODES[m]['name']})" for m in matches[:5])
        raise ValueError(
            f"'{query}' matches multiple rooms: {options}. "
            f"Please specify a room ID."
        )
    else:
        raise ValueError(
            f"No room found matching '{query}'. "
            f"Try a room ID like 'G09' or a clearer name."
        )


def dijkstra(src: str, dst: str):
    """Standard Dijkstra shortest path over the graph."""
    if src not in NODES:
        raise ValueError(f"Unknown source room ID: {src}")
    if dst not in NODES:
        raise ValueError(f"Unknown destination room ID: {dst}")

    dist = {node_id: float("inf") for node_id in NODES}
    prev = {node_id: None for node_id in NODES}
    dist[src] = 0
    pq = [(0, src)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v, w in ADJ[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[dst] == float("inf"):
        return None, float("inf")

    path = []
    node = dst
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, dist[dst]


def build_directions(path):
    """Turn a raw node-ID path into human-readable step directions."""
    if not path or len(path) == 1:
        return "You're already there."

    steps = []
    current_floor = NODES[path[0]]["floor"]
    steps.append(f"Start at {NODES[path[0]]['name']} ({path[0]}), {FLOOR_NAMES[current_floor]}.")

    for i in range(1, len(path)):
        node = NODES[path[i]]
        if node["floor"] != current_floor:
            steps.append(
                f"Take the stairs/lift from {FLOOR_NAMES[current_floor]} up/down to {FLOOR_NAMES[node['floor']]}."
            )
            current_floor = node["floor"]
        if node["category"] == "circulation":
            continue  # don't clutter directions with the stairwell node itself
        steps.append(f"Continue to {node['name']} ({path[i]}).")

    return "\n".join(steps)


def find_path(src_query: str, dst_query: str) -> dict:
    """
    Main entry point for LLM tool-calling.
    Accepts room IDs or fuzzy names for src/dst.
    Returns a dict with path, distance, and human-readable directions.
    """
    src = resolve_room(src_query)
    dst = resolve_room(dst_query)
    path, distance = dijkstra(src, dst)

    if path is None:
        return {
            "src": src,
            "dst": dst,
            "path": [],
            "distance": None,
            "directions": f"No path found between {src} and {dst} (graph may be disconnected).",
        }

    return {
        "src": src,
        "dst": dst,
        "src_name": NODES[src]["name"],
        "dst_name": NODES[dst]["name"],
        "path": path,
        "path_named": [f"{nid} ({NODES[nid]['name']})" for nid in path],
        "distance": distance,
        "directions": build_directions(path),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python navigate.py <source> <destination>")
        sys.exit(1)

    try:
        result = find_path(sys.argv[1], sys.argv[2])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"From: {result.get('src_name', result['src'])} ({result['src']})")
    print(f"To:   {result.get('dst_name', result['dst'])} ({result['dst']})")
    print(f"Distance (units): {result['distance']}")
    print()
    print(result["directions"])
