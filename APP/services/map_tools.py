"""
Map Tool Definitions and Executor for Groq MCP-style Tool Calling.

Exposes three tools the LLM can call:
  - find_path(source, destination)   : Shortest path + step-by-step directions
  - list_rooms(query)                : Search rooms by name/category/floor
  - get_room_info(room_id)           : Details about a specific room
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import navigate.py from App/map/ (local copy inside App/)
# ---------------------------------------------------------------------------

MAP_DIR = Path(__file__).parent.parent / "map"
sys.path.insert(0, str(MAP_DIR))

from navigate import find_path as _find_path, resolve_room, NODES

# ---------------------------------------------------------------------------
# Tool Schemas (Groq / OpenAI-compatible JSON schema format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "find_path",
            "description": (
                "Find the shortest indoor walking route between two locations "
                "inside P.T. Lee Chengalvaraya Naicker College of Engineering "
                "and Technology. Returns step-by-step directions including any "
                "floor changes (stairs). Use this for any question about getting "
                "from one room/lab/facility to another."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Starting location. Can be a room ID (e.g. 'G09') "
                            "or a partial room name (e.g. 'Canteen', 'Principal Chamber')."
                        ),
                    },
                    "destination": {
                        "type": "string",
                        "description": (
                            "Destination location. Can be a room ID (e.g. 'F17') "
                            "or a partial room name (e.g. 'Library', 'ECE IV Year')."
                        ),
                    },
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_rooms",
            "description": (
                "Search for rooms, labs, offices, or facilities in the college "
                "by name keyword, category, or floor number. Use this when the "
                "student asks 'where is the library?', 'which floor is the canteen on?', "
                "or 'list all labs on the first floor'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search keyword to filter rooms. Can be a partial name "
                            "(e.g. 'lab', 'restroom', 'IT', 'CSE'), a category "
                            "('lab', 'lecture_hall', 'admin', 'facility', 'restroom', 'hall'), "
                            "or a floor number as string ('0', '1', '2')."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_room_info",
            "description": (
                "Get detailed information about a specific room including its "
                "full name, floor, and category. Use when the student asks "
                "about a specific room ID like 'What is G09?' or 'tell me about F06'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": (
                            "The room ID (e.g. 'G09', 'F17', 'S01'). "
                            "Also accepts partial names which will be resolved."
                        ),
                    }
                },
                "required": ["room_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool Executor Functions
# ---------------------------------------------------------------------------

FLOOR_NAMES = {0: "Ground Floor", 1: "First Floor", 2: "Second Floor"}
CATEGORY_LABELS = {
    "lab": "Laboratory",
    "lab_complex": "Lab Complex",
    "lecture_hall": "Lecture Hall",
    "admin": "Administrative Office",
    "facility": "Facility",
    "restroom": "Restroom",
    "hall": "Hall / Auditorium",
    "circulation": "Stairwell / Circulation",
    "faculty_room": "Faculty Room",
    "common": "Common Area",
}


def _tool_find_path(source: str, destination: str) -> str:
    """Execute find_path and return a formatted string result."""
    try:
        result = _find_path(source, destination)
        if not result["path"]:
            return result["directions"]

        lines = [
            f"?? Route: {result['src_name']} ({result['src']}) -> {result['dst_name']} ({result['dst']})",
            f"?? Distance: {result['distance']} corridor unit(s)",
            f"???  Path: {' -> '.join(result['path_named'])}",
            "",
            "?? Step-by-step Directions:",
            result["directions"],
        ]
        return "\n".join(lines)

    except ValueError as e:
        return f"Navigation Error: {e}"


def _tool_list_rooms(query: str) -> str:
    """Search rooms by keyword, category, or floor number."""
    q = query.strip().lower()

    matches = []

    # Floor filter: "0", "1", "2"
    if q in ("0", "1", "2"):
        floor_num = int(q)
        matches = [
            (nid, n)
            for nid, n in NODES.items()
            if n["floor"] == floor_num and n.get("category") != "circulation"
        ]
        header = f"Rooms on {FLOOR_NAMES[floor_num]}"
    else:
        # Category filter
        category_match = next(
            (k for k, v in CATEGORY_LABELS.items() if q in k.lower() or q in v.lower()),
            None,
        )
        if category_match:
            matches = [
                (nid, n)
                for nid, n in NODES.items()
                if n.get("category") == category_match
            ]
            header = f"All {CATEGORY_LABELS[category_match]} rooms"
        else:
            # Name substring search
            matches = [
                (nid, n)
                for nid, n in NODES.items()
                if q in n["name"].lower() and n.get("category") != "circulation"
            ]
            header = f"Rooms matching '{query}'"

    if not matches:
        return f"No rooms found matching '{query}'."

    lines = [f"?? {header} ({len(matches)} result(s)):"]
    for nid, n in sorted(matches, key=lambda x: (x[1]["floor"], x[0])):
        cat = CATEGORY_LABELS.get(n.get("category", ""), n.get("category", ""))
        floor = FLOOR_NAMES.get(n["floor"], f"Floor {n['floor']}")
        lines.append(f"  ? [{nid}] {n['name']} -- {cat}, {floor}")

    return "\n".join(lines)


def _tool_get_room_info(room_id: str) -> str:
    """Return detailed info about a specific room."""
    try:
        resolved = resolve_room(room_id)
        n = NODES[resolved]
        cat = CATEGORY_LABELS.get(n.get("category", ""), n.get("category", "Unknown"))
        floor = FLOOR_NAMES.get(n["floor"], f"Floor {n['floor']}")
        verified = "? Verified" if n.get("verified", True) else "?? Unverified (needs on-site confirmation)"
        note = f"\n  Note: {n['note']}" if n.get("note") else ""
        return (
            f"???  Room ID   : {resolved}\n"
            f"?? Name      : {n['name']}\n"
            f"?? Floor     : {floor}\n"
            f"?? Category  : {cat}\n"
            f"?? Status    : {verified}{note}"
        )
    except ValueError as e:
        return f"Room lookup error: {e}"


# ---------------------------------------------------------------------------
# Central dispatcher -- called by tool_caller.py
# ---------------------------------------------------------------------------
#so we have three tools 

TOOL_EXECUTORS = {
    "find_path": lambda args: _tool_find_path(args["source"], args["destination"]),
    "list_rooms": lambda args: _tool_list_rooms(args["query"]),
    "get_room_info": lambda args: _tool_get_room_info(args["room_id"]),
}


def execute_tool(tool_name: str, arguments: str) -> str:
    """
    Execute a tool by name with the JSON-encoded arguments string from Groq.
    Returns the tool result as a string.
    """
    try:
        args = json.loads(arguments)
        executor = TOOL_EXECUTORS.get(tool_name)
        if executor is None:
            return f"Unknown tool: {tool_name}"
        return executor(args)
    except Exception as e:
        return f"Tool execution error ({tool_name}): {e}"

