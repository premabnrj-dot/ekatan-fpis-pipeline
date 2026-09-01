# -*- coding: utf-8 -*-
"""Unit tests for the wall/opening geometry that leaves this pipeline.

WHY THIS FILE EXISTS
--------------------
Everything here guards ONE class of defect: a number that is correct on our side
and means something else on Ekatan's. That defect has now happened three times.

  1. MEP zone positions were sent as `fraction x 20000` and decoded as
     `fraction / 10000`. Both sides were self-consistent; every zone was drawn at
     double its position.
  2. Wall lengths were sent as `fraction x 10000` in a field called
     `wall_length_mm`, and Ekatan's `fpis-geometry.ts` priced wardrobes off it.
  3. Opening offsets were computed against that same placeholder and then NOT
     rescaled when the wall was - so a door detected mid-wall shipped claiming to
     sit ~150 mm from the corner of a 3,861 mm wall.

Number 3 is what this file was written for. The rule it enforces is simple and
worth stating plainly: **an opening and the wall it sits on must always be in the
same unit, and that unit must be millimetres by the time the payload is built.**

Run:  python test_opening_geometry.py
"""
import io
import math
import re
import sys
import types

PATH = "fpis_pipeline.py"

# ─── Load the functions WITHOUT importing the module ─────────────────────────
# fpis_pipeline.py imports modal/cv2/torch at module scope, none of which are
# installed locally. Everything under test is pure Python + math, so it is
# extracted by source and exec'd in a bare namespace - the same approach
# test_refine_room_code.py uses, and for the same reason.
SRC = io.open(PATH, encoding="utf-8").read()


def _grab(pattern, label):
    m = re.search(pattern, SRC, re.S | re.M)
    if not m:
        print("FAIL: could not locate %s in %s" % (label, PATH))
        sys.exit(1)
    return m.group(0)


env = {"math": math}
for pat, label in [
    (r"^def _ensure_clockwise\(.*?\n(?=\n\ndef |\n\n# )", "_ensure_clockwise"),
    (r"^def _assign_openings_to_nearest_wall\(.*?\n(?=\n\ndef |\n\n# )",
     "_assign_openings_to_nearest_wall"),
    (r"^def _r2s_walls_from_polygon\(.*?\n(?=\n\ndef |\n\n# )",
     "_r2s_walls_from_polygon"),
    (r"^def _rescale_walls_to_mm\(.*?\n(?=\n\ndef |\n\n# )", "_rescale_walls_to_mm"),
    (r"^def _resolve_openings_to_mm\(.*?\n(?=\n\ndef |\n\n# )", "_resolve_openings_to_mm"),
]:
    exec(_grab(pat, label), env)

ensure_clockwise = env["_ensure_clockwise"]
walls_from_polygon = env["_r2s_walls_from_polygon"]
assign_openings = env["_assign_openings_to_nearest_wall"]
rescale = env["_rescale_walls_to_mm"]

failures = []


def check(label, cond, detail=""):
    if cond:
        print("  ok   %s" % label)
    else:
        print("  FAIL %s %s" % (label, detail))
        failures.append(label)


def rect(x0, y0, x1, y1):
    """Clockwise-from-top-left rectangle in y-down fractional space."""
    return [{"x": x0, "y": y0}, {"x": x1, "y": y0},
            {"x": x1, "y": y1}, {"x": x0, "y": y1}]


def door_at(cx, cy, w_frac_x=0.04, w_frac_y=0.01, conf=0.9):
    return {"opening_type": "single_door", "_cx": cx, "_cy": cy,
            "_w_frac_x": w_frac_x, "_w_frac_y": w_frac_y,
            "extraction_confidence": conf}


def build_room(polygon, length_mm, width_mm, doors=()):
    """Run the real Step 6 -> Step 7 path over one room."""
    poly = ensure_clockwise(polygon)
    room = {
        "room_type_code": "bedroom",
        "polygon_points": poly,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "walls": walls_from_polygon(poly),
    }
    if doors:
        assign_openings(list(doors), room["walls"])
    rescale([room])
    return room


def all_openings(room):
    return [o for w in room["walls"] for o in (w.get("openings") or [])]


# ─── 1. Winding ───────────────────────────────────────────────────────────────
# The whole opening-offset contract rests on this: Ekatan measures
# offset_from_left_mm from a wall's start vertex, and which end that is depends
# on the winding. A mirrored polygon mirrors every door on every wall.
print("\n1. Polygon winding")

cw = rect(0.1, 0.1, 0.5, 0.3)
ccw = list(reversed(cw))
check("clockwise polygon is left alone", ensure_clockwise(list(cw)) == cw)
check("counter-clockwise polygon is reversed", ensure_clockwise(list(ccw)) == cw)
check("degenerate polygon is returned untouched",
      ensure_clockwise([{"x": 0.0, "y": 0.0}]) == [{"x": 0.0, "y": 0.0}])


def signed_area(pts):
    return sum(pts[i]["x"] * pts[(i + 1) % len(pts)]["y"]
               - pts[(i + 1) % len(pts)]["x"] * pts[i]["y"]
               for i in range(len(pts)))


check("clockwise means positive signed area in y-down space", signed_area(cw) > 0)

# An L-shape, wound counter-clockwise. The bounding-box branch of
# _rectify_contour always emits clockwise, so only irregular rooms can arrive
# mirrored - which is exactly the case that would have shipped broken.
l_shape_cw = [
    {"x": 0.10, "y": 0.10}, {"x": 0.50, "y": 0.10}, {"x": 0.50, "y": 0.25},
    {"x": 0.30, "y": 0.25}, {"x": 0.30, "y": 0.40}, {"x": 0.10, "y": 0.40},
]
check("L-shaped room is normalised to clockwise",
      ensure_clockwise(list(reversed(l_shape_cw))) == l_shape_cw)

# ─── 2. Walls carry their polygon edge index ─────────────────────────────────
# Ekatan's findWallEdge falls back to "the northmost edge is the north wall"
# without this, which is wrong on any non-rectangle. It cannot be recovered from
# list position downstream because short edges are skipped here and long ones are
# trimmed in Step 6.
print("\n2. polygon_edge_index")

walls = walls_from_polygon(rect(0.1, 0.1, 0.5, 0.3))
check("one wall per polygon edge", len(walls) == 4, "got %d" % len(walls))
check("edge indices are 0..n-1 in polygon order",
      [w["polygon_edge_index"] for w in walls] == [0, 1, 2, 3],
      str([w["polygon_edge_index"] for w in walls]))

# A polygon with a near-zero edge: that edge is skipped, so the surviving walls
# must still name their TRUE edge index, not their position in the list.
sliver = [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.1}, {"x": 0.5001, "y": 0.1001},
          {"x": 0.5, "y": 0.3}, {"x": 0.1, "y": 0.3}]
sliver_walls = walls_from_polygon(ensure_clockwise(sliver))
idxs = [w["polygon_edge_index"] for w in sliver_walls]
check("a skipped short edge leaves a gap in the indices, not a renumbering",
      idxs == sorted(set(idxs)) and len(idxs) < len(sliver), str(idxs))

# ─── 3. No wall position Ekatan will silently drop ───────────────────────────
# conversion.ts matches walls over the four cardinals only. A 'custom' wall
# matches nothing and is dropped from the graph WITH ITS OPENINGS.
print("\n3. Wall positions Ekatan can actually consume")

CARDINALS = {"north", "south", "east", "west"}
diamond = ensure_clockwise([
    {"x": 0.30, "y": 0.10}, {"x": 0.50, "y": 0.25},
    {"x": 0.30, "y": 0.40}, {"x": 0.10, "y": 0.25},
])
diag_walls = walls_from_polygon(diamond)
check("a fully diagonal room emits only cardinal positions",
      all(w["wall_position"] in CARDINALS for w in diag_walls),
      str([w["wall_position"] for w in diag_walls]))
check("diagonal walls are marked so they keep the geometric-mean scale",
      all(w["_axis"] == "d" for w in diag_walls),
      str([w["_axis"] for w in diag_walls]))
check("an axis-aligned room still names its walls by compass",
      {w["wall_position"] for w in walls} == CARDINALS,
      str([w["wall_position"] for w in walls]))

# ─── 4. THE ONE THAT MATTERS: openings land in millimetres ───────────────────
print("\n4. Openings are millimetres on the same wall they sit on")

# A 4,000 x 2,000 mm room drawn 0.40 x 0.20 of the image, door at the middle of
# the north wall.
room = build_room(rect(0.1, 0.1, 0.5, 0.3), 4000, 2000,
                  doors=[door_at(0.30, 0.10)])
north = [w for w in room["walls"] if w["wall_position"] == "north"][0]
ops = all_openings(room)

check("the door landed on the north wall", len(north.get("openings") or []) == 1)
check("north wall measures the room's long side",
      abs(north["wall_length_mm"] - 4000) <= 20, str(north["wall_length_mm"]))

door = ops[0]
centre = door["offset_from_left_mm"] + door["rough_width_mm"] / 2.0
check("a door at mid-wall reports its centre at mid-wall",
      abs(centre - 2000) <= 60, "centre=%s" % centre)
# The old bug: offset computed against the fraction x 10000 placeholder. A wall
# of frac 0.40 became "4000", the door's centre landed near 2000 - and then the
# wall was rescaled and the door was not. Where the two units coincide the test
# must still fail on a wall whose scale is NOT 10000/unit, so check a second
# room whose mm-per-fraction is deliberately different.
tall = build_room(rect(0.1, 0.1, 0.5, 0.3), 12000, 6000,
                  doors=[door_at(0.30, 0.10)])
tall_north = [w for w in tall["walls"] if w["wall_position"] == "north"][0]
tall_door = (tall_north.get("openings") or [None])[0]
check("the same door in a 12 m room scales with the room, not with 10000",
      tall_door is not None
      and abs(tall_door["offset_from_left_mm"] + tall_door["rough_width_mm"] / 2.0
              - 6000) <= 180,
      "offset=%s width=%s" % (tall_door and tall_door["offset_from_left_mm"],
                              tall_door and tall_door["rough_width_mm"]))

check("the door is never wider than its own wall",
      all(o["rough_width_mm"] <= w["wall_length_mm"]
          for w in tall["walls"] for o in (w.get("openings") or [])))
check("the door never hangs off the end of its wall",
      all(o["offset_from_left_mm"] + o["rough_width_mm"] <= w["wall_length_mm"] + 1
          for w in tall["walls"] for o in (w.get("openings") or [])))
check("the door never starts before the wall does",
      all(o["offset_from_left_mm"] >= 0
          for w in tall["walls"] for o in (w.get("openings") or [])))

# A door on a VERTICAL wall must be measured by its y-extent. Taking the
# x-extent there reports the wall's THICKNESS as the door's width.
# The two extents are chosen far apart on purpose: at 10,000 mm per fraction the
# y-extent is a 900 mm door and the x-extent is an 80 mm sliver, so reading the
# wrong axis cannot coincidentally land on the right answer.
side = build_room(rect(0.1, 0.1, 0.5, 0.3), 4000, 2000,
                  doors=[door_at(0.50, 0.20, w_frac_x=0.008, w_frac_y=0.09)])
east = [w for w in side["walls"] if w["wall_position"] == "east"][0]
east_door = (east.get("openings") or [None])[0]
check("a door on a vertical wall is measured along that wall",
      east_door is not None and abs(east_door["rough_width_mm"] - 900) <= 40,
      "width=%s (80mm would mean the x-extent was used)"
      % (east_door and east_door["rough_width_mm"]))

# ─── 4b. Windows ride the same rails as doors ────────────────────────────────
# Step 6b adds windows through the same assigner, so they must resolve to
# millimetres identically. The floor differs by kind: flooring a ventilator at
# 600 would silently double it.
print("\n4b. Windows and ventilators")


def window_at(cx, cy, w_frac_x=0.05, w_frac_y=0.0, kind="window_standard"):
    return {"opening_type": kind, "_cx": cx, "_cy": cy,
            "_w_frac_x": w_frac_x, "_w_frac_y": w_frac_y,
            "extraction_confidence": 0.8}


win = build_room(rect(0.1, 0.1, 0.5, 0.3), 4000, 2000,
                 doors=[window_at(0.30, 0.10, w_frac_x=0.12)])
win_op = all_openings(win)[0]
check("a window is labelled W, not D", win_op["opening_label"].startswith("W"),
      win_op["opening_label"])
check("a window resolves to millimetres like a door",
      abs(win_op["rough_width_mm"] - 1200) <= 40, str(win_op["rough_width_mm"]))

# 0.02 of a 10,000 mm-per-fraction axis is 200 mm — under both floors, so this
# separates the ventilator floor (300) from the default one (600).
vent = build_room(rect(0.1, 0.1, 0.5, 0.3), 4000, 2000,
                  doors=[window_at(0.30, 0.10, w_frac_x=0.02, kind="ventilator")])
vent_op = all_openings(vent)[0]
check("a narrow ventilator floors at 300mm, not 600mm",
      vent_op["rough_width_mm"] == 300, str(vent_op["rough_width_mm"]))

narrow = build_room(rect(0.1, 0.1, 0.5, 0.3), 4000, 2000,
                    doors=[window_at(0.30, 0.10, w_frac_x=0.02)])
check("a too-narrow window still floors at 600mm",
      all_openings(narrow)[0]["rough_width_mm"] == 600,
      str(all_openings(narrow)[0]["rough_width_mm"]))

# ─── 5. Unknown scale produces nulls, never invented numbers ─────────────────
# Ekatan reads a null offset as "unpositioned" and still draws the room
# (room-shape.ts skips it, fpis-geometry.ts counts it as unpositionedMm). A
# fabricated number would instead be priced.
print("\n5. An unmeasurable room says so")

blind = build_room(rect(0.1, 0.1, 0.5, 0.3), None, None,
                   doors=[door_at(0.30, 0.10)])
blind_ops = all_openings(blind)
check("the room is flagged",
      "wall_lengths_unscaled:room_dimensions_unknown"
      in (blind.get("validation_flags") or []),
      str(blind.get("validation_flags")))
check("openings report null width rather than a guess",
      all(o["rough_width_mm"] is None for o in blind_ops))
check("openings report null offset rather than a guess",
      all(o["offset_from_left_mm"] is None for o in blind_ops))

# ─── 6. The Claude fallback path is left strictly alone ──────────────────────
# It returns real millimetres in its own JSON. Rescaling those would be the
# original bug wearing the other hat.
print("\n6. Claude-supplied geometry is not touched")

claude_room = {
    "room_type_code": "kitchen",
    "polygon_points": rect(0.1, 0.1, 0.4, 0.4),
    "length_mm": 3000, "width_mm": 3000,
    "walls": [{
        "wall_position": "north", "wall_length_mm": 3000,
        "openings": [{"opening_type": "single_door", "rough_width_mm": 900,
                      "offset_from_left_mm": 300}],
    }],
}
rescale([claude_room])
w0 = claude_room["walls"][0]
check("a Claude wall keeps its own millimetres", w0["wall_length_mm"] == 3000)
check("a Claude opening keeps its own millimetres",
      w0["openings"][0]["offset_from_left_mm"] == 300
      and w0["openings"][0]["rough_width_mm"] == 900)

# ─── 7. Nothing private escapes into the payload ─────────────────────────────
# These keys are bookkeeping. Ekatan's Zod schema would not reject them (it
# ignores unknown keys), which is exactly why a leak here would go unnoticed.
print("\n7. Internal bookkeeping is stripped")

PRIVATE = ("_seg", "_frac_len", "_placeholder", "_axis", "_along_frac", "_width_frac")
leaked = set()
for r in (room, tall, side, blind):
    for w in r["walls"]:
        leaked |= {k for k in w if k in PRIVATE}
        for o in (w.get("openings") or []):
            leaked |= {k for k in o if k in PRIVATE}
check("no underscore-prefixed keys survive to the payload", not leaked, str(leaked))

# ─── Result ───────────────────────────────────────────────────────────────────
print("\n%d checks failed" % len(failures))
if failures:
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("All opening-geometry checks passed.")
