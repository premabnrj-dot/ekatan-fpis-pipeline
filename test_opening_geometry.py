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

Number 3 is what this file was written for. The rule it enforced was simple:
**an opening and the wall it sits on must always be in the same unit, and that
unit must be millimetres by the time the payload is built.**

⚠️ THAT RULE IS NOW ENFORCED BY HAVING NO OPENINGS. Extraction of doors and
windows was removed on 2026-09-02 — the owner judged the data unreliable, and
unreliable here is not cosmetic: an offset feeds `computeWallRun`, which measures
the longest free stretch of a wall, which is what a wardrobe is priced against.
Section 4 was rewritten to prove the payload carries none, and that every wall
still carries the empty `openings` key. Sections 4b, 5 and 6 went with it.

The name is kept. This file's remaining subject — winding, edge indices, wall
positions, the private-key strip — is still exactly the contract that made
opening offsets land correctly, and it is still what a reviewer's manually placed
door depends on: `polygon_edge_index` binds a wall to its polygon edge, and a
mirrored polygon sends every door a person places to the wrong end.

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


env = {"math": math, "re": re}
for pat, label in [
    (r"^def _ensure_clockwise\(.*?\n(?=\n\ndef |\n\n# )", "_ensure_clockwise"),
    (r"^def _r2s_walls_from_polygon\(.*?\n(?=\n\ndef |\n\n# )",
     "_r2s_walls_from_polygon"),
    (r"^def _rescale_walls_to_mm\(.*?\n(?=\n\ndef |\n\n# )", "_rescale_walls_to_mm"),
]:
    exec(_grab(pat, label), env)

ensure_clockwise = env["_ensure_clockwise"]
walls_from_polygon = env["_r2s_walls_from_polygon"]
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


def build_room(polygon, length_mm, width_mm):
    """Run the real Step 6 -> Step 7 path over one room."""
    poly = ensure_clockwise(polygon)
    room = {
        "room_type_code": "bedroom",
        "polygon_points": poly,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "walls": walls_from_polygon(poly),
    }
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

# ─── 4. THIS PIPELINE EMITS NO OPENINGS ──────────────────────────────────────
#
# Sections 4, 4b, 5 and 6 used to prove that a detected door landed on the right
# wall in the right unit. Opening extraction was removed on 2026-09-02 (owner:
# the data was not reliable), so what needs proving inverted: the payload must
# carry NO opening at all, and every wall must still carry the KEY.
#
# ⚠️ THE KEY MATTERS AS MUCH AS THE EMPTINESS. Ekatan reads `wall.openings ?? []`
# so an absent key would not break it — which is exactly why dropping it would go
# unnoticed until somebody assumed the pipeline had simply not been asked.
#
# ⚠️ AND AN EMPTY LIST IS NOT A DEGRADED ANSWER. An opening's offset feeds
# `computeWallRun`, which measures the longest free stretch of a wall, which is
# what a wardrobe is priced against. A door 300 mm out of place changes what
# fits and nothing downstream can tell that from a correct one. Emitting nothing
# removes a confident wrong answer; a reviewer supplies the right one.
print("\n4. No openings leave this pipeline")

room = build_room(rect(0.1, 0.1, 0.5, 0.3), 4000, 2000)
tall = build_room(rect(0.1, 0.1, 0.3, 0.6), 2000, 5000)
blind = build_room(rect(0.1, 0.1, 0.5, 0.3), None, None)

for label, r in (("a measured room", room), ("a tall room", tall),
                 ("an unmeasurable room", blind)):
    check("%s carries an openings key on every wall" % label,
          all("openings" in w for w in r["walls"]),
          str([sorted(w) for w in r["walls"]]))
    check("%s carries no openings" % label,
          all_openings(r) == [], str(all_openings(r)))

# The flag still fires: a room whose dimensions never resolved must say so
# rather than ship the `frac x 10000` placeholder as millimetres.
check("an unmeasurable room is still flagged",
      "wall_lengths_unscaled:room_dimensions_unknown"
      in (blind.get("validation_flags") or []),
      str(blind.get("validation_flags")))
check("and its walls report null length rather than a guess",
      all(w["wall_length_mm"] is None for w in blind["walls"]),
      str([w["wall_length_mm"] for w in blind["walls"]]))

# ⚠️ THE FALLBACK PROMPT'S OWN OPENINGS ARE DISCARDED TOO. The prompt tells the
# model not to report any, but a prompt is a request; the sanitiser is the
# guarantee. A model that volunteers a door must not be able to put it in a
# quote.
sanitize = None
m = re.search(r"for wall in \(room\.get\(\"walls\"\) or \[\]\):\n"
              r"\s+wall\[\"openings\"\] = \[\]", SRC)
check("the Claude fallback forcibly empties every wall's openings",
      m is not None,
      "the sanitiser that discards volunteered openings is gone")

# ─── 7. Nothing private escapes into the payload ─────────────────────────────
# These keys are bookkeeping. Ekatan's Zod schema would not reject them (it
# ignores unknown keys), which is exactly why a leak here would go unnoticed.
print("\n7. Internal bookkeeping is stripped")

PRIVATE = ("_seg", "_frac_len", "_placeholder", "_axis", "_along_frac", "_width_frac")
leaked = set()
for r in (room, tall, blind):
    for w in r["walls"]:
        leaked |= {k for k in w if k in PRIVATE}
        for o in (w.get("openings") or []):
            leaked |= {k for k in o if k in PRIVATE}
check("no underscore-prefixed keys survive to the payload", not leaked, str(leaked))

# ─── 8. Polygon sanitising, against REAL production polygons ─────────────────
# Every polygon below was read out of the production database on 2026-09-01,
# verbatim. Six of the nine rooms on that plan were spirals or had diagonal
# chords slicing across them; the designer's screenshot showed a kitchen with a
# spike stabbing into its own middle. Invented fixtures would not have caught
# this, so these are the real ones.
print("\n8. Polygon sanitising (real production polygons, 2026-09-01)")

try:
    import shapely  # noqa: F401
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False

if not HAVE_SHAPELY:
    # The rest of this file runs on bare CPython by design; only this section
    # needs Shapely, so it is skipped rather than failing the run.
    print("  skipped — shapely not installed locally (it is in the Modal image)")
else:
    for pat, dot in [
        (r"^RECTILINEAR_MIN_EDGE_FRAC = [^\n]*$", False),
        (r"^RECTILINEAR_SIN_TOLERANCE = [^\n]*$", False),
        (r"^DUPLICATE_ROOM_IOU = [^\n]*$", False),
        (r"^def _boxify\(.*?\n(?=\n\ndef |\n\n# |\n\n@|\n\n[A-Z_]+ =)", True),
        (r"^def _sanitize_room_polygon\(.*?\n(?=\n\ndef |\n\n# |\n\n@|\n\n[A-Z_]+ =)", True),
        (r"^def _dedupe_overlapping_rooms\(.*?\n(?=\n\ndef |\n\n# |\n\n@|\n\n[A-Z_]+ =)", True),
    ]:
        m = re.search(pat, SRC, re.M | (re.S if dot else 0))
        if not m:
            print("  FAIL could not extract %s" % pat[:44])
            failures.append("extract " + pat[:44])
        else:
            exec(m.group(0), env)

    sanitize = env["_sanitize_room_polygon"]
    dedupe = env["_dedupe_overlapping_rooms"]

    def pts(*xy):
        return [{"x": x, "y": y} for x, y in xy]

    # A bowtie: the closing edge cuts diagonally back through the shape.
    master_bedroom = pts((0.625, 0.8184), (0.625, 0.9852), (0.4647, 0.9852),
                         (0.4647, 0.7466), (0.8546, 0.7466), (0.8546, 0.9852))
    # Shapely calls this VALID. It is still not a room — a long diagonal chord.
    dining = pts((0.4579, 0.3316), (0.4579, 0.3696), (0.5815, 0.4266),
                 (0.5815, 0.6304), (0.3478, 0.5956), (0.2772, 0.6135),
                 (0.2772, 0.3316))
    # Visits (0.0897, 0.5723) twice in one ring.
    kitchen = pts((0.0897, 0.5723), (0.0897, 0.397), (0.1997, 0.4403),
                  (0.2758, 0.397), (0.2758, 0.4509), (0.2024, 0.4509),
                  (0.2704, 0.5111), (0.2704, 0.6209), (0.0897, 0.5723),
                  (0.0272, 0.6209))
    living = pts((0.4959, 0.2566), (0.8505, 0.2566),
                 (0.8505, 0.5164), (0.4959, 0.5164))
    # A rectilinear notch, where an attached bathroom cuts into the bedroom.
    notched_bedroom = pts((0.9837, 0.5174), (0.9837, 0.7434), (0.7554, 0.7434),
                          (0.8913, 0.7413), (0.8913, 0.6019), (0.7473, 0.6019),
                          (0.7473, 0.7413), (0.6495, 0.7413), (0.6495, 0.5174))

    check("a self-intersecting bowtie becomes its box",
          len(sanitize(master_bedroom)) == 4)
    check("a VALID polygon with a diagonal chord still becomes its box",
          len(sanitize(dining)) == 4,
          "shapely validity alone would have kept this")
    check("a ring that revisits a vertex becomes its box",
          len(sanitize(kitchen)) == 4)
    check("a clean rectangle is left alone",
          sanitize(living) == living)
    check("a RECTILINEAR notch survives — irregular rooms stay irregular",
          len(sanitize(notched_bedroom)) == len(notched_bedroom),
          "this is the requirement, not a bug: an L-shaped room must stay L-shaped")

    # ⚠️ FOUR POINTS ARE NOT AUTOMATICALLY A BOX. There was an early return here
    # saying they were, and a designer kept seeing one kitchen with a single
    # diagonal wall after everything else had been squared up: a four-point
    # trapezoid skipped the rectilinear check entirely.
    trapezoid = pts((0.10, 0.10), (0.50, 0.20), (0.50, 0.40), (0.10, 0.40))
    check("a FOUR-POINT trapezoid is boxed, not waved through",
          sanitize(trapezoid) != trapezoid,
          "four points are not automatically a rectangle")
    wobble = pts((0.10, 0.10), (0.50, 0.1005), (0.50, 0.40), (0.10, 0.40))
    check("a rectangle with sub-pixel raster wobble is still kept",
          sanitize(wobble) == wobble,
          "raster stair-stepping is noise, not a diagonal wall")

    box = sanitize(master_bedroom)
    check("the box spans the original's full extent",
          abs(min(p["x"] for p in box) - 0.4647) < 1e-9
          and abs(max(p["x"] for p in box) - 0.8546) < 1e-9
          and abs(min(p["y"] for p in box) - 0.7466) < 1e-9
          and abs(max(p["y"] for p in box) - 0.9852) < 1e-9)
    check("the box is wound clockwise, like every other polygon we emit",
          signed_area(box) > 0)

    # The same kitchen, detected twice — rooms 7 and 9 of that plan.
    kitchen_b = pts((0.091, 0.5723), (0.091, 0.397), (0.1427, 0.4424),
                    (0.2717, 0.397), (0.2717, 0.4498), (0.2024, 0.4615),
                    (0.2717, 0.5111), (0.2717, 0.6209), (0.0897, 0.5723),
                    (0.0272, 0.6209))
    dupes = [
        {"room_type_code": "kitchen", "extraction_confidence": 0.7,
         "polygon_points": sanitize(kitchen)},
        {"room_type_code": "kitchen", "extraction_confidence": 0.9,
         "polygon_points": sanitize(kitchen_b)},
    ]
    kept = dedupe(dupes)
    check("one kitchen detected twice collapses to one",
          len(kept) == 1, "got %d" % len(kept))
    check("the more confident detection is the one kept",
          kept and kept[0]["extraction_confidence"] == 0.9)

    # A bathroom's box legitimately sits inside a bedroom's. Different types
    # must never be merged — the door assigner depends on that nesting.
    nested = [
        {"room_type_code": "bedroom", "extraction_confidence": 0.9,
         "polygon_points": pts((0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5))},
        {"room_type_code": "bathroom", "extraction_confidence": 0.8,
         "polygon_points": pts((0.15, 0.15), (0.3, 0.15), (0.3, 0.3), (0.15, 0.3))},
    ]
    check("a bathroom inside a bedroom is left alone — different types",
          len(dedupe(nested)) == 2)

# ─── 9. Room names vs dimensions, from the same live plan ────────────────────
# Six of nine rooms were named after their own dimensions ("14'2x11'0"") because
# the old filter stripped quotes and checked isdigit() — and the `x` survived
# that strip. The picker then preferred the LONGEST candidate, and a printed
# dimension is longer than "BEDROOM", so the wrong string won twice over.
print("\n9. A dimension is never a room's name")

for pat, dot in [
    (r"^_QUOTE = [^\n]*$", False),
    (r"^_DIM_PAIR_FT = re\.compile\(.*?\n\)", True),
    (r"^_DIM_PAIR_MM = [^\n]*$", False),
    (r"^def _parse_dimension_pair\(.*?\n(?=\n\ndef |\n\n# |\n\n@|\n\n[A-Z_]+ =)", True),
    (r"^def _is_dimension_text\(.*?\n(?=\n\ndef |\n\n# |\n\n@|\n\n[A-Z_]+ =)", True),
]:
    m = re.search(pat, SRC, re.M | (re.S if dot else 0))
    if not m:
        print("  FAIL could not extract %s" % pat[:44])
        failures.append("extract " + pat[:44])
    else:
        exec(m.group(0), env)

is_dim = env["_is_dimension_text"]
parse_pair = env["_parse_dimension_pair"]

# Every one of these was stored as a room_label in production.
for s in ["14'2x11'0\"", "13'6x10'6", "11'6x110", "9'0'x10'6\"", "5'6x7'8\"",
          "5'0\"WIDE"]:
    check("%-14r is recognised as a measurement" % s, is_dim(s))

for s in ["M.BEDROOM", "LIVING", "BEDROOM", "TOILET", "KITCHEN",
          "UTILITY 5'9\"WIDE"]:
    check("%-18r is recognised as a name" % s, not is_dim(s))

# The kitchen's own dimensions sat in the text beside it and were discarded,
# because OCR read the inches mark as an apostrophe: 9'0' rather than 9'0".
kitchen = parse_pair("9'0'x10'6\"")
check("a stray apostrophe no longer costs a room its dimensions",
      kitchen is not None and abs(kitchen[0] - 2743) < 2 and abs(kitchen[1] - 3200) < 2,
      str(kitchen))
check("a well-formed pair still parses",
      parse_pair("14'2x11'0\"") is not None)
check("a pair with an apostrophe dropped entirely is NOT guessed at",
      parse_pair("11'6x110") is None,
      "110 could be 11'0\" or 1'10\" — no dimension beats an invented one")

# ─── 10. The drawing outranks the model ──────────────────────────────────────
# A live run logged this twice and lost the room both times:
#   rejected 26-toilet->bathroom: label disagrees ('UTILITY 5'9"WIDE')
# The model saw a sink and a washing machine and said "toilet"; the gate refused
# to store a bathroom over a space the drawing calls UTILITY, and then dropped
# it. The gate was right and the outcome was still wrong — the plan came back
# with no utility at all.
print("\n10. A disagreeing label retypes the room instead of deleting it")

for pat in [r"^YOLO_ROOM_TO_CODE = \{.*?^\}",
            r"^YOLO_LABEL_AGREEMENT = \{.*?^\}",
            r"^REFINEMENT_ONLY_CODES = \{.*?^\}",
            r"^def _rescue_code_from_label\(.*?\n(?=\n\ndef |\n\n# |\n\n@|\n\n[A-Z_]+ =)"]:
    m = re.search(pat, SRC, re.M | re.S)
    if not m:
        print("  FAIL could not extract %s" % pat[:40])
        failures.append("extract " + pat[:40])
    else:
        exec(m.group(0), env)

rescue = env["_rescue_code_from_label"]

check("the text that deleted a utility twice now recovers it",
      rescue('UTILITY 5\'9"WIDE') == "utility")
check("a toilet label resolves to bathroom",
      rescue("TOILET 8'6X5'6") == "bathroom")
check("a balcony label resolves to balcony",
      rescue('BALCONY 5\'0"WIDE') == "balcony")
check("two room names in one polygon rescues nothing",
      rescue("BEDROOM TOILET") is None,
      "that polygon spans two rooms; guessing would reintroduce mislabelling")
check("text naming no room rescues nothing",
      rescue("GODREJ ROYALE WOODS") is None)
# Refinement-only codes must not be reachable here — TOILET matches bathroom,
# master_bathroom and powder_room, and without the predictable-only filter the
# result would be ambiguous and nothing would ever be rescued.
check("refinement-only codes are not candidates",
      rescue("MASTER TOILET") in (None, "bathroom"),
      "got %r" % rescue("MASTER TOILET"))
# ⚠️ The candidate set is NOT "what the model predicts". It was, briefly, and
# `store` broke the moment 29-walkin was retyped to walk_in_wardrobe: store
# stopped being any class's output, so a polygon plainly labelled STORE could no
# longer be rescued to it.
check("a code no class predicts can still be rescued from its printed name",
      rescue("STORE") == "store",
      "store has no YOLO class of its own since 29-walkin was retyped")
check("a walk-in wardrobe is rescued from the abbreviation plans actually use",
      rescue("WWR") == "walk_in_wardrobe")

# ─── Result ───────────────────────────────────────────────────────────────────
print("\n%d checks failed" % len(failures))
if failures:
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("All opening-geometry checks passed.")
