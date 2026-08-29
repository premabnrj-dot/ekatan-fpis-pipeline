# -*- coding: utf-8 -*-
"""Unit tests for `_refine_room_code` and the tables it depends on.

WHY THIS FILE EXISTS
--------------------
`_refine_room_code` types seven rooms the model has no class for, by reading the
words printed on the drawing. Five of those codes were added on 2026-08-29 after
a measurement against the live catalogue: `kids_room`, `guest_room`,
`home_theatre`, `powder_room` and `master_bathroom` carry **17 published L3
systems between them and had been detected zero times**, because nothing in this
pipeline ever emitted the codes. `kids_room` alone has more published systems
than `kitchen`.

The dangerous part is not the refinement, it is the GATE THAT RUNS AFTER IT.
`extract_rooms_yolo` does:

    code = _refine_room_code(code, texts)
    if not any(w in joined for w in YOLO_LABEL_AGREEMENT.get(code, ())):
        rejected.append(...)          # <- the room is DROPPED

`.get(code, ())` means a promotion to a code that is absent from
YOLO_LABEL_AGREEMENT matches the empty tuple and the room disappears — a strictly
worse outcome than never promoting, and a silent one. Test 1 is that closure, and
it is the reason this file exists at all; the rest are ordinary behaviour checks.

Run:  python test_refine_room_code.py
"""
import re
import sys
import io
import types

PATH = "fpis_pipeline.py"

# ─── Load the tables + function WITHOUT importing the module ─────────────────
# fpis_pipeline.py imports modal/cv2/torch at module scope, none of which are
# installed locally. Everything under test is pure Python with no dependencies,
# so it is extracted by source and exec'd in a bare namespace. That keeps this
# runnable on a laptop with nothing but CPython — the alternative (mocking modal)
# tests the mock as much as the code.
SRC = io.open(PATH, encoding="utf-8").read()


def _grab(pattern, label):
    m = re.search(pattern, SRC, re.S | re.M)
    if not m:
        print("FAIL: could not locate %s in %s" % (label, PATH))
        sys.exit(1)
    return m.group(0)


ns = types.SimpleNamespace()
env = {}
for pat, label in [
    (r"^VALID_ROOM_CODES = \{.*?^\}", "VALID_ROOM_CODES"),
    (r"^WET_AREA_CODES = \{.*?\}", "WET_AREA_CODES"),
    (r"^ROOM_LABEL_WORDS = \(.*?^\)", "ROOM_LABEL_WORDS"),
    (r"^YOLO_LABEL_AGREEMENT = \{.*?^\}", "YOLO_LABEL_AGREEMENT"),
    (r"^def _refine_room_code\(.*?\n(?=\n\ndef |\n\n# )", "_refine_room_code"),
]:
    exec(_grab(pat, label), env)

refine = env["_refine_room_code"]
VALID = env["VALID_ROOM_CODES"]
WET = env["WET_AREA_CODES"]
WORDS = env["ROOM_LABEL_WORDS"]
AGREE = env["YOLO_LABEL_AGREEMENT"]

failures = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        failures.append(msg)


def refined(code, *texts):
    return refine(code, list(texts))


# ─── 1. THE CLOSURE RULE — every reachable output survives the gate ──────────
# This is the test that matters. It reproduces the caller's own gate rather than
# describing it, so it fails if either side drifts.
print("\n1. every code _refine_room_code can return is gated, valid, and reachable")

REACHABLE = {
    # (base code, a label that triggers it) -> expected code
    ("bedroom", "MASTER BEDROOM"):        "master_bedroom",
    ("bedroom", "SERVANT ROOM"):          "servant_room",
    ("bedroom", "KIDS BEDROOM"):          "kids_room",
    ("bedroom", "GUEST BEDROOM"):         "guest_room",
    ("bedroom", "BEDROOM"):               "bedroom",
    ("bathroom", "M.TOILET"):             "master_bathroom",
    ("bathroom", "POWDER ROOM"):          "powder_room",
    ("bathroom", "TOILET"):               "bathroom",
    ("living_room", "HOME THEATRE"):      "home_theatre",
    ("living_room", "LIVING"):            "living_room",
}

for (base, label), expected in REACHABLE.items():
    got = refined(base, label)
    check(got == expected, "%-11s + %-15r -> %s" % (base, label, expected)
          + ("" if got == expected else "   (got %r)" % got))

    # (a) the refined code must be a real room_types code
    check(got in VALID, "   %-15s is in VALID_ROOM_CODES" % got)

    # (b) it must be a KEY in the agreement table, or .get() returns () and the
    #     caller drops the room. This is the trap the whole file guards.
    check(got in AGREE, "   %-15s is a key in YOLO_LABEL_AGREEMENT" % got)

    # (c) and the very label that triggered the promotion must satisfy that
    #     entry — otherwise the promotion fires and the gate immediately
    #     rejects the room it just typed correctly.
    joined = label.upper()
    check(any(w in joined for w in AGREE.get(got, ())),
          "   %-15s agreement accepts %r" % (got, label))

    # (d) and the label must clear the EARLIER gate too. A word missing from
    #     ROOM_LABEL_WORDS throws the polygon out before the refinement ever
    #     runs, so the refinement would be dead code.
    check(any(w in joined for w in WORDS),
          "   %-15s label clears ROOM_LABEL_WORDS" % got)


# ─── 2. existing behaviour is unchanged ─────────────────────────────────────
print("\n2. the two pre-existing promotions still behave exactly as before")
check(refined("bedroom", "MASTER BEDROOM 12'0\" X 11'8\"") == "master_bedroom",
      "MASTER BEDROOM with dimensions -> master_bedroom")
check(refined("bedroom", "M.BED") == "master_bedroom", "M.BED -> master_bedroom")
check(refined("bedroom", "M. BED") == "master_bedroom", "M. BED -> master_bedroom")
check(refined("bedroom", "MAID ROOM") == "servant_room", "MAID -> servant_room")
check(refined("bedroom", "BEDROOM 10'6\"X11'6\"") == "bedroom", "plain bedroom untouched")


# ─── 3. codes with no refinement rule pass straight through ─────────────────
print("\n3. untouched codes pass through unchanged")
for code in ("kitchen", "balcony", "utility", "dining_area", "foyer_entrance",
             "passage", "staircase", "store", "pooja_room", "terrace", "other"):
    check(refined(code, "ANYTHING AT ALL") == code, "%s passes through" % code)


# ─── 4. precedence inside each base type ────────────────────────────────────
print("\n4. most-specific-first precedence")
check(refined("bedroom", "MASTER BEDROOM", "KIDS") == "master_bedroom",
      "MASTER beats KIDS (existing behaviour wins)")
check(refined("bathroom", "MASTER TOILET", "POWDER") == "master_bathroom",
      "MASTER TOILET beats POWDER")
check(refined("bedroom", "SERVANT", "GUEST") == "servant_room",
      "SERVANT beats GUEST")


# ─── 5. the real-world labels this was built from ───────────────────────────
# Taken from plans in the annotation kit rather than invented.
print("\n5. labels observed on real Bengaluru brochure plans")
for label, base, expected in [
    ("M.TOILET 8'X5'1\"",   "bathroom",    "master_bathroom"),
    ("M.BEDROOM 15'1\"X11'8\"", "bedroom", "master_bedroom"),
    ("C TOILET 8'X5'1\"",   "bathroom",    "bathroom"),
    ("CHILDREN BED ROOM",   "bedroom",     "kids_room"),
    ("GUEST BED 10'X10'",   "bedroom",     "guest_room"),
    ("SLEEP",               "bedroom",     "bedroom"),
]:
    got = refined(base, label)
    check(got == expected, "%-24r -> %s%s" % (label, expected,
          "" if got == expected else "   (got %r)" % got))


# ─── 6. the wet-area flag the caller stamps from WET_AREA_CODES ─────────────
print("\n6. wet-area flag follows the promotion")
check("master_bathroom" in WET, "master_bathroom is a wet area")
check("powder_room" in WET, "powder_room is a wet area")
check("kids_room" not in WET, "kids_room is NOT a wet area")
check("guest_room" not in WET, "guest_room is NOT a wet area")
check("home_theatre" not in WET, "home_theatre is NOT a wet area")


# ─── 7. a promotion that is NOT wired up must not silently appear ───────────
# Guards the reverse drift: someone adds a return to _refine_room_code and
# forgets the two tables. Scans the function's own source for returned literals.
print("\n7. no returned literal escapes the tables")
fn_src = _grab(r"^def _refine_room_code\(.*?\n(?=\n\ndef |\n\n# )", "_refine_room_code")
returned = set(re.findall(r'return "([a-z_]+)"', fn_src))
for code in sorted(returned):
    check(code in VALID, "returned %-16s is in VALID_ROOM_CODES" % code)
    check(code in AGREE, "returned %-16s is a key in YOLO_LABEL_AGREEMENT" % code)


print("\n" + "=" * 66)
if failures:
    print("%d FAILURE(S):" % len(failures))
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("all checks passed")
