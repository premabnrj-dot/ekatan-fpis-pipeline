# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Ekatan
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. It is distributed WITHOUT ANY WARRANTY; see the licence for
# details: <https://www.gnu.org/licenses/>.
#
# ⚠️ THIS FILE IS PUBLIC — https://github.com/premabnrj-dot/ekatan-fpis-pipeline
# It uses Ultralytics YOLO (AGPL-3.0) and Ekatan serves inference over a network,
# so section 13 applies and this source is published (ADR-171). Two consequences,
# both load-bearing:
#   • NEVER put a secret, credential or customer identifier in this file. Every
#     secret arrives through modal.Secret / os.environ, and that is a rule now.
#   • A STALE PUBLIC COPY IS NOT COMPLIANCE. `modal deploy` ships from a working
#     copy and forces no commit, so deploy and publish in the same motion.
# =============================================================================
# FILE: fpis_pipeline.py
# Ekatan FPIS — Floor Plan Intelligence System
# Modal.com serverless pipeline — Phase 1 (Claude Sonnet 4.6)
#
# Deploy:   modal deploy fpis_pipeline.py
# Test:     modal run fpis_pipeline.py
# Logs:     modal app logs ekatan-fpis --follow
# Rollback: modal app rollback ekatan-fpis
#
# The pipeline, as it actually runs. Keep this block true: it is the first
# thing a new reader (or agent) trusts, and it was wrong for months - it called
# Step 6 a Claude step long after YOLO took over, which is how an architecture
# review can start from a false premise.
#   0.5 PDF normalisation   — PyMuPDF (render page 1; use the exact text layer)
#   0  Pre-flight Quality    — Heuristics (file size, image dims, aspect ratio)
#   1  Quality Gate          — Claude Sonnet 4.6 (reject non-floor-plans early)
#   2  Image Preprocessing   — OpenCV (deskew, CLAHE, longest side to 2000px)
#   3  OCR Extraction        — PaddleOCR (room labels, dimension strings)
#   4  Scale Detection       — Claude Sonnet 4.6 (title block: unit system, north)
#   6  Room Extraction       — YOLO26n-seg on GPU, gated against the drawing's
#                              printed labels; Claude Sonnet 4.6 is the FALLBACK
#                              when the model is absent, errors, or disagrees
#   7  Spatial Reconciliation — Shapely (link OCR dims → room polygons, then
#                              resolve walls AND their openings into millimetres)
#   8  Rules Engine          — Python (plausibility checks, confidence scoring)
#   9  Webhook Callback      — HTTP POST to /api/fpis/callback
#
# Step 5 (MEP zone detection) was removed on 2026-09-01 — see _run_all_steps.
# The step numbers are deliberately NOT renumbered: they appear in production log
# lines going back months, and renumbering would silently break that history.
#
# Env vars required (set via: modal secret create ekatan-fpis-secrets ...):
#   ANTHROPIC_API_KEY     — Anthropic Claude API key
#   FPIS_WEBHOOK_SECRET   — Shared secret, must match Vercel FPIS_WEBHOOK_SECRET
#
# Code conventions:
#   Room type codes   — match room_types.code in Supabase (snake_case lowercase)
#   Opening type codes — match opening_types.code in Supabase (snake_case lowercase)
# =============================================================================

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import math
import re
import time
import traceback
from typing import Any

import modal

# ─── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("ekatan-fpis")

# Shared secret loaded from Modal secret store
fpis_secrets = modal.Secret.from_name("ekatan-fpis-secrets")

# CPU image — sufficient for Phase 1 (Claude handles all heavy lifting)
def _download_ocr_models() -> None:
    """
    Runs ONCE at image build time. Constructing PaddleOCR downloads its
    detection / recognition / angle-classifier weights from
    paddleocr.bj.bcebos.com - which was measured on 2026-08-28 delivering the
    4 MB en_PP-OCRv3 detector at 2-11 kiB/s (ETA 15-30 minutes) from Modal's
    region. Doing that per request is what made every cold run miss its
    delivery window. Baked into the layer, it costs zero at request time.
    """
    from paddleocr import PaddleOCR

    PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


_OCR = None


def _get_ocr():
    """One PaddleOCR instance per container, not one per request."""
    global _OCR
    if _OCR is None:
        from paddleocr import PaddleOCR

        _OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _OCR


cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
.apt_install("libgl1-mesa-glx", "libglib2.0-0")   # ← add this line
    .pip_install(
        "fastapi[standard]",
        "anthropic>=0.39.0",
        "paddlepaddle",
        "paddleocr==2.8.1",
        "opencv-python-headless==4.10.0.84",
        "Pillow==10.4.0",
        "numpy==1.26.4",
        "Shapely==2.0.6",
        "httpx==0.27.2",
        "PyPDF2>=3.0.0",   # Step 0 PDF page-count pre-flight
        "pymupdf>=1.24",   # Step 0.5 vector-PDF fast path
    )
    # Bake the OCR weights into the image - see _download_ocr_models above.
    .run_function(_download_ocr_models, timeout=3600)
)

# Web image - the endpoint only validates and spawns, so it carries nothing
# heavy. A cold start here is a couple of seconds, not a paddle image pull;
# the caller (triggerFpisExtraction) aborts at 8s.
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]"
)

# GPU image — Step 6 room segmentation (Ultralytics YOLO26n-seg, not CubiCasa5K;
# the name survived from a design that was never shipped)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "anthropic>=0.39.0",
        "torchvision>=0.16.0",
        "Pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "Shapely==2.0.6",
        "httpx==0.27.2",
        "fastapi[standard]",
        # Pinned to the version the checkpoint was trained with (the .pt records
        # ultralytics 8.4.43, 2026-04-29). An Ultralytics .pt is a pickled model
        # OBJECT, not a state_dict - a version drift here is a load failure, not
        # a warning. AGPL-3.0 / commercial dual licence, accepted by the owner.
        "ultralytics==8.4.43",
    )
    # Ultralytics writes a settings file on first import; /root/.config is not
    # writable in the container, which produced a WARNING on every run.
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics"})
)

# Modal Volume — holds the fine-tuned segmentation weights. The file the code
# actually loads is YOLO_WEIGHTS_PATH (/models/floor_plan_indian_best.pt); the
# "cubicasa5k_bangalore_v1.pt" this comment used to name has never existed.
model_volume = modal.Volume.from_name("fpis-model-weights", create_if_missing=True)

# ─── Canonical code tables (must match Supabase room_types + opening_types) ──
#
# room_types.code values in DB:
#   master_bedroom, bedroom, living_room, dining_area, kitchen, bathroom,
#   pooja_room, home_office, foyer_entrance, balcony, utility,
#   passage, staircase, servant_room, store, terrace, other,
#   kids_room, guest_room, home_theatre, powder_room, master_bathroom
#
# opening_types.code values in DB:
#   single_door, double_door, sliding_door, french_door, pocket_door,
#   window_standard, window_bay, window_corner, ventilator,
#   arched_opening, niche_shallow, niche_deep,
#   exhaust_opening, duct_access, pass_through, meter_box

# ⚠️ `utility_balcony` IS NOT A ROOM TYPE. It is a legacy code, dropped in
# Ekatan migration 0088, and `room-type-normalize.ts` aliases it to `balcony`.
# Emitting it meant every UTILITY this pipeline found was stored as a BALCONY —
# and per ADR-133 those are opposites in the only way that matters commercially:
#
#     balcony   outdoor · 80 rooms in the DB · 0 L3 systems mapped · sells nothing
#     utility   indoor work area · 24 rooms · 2 L3 systems · sells storage
#
# The resolver's label rescue only ever promotes utility -> balcony, never back,
# so nothing downstream could undo it. Verified against the live room_types
# table on 2026-08-28. These two are now separate, and must stay separate.
VALID_ROOM_CODES = {
    "master_bedroom", "bedroom", "living_room", "dining_area",
    "kitchen", "bathroom", "pooja_room", "home_office",
    "foyer_entrance", "balcony", "utility", "passage", "staircase",
    "servant_room", "store", "terrace", "other",
    # These five are reached ONLY by `_refine_room_code`, never by the model.
    # Each is a bedroom or a bathroom in SHAPE - nothing about the outline tells
    # them apart, only the printed name does - so they are deliberately NOT
    # training classes. Measured against the live catalogue on 2026-08-29 they
    # carry 17 published L3 systems between them and had been detected ZERO
    # times, because nothing in this pipeline ever emitted the codes:
    #     kids_room 7 systems - guest_room 4 - home_theatre 2 -
    #     powder_room 2 - master_bathroom 2
    # kids_room alone has more published systems than kitchen.
    "kids_room", "guest_room", "home_theatre", "powder_room", "master_bathroom",
}

VALID_OPENING_CODES = {
    "single_door", "double_door", "sliding_door", "french_door",
    "pocket_door", "window_standard", "window_bay", "window_corner",
    "ventilator", "arched_opening", "niche_shallow", "niche_deep",
    "exhaust_opening", "duct_access", "pass_through", "meter_box",
}

# Rooms that must always have is_wet_area = true
WET_AREA_CODES = {"bathroom", "kitchen", "utility", "balcony",
                  "master_bathroom", "powder_room"}

# Rooms exempt from the "zero openings" rule
NO_OPENING_EXEMPT = {"passage", "staircase", "store"}

# Rooms excluded from carpet area total (external / non-habitable).
# ⚠️ A UTILITY IS NOT EXTERNAL. It is the indoor work area beside the kitchen
# (ADR-133), so it belongs in carpet area; only the balcony is outside. While
# both wore the `utility_balcony` code this list silently excluded every utility
# from the plan-area check, which is how Rule 8 could pass on a plan whose
# internal area was understated.
EXTERNAL_ROOM_CODES = {"balcony", "staircase", "terrace"}

# ─── Entry point ─────────────────────────────────────────────────────────────

@app.function(image=web_image, timeout=60, memory=512)
@modal.fastapi_endpoint(method="POST")
def run_pipeline(payload: dict) -> dict:
    """
    Main entry point. Called by Ekatan's triggerFpisExtraction server action.

    THIS FUNCTION MUST STAY FAST. It validates, spawns `_run_job`, and
    returns - nothing else.

    It used to run all ten steps inline, and that is what broke delivery
    (diagnosed 2026-08-28 from `modal app logs ekatan-fpis`). The caller fires
    the POST with an 8-second abort and treats the abort as "accepted", but
    Modal CANCELS THE INPUT when the HTTP client disconnects:

        [modal-client] Received a cancellation signal while processing input
        Runner failed with exception: Runner has been shutting down for too
        long (grace period: 30 seconds)

    So every upload got a kill signal 8 seconds in and then raced a 30-second
    grace period it usually lost. The designer pressed Retry, each retry wrote
    a fresh `fpis_job_id` on the plan, and when an earlier run DID finish it
    posted a perfectly good result that /api/fpis/callback then correctly
    refused with 409 "Job ID mismatch - stale callback ignored". Recent window
    before the fix: 14 rejected callbacks, 1 accepted, 6 cancellations.

    A spawned call is owned by Modal, not by the HTTP connection, so there is
    no disconnect to cancel and no 150-second web-endpoint cap to outrun.

    Expected payload:

    Expected payload:
        job_id           str   — unique job identifier (fpis_<planId>_<ts>)
        plan_id          str   — UUID of unit_type_floor_plans row
        image_url        str   — public Supabase Storage URL
        callback_url     str   — https://ekatan.vercel.app/api/fpis/callback
        property_type    str   — 'apartment' | 'villa' | etc.
        bedroom_count    int?  — hint for Room Extraction prompt
        carpet_area_sqft float?— hint for Rules Engine area check
        floor_number     int?  — hint for title block parsing
    """
    import os

    job_id       = payload.get("job_id", "")
    plan_id      = payload.get("plan_id", "")
    image_url    = payload.get("image_url", "")
    callback_url = payload.get("callback_url", "")

    if not all([job_id, plan_id, image_url, callback_url]):
        return {"ok": False, "error": "Missing required fields: job_id, plan_id, image_url, callback_url"}

    call = _run_job.spawn(payload)
    print(f"[FPIS] Accepted job {job_id} for plan {plan_id} as call {call.object_id}")
    return {"ok": True, "job_id": job_id, "call_id": call.object_id, "accepted": True}


# --- Worker - the actual ten steps -------------------------------------------

@app.function(
    image=cpu_image,
    secrets=[fpis_secrets],
    timeout=1800,          # generous: nothing is waiting on an HTTP connection
    memory=2048,
    cpu=2,
)
def _run_job(payload: dict) -> dict:
    """
    Runs the pipeline and delivers the result to Ekatan's webhook.

    Spawned by `run_pipeline`. Never called over HTTP, so it cannot be
    cancelled by a client disconnect.
    """
    import os

    job_id       = payload.get("job_id", "")
    plan_id      = payload.get("plan_id", "")
    callback_url = payload.get("callback_url", "")

    try:
        result = _run_all_steps(payload)
        _post_callback(callback_url, result, os.environ["FPIS_WEBHOOK_SECRET"])
        return {"ok": True, "job_id": job_id, "rooms_detected": len(result.get("rooms", []))}
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[FPIS] Pipeline failed: {exc}\n{tb}")
        _post_callback(callback_url, {
            "job_id":        job_id,
            "plan_id":       plan_id,
            "status":        "failed",
            "error_message": f"Pipeline error: {exc}",
        }, os.environ.get("FPIS_WEBHOOK_SECRET", ""))
        return {"ok": False, "error": str(exc)}


# ─── Pipeline orchestrator ───────────────────────────────────────────────────

def _run_all_steps(payload: dict) -> dict:
    import httpx

    job_id        = payload["job_id"]
    plan_id       = payload["plan_id"]
    image_url     = payload["image_url"]
    bedroom_count = payload.get("bedroom_count")
    carpet_area   = payload.get("carpet_area_sqft")
    property_type = payload.get("property_type", "apartment")

    print(f"[Step 0] Starting job {job_id} for plan {plan_id}")

    # ── Download image ────────────────────────────────────────────────────────
    resp = httpx.get(image_url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    raw_bytes = resp.content
    print(f"[Step 0] Downloaded {len(raw_bytes):,} bytes from Supabase")

    # -- Step 0.5: a PDF becomes a raster (plus its exact text layer) -----
    pdf_text_regions = None
    if raw_bytes[:4] in (b"%PDF", b"%pdf"):
        try:
            raw_bytes, pdf_text_regions = _pdf_to_raster_and_text(raw_bytes)
        except Exception as exc:
            return {
                "job_id":        job_id,
                "plan_id":       plan_id,
                "status":        "failed",
                "error_message": f"Could not read the PDF: {exc}",
            }

    # ── Step 0: Pre-flight Quality Check ────────────────────────────────────────
    preflight = _step0_preflight_check(raw_bytes)
    if not preflight['proceed']:
        return {
            "job_id":        job_id,
            "plan_id":       plan_id,
            "status":        "failed",
            "error_message": preflight['reason'],
        }
    print(f"[Step 0] Pre-flight check passed")

    # ── Step 1: Quality Gate ─────────────────────────────────────────────────
    quality = _step1_quality_gate(raw_bytes)
    print(f"[Step 1] Quality: {quality}")
    if quality["classification"] in ("low_quality", "not_a_floor_plan"):
        return {
            "job_id":        job_id,
            "plan_id":       plan_id,
            "status":        "failed",
            "error_message": f"Image rejected by quality gate: {quality['classification']}. {quality.get('reason', '')}",
        }

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    processed_bytes = _step2_preprocess(raw_bytes, is_photo=(quality["classification"] == "photo_of_drawing"))
    print(f"[Step 2] Preprocessed: {len(processed_bytes):,} bytes")

    # ── Step 3: OCR (runs first — Steps 4+5 need text hints) ─────────────────
    # A vector PDF already handed us its text with exact coordinates;
    # running PaddleOCR over a render of it would only add recognition error.
    if pdf_text_regions is not None:
        ocr_results = pdf_text_regions
        print(f"[Step 3] Using {len(ocr_results)} exact PDF text regions (OCR skipped)")
    else:
        ocr_results = _step3_ocr(processed_bytes)
        print(f"[Step 3] OCR found {len(ocr_results)} text regions")

    # ── Step 4: Scale / title block ──────────────────────────────────────────
    # Step 5 (MEP zone detection) was REMOVED on 2026-09-01 by the owner's
    # decision: this pipeline's job is rooms, dimensions and openings, and MEP
    # zones are fixtures inside a room rather than the room itself. Nothing in
    # Ekatan prices them - the only consumer that reaches a live flow is a
    # `ruleType: 'warn'` advisory on a quote line
    # (src/domains/constraint-engine/geometry-constraints.ts), plus a display-only
    # `acProvisionCount`. `mep_zones` is optional in Ekatan's callback schema, so
    # omitting the key is a clean no-op there; a designer can still add zones by
    # hand in the GFC editor.
    scale_handle = _step4_scale_detect.spawn(processed_bytes, ocr_results)
    scale_info   = scale_handle.get(timeout=120)   # must stay under run_pipeline's 600s budget
    print(f"[Step 4] Scale: {scale_info}")

    # ── Step 6: Room Extraction (CubiCasa5K segmentation) ────────────────────
    # Runs in a separate GPU container via .remote() — returns exact polygons.
    # Falls back to Claude extraction if model weights not yet deployed.
    rooms_raw = _step6_raster2seq_extract.remote(
        processed_bytes, ocr_results, scale_info, bedroom_count, property_type
    )
    print(f"[Step 6] Raster2Seq extracted {len(rooms_raw)} rooms")

    # ── Step 6b: Windows ──────────────────────────────────────────────────────
    # Only on the YOLO path: the model has no window class, so without this a
    # plan ships with doors and nothing else. The Claude fallback already
    # returns windows in its own geometry, and asking twice would double them.
    if _rooms_came_from_yolo(rooms_raw):
        rooms_raw = _step6b_windows_detect.remote(processed_bytes, rooms_raw)
    else:
        print("[Step 6b] Skipped - the Claude fallback already returns windows")

    # ── Step 7: Spatial Reconciliation ────────────────────────────────────────
    rooms_reconciled = _step7_reconcile(rooms_raw, ocr_results, scale_info)
    print(f"[Step 7] Reconciled rooms: {len(rooms_reconciled)}")

    # ── Step 7b: read the dimensions OCR could not ────────────────────────────
    rooms_reconciled = _step7b_verify_dimensions(processed_bytes, rooms_reconciled)

    # Walls and openings become real millimetres LAST, once every source of a
    # room dimension has been tried - OCR in Step 7, then the drawing itself in
    # 7b. Doing this inside Step 7 measured walls against the OCR answer and
    # never revisited them when 7b recovered a better one.
    _rescale_walls_to_mm(rooms_reconciled)

    # ── Step 8: Rules Engine ─────────────────────────────────────────────────
    rooms_final = _step8_rules_engine(rooms_reconciled, carpet_area)
    print(f"[Step 8] Final rooms: {len(rooms_final)}")

    return {
        "job_id":              job_id,
        "plan_id":             plan_id,
        "status":              "completed",
        "unit_system":         scale_info.get("unit_system", "mm"),
        "drawing_scale":       scale_info.get("drawing_scale"),
        "scale_confidence":    scale_info.get("confidence", 0.0),
        "north_direction_deg": scale_info.get("north_direction_deg"),
        "rooms":               rooms_final,
        "raw_extraction": {
            "ocr_regions_count":  len(ocr_results),
            "scale_info":         scale_info,
            "rooms_before_rules": len(rooms_reconciled),
        },
    }


# --- Step 0.5: PDF normalisation (vector fast path) --------------------------
#
# WHY THIS EXISTS. Both upload surfaces accept application/pdf (see
# client/(portal)/upload/_components/floor-plan-dropzone.tsx). Nothing in this
# pipeline ever converted one: Step 1 handed raw PDF bytes to
# _image_for_claude, PIL could not open them, and the run died with
# "Pipeline error". A real customer upload sat at fpis_status='processing'
# from 2026-07-14 to 2026-08-28 for exactly this reason.
#
# A vector PDF also carries its own text layer with EXACT coordinates - room
# labels, dimension strings, the title block. That is strictly better than
# OCR: no model download, no misread, no confidence below 1.0. When one is
# present we use it and skip PaddleOCR entirely.
#
# NOT DONE HERE: recovering room POLYGONS from the vector wall geometry. The
# walls are there (filled paths, identifiable by fill colour - 101 of them on
# the sample plan) but turning them into rooms needs page-rotation handling,
# complete wall-fill identification, doorway bridging and seed filtering.
# Measured against a real Bengaluru plan on 2026-08-28: watershed seeded on
# label text leaked across rooms because the wall mask was incomplete.
# Deliberately left as a follow-up rather than shipped half-working.

MAX_PDF_RENDER_PX = 2600


def _pdf_to_raster_and_text(pdf_bytes: bytes):
    """
    Render page 1 of a PDF to PNG and, when the file is a real vector PDF,
    return its text layer in the same shape _step3_ocr produces.

    Returns (png_bytes, text_regions_or_None). Raises on an unreadable PDF so
    the caller can fail the job with a message a human can act on.
    """
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")
    page = doc[0]

    long_pt = max(page.rect.width, page.rect.height) or 1.0
    dpi = max(72, min(300, int(MAX_PDF_RENDER_PX * 72 / long_pt)))
    pix = page.get_pixmap(dpi=dpi)
    png = pix.tobytes("png")
    print("[Step 0.5] PDF rendered at %d dpi -> %dx%dpx" % (dpi, pix.width, pix.height))

    words = page.get_text("words")   # (x0, y0, x1, y1, text, block, line, word)
    if not words:
        print("[Step 0.5] No text layer (scanned PDF) - OCR will run normally")
        return png, None

    # ⚠️ ROTATION. get_text() and get_drawings() report coordinates in the
    # UNROTATED mediabox space; get_pixmap() renders the ROTATED page. On a
    # /Rotate 90 or 270 sheet those are transposed, and page.rotation_matrix is
    # what maps the former onto the latter.
    #
    # The first version of this function tried to paper over the mismatch by
    # widening the divisor until every word fit. That produced coordinates in
    # neither space: on the 2026-07-14 sample (rotation 270, mediabox 595x842,
    # render 842x595) the word "KITCHEN" came out at (0.503, 0.246) when it
    # renders at (0.217, 0.289). Every room label and every printed dimension
    # was then matched against the wrong part of the plan — silently, because
    # the numbers were still inside 0..1 and nothing downstream could tell.
    #
    # On an unrotated page rotation_matrix is the identity, so this is simply
    # correct in both cases rather than a special case for rotated files.
    rot = page.rotation_matrix
    pw, ph = page.rect.width, page.rect.height
    if page.rotation:
        print("[Step 0.5] Page rotation %d - mapping text through the rotation matrix"
              % page.rotation)

    regions = []
    for w in words:
        t = (w[4] or "").strip()
        if not t:
            continue
        r = pymupdf.Rect(w[0], w[1], w[2], w[3]) * rot
        x0, y0, x1, y1 = min(r.x0, r.x1), min(r.y0, r.y1), max(r.x0, r.x1), max(r.y0, r.y1)
        regions.append({
            "text":       t,
            "confidence": 1.0,          # printed text, not a recognition guess
            "bbox":       [[x0 / pw, y0 / ph], [x1 / pw, y0 / ph],
                           [x1 / pw, y1 / ph], [x0 / pw, y1 / ph]],
            "centroid":   {"x": round((x0 + x1) / 2 / pw, 4),
                           "y": round((y0 + y1) / 2 / ph, 4)},
            "source":     "pdf_text",
        })

    out_of_bounds = sum(
        1 for r in regions
        if not (0.0 <= r["centroid"]["x"] <= 1.0 and 0.0 <= r["centroid"]["y"] <= 1.0)
    )
    if out_of_bounds:
        # Every renderer assumes 0..1. Rather than ship coordinates that will be
        # drawn off-canvas, fall back to OCR, which cannot be wrong in this way.
        print("[Step 0.5] %d/%d text regions fell outside the page after rotation "
              "- discarding the text layer and letting OCR run"
              % (out_of_bounds, len(regions)))
        return png, None
    print("[Step 0.5] Vector PDF: %d exact text regions, OCR skipped" % len(regions))
    return png, regions


# --- Dimension pairs ---------------------------------------------------------
#
# Indian plans print a room's size as a single token: KITCHEN AREA 8'1"X9'11".
# _step7_reconcile's ocr_to_mm reads one value at a time, so a printed pair was
# previously half-read or ignored. Parsing the pair directly yields length AND
# width from one string - the most trustworthy dimension source there is,
# because it is what the architect wrote rather than anything inferred from
# pixels.

# \u26a0\ufe0f THE QUOTE MARKS ARE WHATEVER OCR DECIDED THEY WERE, so both positions
# accept any of them. This regex used to allow only a straight `"` for inches,
# and required a straight `'` for feet \u2014 which cost a real room its dimensions:
# a KITCHEN printed 9'0"x10'6" was READ correctly as `9'0'x10'6"` (OCR called the
# inches mark an apostrophe), the pattern then met `'` where it wanted `x`, and
# the whole pair was discarded. That room shipped with null length AND null
# width while its own dimensions sat in the text beside it.
# Feet and inches marks are deliberately the same permissive class: telling a
# misread apostrophe from a misread double-quote is not possible here, and the
# `x` between the two halves is what actually anchors the match.
_QUOTE = "[\"'\u2019\u2018`\u00b4\u201d\u201c\u2033\u2032]"
_DIM_PAIR_FT = re.compile(
    "(\\d+)\\s*" + _QUOTE + "\\s*(\\d+)?\\s*" + _QUOTE + "?"
    "\\s*[xX\u00d7]\\s*"
    "(\\d+)\\s*" + _QUOTE + "\\s*(\\d+)?\\s*" + _QUOTE + "?"
)
_DIM_PAIR_MM = re.compile("(\\d{3,5})\\s*[xX\u00d7]\\s*(\\d{3,5})")


def _is_dimension_text(text: str) -> bool:
    """
    True when a piece of OCR text is a measurement rather than a room's name.

    ⚠️ THIS IS WHY ROOMS WERE NAMED "14'2x11'0"". The label picker used to reject
    a candidate only if stripping `. , - ' "` left it all digits — and the `x` in
    a dimension pair survives that strip, so `14'2x11'0"` came through as a
    perfectly good room name. It then WON, because the picker preferred the
    longest string and a printed dimension is longer than "BEDROOM". Measured on
    a live plan 2026-09-01: six of nine rooms were named after their own
    dimensions.

    Two tests, because neither alone is enough:
      • it parses as a dimension pair (the authoritative case), or
      • stripping digits, quote marks, separators and the unit words leaves
        nothing behind — which catches the malformed pairs OCR actually
        produces, like `11'6x110` and `9'0'x10'6"`, that no pair regex matches.
    """
    s = (text or "").strip()
    if not s:
        return True
    if _parse_dimension_pair(s):
        return True
    # Remove everything a measurement is made of; a room NAME leaves letters.
    core = re.sub(r"[0-9\s'\"×xX.,\-–/]", "", s).upper()
    if not core:
        return True
    # What survives on a dimension annotation is its unit word, not a name.
    return core in {"WIDE", "MM", "M", "CM", "FT", "FEET", "IN", "INCH",
                    "SQFT", "SQ", "SQM", "SFT", "DIA", "THK", "HT", "H", "W"}


def _parse_dimension_pair(text: str):
    """Return (a_mm, b_mm) from a printed pair, or None."""
    s = text or ""
    m = _DIM_PAIR_FT.search(s)
    if m:
        ft1, in1, ft2, in2 = m.groups()
        a = int(ft1) * 304.8 + (int(in1) if in1 else 0) * 25.4
        b = int(ft2) * 304.8 + (int(in2) if in2 else 0) * 25.4
        if 300 <= a <= 30000 and 300 <= b <= 30000:
            return (a, b)
    m = _DIM_PAIR_MM.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if 300 <= a <= 30000 and 300 <= b <= 30000:
            return (a, b)
    return None


# ─── Step 0: Pre-flight Quality Check ──────────────────────────────────────────

def _step0_preflight_check(image_bytes: bytes) -> dict:
    """
    Reject obviously bad files BEFORE expensive API calls.
    Returns: {proceed: bool, reason: str}
    """
    from PIL import Image
    import io

    # Check file size (reject <10KB or >50MB)
    if len(image_bytes) < 10_000:
        return {'proceed': False, 'reason': 'File too small — must be at least 10 KB.'}
    if len(image_bytes) > 50_000_000:
        return {'proceed': False, 'reason': 'File too large — must be under 50 MB.'}

    # Detect PDF and check page count
    if image_bytes[:4] in (b'%PDF', b'%pdf'):
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(image_bytes))
            page_count = len(pdf_reader.pages)
            if page_count > 5:
                return {
                    'proceed': False,
                    'reason': f'PDF has {page_count} pages. Please upload a single-page floor plan or an image instead.'
                }
        except Exception as e:
            # If PDF parsing fails, try to proceed anyway (might be a corrupted PDF header)
            print(f"[Step 0] PDF parsing warning: {e}")

    # Load image and check dimensions
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        w, h = pil_img.size
    except Exception as e:
        return {'proceed': False, 'reason': f'Could not read file as image: {e}'}

    # Check minimum dimensions
    if w < 500 or h < 400:
        return {'proceed': False, 'reason': f'Image too small ({w}×{h}). Minimum is 500×400 pixels.'}

    # Check maximum dimensions
    if w > 20_000 or h > 20_000:
        return {'proceed': False, 'reason': f'Image too large ({w}×{h}). Maximum is 20,000×20,000 pixels.'}

    # Check aspect ratio (floor plans are roughly square, reject very wide/tall)
    aspect = max(w, h) / min(w, h)
    if aspect > 5.0:
        return {'proceed': False, 'reason': f'Image aspect ratio too extreme ({aspect:.1f}:1). Floor plans should be roughly square.'}

    return {'proceed': True, 'reason': None}


# ─── Step 1: Quality Gate ─────────────────────────────────────────────────────

def _step1_quality_gate(image_bytes: bytes) -> dict:
    """
    Classify image type using Claude Sonnet 4.6 (fast + accurate for this gate).
    Returns classification + reason + confidence.
    """
    client = _get_claude()
    b64, mime = _image_for_claude(image_bytes)

    prompt = """You are a quality gate for an architectural floor plan processing system.

Classify this image into exactly one of these categories:
- architectural_drawing: A top-down floor plan showing rooms, walls, dimensions, and/or a title block. Includes CAD drawings, scanned blueprints, PDFs converted to images.
- photo_of_drawing: A photograph of a physical floor plan drawing (may be skewed, distorted, or low contrast).
- low_quality: A floor plan image but too blurry, too low resolution, or too incomplete to extract meaningful data from.
- not_a_floor_plan: Not a floor plan at all (photo of a room, 3D render, elevation, section, site plan, etc.).

Respond with JSON only, no markdown, no preamble:
{"classification": "<one of the four codes>", "reason": "<one sentence explanation>", "confidence": <0.0 to 1.0>}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }],
    )

    return _parse_json_response(response.content[0].text, {
        "classification": "architectural_drawing",
        "reason": "Could not classify",
        "confidence": 0.5,
    })


# ─── Step 2: Preprocessing ────────────────────────────────────────────────────

def _step2_preprocess(image_bytes: bytes, is_photo: bool = False) -> bytes:
    """
    Normalise image for optimal Claude + OCR accuracy:
    - Photos: deskew via Hough transform, perspective correction
    - All: CLAHE contrast enhancement (especially for faded scans)
    - Normalise longest side to 2000px (approx 300 DPI for A3)
    Returns JPEG bytes.
    """
    import cv2
    import numpy as np
    from PIL import Image

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(pil)[:, :, ::-1]  # RGB → BGR

    if is_photo:
        img = _deskew(img)

    # CLAHE contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Resize: longest side = 2000px
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bytes(buf)


def _deskew(img):
    """Straighten a photo of a drawing using Hough line detection."""
    import cv2
    import numpy as np

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold=100)
    if lines is None:
        return img

    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = math.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = sorted(angles)[len(angles) // 2]
    if abs(median_angle) < 0.5:
        return img

    h, w    = img.shape[:2]
    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


# ─── Step 3: OCR Extraction ───────────────────────────────────────────────────

def _step3_ocr(image_bytes: bytes) -> list[dict]:
    """
    Extract all text regions using PaddleOCR.
    Returns list of: {text, confidence, bbox, centroid}
    All coordinates are fractional (0.0–1.0) relative to image dimensions.
    """
    import cv2
    import numpy as np

    ocr   = _get_ocr()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w  = img.shape[:2]

    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return []

    regions = []
    for line in result[0]:
        bbox_px, (text, conf) = line
        norm_bbox = [[pt[0] / w, pt[1] / h] for pt in bbox_px]
        cx = sum(pt[0] for pt in norm_bbox) / 4
        cy = sum(pt[1] for pt in norm_bbox) / 4
        regions.append({
            "text":       text.strip(),
            "confidence": round(float(conf), 3),
            "bbox":       norm_bbox,
            "centroid":   {"x": round(cx, 4), "y": round(cy, 4)},
        })
    return regions


# ─── Step 4: Scale & Title Block Detection ────────────────────────────────────

@app.function(image=cpu_image, secrets=[fpis_secrets], timeout=180, memory=512)
def _step4_scale_detect(image_bytes: bytes, ocr_results: list[dict]) -> dict:
    """
    Parse the title block (bottom-right 25%×20% crop) for scale, unit system,
    north direction. Also scans OCR results for explicit scale text.

    ⚠️ WHAT THIS STEP IS ACTUALLY FOR: `unit_system`. That value picks the
    parsing branch in Step 7's `ocr_to_mm`, and getting it wrong turns every
    dimension on the plan into a different number.

    `drawing_scale` is NOT load-bearing. Nothing in this pipeline multiplies a
    pixel measurement by a 1:N ratio - every real dimension comes from text
    printed on the drawing, and wall lengths are scaled from the room's own
    resolved size. It is passed through to Ekatan as metadata and shown to a
    human. That is precisely why the prompt no longer tells the model to
    "assume 1:100 with confidence 0.4" when it finds nothing: a fabricated
    ratio carrying a plausible-looking confidence is worse than a null, because
    a reader downstream cannot tell it was invented.
    """
    import base64
    client     = _get_claude()
    title_crop = _crop_title_block(image_bytes)
    b64_crop   = base64.b64encode(title_crop).decode()
    b64_full   = base64.b64encode(image_bytes).decode()
    mime       = _detect_mime(image_bytes)

    scale_hints = [r["text"] for r in ocr_results if re.search(r"1\s*:\s*\d+|scale|नक्शा", r["text"], re.I)]

    prompt = f"""You are analysing a floor plan drawing to extract its scale and orientation metadata.

OCR scale hints found in the image: {json.dumps(scale_hints[:10])}

Examine the title block crop (first image) and the full floor plan (second image).
Extract:
1. drawing_scale — the drawing scale ratio as a string, e.g. "1:100", "1:50", "NTS" if not to scale
2. unit_system — the measurement unit: "mm" (millimetres), "feet_inches" (feet and inches), or "m" (metres)
3. north_direction_deg — integer 0–359 where 0 = north is up, 90 = north is right. Use the north arrow if present.
4. confidence — your confidence in the scale extraction, 0.0 to 1.0

If NO scale is printed on the drawing, return drawing_scale: null with confidence 0.0.
Do not guess a common scale — a missing scale must read as missing.
If dimensions appear to be in feet-inches format (e.g. 12'-6"), set unit_system to "feet_inches".

Respond with JSON only, no markdown:
{{"drawing_scale": "1:100", "unit_system": "mm", "north_direction_deg": 0, "confidence": 0.9}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_crop},
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64_full},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }],
    )

    parsed = _parse_json_response(response.content[0].text, {})
    if not isinstance(parsed, dict):
        parsed = {}

    # ⚠️ A DEFAULT DOES NOT CATCH AN EXPLICIT NULL. `d.get(k, default)` returns
    # the default only when the key is ABSENT - a key present with the value
    # None returns None, and the caller ships that. Normalising here rather
    # than at each call site, because the value travels a long way: unit_system
    # picks the parsing branch in Step 7's ocr_to_mm, and getting it wrong turns
    # every dimension on the plan into a different number.
    #
    # `drawing_scale` is the exception and stays nullable on purpose - null is
    # the honest answer for a drawing that prints no scale, and Ekatan's
    # callback accepts it as data rather than as a malformed payload.
    unit = parsed.get("unit_system")
    if unit not in ("mm", "feet_inches", "m"):
        unit = "mm"

    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    north = parsed.get("north_direction_deg")
    if not isinstance(north, (int, float)) or not (0 <= north <= 360):
        north = 0

    scale = parsed.get("drawing_scale")
    if scale is not None and not isinstance(scale, str):
        scale = None

    return {
        "drawing_scale":       scale[:50] if isinstance(scale, str) else None,
        "unit_system":         unit,
        "north_direction_deg": int(north),
        "confidence":          confidence,
    }


def _crop_title_block(image_bytes: bytes) -> bytes:
    """Crop bottom 20%, right 25% of the image for title block parsing."""
    import cv2
    import numpy as np

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w  = img.shape[:2]
    crop  = img[int(h * 0.80):h, int(w * 0.75):w]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf)


# ─── YOLO room model — mapping and gates ──────────────────────────────────────
#
# `floor_plan_indian_best.pt` is a YOLO26n-seg checkpoint trained on 249
# annotated Indian floor plans (mAP50 0.868). Measured against real plans on
# 2026-08-28:
#
#   ordinary builder plans   17 / 11 / 9 room instances, conf 0.82-0.94,
#                            polygons tight to the rooms
#   interior presentation    0 usable rooms - every detection disagreed with
#   drawings                 the printed label under it
#
# So the model is USED WHERE IT WORKS and stood down where it does not. The
# gates below are what tell the two cases apart; without them this is a
# regression, because a spiky polygon spanning two rooms is worse for a
# designer than an honest rectangle.

# YOLO class name -> Supabase room_types.code.
# ⚠️ AUTHORED HERE, deliberately. Neither existing table fits: `v5 plan/
# class_map.py` targets an 11-value canonical set with no master_bedroom /
# bathroom / pooja_room / staircase / servant_room / home_office /
# foyer_entrance / terrace split, and VALID_ROOM_CODES needs exactly those.
YOLO_ROOM_TO_CODE = {
    "0-Balcony":         "balcony",
    "2-Bedroom":         "bedroom",
    "12-attachBedroom":  "bedroom",
    "3.Dining Room":     "dining_area",
    "5-Foyer":           "foyer_entrance",
    "6-Kitchen":         "kitchen",
    "7-Living Room":     "living_room",
    "9-Terrace":         "terrace",
    "10-Terrace Lounge": "terrace",
    "19-garage":         "other",          # room_types has no garage code
    "22-lobby":          "passage",
    "25-study":          "home_office",
    "26-toilet":         "bathroom",
    "28-utility":        "utility",
    "29-walkin":         "store",
}

YOLO_DOOR_CLASS      = "15-door"
YOLO_WEIGHTS_PATH    = "/models/floor_plan_indian_best.pt"
YOLO_MIN_CONF        = 0.30
YOLO_MIN_AREA_FRAC   = 0.004
YOLO_MAX_AREA_FRAC   = 0.45
YOLO_MIN_ROOMS       = 3      # below this we do not trust the run at all
RECTIFY_FILL_RATIO   = 0.86

# ⚠️ HARD CONTRACT, NOT A STYLE CHOICE. Ekatan's callback validates each room
# with FpisWallSchema as `nullish array, max 10` (src/app/api/fpis/callback/
# route.ts). _r2s_walls_from_polygon emits one wall per polygon EDGE, so a
# polygon with more than 10 edges makes the whole payload fail Zod with
# 400 "Array must contain at most 10 element(s)" - the entire extraction is
# thrown away over geometry nobody asked for. Caught on a live run against a
# real builder plan on 2026-08-28: a 16-vertex living room killed a run that
# had otherwise produced 6 good rooms and 9 doors.
MAX_ROOM_POLYGON_VERTS = 10

# An edge shorter than this fraction of the room's bounding diagonal is raster
# stair-stepping, not a wall, and is exempt from the rectilinear test below.
RECTILINEAR_MIN_EDGE_FRAC = 0.06
# sin(8.6°). A substantial edge further off-axis than this means the outline is a
# mask artefact rather than a room, and the whole polygon falls back to its box.
# Deliberately TIGHTER than `_rectify_contour`'s own snap (slope ratio 0.18, or
# 10.2°), so the rule composes into something simple: if rectification could not
# square an edge, this boxes the room. A looser value here leaves a band of
# angles that neither pass flattens, which is exactly where the kitchen with one
# diagonal wall lived.
RECTILINEAR_SIN_TOLERANCE = 0.15

# Two detections of the SAME room type overlapping by at least this much are one
# room found twice, not two rooms. Set well above the incidental overlap of two
# genuinely adjacent rooms of one type (two bedrooms sharing a wall overlap at
# ~0), and below the near-identical duplicates the model actually emits.
DUPLICATE_ROOM_IOU = 0.60

# ─── Furniture absorption (see _absorb_furniture) ─────────────────────────────
# Classes that are FURNITURE STANDING ON A ROOM'S FLOOR. The room mask stops at
# each of these instead of at the wall, so unioning them back repairs the
# outline. The model already produces them and Step 6 used to discard all of it.
FURNITURE_ABSORB_CLASSES = {
    "1-Bed", "4-Dining table", "8-Sofa", "11-Wardrobe", "14-commode",
    "16-dress", "18-fridge", "20-kitchen-slab", "23-sink", "24-stove",
    "27-tv", "30-wash", "31-washing-machine",
}
# ⚠️ `17-duct` AND `21-lift` ARE DELIBERATELY ABSENT. A shaft or a lift core is a
# genuine HOLE in the plan, not occluded floor — it is not part of any room and
# absorbing it would grow a room into a void nobody can use or sell. "Everything
# that is not a room or a door" would be the obvious rule and the wrong one.
FURNITURE_MIN_CONF = 0.25
# Fraction of a furniture detection that must fall inside the room's bounding
# box. Measured against the BOX, not the room: the furniture is precisely the
# region the room mask excluded, so it barely overlaps the room itself.
FURNITURE_CONTAINMENT = 0.75
# How close furniture must sit to the room's own mask, as a fraction of the
# room's bounding diagonal. This is the guard that stops a room absorbing a
# wardrobe standing in the room NEXT DOOR and growing through its own wall.
FURNITURE_ADJACENCY_FRAC = 0.03
# A repair that grows a room by more than this is not a repair.
FURNITURE_MAX_GROWTH = 1.35

# Rooms handed to Step 7b's dimension re-read. A bound, not a budget: a plan
# with fifty unmeasured rooms has a problem this step cannot fix.
VERIFY_DIMENSIONS_MAX_ROOMS = 20

# Words that mean "this polygon is over a room", used to reject detections
# that landed on a title block or a legend.
# ⚠️ BUILDERS RENAME ROOMS FOR MARKETING, and this list is what decides whether
# a detected polygon is over a room at all. Measured on a real Godrej Royale
# Woods plan (2026-08-28): the model found 9 rooms correctly and **all 9 were
# thrown away**, because the drawing says SLEEP, LOUNGE, COOK/DINE, SPLASH,
# CLEANSE and SITOUT rather than bedroom, living, kitchen, bathroom, utility and
# balcony. Nothing logged a problem — the plan simply fell through to the Claude
# fallback and looked like the model had failed.
#
# Add a builder's vocabulary here the first time one of their plans is seen.
# The cost of a missing word is silent and total; the cost of an extra one is
# almost nil, because the agreement table below still has to match the class.
ROOM_LABEL_WORDS = (
    "BEDROOM", "BED ROOM", "KITCHEN", "LIVING", "HALL", "DINING", "BALCONY",
    "UTILITY", "TOILET", "BATH", "W/C", "WC", "FOYER", "ENTRY", "ENTRANCE",
    "LOBBY", "STUDY", "OFFICE", "POOJA", "PUJA", "STORE", "PASSAGE",
    "CORRIDOR", "TERRACE", "DECK", "MASTER", "GUEST", "PARENT", "SERVANT",
    "MAID", "WALKIN", "WALK-IN", "DRESS", "M.BED", "M. BED",
    # Godrej Royale Woods, observed 2026-08-28
    "SLEEP", "LOUNGE", "COOK", "DINE", "SPLASH", "CLEANSE", "SITOUT", "SIT OUT",
    # other Indian-plan variants seen across the 18-brochure harvest
    "CHILDREN BED", "COM.TOILET", "COMMON TOILET", "WASH", "SERVICE",
    "PANTRY", "POWDER", "VERANDAH", "VERANDA", "DRAWING", "FAMILY",
    # Words that reach the five refinement-only codes (see VALID_ROOM_CODES).
    # Without them the polygon is thrown out one gate EARLIER than the
    # refinement, so adding the refinement alone would have changed nothing.
    "KIDS", "KID'S", "CHILDREN", "CHILD", "THEATRE", "THEATER",
    "CINEMA", "MEDIA", "M.TOILET", "M. TOILET",
    # Walk-in wardrobe. Plans abbreviate it every possible way, and the one seen
    # on a real Bengaluru plan (2026-09-01) was "WWR 8'6"x5'0"" — a well
    # dimensioned, sellable space that this pipeline could not name, so its
    # detection was thrown out one gate before it could be typed.
    "WWR", "WIW", "W.I.W", "WALK IN WARDROBE", "WALK-IN WARDROBE", "WARDROBE",
)

# The printed label must AGREE with the predicted class. This is the gate that
# matters: on the presentation drawing every survivor of the earlier gates was
# still wrong - a "living_room" polygon sitting over PARENT'S BEDROOM, a
# "bathroom" over UTILITY. Confidence could not see it (0.83, 0.95) and neither
# could area. Only the drawing's own words could.
# This is the gate that matters: it catches a polygon in the WRONG PLACE — a
# "living_room" sitting over PARENT'S BEDROOM — which confidence and area cannot
# see. It is NOT meant to arbitrate fine distinctions between adjacent classes.
# That is why TERRACE and SITOUT are accepted for `balcony` below: calling
# a private terrace a balcony is a small labelling imprecision, while REJECTING
# it loses the room entirely, and losing rooms is the failure this whole gate
# stack is trying to avoid.
YOLO_LABEL_AGREEMENT = {
    "bedroom":         ("BEDROOM", "BED ROOM", "M.BED", "M. BED", "GUEST", "PARENT",
                        "MASTER", "SLEEP", "CHILDREN BED"),
    "master_bedroom":  ("MASTER", "M.BED", "M. BED"),
    "servant_room":    ("SERVANT", "MAID"),
    # These five exist because the gate below runs AFTER _refine_room_code and
    # reads `.get(code, ())`. A refined code missing from this table matches the
    # empty tuple and the room is DROPPED - so a promotion without an entry here
    # would LOSE rooms rather than type them better.
    # They are permissive on purpose: the refinement only fires when the
    # distinguishing word is already present, so the marker is always found, and
    # the base words are here so a near-miss keeps the room.
    "kids_room":       ("KIDS", "KID'S", "CHILDREN", "CHILD",
                        "BEDROOM", "BED ROOM"),
    "guest_room":      ("GUEST", "BEDROOM", "BED ROOM"),
    "home_theatre":    ("THEATRE", "THEATER", "CINEMA", "MEDIA"),
    "master_bathroom": ("MASTER", "M.TOILET", "M. TOILET", "MASTER TOILET",
                        "TOILET", "BATH"),
    "powder_room":     ("POWDER", "TOILET", "BATH", "WC", "W/C"),
    "bathroom":        ("TOILET", "BATH", "W/C", "WC", "SPLASH", "POWDER",
                        "COM.TOILET", "COMMON TOILET"),
    "kitchen":         ("KITCHEN", "COOK", "PANTRY"),
    "living_room":     ("LIVING", "HALL", "DRAWING", "LOUNGE", "FAMILY"),
    "dining_area":     ("DINING", "DINE"),
    # Split deliberately — see the note on VALID_ROOM_CODES. A drawing that
    # says UTILITY must not satisfy a balcony prediction, or the distinction
    # this split exists to protect is undone one gate later.
    "balcony":         ("BALCONY", "SITOUT", "SIT OUT", "VERANDAH", "VERANDA",
                        "TERRACE", "DECK"),
    "utility":         ("UTILITY", "WASH", "CLEANSE", "SERVICE", "LAUNDRY"),
    "terrace":         ("TERRACE", "DECK", "SITOUT", "SIT OUT"),
    "passage":         ("LOBBY", "PASSAGE", "CORRIDOR"),
    "foyer_entrance":  ("FOYER", "ENTRY", "ENTRANCE"),
    "home_office":     ("STUDY", "OFFICE"),
    # WWR / WIW sit here TEMPORARILY. The model's `29-walkin` class maps to
    # `store` today, so this is what keeps a walk-in wardrobe from being dropped
    # at the agreement gate. They move to a `walk_in_wardrobe` code as soon as
    # that row exists in Ekatan's room_types — emitting the code before the row
    # exists would fail GUARD #1 and discard the WHOLE extraction, not just the
    # one room.
    "store":           ("STORE", "WALKIN", "WALK-IN", "WWR", "WIW", "WARDROBE"),
    "pooja_room":      ("POOJA", "PUJA", "MANDIR"),
}


def _point_in_fractional_polygon(polygon_points: list[dict], x: float, y: float) -> bool:
    """Ray cast against a list of {x, y} fractional points."""
    inside = False
    n = len(polygon_points)
    j = n - 1
    for i in range(n):
        xi, yi = polygon_points[i]["x"], polygon_points[i]["y"]
        xj, yj = polygon_points[j]["x"], polygon_points[j]["y"]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
        j = i
    return inside


def _texts_inside_polygon(polygon_points: list[dict], ocr_results: list[dict]) -> list[str]:
    out = []
    for r in ocr_results or []:
        c = r.get("centroid") or {}
        if "x" in c and "y" in c and _point_in_fractional_polygon(polygon_points, c["x"], c["y"]):
            t = (r.get("text") or "").strip()
            if t:
                out.append(t)
    return out


def _rectify_contour(contour, cv2, np):
    """
    Turn a segmentation contour into architectural geometry.

    A mask contour wobbles along every wall and throws diagonal spurs across
    the plan wherever the mask leaked through a doorway. Two steps:
      1. Simplify hard (Douglas-Peucker at 1.5% of perimeter) - removes spurs.
      2. If the result fills its own axis-aligned bounding box well enough,
         REPLACE it with that box, so a rectangular room gets genuinely
         straight walls rather than an approximation of them.
    Anything left keeps its simplified outline with near-axis edges snapped
    square.
    """
    peri = cv2.arcLength(contour, True)
    simp = cv2.approxPolyDP(contour, 0.015 * peri, True)
    if len(simp) < 3:
        return None

    # Simplify harder until the room fits inside the wall-count contract.
    # Doubling epsilon each pass converges in a handful of steps; the bound is
    # there so a pathological contour cannot spin.
    eps = 0.015
    for _ in range(8):
        if len(simp) <= MAX_ROOM_POLYGON_VERTS:
            break
        eps *= 1.7
        harder = cv2.approxPolyDP(contour, eps * peri, True)
        if len(harder) < 3:
            break
        simp = harder
    area = abs(cv2.contourArea(simp))
    if area <= 0:
        return None
    x, y, w, h = cv2.boundingRect(simp)
    if w <= 0 or h <= 0:
        return None
    if area / float(w * h) >= RECTIFY_FILL_RATIO:
        return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
    pts = [p[0].astype(float) for p in simp]
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        if abs(dy) <= 0.18 * abs(dx):
            a[1] = b[1] = (a[1] + b[1]) / 2.0
        elif abs(dx) <= 0.18 * abs(dy):
            a[0] = b[0] = (a[0] + b[0]) / 2.0
    return np.array([[[int(round(p[0])), int(round(p[1]))]] for p in pts], dtype=np.int32)


def _refine_room_code(code: str, texts: list[str]) -> str:
    """Seven room types the model has no class for; the drawing says so in words.

    A kids room is shaped exactly like a bedroom and a powder room exactly like a
    bathroom. Nothing about the OUTLINE separates them, so training a class for
    each would split the examples across shapes the model cannot tell apart and
    still get them wrong. The printed name is the only real signal, and it is
    free - so the shape is detected and the words do the sorting.

    Every code returned here MUST be a key in YOLO_LABEL_AGREEMENT. The caller
    gates on `.get(code, ())` immediately after this returns, so a promotion to a
    code absent there matches the empty tuple and the room is DROPPED - strictly
    worse than never promoting. `test_refine_room_code()` asserts that closure.

    Order is most-specific-first within each base type. MASTER is tested before
    the additions so existing behaviour is unchanged.
    """
    joined = " ".join(texts).upper()

    if code == "bedroom":
        if "MASTER" in joined or "M.BED" in joined or "M. BED" in joined:
            return "master_bedroom"
        if "SERVANT" in joined or "MAID" in joined:
            return "servant_room"
        if ("KIDS" in joined or "KID'S" in joined
                or "CHILDREN" in joined or "CHILD" in joined):
            return "kids_room"
        if "GUEST" in joined:
            return "guest_room"
        return "bedroom"

    if code == "bathroom":
        # MASTER is tested first: an "M.TOILET" beside the master bedroom is a
        # master bathroom, and a powder room is never also a master bath.
        if ("M.TOILET" in joined or "M. TOILET" in joined
                or "MASTER TOILET" in joined or "MASTER BATH" in joined
                or "M.BATH" in joined):
            return "master_bathroom"
        if "POWDER" in joined:
            return "powder_room"
        return "bathroom"

    if code == "living_room":
        # A home theatre is a room the model reads as a living room; only the
        # printed name says otherwise. Rare in apartment brochures, common in
        # villa plans, and worth 2 published systems that have never sold.
        if ("THEATRE" in joined or "THEATER" in joined
                or "CINEMA" in joined or "MEDIA ROOM" in joined):
            return "home_theatre"
        return "living_room"

    return code


def _rescue_code_from_label(joined: str) -> str | None:
    """
    When the model's class disagrees with the drawing, believe the drawing.

    ⚠️ THIS RECOVERS ROOMS THE LABEL GATE USED TO DELETE. Measured on a live
    plan 2026-09-01, twice in one run:

        rejected 26-toilet->bathroom: label disagrees ('UTILITY 5\\'9"WIDE')

    The model looked at the utility — which has a sink and a washing machine in
    it — and called it a toilet. The gate correctly refused to store a bathroom
    over a space the drawing calls UTILITY, and then dropped the detection
    entirely. So the gate was right and the outcome was still wrong: the plan
    came back with no utility at all, and a designer had to add it by hand.

    The drawing had already answered the question. `_refine_room_code` acts on
    exactly this principle for bedrooms and bathrooms; this extends it to the
    case where the printed name contradicts the class outright rather than
    refining it.

    ONLY CODES THE MODEL CAN ITSELF PREDICT are candidates. Without that filter
    the word TOILET matches bathroom, master_bathroom AND powder_room, three
    codes become ambiguous, and nothing is ever rescued. The refinement-only
    codes are reached afterwards, from the same text, by the existing path.

    Returns a code only when EXACTLY ONE matches. A polygon carrying two room
    names is a polygon spanning two rooms, and guessing which one it is would
    reintroduce the mislabelling this gate exists to stop.
    """
    predictable = set(YOLO_ROOM_TO_CODE.values()) - {"other"}
    hits = {
        code for code in predictable
        if any(w in joined for w in YOLO_LABEL_AGREEMENT.get(code, ()))
    }
    return hits.pop() if len(hits) == 1 else None


def _fractional_polygon_area(polygon_points: list[dict]) -> float:
    """Shoelace area of a fractional polygon. Used to pick the SMALLEST room
    containing a door, so a door inside a bedroom that also falls inside a
    larger overlapping detection is credited to the bedroom."""
    n = len(polygon_points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        a = polygon_points[i]
        b = polygon_points[(i + 1) % n]
        total += a["x"] * b["y"] - b["x"] * a["y"]
    return abs(total) / 2.0


def _boxify(polygon_points: list[dict]) -> list[dict]:
    """The axis-aligned bounding box of a polygon, wound clockwise from top-left."""
    xs = [p["x"] for p in polygon_points]
    ys = [p["y"] for p in polygon_points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [{"x": x0, "y": y0}, {"x": x1, "y": y0},
            {"x": x1, "y": y1}, {"x": x0, "y": y1}]


def _sanitize_room_polygon(polygon_points: list[dict]) -> list[dict]:
    """
    A room outline that crosses itself, or wanders off-axis, is not a room.
    Replace it with its box.

    ⚠️ THIS IS THE SINGLE BIGGEST QUALITY PROBLEM IN THE SEGMENTATION PATH, and
    it was invisible until the geometry started being rendered faithfully.
    Measured on a live Bengaluru plan, 2026-09-01 — the master bedroom shipped as

        (0.625,0.818) (0.625,0.985) (0.465,0.985) (0.465,0.747)
        (0.855,0.747) (0.855,0.985)

    whose closing edge cuts diagonally back through the middle of the shape. A
    bowtie. Another room visited one vertex TWICE in the same ring. Of nine
    rooms on that plan, most were spirals like this.

    `_rectify_contour`'s bounding-box snap is supposed to catch the hopeless
    cases, and it structurally cannot catch THESE: it compares `cv2.contourArea`
    against the box, and contourArea on a self-intersecting ring returns the
    ALGEBRAIC area, where the clockwise and counter-clockwise lobes cancel. A
    spiral therefore reports a near-zero area, the fill ratio comes out tiny,
    and the snap never fires — so the worst polygons are precisely the ones that
    keep their shape.

    Shapely is the right judge (`is_valid` is false for a self-intersecting
    ring) and it is already in the GPU image. An honest rectangle in the right
    place beats an accurate-looking spiral: the box is what a designer would
    have drawn anyway, and every consumer — the review overlay, the 3D view,
    `findWallEdge` — behaves sanely on four edges.

    ── AND VALIDITY ALONE IS NOT ENOUGH ────────────────────────────────────────
    The same plan produced a DINING room that was a perfectly valid quadrilateral
    and still nonsense: a long diagonal chord sliced across it, because
    Douglas-Peucker cut a corner off a mask that had leaked through a doorway.
    Shapely is happy with that shape. A designer would not be.

    Indian residential plans are overwhelmingly RECTILINEAR — walls meet at right
    angles. A significant edge running at 40° is therefore evidence of a mask
    artefact, not of an unusual room. So a non-rectangular outline is kept only
    when every substantial edge is near an axis; anything else becomes its box.

    This deliberately keeps genuine L-shapes, which are rectilinear, and discards
    diagonal-chord artefacts, which are not. Short edges are exempt: a two-pixel
    stair-step on an otherwise square corner is noise, not a diagonal wall.
    """
    if len(polygon_points) < 4:
        return polygon_points

    try:
        from shapely.geometry import Polygon
    except ImportError:
        return polygon_points

    try:
        poly = Polygon([(p["x"], p["y"]) for p in polygon_points])
        if not poly.is_valid or poly.area <= 0:
            return _boxify(polygon_points)
    except Exception:
        return _boxify(polygon_points)

    # ⚠️ NO EARLY RETURN FOR FOUR POINTS. There used to be one — "four points are
    # already a box" — and it was simply false: `_rectify_contour` readily
    # produces a four-point TRAPEZOID with one diagonal side, which then skipped
    # this check entirely. That is why a designer still saw a kitchen with one
    # diagonal wall after the rest of the plan had been squared up. A genuine
    # axis-aligned rectangle passes the test below anyway, so the shortcut
    # bought nothing and hid the one shape it should have caught.
    xs = [p["x"] for p in polygon_points]
    ys = [p["y"] for p in polygon_points]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diag <= 1e-9:
        return _boxify(polygon_points)

    n = len(polygon_points)
    for i in range(n):
        a, b = polygon_points[i], polygon_points[(i + 1) % n]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        edge = math.hypot(dx, dy)
        # Ignore short edges - they are the stair-stepping of a raster mask,
        # not walls anyone drew.
        if edge < RECTILINEAR_MIN_EDGE_FRAC * diag:
            continue
        # Angle to the nearer axis, in degrees.
        off_axis = min(abs(dx), abs(dy)) / edge      # sin of that angle
        if off_axis > RECTILINEAR_SIN_TOLERANCE:
            return _boxify(polygon_points)

    return polygon_points


def _absorb_furniture(room_xy, furniture_xy, np, label: str = ""):
    """
    Fill the furniture back into a room outline, so it traces walls not wardrobes.

    ⚠️ THE MODEL SEGMENTS VISIBLE FLOOR, NOT THE ROOM. A wardrobe drawn against
    the bedroom wall occludes the floor, so the mask stops at the wardrobe and
    the traced outline takes a bite out of the room exactly where the wardrobe
    is. Same for a bed, a sofa, the kitchen slab. Those bites are what produced
    the spiky outlines a designer saw on 2026-09-01.

    The repair is free: the model already detects that furniture and Step 6 threw
    every one of those detections away. Union the room's mask with the furniture
    standing in it and the outline runs to the wall again.

    ── WHY CONTAINMENT IS TESTED AGAINST THE BOUNDING BOX ──────────────────────
    You cannot ask "does this wardrobe overlap the room?" — it barely does, and
    that is the entire problem. The wardrobe IS the region the room mask
    excluded, so the two are ADJACENT, not overlapping. Containment is therefore
    measured against the room's bounding box.

    ── AND WHY ADJACENCY IS TESTED SEPARATELY ──────────────────────────────────
    A bounding box is generous, especially for an L-shaped room, so the box test
    alone would happily swallow a wardrobe standing in the NEXT room and grow
    this room straight through its own wall. The second test is what prevents
    that: furniture bitten out of THIS room touches this room's mask, while
    furniture in the neighbouring room is separated from it by the wall.

    Three guards, and every decision is logged so the next run's Modal logs show
    exactly what this did on real plans rather than what I assumed it would do.
    """
    if len(furniture_xy) == 0 or room_xy is None or len(room_xy) < 3:
        return room_xy
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return room_xy

    def _poly(xy):
        try:
            p = Polygon([(float(a), float(b)) for a, b in xy])
            if not p.is_valid:
                p = p.buffer(0)
            return p if (not p.is_empty and p.area > 0) else None
        except Exception:
            return None

    room = _poly(room_xy)
    if room is None:
        return room_xy

    envelope = room.envelope
    minx, miny, maxx, maxy = room.bounds
    reach = math.hypot(maxx - minx, maxy - miny) * FURNITURE_ADJACENCY_FRAC

    absorbed, refused = [], []
    for f_xy in furniture_xy:
        fp = _poly(f_xy)
        if fp is None:
            continue
        inside = fp.intersection(envelope).area / fp.area
        if inside < FURNITURE_CONTAINMENT:
            refused.append(f"outside ({inside:.0%} in box)")
            continue
        if fp.distance(room) > reach:
            # In this room's box, but not touching this room's floor — almost
            # always a fitting in the room next door.
            refused.append("not adjacent")
            continue
        absorbed.append(fp)

    if not absorbed:
        if refused:
            print(f"[Step 6]   {label}: absorbed 0 furniture "
                  f"({', '.join(refused[:4])})")
        return room_xy

    try:
        merged = unary_union([room] + absorbed)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        if merged.is_empty or merged.area <= 0:
            return room_xy
    except Exception:
        return room_xy

    growth = merged.area / room.area
    if growth > FURNITURE_MAX_GROWTH:
        # A jump this large is a room eating its neighbour, not a wardrobe being
        # filled in. Refuse the whole repair rather than pick which part to keep.
        print(f"[Step 6]   {label}: REFUSED furniture repair, would grow room "
              f"{growth:.0%} (cap {FURNITURE_MAX_GROWTH:.0%})")
        return room_xy

    print(f"[Step 6]   {label}: absorbed {len(absorbed)} furniture item(s), "
          f"room area +{(growth - 1) * 100:.0f}%"
          + (f", refused {len(refused)}" if refused else ""))
    return np.array(merged.exterior.coords[:-1], dtype=np.float32)


def _dedupe_overlapping_rooms(rooms: list[dict]) -> list[dict]:
    """
    One room detected twice is one room. Keep the more confident copy.

    The model returns overlapping instances for the same space, and nothing
    downstream merges them. Measured on a live plan 2026-09-01: the KITCHEN came
    through twice, as rooms 7 and 9, with polygons differing by a few hundredths
    of a fraction. Both were stored, both drew, and both would have been priced.

    Only same-typed rooms are compared. Two DIFFERENT room types overlapping is
    ordinary and must not be touched - a bathroom's box legitimately sits inside
    a bedroom's, and the door assigner already relies on that nesting.
    """
    if len(rooms) < 2:
        return rooms
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return rooms

    def _poly(room):
        pts = room.get("polygon_points") or []
        if len(pts) < 3:
            return None
        try:
            p = Polygon([(q["x"], q["y"]) for q in pts])
            return p if p.is_valid and p.area > 0 else None
        except Exception:
            return None

    dropped: set[int] = set()
    for i in range(len(rooms)):
        if i in dropped:
            continue
        pi = _poly(rooms[i])
        if pi is None:
            continue
        for j in range(i + 1, len(rooms)):
            if j in dropped:
                continue
            if rooms[j].get("room_type_code") != rooms[i].get("room_type_code"):
                continue
            pj = _poly(rooms[j])
            if pj is None:
                continue
            union = pi.union(pj).area
            if union <= 0:
                continue
            if (pi.intersection(pj).area / union) < DUPLICATE_ROOM_IOU:
                continue
            # Same room, twice. The more confident detection wins; on a tie the
            # earlier one does, so the result does not depend on iteration order.
            loser = j if (rooms[j].get("extraction_confidence") or 0) <= \
                         (rooms[i].get("extraction_confidence") or 0) else i
            dropped.add(loser)
            print(f"[Step 6]   dropped duplicate {rooms[loser]['room_type_code']} "
                  f"(IoU >= {DUPLICATE_ROOM_IOU})")
            if loser == i:
                break

    return [r for k, r in enumerate(rooms) if k not in dropped]


def _ensure_clockwise(polygon_points: list[dict]) -> list[dict]:
    """
    Wind a fractional polygon clockwise as seen on the page, reversing if not.

    ⚠️ THIS IS A CONTRACT WITH EKATAN, NOT A TIDINESS RULE. `offset_from_left_mm`
    is measured from a wall's START vertex, and which end that is depends
    entirely on the winding direction. Ekatan builds its own walls "clockwise
    from top-left in y-down space" (src/domains/plan-graph/conversion.ts) - so
    north runs left-to-right, east top-to-bottom, south right-to-left, west
    bottom-to-top. Hand it a counter-clockwise polygon and every opening is
    mirrored to the wrong end of every wall, consistently and invisibly.

    `_rectify_contour`'s bounding-box branch already emits clockwise, but the
    irregular branch inherits the winding of the YOLO mask contour, which is not
    guaranteed - so an L-shaped room could ship mirrored while a rectangular one
    beside it was correct.

    Image coordinates run y-DOWN, so a clockwise loop has a POSITIVE signed
    shoelace area (the opposite of the y-up convention).
    """
    n = len(polygon_points)
    if n < 3:
        return polygon_points
    signed = 0.0
    for i in range(n):
        a = polygon_points[i]
        b = polygon_points[(i + 1) % n]
        signed += a["x"] * b["y"] - b["x"] * a["y"]
    return polygon_points if signed >= 0 else list(reversed(polygon_points))


def _assign_openings_to_nearest_wall(openings: list[dict], walls: list[dict]) -> None:
    """
    Put each opening on the wall whose segment it is actually closest to.

    ⚠️ THE FUNCTION THIS REPLACES WAS BROKEN. `_r2s_assign_openings_to_walls`
    computed `dist = abs(cx_frac - 0.5) + abs(cy_frac - 0.5)`, which does not
    reference `wall` at all - the value was identical for every wall, so
    `dist < best_dist` was true only on the first iteration and EVERY opening
    in a room landed on wall #1. It never fired in production because the model
    weights were missing, but it would have the moment they arrived.

    Walls carry their own segment endpoints here (_seg), stripped before the
    payload leaves this module.

    ⚠️ NOTHING HERE IS IN MILLIMETRES, AND THAT IS THE POINT. At Step 6 a wall
    is still a fraction of the image (`_frac_len`), so an offset computed now
    would be a fraction dressed up as mm. It used to be exactly that: this
    function wrote `best_t * wall_length_mm` while `wall_length_mm` was the
    `frac x 10000` placeholder, and `_rescale_walls_to_mm` later converted the
    WALL to real mm and left the OPENING behind. The two then sat on the same
    object in different units - a door at mid-wall rendered hard against the
    left corner, and Ekatan's `fpis-geometry.ts` fed that offset into the span
    maths that sizes a wardrobe on a quote.
    So we record WHERE ALONG THE WALL the opening sits (0-1) and how wide it is
    as a fraction of the image on each axis, and let `_rescale_walls_to_mm`
    turn all three into millimetres in one place, once, when the room's true
    size is known.
    """
    for op in openings:
        # ⚠️ READ, NEVER POP. This used to `op.pop("_cx")`, which mutated a dict
        # shared with the caller's candidate list. A door whose centroid falls
        # inside two overlapping room polygons is visited twice, and the second
        # visit died with KeyError: '_cx' - caught on a live run against a real
        # builder plan on 2026-08-28, and missed by the unit test because that
        # test copied the dicts before calling. Each door is now assigned to
        # exactly one room by the caller, but this stays non-destructive so the
        # function is safe to call more than once regardless.
        cx, cy = op.get("_cx"), op.get("_cy")
        if cx is None or cy is None:
            continue
        best, best_d, best_t = None, float("inf"), 0.0
        for wall in walls:
            seg = wall.get("_seg")
            if not seg:
                continue
            (x0, y0), (x1, y1) = seg
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            if L2 <= 1e-12:
                continue
            t = max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / L2))
            px, py = x0 + t * dx, y0 + t * dy
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best, best_d, best_t = wall, d, t
        if best is None:
            continue
        is_door = str(op.get("opening_type", "")).endswith("door")
        # Measure the opening along the wall it landed on, not along the image.
        # A door on a vertical wall is WIDE in y and thin in x; taking the
        # x-extent for it would report the wall's thickness as the door's width.
        axis = best.get("_axis")
        width_frac = op.get("_w_frac_y") if axis == "v" else op.get("_w_frac_x")
        best["openings"].append({
            "opening_type":          op["opening_type"],
            "opening_label":         f"{'D' if is_door else 'W'}{len(best['openings']) + 1}",
            # Resolved by _rescale_walls_to_mm, together with the wall itself.
            "rough_width_mm":        None,
            "offset_from_left_mm":   None,
            "extraction_confidence": op["extraction_confidence"],
            "_along_frac":           best_t,
            "_width_frac":           width_frac,
        })


# ─── Step 6: Room Extraction — YOLO26n-seg (Indian plans) ─────────────────────

@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=900,
    # ⚠️ WITHOUT THIS, `timeout` ALSO CAPS CONTAINER STARTUP (Modal docs: "if
    # startup_timeout is not set, timeout will still configure both times").
    # This image carries torch + the full CUDA toolkit + ultralytics; a cold
    # pull can outrun a 180s budget, which is how a run dies before it begins.
    startup_timeout=900,
    memory=8192,
    cpu=2,
    secrets=[fpis_secrets],
    volumes={"/models": model_volume},
)
def _step6_raster2seq_extract(
    image_bytes:   bytes,
    ocr_results:   list[dict],
    scale_info:    dict,
    bedroom_count: int | None,
    property_type: str,
) -> list[dict]:
    """
    Extract rooms with the fine-tuned YOLO26n-seg model, gated against the
    drawing's own printed labels. Falls back to Claude whenever the model is
    absent, errors, or fails to agree with the page.

    Returns the same schema Steps 7/8 already consume.
    """
    import os
    import numpy as np
    import cv2

    if not os.path.exists(YOLO_WEIGHTS_PATH):
        print(f"[Step 6] Model weights not found at {YOLO_WEIGHTS_PATH}")
        print("[Step 6] Falling back to Claude room extraction")
        return _step6_claude_fallback(image_bytes, ocr_results, scale_info,
                                      bedroom_count, property_type)

    try:
        from ultralytics import YOLO

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image for YOLO")
        H, W = img.shape[:2]

        model = YOLO(YOLO_WEIGHTS_PATH)
        result = model.predict(img, conf=0.20, iou=0.45, verbose=False)[0]
        names = model.names
        if result.masks is None or result.boxes is None:
            raise ValueError("model returned no masks")
        print(f"[Step 6] YOLO returned {len(result.boxes)} detections on {W}x{H}")
    except Exception as exc:
        print(f"[Step 6] YOLO inference failed: {exc} - falling back to Claude")
        return _step6_claude_fallback(image_bytes, ocr_results, scale_info,
                                      bedroom_count, property_type)

    rooms: list[dict] = []
    door_candidates: list[dict] = []
    rejected: list[str] = []

    # PASS 1 — collect the furniture. It has to happen before any room is
    # traced, because a room's outline is repaired using the furniture standing
    # in it, and detections arrive in no particular order.
    furniture_xy = [
        mask_xy
        for box, mask_xy in zip(result.boxes, result.masks.xy)
        if names[int(box.cls[0])] in FURNITURE_ABSORB_CLASSES
        and float(box.conf[0]) >= FURNITURE_MIN_CONF
    ]
    print(f"[Step 6] {len(furniture_xy)} furniture detection(s) available to "
          f"repair room outlines")

    # PASS 2 — rooms and doors.
    for box, mask_xy in zip(result.boxes, result.masks.xy):
        name = names[int(box.cls[0])]
        conf = float(box.conf[0])

        if name == YOLO_DOOR_CLASS and conf >= YOLO_MIN_CONF:
            xs, ys = mask_xy[:, 0], mask_xy[:, 1]
            door_candidates.append({
                "opening_type":          "single_door",
                "_cx":                   float(xs.mean()) / W,
                "_cy":                   float(ys.mean()) / H,
                # Both axes: the assigner picks whichever runs ALONG the wall
                # this door ends up on. Keeping only the x-extent reported a
                # vertical door's width as the wall's thickness.
                "_w_frac_x":             float(xs.max() - xs.min()) / W,
                "_w_frac_y":             float(ys.max() - ys.min()) / H,
                "extraction_confidence": round(conf, 3),
            })
            continue

        code = YOLO_ROOM_TO_CODE.get(name)
        if code is None or conf < YOLO_MIN_CONF:
            continue

        repaired = _absorb_furniture(mask_xy, furniture_xy, np, label=name)
        contour = repaired.astype(np.int32).reshape(-1, 1, 2)
        rect = _rectify_contour(contour, cv2, np)
        if rect is None:
            rejected.append(f"{name}: degenerate"); continue

        area_frac = abs(cv2.contourArea(rect)) / float(W * H)
        if not (YOLO_MIN_AREA_FRAC <= area_frac <= YOLO_MAX_AREA_FRAC):
            rejected.append(f"{name}: area {area_frac:.2%}"); continue

        polygon_points = _ensure_clockwise(_sanitize_room_polygon([
            {"x": round(float(p[0][0]) / W, 4), "y": round(float(p[0][1]) / H, 4)}
            for p in rect
        ]))
        texts = _texts_inside_polygon(polygon_points, ocr_results)
        joined = " ".join(texts).upper()

        if not any(w in joined for w in ROOM_LABEL_WORDS):
            rejected.append(f"{name}: no room label inside"); continue

        # ⚠️ A REFINEMENT MUST NEVER COST US THE ROOM. `.get(code, ())` used to
        # sit here, so a code returned by `_refine_room_code` but missing from
        # YOLO_LABEL_AGREEMENT matched the empty tuple and the room was DROPPED
        # - strictly worse than never promoting it, and silent. Four tables have
        # to move together for a taxonomy change (ROOM_LABEL_WORDS,
        # YOLO_LABEL_AGREEMENT, VALID_ROOM_CODES, WET_AREA_CODES) and only a
        # hand-written test enforces that; this is the runtime half of the
        # guard. If the promotion has no entry we say so loudly and gate on the
        # code the model actually predicted, which always has one.
        base_code = code
        code = _refine_room_code(code, texts)
        if code not in YOLO_LABEL_AGREEMENT:
            print(f"[Step 6]   ⚠️ BUG: _refine_room_code returned {code!r}, which is "
                  f"not a key in YOLO_LABEL_AGREEMENT - keeping the room and gating "
                  f"on {base_code!r} instead. Add {code!r} to that table.")
            code = base_code
        if not any(w in joined for w in YOLO_LABEL_AGREEMENT.get(code, ())):
            # The drawing disagrees with the model. Before dropping the room,
            # ask whether the drawing named something else we recognise — a
            # retyped room beats a missing one, and the printed name is the
            # better authority anyway.
            rescued = _rescue_code_from_label(joined)
            if rescued and rescued != code:
                print(f"[Step 6]   {name}->{code} retyped to {rescued}: "
                      f"the drawing says so ({joined[:28]!r})")
                code = _refine_room_code(rescued, texts)
                if code not in YOLO_LABEL_AGREEMENT:
                    code = rescued
            else:
                rejected.append(f"{name}->{code}: label disagrees ({joined[:28]!r})")
                continue

        rooms.append({
            "room_type_code":        code,
            "room_label":            None,   # Step 7 reads it off the drawing
            "length_mm":             None,   # Step 7 reconciles
            "width_mm":              None,
            "is_wet_area":           code in WET_AREA_CODES,
            "sort_order":            len(rooms) + 1,
            "polygon_points":        polygon_points,
            "extraction_confidence": round(conf, 3),
            "walls":                 _r2s_walls_from_polygon(polygon_points),
        })

    for line in rejected:
        print(f"[Step 6]   rejected {line}")

    # Before the minimum-rooms gate, so a plan is never rescued by counting the
    # same room twice — and before door assignment, so a door is not credited to
    # a duplicate that is about to disappear.
    rooms = _dedupe_overlapping_rooms(rooms)
    for idx, room in enumerate(rooms):
        room["sort_order"] = idx + 1

    if len(rooms) < YOLO_MIN_ROOMS:
        print(f"[Step 6] Only {len(rooms)} room(s) survived the gates "
              f"(minimum {YOLO_MIN_ROOMS}) - falling back to Claude")
        return _step6_claude_fallback(image_bytes, ocr_results, scale_info,
                                      bedroom_count, property_type)

    if door_candidates:
        # Each door belongs to exactly ONE room - the smallest polygon that
        # contains it. Detections overlap (a bathroom sits inside a bedroom's
        # box often enough), and assigning the same door to both rooms both
        # double-counts openings and re-enters the assigner on a dict it has
        # already consumed.
        for door in door_candidates:
            containing = [
                r for r in rooms
                if _point_in_fractional_polygon(r["polygon_points"], door["_cx"], door["_cy"])
            ]
            door["_room_index"] = (
                min(range(len(rooms)),
                    key=lambda i: _fractional_polygon_area(rooms[i]["polygon_points"])
                    if rooms[i] in containing else float("inf"))
                if containing else None
            )

        for idx, room in enumerate(rooms):
            mine = [d for d in door_candidates if d.get("_room_index") == idx]
            if mine:
                _assign_openings_to_nearest_wall(mine, room["walls"])

    # Segment endpoints were an implementation detail of door assignment.
    for room in rooms:
        for wall in room["walls"]:
            wall.pop("_seg", None)

    # Last line of defence. Rectification should already have kept every
    # polygon under the cap, but a room that arrives here with more walls than
    # Ekatan will accept would fail the WHOLE payload, not just itself - so the
    # longest walls win and the rest are dropped rather than losing the run.
    for room in rooms:
        if len(room["walls"]) > MAX_ROOM_POLYGON_VERTS:
            before = len(room["walls"])
            room["walls"].sort(key=lambda w: -(w.get("wall_length_mm") or 0))
            del room["walls"][MAX_ROOM_POLYGON_VERTS:]
            print(f"[Step 6]   trimmed {room['room_type_code']} walls "
                  f"{before} -> {len(room['walls'])} (callback caps at "
                  f"{MAX_ROOM_POLYGON_VERTS})")

    print(f"[Step 6] YOLO produced {len(rooms)} gated rooms, "
          f"{len(door_candidates)} doors")
    return rooms


# ─── Step 6 helpers ───────────────────────────────────────────────────────────

def _r2s_walls_from_polygon(polygon_points: list[dict]) -> list[dict]:
    """
    Convert polygon edges → wall list with cardinal position (N/S/E/W/custom).
    wall_length_mm is a placeholder (Step 7 Shapely reconciliation overwrites dims).
    """
    walls = []
    n     = len(polygon_points)
    if n == 0:
        return walls

    # A wall is named for WHICH SIDE OF THE ROOM IT IS ON, not for the
    # direction the polygon happens to be wound in.
    #
    # ⚠️ THIS USED TO BE INVERTED. The previous version read
    #     angle = degrees(atan2(dy, dx))
    #     if -22.5 <= angle <= 22.5 ...:  position = "east" if dx > 0 else "west"
    # which labels a HORIZONTAL edge east/west and a VERTICAL edge north/south -
    # exactly backwards. A horizontal edge is the north or south wall.
    # It never bit because the segmentation path never ran (the Claude fallback
    # supplies its own, correct, wall_position), but Ekatan's `findWallEdge`
    # (src/lib/fpis/polygon-utils.ts) matches a wall back to a polygon edge by
    # this very label, so every opening would have been drawn on the wrong wall
    # of every room the moment the model went live.
    cx = sum(p["x"] for p in polygon_points) / n
    cy = sum(p["y"] for p in polygon_points) / n

    for i in range(n):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % n]
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        length_frac = math.sqrt(dx * dx + dy * dy)
        if length_frac < 0.008:  # skip near-zero edges
            continue

        mx, my = (p1["x"] + p2["x"]) / 2, (p1["y"] + p2["y"]) / 2
        angle = math.degrees(math.atan2(dy, dx))
        axis = None
        if   -22.5 <= angle <=  22.5 or abs(angle) >= 157.5:   # horizontal edge
            position = "north" if my <= cy else "south"
            axis = "h"
        elif 67.5  <= abs(angle) <= 112.5:                      # vertical edge
            position = "west"  if mx <= cx else "east"
            axis = "v"
        else:
            # ⚠️ A DIAGONAL EDGE IS STILL GIVEN A CARDINAL NAME, DELIBERATELY.
            # This used to emit "custom", which Ekatan accepts, stores, and then
            # loses: `plan-graph/conversion.ts` matches walls with
            # `roomWalls.find(w => w.wallPosition === pos)` over the four
            # cardinals only, so a custom wall matches nothing and is dropped
            # from the graph WITH EVERY OPENING ON IT. `room-shape.ts` also
            # treats any non-north/south position as a Y-axis wall, so the run
            # length would be wrong even where the wall survived.
            # Losing the door is worse than approximating the wall's compass
            # direction, so the dominant axis names it and `_axis = "d"` keeps
            # the honest geometric-mean length in _rescale_walls_to_mm.
            if abs(dx) >= abs(dy):
                position = "north" if my <= cy else "south"
            else:
                position = "west" if mx <= cx else "east"
            axis = "d"
        # Estimate wall length in fractional units × 10000 as mm placeholder
        walls.append({
            "wall_position":  position,
            # WHICH POLYGON EDGE THIS WALL IS, said out loud. Ekatan's
            # `findWallEdge` (src/lib/fpis/polygon-utils.ts) otherwise guesses by
            # cardinal label - "the northmost edge is the north wall" - which is
            # only correct for an axis-aligned rectangle and silently picks the
            # wrong edge on an L-shaped room, taking that wall's openings with
            # it. The column has existed since migration 0218 and was measured
            # at 0 of 1684 rows populated, because nothing ever sent one.
            # It cannot be inferred downstream from list position either: the
            # near-zero-edge skip below, and the wall-count trim in Step 6, both
            # drop walls, so walls[i] is not polygon_points[i].
            "polygon_edge_index": i,
            # ⚠️ PLACEHOLDER, NOT MILLIMETRES. This is fraction x 10000, and it
            # is only ever correct after _rescale_walls_to_mm has run. The flag
            # below is what tells that pass which walls it owns - a wall that
            # came from Claude already carries real mm and must be left alone.
            "wall_length_mm": int(length_frac * 10000),
            "_frac_len":      length_frac,
            "_axis":          axis,
            "_placeholder":   True,
            "openings":       [],
            # Kept only long enough for _assign_openings_to_nearest_wall to
            # measure real distances; stripped before the payload is built.
            "_seg":           ((p1["x"], p1["y"]), (p2["x"], p2["y"])),
        })
    return walls


# ─── Step 6 Claude fallback (used before model weights are deployed) ──────────

def _step6_claude_fallback(
    image_bytes:   bytes,
    ocr_results:   list[dict],
    scale_info:    dict,
    bedroom_count: int | None,
    property_type: str,
) -> list[dict]:
    """
    Claude Sonnet 4.6 room extraction — fallback until
    CubiCasa5K weights are available in /models volume.
    Produces rectangular polygons only (not exact boundaries).
    """
    import base64
    client = _get_claude()
    b64   = base64.b64encode(image_bytes).decode()
    mime  = _detect_mime(image_bytes)

    unit_system   = scale_info.get("unit_system", "mm")
    drawing_scale = scale_info.get("drawing_scale") or "unknown"
    bedroom_hint  = f"Expected bedroom count: {bedroom_count}" if bedroom_count else "Bedroom count: unknown"
    ocr_texts     = [r["text"] for r in ocr_results if r["confidence"] > 0.7]

    prompt = f"""You are an expert architectural drawing analyst specialising in Indian residential floor plans.

Context:
- Property type: {property_type}
- {bedroom_hint}
- Drawing scale: {drawing_scale}
- Measurement unit system: {unit_system}
- OCR text found in drawing: {json.dumps(ocr_texts[:40])}

TASK: Extract every room from this floor plan.

Valid room_type_code values (use EXACTLY these strings):
{json.dumps(sorted(VALID_ROOM_CODES))}

Valid opening_type values (use EXACTLY these strings):
{json.dumps(sorted(VALID_OPENING_CODES))}

Room label guidance:
- "Master Bedroom", "M. Bed", "MBR"              → master_bedroom
- "Bedroom", "Bed Room", "BR"                    → bedroom
- "Living", "Hall", "Drawing Room", "L/D"        → living_room
- "Dining", "Dining Room", "D.R"                 → dining_area
- "Kitchen", "K", "Modular Kitchen"              → kitchen
- "Bathroom", "Bath", "W/C", "Toilet", "WC"      → bathroom
- "Pooja", "Puja", "Prayer Room", "Mandir"       → pooja_room
- "Study", "Office", "Work Room"                 → home_office
- "Foyer", "Entry", "Entrance", "Lobby"          → foyer_entrance
- "Balcony", "Sit Out", "Deck", "Terrace"        → balcony
- "Utility", "Service", "Wash Area", "Cleanse"   → utility
- "Passage", "Corridor", "Gallery"               → passage
- "Staircase", "Stair", "Lift"                   → staircase
- "Servant", "Maid", "Staff"                     → servant_room
- "Kids", "Children", "Child Bedroom"            → kids_room
- "Guest Bedroom", "Guest Room"                  → guest_room
- "Home Theatre", "Theater", "Media Room"        → home_theatre
- "Powder Room", "Powder"                        → powder_room
- "Master Toilet", "M.Toilet", "Master Bath"     → master_bathroom
- "Store", "Storage", "Dry Balcony"              → store
- "Terrace", "Deck", "Roof"                      → terrace
- Anything else                                  → other

Convert ALL dimensions to MILLIMETRES. If unreadable, set null — never invent.

Output JSON array only. polygon_points are fractional (0.0–1.0), clockwise, min 4 points.
[
  {{
    "room_type_code": "master_bedroom",
    "room_label": "Master Bedroom",
    "length_mm": 4500,
    "width_mm": 3600,
    "is_wet_area": false,
    "sort_order": 1,
    "polygon_points": [{{"x": 0.10, "y": 0.15}}, {{"x": 0.35, "y": 0.15}}, {{"x": 0.35, "y": 0.40}}, {{"x": 0.10, "y": 0.40}}],
    "walls": [
      {{"wall_position": "north", "wall_length_mm": 4500, "openings": [{{"opening_type": "single_door", "rough_width_mm": 900, "offset_from_left_mm": 300, "door_swing": "inward_right", "extraction_confidence": 0.88}}]}},
      {{"wall_position": "east",  "wall_length_mm": 3600, "openings": [{{"opening_type": "window_standard", "rough_width_mm": 1200, "offset_from_left_mm": 600}}]}},
      {{"wall_position": "south", "wall_length_mm": 4500, "openings": []}},
      {{"wall_position": "west",  "wall_length_mm": 3600, "openings": []}}
    ]
  }}
]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,   # full-geometry dump for a multi-room plan exceeds 4000
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }],
    )

    rooms_raw = _parse_json_response(response.content[0].text, [])
    if not isinstance(rooms_raw, list):
        return []

    for room in rooms_raw:
        if not isinstance(room, dict):
            continue
        if room.get("room_type_code") not in VALID_ROOM_CODES:
            room["room_type_code"] = "other"
        # The prompt asks for clockwise points; asking is not enforcing. Ekatan's
        # renderers read this polygon directly (the FPIS review overlays and the
        # customer 3D view), so normalise it the same way the model path does
        # rather than trusting the model to have complied.
        pts = room.get("polygon_points")
        if isinstance(pts, list) and len(pts) >= 3:
            try:
                room["polygon_points"] = _ensure_clockwise(pts)
            except (KeyError, TypeError):
                pass   # malformed points: leave them, the rules engine will flag
        for wall in (room.get("walls") or []):
            for opening in (wall.get("openings") or []):
                if opening.get("opening_type") not in VALID_OPENING_CODES:
                    opening["opening_type"] = "single_door"

    return [r for r in rooms_raw if isinstance(r, dict)]


# ─── Step 6b: Windows ─────────────────────────────────────────────────────────
#
# WHY A SEPARATE PASS, AND WHY CLAUDE. The segmentation model has 31 classes and
# exactly one of them is an opening: `15-door`. There is no window class, so on
# the YOLO path a window could not be detected at all — every room shipped with
# doors only, and Rule 6 then flagged the room for the missing openings it was
# never able to find.
#
# This is deliberately the NARROWEST possible use of a model: it is asked for
# windows and nothing else. Doors stay with YOLO, which is trained and measured
# on them (mAP50 0.868). One source per opening kind means there is nothing to
# de-duplicate and no arbitration to get wrong.
#
# It does NOT run on the Claude fallback path — that prompt already returns
# windows with the rest of its geometry, and asking twice would double them.
#
# The output feeds the same assigner and the same rescaling as doors, so a
# window cannot end up in different units from the wall it sits on.

WINDOW_OPENING_CODES = ("window_standard", "window_bay", "window_corner", "ventilator")


def _rooms_came_from_yolo(rooms: list[dict]) -> bool:
    """
    True when Step 6 produced these rooms from the segmentation model.

    The tell is `_placeholder`: `_r2s_walls_from_polygon` stamps it on every wall
    it derives from a polygon, and the Claude fallback returns walls carrying
    real millimetres with no such marker. Step 7 already relies on exactly this
    distinction to decide which walls it owns.
    """
    return any(
        w.get("_placeholder")
        for r in rooms
        for w in (r.get("walls") or [])
    )


@app.function(image=cpu_image, secrets=[fpis_secrets], timeout=180, memory=1024)
def _step6b_windows_detect(image_bytes: bytes, rooms: list[dict]) -> list[dict]:
    """
    Find windows and add them to the rooms Step 6 already found.

    Returns the rooms, mutated. On any failure it returns them untouched: a
    plan with doors and no windows is a smaller loss than a failed extraction.
    """
    if not rooms:
        return rooms

    try:
        client = _get_claude()
        b64, mime = _image_for_claude(image_bytes)

        prompt = f"""You are reading an Indian residential floor plan to locate its WINDOWS.

Find every window, ventilator and similar wall opening that is NOT a door.
Ignore doors entirely — those are already known.

A window is drawn as a break in a wall, usually with thin parallel lines across
the gap. It sits ON a wall line, never in open floor space.

For each one give the two endpoints where it meets the wall, as fractions of the
image (0.0-1.0, x from left, y from top). The line from (x1,y1) to (x2,y2) must
run ALONG the wall, so a window on a vertical wall has x1 approximately equal to x2.

Valid opening_type values (use EXACTLY these strings):
{json.dumps(list(WINDOW_OPENING_CODES))}
  window_standard - an ordinary window
  window_bay      - a window that projects out of the wall
  window_corner   - a window wrapping a corner
  ventilator      - a small high vent, common above bathroom doors

Return a JSON array only, no markdown. Return [] if you find none.
Do not guess: a window you are unsure about is better left out.
[
  {{"opening_type": "window_standard", "x1": 0.21, "y1": 0.14, "x2": 0.29, "y2": 0.14, "confidence": 0.9}},
  {{"opening_type": "ventilator",      "x1": 0.55, "y1": 0.40, "x2": 0.55, "y2": 0.44, "confidence": 0.7}}
]"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        found = _parse_json_response(response.content[0].text, [])
    except Exception as exc:
        print(f"[Step 6b] Window detection failed: {exc} - continuing with doors only")
        return rooms

    if not isinstance(found, list) or not found:
        print("[Step 6b] No windows found")
        return rooms

    # Rebuild each wall's segment from the polygon. Step 6 strips `_seg` before
    # returning, and `polygon_edge_index` is what makes this recoverable —
    # walls[i] is NOT polygon_points[i] once short edges are skipped.
    for room in rooms:
        pts = room.get("polygon_points") or []
        n = len(pts)
        for wall in (room.get("walls") or []):
            i = wall.get("polygon_edge_index")
            if isinstance(i, int) and 0 <= i < n:
                a, b = pts[i], pts[(i + 1) % n]
                wall["_seg"] = ((a["x"], a["y"]), (b["x"], b["y"]))

    candidates = []
    for w in found[:40]:
        if not isinstance(w, dict):
            continue
        code = w.get("opening_type")
        if code not in WINDOW_OPENING_CODES:
            code = "window_standard"
        try:
            x1, y1 = float(w["x1"]), float(w["y1"])
            x2, y2 = float(w["x2"]), float(w["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            continue
        candidates.append({
            "opening_type":          code,
            "_cx":                   (x1 + x2) / 2.0,
            "_cy":                   (y1 + y2) / 2.0,
            "_w_frac_x":             abs(x2 - x1),
            "_w_frac_y":             abs(y2 - y1),
            "extraction_confidence": round(float(w.get("confidence") or 0.7), 3),
        })

    # A window sits ON a boundary, so containment cannot place it the way it
    # places a door. Give it to the room whose nearest wall it is actually
    # closest to, and only if it is close enough to be on that wall at all.
    MAX_WALL_DISTANCE_FRAC = 0.03
    placed = 0
    for cand in candidates:
        best_room, best_d = None, float("inf")
        for room in rooms:
            for wall in (room.get("walls") or []):
                seg = wall.get("_seg")
                if not seg:
                    continue
                (x0, y0), (x1, y1) = seg
                dx, dy = x1 - x0, y1 - y0
                L2 = dx * dx + dy * dy
                if L2 <= 1e-12:
                    continue
                t = max(0.0, min(1.0, ((cand["_cx"] - x0) * dx + (cand["_cy"] - y0) * dy) / L2))
                d = math.hypot(cand["_cx"] - (x0 + t * dx), cand["_cy"] - (y0 + t * dy))
                if d < best_d:
                    best_room, best_d = room, d
        if best_room is not None and best_d <= MAX_WALL_DISTANCE_FRAC:
            _assign_openings_to_nearest_wall([cand], best_room["walls"])
            placed += 1

    for room in rooms:
        for wall in (room.get("walls") or []):
            wall.pop("_seg", None)

    print(f"[Step 6b] Claude found {len(candidates)} window(s), placed {placed} on walls")
    return rooms


# ─── Step 7: Spatial Reconciliation ───────────────────────────────────────────

def _step7_reconcile(rooms: list[dict], ocr_results: list[dict], scale_info: dict) -> list[dict]:
    """
    Link OCR dimension strings → room polygons via Shapely spatial join.
    Fills null length_mm/width_mm from spatially associated OCR text.
    Resolves L vs W ambiguity (longer dim = length by convention).
    """
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        return rooms

    unit = scale_info.get("unit_system", "mm")

    def ocr_to_mm(text: str) -> float | None:
        text = text.strip()
        # Feet-inches: 12'-6" or 12'6"
        m = re.match(r"""(\d+)['\s]*(?:ft|feet)?[-\s]*(\d+)?["\s]*(?:in|inch)?""", text, re.I)
        if m and unit == "feet_inches":
            ft   = int(m.group(1))
            inch = int(m.group(2)) if m.group(2) else 0
            return ft * 304.8 + inch * 25.4
        # Metres: 3.6m
        m = re.match(r"(\d+\.?\d*)\s*m$", text, re.I)
        if m:
            return float(m.group(1)) * 1000
        # Plain number
        m = re.match(r"(\d+\.?\d*)$", text)
        if m:
            val = float(m.group(1))
            return val if (unit == "mm" or val >= 100) else val * 1000
        return None

    dim_tokens: list[dict] = []
    pair_tokens: list[dict] = []
    for region in ocr_results:
        pair = _parse_dimension_pair(region["text"])
        if pair:
            # A printed "12'8\"X16'5\"" is authoritative - it is the architect's
            # own figure. Kept separate from single-value tokens so it can win.
            pair_tokens.append({"pair": pair, "centroid": region["centroid"]})
            continue
        val = ocr_to_mm(region["text"])
        if val and 500 <= val <= 20000:
            dim_tokens.append({"mm": val, "centroid": region["centroid"]})

    # Anything already carrying dimensions at this point read them off the
    # drawing itself in Step 6 - that only happens on the Claude fallback path,
    # which returns its own millimetres. Claim it now, so a room whose numbers
    # nothing here improves still says where they came from. A printed pair
    # found below outranks it and overwrites.
    for room in rooms:
        if room.get("length_mm") is not None or room.get("width_mm") is not None:
            room.setdefault("dimension_source", "model_extraction")

    for room in rooms:
        poly_pts = room.get("polygon_points")
        if not poly_pts or len(poly_pts) < 3:
            continue
        try:
            poly = Polygon([(p["x"], p["y"]) for p in poly_pts])
        except Exception:
            continue
        if not poly.is_valid:
            continue

        # A printed pair inside this room wins outright over loose numbers.
        inside_pairs = [
            tok["pair"] for tok in pair_tokens
            if poly.contains(Point(tok["centroid"]["x"], tok["centroid"]["y"]))
        ]
        if inside_pairs:
            # ⚠️ PICK THE LARGEST, NEVER THE FIRST. An interior drawing prints
            # furniture sizes inside the room too - the sample plan carries
            # 6'6"X6'0" (a bed cot) and 7'0"x3'3" (a sofa) sitting inside the
            # very polygons whose room dimensions we are trying to read. A room
            # is always larger than the furniture standing in it, so max-by-area
            # picks the room and demotes the props. Taking [0] sized a master
            # bedroom as a bed in testing on 2026-08-28.
            a, b = max(inside_pairs, key=lambda p: p[0] * p[1])
            room["length_mm"] = int(max(a, b))
            room["width_mm"] = int(min(a, b))
            room["dimension_confidence"] = "high"
            room["dimension_source"] = "printed_pair"
            continue

        inside_dims = [
            tok["mm"] for tok in dim_tokens
            if poly.contains(Point(tok["centroid"]["x"], tok["centroid"]["y"]))
            or poly.distance(Point(tok["centroid"]["x"], tok["centroid"]["y"])) < 0.02
        ]
        if not inside_dims:
            continue

        inside_dims.sort(reverse=True)  # largest = length

        # These are loose numbers lying inside or near the polygon, not a pair
        # the architect wrote as this room's size. They are worth using and
        # worth labelling as weaker - a designer deciding whether to re-measure
        # needs to know which of the two they are looking at.
        took_token = False
        if room.get("length_mm") is None and len(inside_dims) >= 1:
            room["length_mm"] = int(inside_dims[0])
            took_token = True
        if room.get("width_mm") is None and len(inside_dims) >= 2:
            room["width_mm"] = int(inside_dims[1])
            took_token = True
        if took_token:
            room["dimension_source"] = "ocr_tokens"

    # Enforce: length >= width
    for room in rooms:
        l, w = room.get("length_mm"), room.get("width_mm")
        if l and w and w > l:
            room["length_mm"], room["width_mm"] = w, l

        # Set dimension_confidence based on how many dims were resolved.
        # Without this, _compute_confidence defaults to "high" and inflates
        # extraction_confidence → quotes.ai_confidence_avg.
        if room.get("dimension_confidence") is None:
            dims_found = (room.get("length_mm") is not None) + (room.get("width_mm") is not None)
            room["dimension_confidence"] = {2: "high", 1: "medium", 0: "low"}[dims_found]

        # Say "none" rather than leaving the field absent. An absent key reads
        # as "an older pipeline that never reported this"; "none" is the
        # positive statement that we looked and found nothing.
        if room.get("dimension_source") is None:
            room["dimension_source"] = "none"

    # ── Assign room labels from OCR text inside each polygon ─────────────────
    # (Critical for Raster2Seq path: segmentation gives type but no label text)
    for room in rooms:
        if room.get("room_label"):
            # Already labelled by the Claude extraction path — high label confidence.
            room.setdefault("label_confidence", "high")
            continue

        poly_pts = room.get("polygon_points")
        if not poly_pts or len(poly_pts) < 3:
            room["room_label"] = room["room_type_code"].replace("_", " ").title()
            room["label_confidence"] = "low"  # fell back to room_type_code as label
            continue
        try:
            poly = Polygon([(p["x"], p["y"]) for p in poly_pts])
        except Exception:
            room["room_label"] = room["room_type_code"].replace("_", " ").title()
            room["label_confidence"] = "low"
            continue

        # Collect OCR text whose centroid falls inside the polygon.
        # A measurement is never a room's name - see _is_dimension_text for what
        # this used to let through, and why the old digits-only test could not.
        label_candidates = [
            r["text"].strip() for r in ocr_results
            if r["confidence"] > 0.65
            and len(r["text"].strip()) > 1
            and not _is_dimension_text(r["text"])
            and (
                poly.contains(Point(r["centroid"]["x"], r["centroid"]["y"]))
                or poly.distance(Point(r["centroid"]["x"], r["centroid"]["y"])) < 0.015
            )
        ]

        if label_candidates:
            # ⚠️ LONGEST IS A TIEBREAK, NOT THE RULE. It used to be the rule, and
            # it is half of why rooms were named "14'2x11'0"" - a printed
            # dimension is longer than "BEDROOM", so on any plan that labels its
            # rooms with both, the measurement won. Dimensions are filtered out
            # above now, but the preference still belongs on WORDS THE DRAWING
            # USES FOR ROOMS: that is the same vocabulary the detection gate
            # already trusts, so a label it recognises beats a longer stray
            # string like a builder's watermark or a legend caption.
            named = [t for t in label_candidates
                     if any(w in t.upper() for w in ROOM_LABEL_WORDS)]
            room["room_label"] = max(named or label_candidates, key=len)
            room["label_confidence"] = "high" if named else "medium"
        else:
            room["room_label"] = room["room_type_code"].replace("_", " ").title()
            room["label_confidence"] = "low"  # fell back to room_type_code as label

    # ⚠️ WALL RESCALING DELIBERATELY DOES NOT HAPPEN HERE ANY MORE. It is the
    # LAST thing that may run, because it converts walls and openings using the
    # room's dimensions — and Step 7b can still recover a dimension this step
    # failed to read. Rescaling here would measure every wall against the OCR
    # answer and then never revisit it. The orchestrator calls it after 7b.
    return rooms



def _step7b_verify_dimensions(image_bytes: bytes, rooms: list[dict]) -> list[dict]:
    """
    Ask Claude to read the dimensions OCR could not, for rooms still missing them.

    ⚠️ WHY A MODEL AND NOT A BETTER REGEX. PaddleOCR mangles the feet and inch
    marks, and the damage is not always recoverable by pattern. Measured on a
    live plan 2026-09-01, the DINING room prints `11'6"x11'0"` and OCR returned:

        11'6x110

    Both inch marks are gone. `110` is either 11'0" or 1'10", and no rule can
    tell which without looking at the drawing — so the parser correctly refuses
    it and the room ships with no dimensions, while the number sits legibly in
    the image. The LIVING room prints 14'6"x12'0" and came back as a single
    stray token that reconciled to 19812 mm: a 65-foot living room.

    Reading text off an image is the one thing a vision model is unambiguously
    better at than the pipeline around it, so this is a narrow use of one: it is
    given ONLY the rooms that are still unmeasured, it is asked ONLY to copy the
    printed text back verbatim, and the string it returns is parsed by the SAME
    deterministic parser everything else uses. The model never returns a
    millimetre value and is never trusted to compute one — it reads, we measure.

    A room whose dimensions came from here is recorded as `printed_pair`,
    because that is what it is: a pair printed on the drawing. Which reader
    recovered it is a mechanism detail, and the distinction a designer needs is
    "off the drawing" versus "inferred".
    """
    needy = [
        (i, r) for i, r in enumerate(rooms)
        if r.get("length_mm") is None or r.get("width_mm") is None
        or r.get("dimension_source") in ("ocr_tokens", "none", None)
    ]
    if not needy:
        return rooms
    needy = needy[:VERIFY_DIMENSIONS_MAX_ROOMS]

    def _centroid(room):
        pts = room.get("polygon_points") or []
        if not pts:
            return 0.5, 0.5
        return (sum(p["x"] for p in pts) / len(pts),
                sum(p["y"] for p in pts) / len(pts))

    listing = []
    for slot, (_, room) in enumerate(needy):
        cx, cy = _centroid(room)
        listing.append(f"  {slot}. {room.get('room_label') or room['room_type_code']}"
                       f"  ({room['room_type_code']}) at ({cx:.2f}, {cy:.2f})")

    try:
        client = _get_claude()
        b64, mime = _image_for_claude(image_bytes)
        prompt = f"""You are reading an Indian residential floor plan.

For each room below, find the dimension text PRINTED ON THE DRAWING for that
room and copy it back EXACTLY as printed. Positions are fractions of the image,
x from the left, y from the top.

{chr(10).join(listing)}

Rules:
- Copy the text VERBATIM, including the feet and inch marks: 14'6"x12'0"
- The dimensions of a ROOM, never of the furniture standing in it. A bed printed
  6'6"x6'0" inside a bedroom is not the bedroom.
- Some rooms print only one dimension, e.g. 5'9"WIDE. Copy that as it appears.
- If a room prints no dimensions at all, return null for it. NEVER estimate,
  never measure, never infer from the drawing's scale. A null is useful; a
  guess is not.

Return a JSON array only, no markdown, one entry per room above:
[{{"index": 0, "printed": "14'6\\"x12'0\\""}}, {{"index": 1, "printed": null}}]"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        found = _parse_json_response(response.content[0].text, [])
    except Exception as exc:
        print(f"[Step 7b] Dimension verification failed: {exc} - keeping OCR result")
        return rooms

    if not isinstance(found, list):
        return rooms

    applied = 0
    for entry in found:
        if not isinstance(entry, dict):
            continue
        slot = entry.get("index")
        printed = entry.get("printed")
        if not isinstance(slot, int) or not (0 <= slot < len(needy)):
            continue
        if not isinstance(printed, str) or not printed.strip():
            continue

        room = needy[slot][1]
        before = (room.get("length_mm"), room.get("width_mm"))

        pair = _parse_dimension_pair(printed)
        if pair:
            a, b = pair
            room["length_mm"] = int(max(a, b))
            room["width_mm"] = int(min(a, b))
            room["dimension_source"] = "printed_pair"
            room["dimension_confidence"] = "high"
            applied += 1
            print(f"[Step 7b]   {room['room_type_code']}: read {printed!r} -> "
                  f"{room['length_mm']}x{room['width_mm']}mm (was {before})")
            continue

        single = _parse_single_dimension(printed)
        if single and room.get("length_mm") is None:
            # One printed dimension is half an answer, and half is worth having:
            # a wall run is better measured against it than against nothing.
            room["length_mm"] = int(single)
            room["dimension_source"] = "printed_pair"
            room["dimension_confidence"] = "medium"
            applied += 1
            print(f"[Step 7b]   {room['room_type_code']}: read {printed!r} -> "
                  f"{room['length_mm']}mm (one dimension only)")
        else:
            print(f"[Step 7b]   {room['room_type_code']}: {printed!r} did not parse "
                  f"- left as it was")

    print(f"[Step 7b] Verified {applied} of {len(needy)} unmeasured room(s)")
    return rooms


def _parse_single_dimension(text: str) -> float | None:
    """One printed dimension in mm, e.g. `5'9"WIDE` -> 1752. None if absent."""
    s = (text or "").strip()
    if _parse_dimension_pair(s):
        return None      # a pair belongs to the pair parser, not here
    m = re.search(r"(\d+)\s*" + _QUOTE + r"\s*(\d+)?", s)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(2)) if m.group(2) else 0
        mm = ft * 304.8 + inch * 25.4
        return mm if 300 <= mm <= 30000 else None
    m = re.fullmatch(r"\s*(\d{3,5})\s*(?:mm)?\s*", s, re.I)
    if m:
        mm = float(m.group(1))
        return mm if 300 <= mm <= 30000 else None
    return None


def _rescale_walls_to_mm(rooms: list[dict]) -> None:
    """
    Convert polygon-derived wall lengths from fractional units into real mm.

    WHY THIS EXISTS. `_r2s_walls_from_polygon` can only measure a wall as a
    fraction of the image - at Step 6 nobody knows how many millimetres a
    fraction is worth. It wrote `int(length_frac * 10000)` into a field called
    `wall_length_mm` and moved on. Nothing downstream knew the difference:
    Ekatan's `buildRoomGeometry` (src/domains/geometry-engine/fpis-geometry.ts)
    reads `wallLengthMm` as millimetres and derives `tallRunMm` / `baseRunMm`
    from it, which is what sizes a wardrobe on a quote. A wall reported as
    1,200 "mm" when it is really 0.12 of the image is a priced line item built
    on a number that means nothing.

    It never bit before because the segmentation path never ran - every
    extraction fell through to Claude, which returns real millimetres in its
    JSON. The moment the YOLO path went live it would have.

    By Step 7 the room's true size is usually known - printed on the drawing
    ("MASTER BEDROOM 12'8\"X16'5\"") or reconciled from OCR. That gives a
    millimetres-per-fraction scale per room, per axis, which is all this needs.
    A room whose dimensions are still unknown keeps its placeholder and is
    flagged, rather than being silently handed a fabricated number.

    THE OPENINGS ARE RESOLVED HERE TOO, and they must be. They were not, and
    that was a bug with a straight line to a customer's quote: Step 6 wrote
    `offset_from_left_mm` against the fraction placeholder, this function
    converted the WALL and not the OPENING, and the two shipped in different
    units on the same object. A door detected mid-wall arrived at Ekatan
    claiming to sit ~150 mm from the corner of a 3,861 mm wall. `room-shape.ts`
    clamps that offset against the wall run, `fpis-geometry.ts` turns it into
    the spans that size a wardrobe, and the 3D portal draws it - three
    consumers, all of them wrong, none of them able to notice.
    """
    for room in rooms:
        walls = room.get("walls") or []
        placeholder = [w for w in walls if w.get("_placeholder")]
        if not placeholder:
            continue

        pts = room.get("polygon_points") or []
        l_mm, w_mm = room.get("length_mm"), room.get("width_mm")

        if len(pts) >= 3 and l_mm and w_mm:
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            bw, bh = (max(xs) - min(xs)), (max(ys) - min(ys))
            if bw > 1e-6 and bh > 1e-6:
                # The longer printed dimension belongs to the longer bbox side.
                long_mm, short_mm = max(l_mm, w_mm), min(l_mm, w_mm)
                if bw >= bh:
                    mm_per_x, mm_per_y = long_mm / bw, short_mm / bh
                else:
                    mm_per_x, mm_per_y = short_mm / bw, long_mm / bh
                for wall in placeholder:
                    frac = wall.get("_frac_len") or 0.0
                    # Scale by the edge's own axis. Deliberately NOT by
                    # wall_position: that is a derived label, and a "custom"
                    # (diagonal) edge has no single axis to scale by - it takes
                    # the geometric mean so it is never wildly wrong either way.
                    axis = wall.get("_axis")
                    if axis == "h":
                        scale = mm_per_x
                    elif axis == "v":
                        scale = mm_per_y
                    else:
                        scale = math.sqrt(mm_per_x * mm_per_y)
                    wall["wall_length_mm"] = int(round(frac * scale))
                    wall["_placeholder"] = False
                    _resolve_openings_to_mm(wall, scale)
                continue

        # No usable dimensions: say so rather than invent a length.
        #
        # ⚠️ THE WALL ITSELF MUST GO OUT NULL TOO, AND FOR A WHILE IT DID NOT.
        # This branch used to null only the openings and leave `wall_length_mm`
        # holding the `frac x 10000` placeholder, which then shipped as
        # millimetres — the exact defect this function exists to prevent, left
        # standing in the one branch that admits it cannot measure. Caught on a
        # live plan on 2026-09-01: a master bedroom whose width never resolved
        # reported a 1667 mm wall for an edge that was 0.1668 of the image, and
        # the living room reported 3546 for 0.3546. Every one of those numbers
        # was the image fraction wearing a millimetre label.
        #
        # A room reaches here whenever EITHER dimension is missing, which is
        # common — five of nine rooms on that plan. Ekatan takes a null length
        # without complaint (`wallLengthMm: number | null`) and falls back to the
        # room's own extent when it renders, so null costs nothing and a
        # fabricated number costs a wardrobe sized against the image.
        for wall in placeholder:
            wall["wall_length_mm"] = None
            for op in wall.get("openings") or []:
                if "_along_frac" in op:
                    op["rough_width_mm"] = None
                    op["offset_from_left_mm"] = None
        room.setdefault("validation_flags", []).append(
            "wall_lengths_unscaled:room_dimensions_unknown"
        )

    # Strip the bookkeeping before the payload is built.
    for room in rooms:
        for wall in room.get("walls") or []:
            wall.pop("_frac_len", None)
            wall.pop("_placeholder", None)
            wall.pop("_axis", None)
            wall.pop("_seg", None)
            for op in wall.get("openings") or []:
                op.pop("_along_frac", None)
                op.pop("_width_frac", None)


def _resolve_openings_to_mm(wall: dict, scale: float) -> None:
    """
    Turn one wall's fractional openings into millimetres along that wall.

    `scale` is the same millimetres-per-fraction the wall itself was just
    measured with, so the opening and the wall it sits on can never again be
    expressed in different units.

    Only openings carrying `_along_frac` are touched: the Claude fallback path
    returns real millimetres in its own JSON and must be left exactly alone.
    """
    wall_mm = wall.get("wall_length_mm") or 0
    for op in wall.get("openings") or []:
        if "_along_frac" not in op:
            continue

        width_frac = op.get("_width_frac")
        width_mm = int(round(width_frac * scale)) if width_frac else 0
        # A detection narrower than the narrowest real opening of its kind is an
        # artefact, not a measurement. The floors come from the widths Ekatan
        # seeds in opening_types.min_clear_width_mm: a ventilator starts at 300,
        # an exhaust at 150, a single door at 700, a standard window at 600.
        # Flooring everything at 600 would have quietly doubled every ventilator.
        kind = str(op.get("opening_type", ""))
        floor_mm = 300 if kind in ("ventilator", "exhaust_opening") else 600
        if width_mm < floor_mm:
            width_mm = floor_mm
        if wall_mm > 0:
            width_mm = min(width_mm, wall_mm)

        # `_along_frac` is where the opening's CENTRE sits along the wall, 0 at
        # the wall's start vertex. Ekatan measures `offset_from_left_mm` from
        # that same start (plan-graph/conversion.ts winds every room clockwise
        # from the top-left), so the two agree as long as the polygon does.
        centre_mm = (op.get("_along_frac") or 0.0) * wall_mm
        offset = int(round(centre_mm - width_mm / 2.0))
        # Keep the opening on its own wall: Ekatan clamps this anyway
        # (room-shape.ts), but a negative offset would be stored as-is.
        offset = max(0, min(offset, max(0, wall_mm - width_mm)))

        op["rough_width_mm"] = width_mm
        op["offset_from_left_mm"] = offset


# ─── Step 8: Rules Engine ─────────────────────────────────────────────────────

def _step8_rules_engine(rooms: list[dict], carpet_area_sqft: float | None) -> list[dict]:
    """
    Apply the plausibility rules. Flags but never discards.
    Computes extraction_confidence per room.
    All code comparisons use DB lowercase codes.

    Rules 2 and 7 were removed on 2026-09-01 — both tested fields no code path
    ever set, so neither could fire. The numbering of the survivors is left
    alone because these strings are stored in `validation_flags` and read by
    humans in the FPIS review screen; renumbering would orphan every flag
    already in the database.
    """
    for i, room in enumerate(rooms):
        flags        = []
        needs_review = False

        l_mm = room.get("length_mm")
        w_mm = room.get("width_mm")
        code = room.get("room_type_code", "other")

        # Rule 1 — Dimension range
        for dim_name, dim_val in [("length_mm", l_mm), ("width_mm", w_mm)]:
            if dim_val is not None and (dim_val < 1200 or dim_val > 15000):
                flags.append(f"dimension_out_of_range:{dim_name}={dim_val}")
                needs_review = True

        # Rule 2 — REMOVED 2026-09-01. It compared a computed area against
        # `room.get("area_sqft")`, and no code path in this pipeline has ever
        # set `area_sqft` on a room - not the YOLO path, not the Claude
        # fallback's JSON schema. The condition could never be true, so the rule
        # was dead weight that read like a working safety net. Ekatan computes
        # its own area from length x width on arrival
        # (src/lib/fpis/room-area.ts::computeRoomAreaSqft), so there is nothing
        # to cross-check against here.

        # Rule 3 — Wet area auto-correction
        if code in WET_AREA_CODES and not room.get("is_wet_area"):
            room["is_wet_area"] = True
            flags.append(f"wet_area_auto_corrected:{code}")

        # Rule 4 — "other" with high label confidence = likely mis-classification
        label_conf = room.get("label_confidence", "medium")
        if code == "other" and label_conf == "high":
            flags.append("possible_misclassification:other_with_high_label_confidence")
            needs_review = True

        # Rule 5 — Both dimensions null
        if l_mm is None and w_mm is None:
            flags.append("both_dimensions_null:cost_engine_blocked")
            needs_review = True

        # Rule 6 — Opening count sanity.
        # ⚠️ DOORS ONLY, ON PURPOSE. A habitable room with no door found is a
        # real miss worth a human's time. A room with no WINDOW found is not
        # evidence of anything: the segmentation model has no window class at
        # all (its only opening class is `15-door`), so windows arrive only from
        # the Claude fallback or the openings pass. Counting all openings here
        # flagged nearly every room on every YOLO run and docked 0.15 off its
        # confidence for a gap in the model's vocabulary rather than a defect in
        # the plan.
        door_count = sum(
            1
            for wall in (room.get("walls") or [])
            for op in (wall.get("openings") or [])
            if str(op.get("opening_type", "")).endswith("door")
        )
        if door_count == 0 and code not in NO_OPENING_EXEMPT:
            flags.append("zero_doors_detected:possible_miss")
            needs_review = True

        # Rule 7 — REMOVED 2026-09-01, for the same reason as Rule 2: it read
        # `room.get("ceiling_height_mm")`, which nothing in this pipeline ever
        # sets. Ceiling height is Stage-2 survey data a human records on site
        # (unit_type_rooms.finished_ceiling_height_mm), never something an image
        # can tell us, so this rule cannot be revived here - it belongs where the
        # measurement is taken.

        room["needs_review"]          = needs_review
        room["validation_flags"]      = flags
        room["extraction_confidence"] = _compute_confidence(room)
        room["sort_order"]            = room.get("sort_order") or i + 1

    # Rule 8 — Plan total area check
    if carpet_area_sqft and carpet_area_sqft > 0:
        internal_codes = VALID_ROOM_CODES - EXTERNAL_ROOM_CODES
        total_computed = sum(
            (r["length_mm"] / 304.8) * (r["width_mm"] / 304.8)
            for r in rooms
            if r.get("length_mm") and r.get("width_mm")
            and r.get("room_type_code") in internal_codes
        )
        if total_computed > 0:
            ratio = total_computed / carpet_area_sqft
            if ratio < 0.70 or ratio > 1.30:
                msg = f"plan_area_mismatch:sum={total_computed:.0f}sqft stated={carpet_area_sqft}sqft ratio={ratio:.2f}"
                for room in rooms:
                    room.setdefault("validation_flags", []).append(msg)

    return rooms


def _compute_confidence(room: dict) -> float:
    """Confidence scoring formula from FPIS architecture plan Section 3.5."""
    score = 1.0
    l, w  = room.get("length_mm"), room.get("width_mm")

    if l is None:
        score -= 0.30
    if w is None:
        score -= 0.30
    if (l is None) != (w is None):  # only one found
        score -= 0.15

    dim_conf = room.get("dimension_confidence", "high")
    score -= {"medium": 0.10, "low": 0.25}.get(dim_conf, 0)

    lbl_conf = room.get("label_confidence", "high")
    score -= {"medium": 0.10, "low": 0.25}.get(lbl_conf, 0)

    score -= 0.15 * min(len(room.get("validation_flags") or []), 3)
    return round(max(0.0, min(1.0, score)), 3)


# ─── Step 9: Webhook Callback ─────────────────────────────────────────────────

def _post_callback(callback_url: str, payload: dict, secret: str) -> None:
    """
    POST results to /api/fpis/callback with shared secret header.
    Retries up to 3 times on transient failures.
    """
    import httpx

    body    = json.dumps(payload).encode()
    headers = {
        "Content-Type":  "application/json",
        "X-FPIS-Secret": secret,
    }

    for attempt in range(3):
        try:
            r = httpx.post(callback_url, content=body, headers=headers, timeout=120)
            if r.status_code == 200:
                print(f"[Step 9] Callback delivered: {r.status_code}")
                return
            print(f"[Step 9] Callback attempt {attempt + 1} returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[Step 9] Callback attempt {attempt + 1} failed: {e}")
        if attempt < 2:
            time.sleep(2 ** attempt)

    print(f"[Step 9] WARNING: All callback attempts failed for plan_id={payload.get('plan_id')}")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _get_claude():
    import os
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _detect_mime(image_bytes: bytes) -> str:
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:4] in (b"%PDF", b"%pdf"):
        return "application/pdf"
    return "image/jpeg"


# Media types the Claude Messages API accepts in an image content block.
_CLAUDE_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _image_for_claude(image_bytes: bytes) -> tuple[str, str]:
    """
    Return (base64_data, media_type) guaranteed valid for a Claude image block.

    Claude accepts jpeg/png/gif/webp. We detect the TRUE format via PIL (more
    reliable than magic-byte sniffing) and transcode anything else (tiff, bmp,
    misdetected uploads) to JPEG. This prevents the media-type-mismatch 400:
    "specified using image/jpeg ... but the image appears to be image/webp".
    """
    import base64
    import io
    from PIL import Image

    _PIL_TO_MIME = {
        "JPEG": "image/jpeg", "PNG": "image/png",
        "GIF": "image/gif",   "WEBP": "image/webp",
    }
    try:
        fmt = Image.open(io.BytesIO(image_bytes)).format
    except Exception:
        fmt = None

    mime = _PIL_TO_MIME.get(fmt or "")
    if mime:
        return base64.b64encode(image_bytes).decode(), mime

    # Unsupported or unrecognised format → transcode to JPEG.
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _parse_json_response(text: str, fallback: Any) -> Any:
    """
    Safely parse a Claude JSON response.
    Strips markdown fences, finds first valid JSON object or array.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"[FPIS] Failed to parse JSON response: {text[:300]}")
        return fallback


# ─── Local test entry point ────────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    """
    Quick local sanity test. Run with: modal run fpis_pipeline.py
    Uses a public test floor plan image.
    """
    test_payload = {
        "job_id":           "test_local_001",
        "plan_id":          "00000000-0000-0000-0000-000000000001",
        "image_url":        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Floor_Plan_Drawings.jpg/800px-Floor_Plan_Drawings.jpg",
        "callback_url":     "https://webhook.site/test",  # Replace with your webhook.site URL
        "property_type":    "apartment",
        "bedroom_count":    2,
        "carpet_area_sqft": 850,
        "floor_number":     3,
    }
    print("Starting local test run...")
    result = _run_job.remote(test_payload)
    print("\n=== Pipeline Result ===")
    print(json.dumps(result, indent=2))