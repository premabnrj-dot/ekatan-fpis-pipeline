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
# 10-step pipeline:
#   0  Pre-flight Quality    — Heuristics (file size, image dims, aspect ratio)
#   1  Quality Gate          — Claude Sonnet 4.6 (reject non-floor-plans early)
#   2  Image Preprocessing   — OpenCV (deskew, CLAHE, normalise to 300 DPI)
#   3  OCR Extraction        — PaddleOCR (room labels, dimension strings)
#   4  Scale Detection       — Claude Sonnet 4.6 (title block: scale, unit, north)  ┐ parallel
#   5  MEP Zone Detection    — Claude Sonnet 4.6 (plumbing, DB boards, AC slots)    ┘
#   6  Room Extraction       — Claude Sonnet 4.6 (rooms, walls, openings, polygon)
#   7  Spatial Reconciliation — Shapely (link OCR dims → room polygons)
#   8  Rules Engine          — Python (8 plausibility checks, confidence scoring)
#   9  Webhook Callback      — HTTP POST to /api/fpis/callback
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

# GPU image — CubiCasa5K inference (Step 6: room segmentation)
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

# Modal Volume — stores fine-tuned CubiCasa5K weights (cubicasa5k_bangalore_v1.pt)
model_volume = modal.Volume.from_name("fpis-model-weights", create_if_missing=True)

# ─── Canonical code tables (must match Supabase room_types + opening_types) ──
#
# room_types.code values in DB:
#   master_bedroom, bedroom, living_room, dining_area, kitchen, bathroom,
#   pooja_room, home_office, foyer_entrance, utility_balcony,
#   passage, staircase, servant_room, store, terrace, other
#
# opening_types.code values in DB:
#   single_door, double_door, sliding_door, french_door, pocket_door,
#   window_standard, window_bay, window_corner, ventilator,
#   arched_opening, niche_shallow, niche_deep,
#   exhaust_opening, duct_access, pass_through, meter_box

VALID_ROOM_CODES = {
    "master_bedroom", "bedroom", "living_room", "dining_area",
    "kitchen", "bathroom", "pooja_room", "home_office",
    "foyer_entrance", "utility_balcony", "passage", "staircase",
    "servant_room", "store", "terrace", "other",
}

VALID_OPENING_CODES = {
    "single_door", "double_door", "sliding_door", "french_door",
    "pocket_door", "window_standard", "window_bay", "window_corner",
    "ventilator", "arched_opening", "niche_shallow", "niche_deep",
    "exhaust_opening", "duct_access", "pass_through", "meter_box",
}

# Rooms that must always have is_wet_area = true
WET_AREA_CODES = {"bathroom", "kitchen", "utility_balcony"}

# Rooms exempt from the "zero openings" rule
NO_OPENING_EXEMPT = {"passage", "staircase", "store"}

# Rooms excluded from carpet area total (external / non-habitable)
EXTERNAL_ROOM_CODES = {"utility_balcony", "staircase", "terrace"}

# ─── CubiCasa5K class index ↔ DB code mappings ────────────────────────────────
# Index order MUST match train_raster2seq.py ROOM_CODES list exactly.

NUM_ROOM_CLASSES = 17
NUM_ICON_CLASSES = 9

ROOM_CLASS_TO_CODE = [
    "other",           # 0  background → fallback
    "living_room",     # 1
    "dining_area",     # 2
    "kitchen",         # 3
    "master_bedroom",  # 4
    "bedroom",         # 5
    "bathroom",        # 6
    "pooja_room",      # 7
    "home_office",     # 8
    "foyer_entrance",  # 9
    "utility_balcony", # 10
    "passage",         # 11
    "staircase",       # 12
    "servant_room",    # 13
    "store",           # 14
    "terrace",         # 15
    "other",           # 16
]

ICON_CLASS_TO_CODE = [
    None,              # 0  background
    "single_door",     # 1
    "double_door",     # 2
    "sliding_door",    # 3
    "window_standard", # 4
    "window_bay",      # 5
    "ventilator",      # 6
    "arched_opening",  # 7
    "single_door",     # 8  other → fallback
]


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

    # ── Steps 4 + 5: Parallel ────────────────────────────────────────────────
    scale_handle = _step4_scale_detect.spawn(processed_bytes, ocr_results)
    mep_handle   = _step5_mep_detect.spawn(processed_bytes, ocr_results)
    scale_info   = scale_handle.get(timeout=120)   # must stay under run_pipeline's 600s budget
    mep_zones    = mep_handle.get(timeout=180)
    print(f"[Step 4] Scale: {scale_info}")
    print(f"[Step 5] MEP zones: {len(mep_zones)}")

    # ── Step 6: Room Extraction (CubiCasa5K segmentation) ────────────────────
    # Runs in a separate GPU container via .remote() — returns exact polygons.
    # Falls back to Claude extraction if model weights not yet deployed.
    rooms_raw = _step6_raster2seq_extract.remote(
        processed_bytes, ocr_results, scale_info, bedroom_count, property_type
    )
    print(f"[Step 6] Raster2Seq extracted {len(rooms_raw)} rooms")

    # ── Step 7: Spatial Reconciliation ────────────────────────────────────────
    rooms_reconciled = _step7_reconcile(rooms_raw, ocr_results, scale_info)
    print(f"[Step 7] Reconciled rooms: {len(rooms_reconciled)}")

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
        "mep_zones":           mep_zones,
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

_DIM_PAIR_FT = re.compile(
    "(\\d+)\\s*'\\s*(\\d+)?\\s*\"?\\s*[xX\u00d7]\\s*(\\d+)\\s*'\\s*(\\d+)?\\s*\"?"
)
_DIM_PAIR_MM = re.compile("(\\d{3,5})\\s*[xX\u00d7]\\s*(\\d{3,5})")


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

For Indian residential floor plans: if no scale is found, assume "1:100" with confidence 0.4.
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

    return _parse_json_response(response.content[0].text, {
        "drawing_scale":       None,
        "unit_system":         "mm",
        "north_direction_deg": 0,
        "confidence":          0.3,
    })


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


# ─── Step 5: MEP Zone Detection ───────────────────────────────────────────────

@app.function(image=cpu_image, secrets=[fpis_secrets], timeout=180, memory=512)
def _step5_mep_detect(image_bytes: bytes, ocr_results: list[dict]) -> list[dict]:
    """
    Detect MEP zones using Claude Sonnet 4.6.
    Returns zones with fractional (x,y) positions converted to mm.
    zone_type codes match unit_type_mep_zones.zone_type in DB.
    """
    import base64
    client = _get_claude()
    b64   = base64.b64encode(image_bytes).decode()
    mime  = _detect_mime(image_bytes)

    mep_hints = [r["text"] for r in ocr_results if re.search(r"WD|DB|AC|gas|exhaust|elect|drain|duct", r["text"], re.I)]

    prompt = f"""You are analysing an Indian residential floor plan to detect MEP (Mechanical, Electrical, Plumbing) zones.

OCR text hints found near potential MEP symbols: {json.dumps(mep_hints[:15])}

Detect ALL of the following MEP zone types that are visible:
- PLUMBING_STACK: Waste/drain pipe symbols, WD markings, soil pipe indicators
- DB_BOARD: Electrical distribution board, DB panel, main switchboard locations
- AC_OUTDOOR_SLOT: Air conditioner outdoor unit ledge, slot, or designated space
- GAS_METER: Piped gas meter location or gas inlet point
- EXHAUST_DUCT: Kitchen/bathroom exhaust duct connection point, exhaust fan cut-out

For each zone found, provide its fractional position (0.0–1.0) relative to image width/height.
Position (0,0) is top-left, (1,1) is bottom-right.

Return JSON array only. If no MEP zones found, return empty array [].
[
  {{"zone_type": "PLUMBING_STACK", "zone_label": "WD", "position_x": 0.35, "position_y": 0.72}},
  {{"zone_type": "DB_BOARD", "zone_label": "DB", "position_x": 0.12, "position_y": 0.08}}
]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
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

    zones_raw = _parse_json_response(response.content[0].text, [])
    if not isinstance(zones_raw, list):
        return []

    zones = []
    for z in zones_raw[:20]:
        if not isinstance(z, dict):
            continue
        fx = float(z.get("position_x", 0.5))
        fy = float(z.get("position_y", 0.5))
        zones.append({
            "zone_type":     z.get("zone_type", "PLUMBING_STACK"),
            "zone_label":    z.get("zone_label"),
            # ⚠️ THE SCALE IS 10000, AND IT IS A CONTRACT, NOT A CHOICE.
            # `position_x_mm` does not hold millimetres despite its name - it
            # holds canvas fraction x 10000, documented in Ekatan at
            # src/lib/types/floor-plan.ts and src/lib/validations/
            # territory-fpis.ts, and decoded as /10000 by every renderer.
            # This wrote x20000 and Ekatan's matchRoomForMepZone decoded /20000,
            # so the two agreed with each other and disagreed with the three
            # renderers: every MEP zone was DRAWN AT DOUBLE its true position,
            # and anything past the halfway mark fell off the image entirely.
            # Neither side could see it alone - each was self-consistent.
            "position_x_mm": int(fx * 10000),
            "position_y_mm": int(fy * 10000),
        })
    return zones



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
    "0-Balcony":         "utility_balcony",
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
    "28-utility":        "utility_balcony",
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
)

# The printed label must AGREE with the predicted class. This is the gate that
# matters: on the presentation drawing every survivor of the earlier gates was
# still wrong - a "living_room" polygon sitting over PARENT'S BEDROOM, a
# "bathroom" over UTILITY. Confidence could not see it (0.83, 0.95) and neither
# could area. Only the drawing's own words could.
# This is the gate that matters: it catches a polygon in the WRONG PLACE — a
# "living_room" sitting over PARENT'S BEDROOM — which confidence and area cannot
# see. It is NOT meant to arbitrate fine distinctions between adjacent classes.
# That is why TERRACE and SITOUT are accepted for utility_balcony below: calling
# a private terrace a balcony is a small labelling imprecision, while REJECTING
# it loses the room entirely, and losing rooms is the failure this whole gate
# stack is trying to avoid.
YOLO_LABEL_AGREEMENT = {
    "bedroom":         ("BEDROOM", "BED ROOM", "M.BED", "M. BED", "GUEST", "PARENT",
                        "MASTER", "SLEEP", "CHILDREN BED"),
    "master_bedroom":  ("MASTER", "M.BED", "M. BED"),
    "servant_room":    ("SERVANT", "MAID"),
    "bathroom":        ("TOILET", "BATH", "W/C", "WC", "SPLASH", "POWDER",
                        "COM.TOILET", "COMMON TOILET"),
    "kitchen":         ("KITCHEN", "COOK", "PANTRY"),
    "living_room":     ("LIVING", "HALL", "DRAWING", "LOUNGE", "FAMILY"),
    "dining_area":     ("DINING", "DINE"),
    "utility_balcony": ("BALCONY", "UTILITY", "WASH", "SERVICE", "CLEANSE",
                        "SITOUT", "SIT OUT", "VERANDAH", "VERANDA", "TERRACE",
                        "DECK"),
    "terrace":         ("TERRACE", "DECK", "SITOUT", "SIT OUT"),
    "passage":         ("LOBBY", "PASSAGE", "CORRIDOR"),
    "foyer_entrance":  ("FOYER", "ENTRY", "ENTRANCE"),
    "home_office":     ("STUDY", "OFFICE"),
    "store":           ("STORE", "WALKIN", "WALK-IN"),
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
    """The model has no master-bedroom class; the drawing says so in words."""
    if code != "bedroom":
        return code
    joined = " ".join(texts).upper()
    if "MASTER" in joined or "M.BED" in joined or "M. BED" in joined:
        return "master_bedroom"
    if "SERVANT" in joined or "MAID" in joined:
        return "servant_room"
    return "bedroom"


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
        length_mm = best.get("wall_length_mm") or 0
        is_door = str(op.get("opening_type", "")).endswith("door")
        best["openings"].append({
            "opening_type":          op["opening_type"],
            "opening_label":         f"{'D' if is_door else 'W'}{len(best['openings']) + 1}",
            "rough_width_mm":        op["rough_width_mm"],
            "offset_from_left_mm":   int(max(0, best_t * length_mm - op["rough_width_mm"] / 2)),
            "extraction_confidence": op["extraction_confidence"],
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

    for box, mask_xy in zip(result.boxes, result.masks.xy):
        name = names[int(box.cls[0])]
        conf = float(box.conf[0])

        if name == YOLO_DOOR_CLASS and conf >= YOLO_MIN_CONF:
            xs, ys = mask_xy[:, 0], mask_xy[:, 1]
            door_candidates.append({
                "opening_type":          "single_door",
                "_cx":                   float(xs.mean()) / W,
                "_cy":                   float(ys.mean()) / H,
                "rough_width_mm":        0,          # filled once scale is known
                "_w_frac":               float(xs.max() - xs.min()) / W,
                "extraction_confidence": round(conf, 3),
            })
            continue

        code = YOLO_ROOM_TO_CODE.get(name)
        if code is None or conf < YOLO_MIN_CONF:
            continue

        contour = mask_xy.astype(np.int32).reshape(-1, 1, 2)
        rect = _rectify_contour(contour, cv2, np)
        if rect is None:
            rejected.append(f"{name}: degenerate"); continue

        area_frac = abs(cv2.contourArea(rect)) / float(W * H)
        if not (YOLO_MIN_AREA_FRAC <= area_frac <= YOLO_MAX_AREA_FRAC):
            rejected.append(f"{name}: area {area_frac:.2%}"); continue

        polygon_points = [
            {"x": round(float(p[0][0]) / W, 4), "y": round(float(p[0][1]) / H, 4)}
            for p in rect
        ]
        texts = _texts_inside_polygon(polygon_points, ocr_results)
        joined = " ".join(texts).upper()

        if not any(w in joined for w in ROOM_LABEL_WORDS):
            rejected.append(f"{name}: no room label inside"); continue

        code = _refine_room_code(code, texts)
        if not any(w in joined for w in YOLO_LABEL_AGREEMENT.get(code, ())):
            rejected.append(f"{name}->{code}: label disagrees ({joined[:28]!r})"); continue

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
            for d in mine:
                d["rough_width_mm"] = max(600, int(d.get("_w_frac", 0.09) * 10000))
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
            position = "custom"
        # Estimate wall length in fractional units × 10000 as mm placeholder
        walls.append({
            "wall_position":  position,
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
- "Balcony", "Utility", "Service", "Wash Area"   → utility_balcony
- "Passage", "Corridor", "Gallery"               → passage
- "Staircase", "Stair", "Lift"                   → staircase
- "Servant", "Maid", "Staff"                     → servant_room
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
        for wall in (room.get("walls") or []):
            for opening in (wall.get("openings") or []):
                if opening.get("opening_type") not in VALID_OPENING_CODES:
                    opening["opening_type"] = "single_door"

    return [r for r in rooms_raw if isinstance(r, dict)]


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

        if room.get("length_mm") is None and len(inside_dims) >= 1:
            room["length_mm"] = int(inside_dims[0])
        if room.get("width_mm") is None and len(inside_dims) >= 2:
            room["width_mm"] = int(inside_dims[1])

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

        # Collect OCR text whose centroid falls inside the polygon
        # Skip pure-number strings (those are dimensions, not labels)
        label_candidates = [
            r["text"] for r in ocr_results
            if r["confidence"] > 0.65
            and len(r["text"]) > 1
            and not r["text"].replace(".", "").replace(",", "").replace("-", "").replace("'", "").replace('"', "").isdigit()
            and (
                poly.contains(Point(r["centroid"]["x"], r["centroid"]["y"]))
                or poly.distance(Point(r["centroid"]["x"], r["centroid"]["y"])) < 0.015
            )
        ]

        if label_candidates:
            # Prefer longer strings (room names > single letters)
            room["room_label"] = max(label_candidates, key=len)
            room["label_confidence"] = "high"
        else:
            room["room_label"] = room["room_type_code"].replace("_", " ").title()
            room["label_confidence"] = "low"  # fell back to room_type_code as label

    # Wall lengths become real millimetres only now, once room dimensions are
    # known. See _rescale_walls_to_mm for why this cannot happen at Step 6.
    _rescale_walls_to_mm(rooms)

    return rooms



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
                continue

        # No usable dimensions: say so rather than invent a length.
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


# ─── Step 8: Rules Engine ─────────────────────────────────────────────────────

def _step8_rules_engine(rooms: list[dict], carpet_area_sqft: float | None) -> list[dict]:
    """
    Apply 8 plausibility rules. Flags but never discards.
    Computes extraction_confidence per room.
    All code comparisons use DB lowercase codes.
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

        # Rule 2 — Area consistency
        if l_mm and w_mm:
            computed_sqft = (l_mm / 304.8) * (w_mm / 304.8)
            stated_sqft   = room.get("area_sqft")
            if stated_sqft and abs(computed_sqft - stated_sqft) / stated_sqft > 0.15:
                flags.append(f"area_inconsistent:computed={computed_sqft:.0f}sqft stated={stated_sqft}sqft")
                needs_review = True

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

        # Rule 6 — Opening count sanity
        total_openings = sum(
            len(wall.get("openings") or [])
            for wall in (room.get("walls") or [])
        )
        if total_openings == 0 and code not in NO_OPENING_EXEMPT:
            flags.append("zero_openings_detected:possible_miss")
            needs_review = True

        # Rule 7 — Ceiling height range
        ch = room.get("ceiling_height_mm")
        if ch is not None and (ch < 2200 or ch > 4500):
            flags.append(f"ceiling_height_out_of_range:{ch}mm")
            needs_review = True

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