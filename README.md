# Ekatan FPIS — floor-plan extraction pipeline

This repository exists to satisfy the **AGPL-3.0** source-availability obligation for the
floor-plan extraction pipeline Ekatan runs on [Modal](https://modal.com).

`fpis_pipeline.py` reads an architectural floor plan and returns rooms, walls, openings and
MEP zones as structured data. It uses [Ultralytics](https://github.com/ultralytics/ultralytics)
YOLO for room segmentation, which is licensed AGPL-3.0. Ekatan serves inference over a
network, so AGPL section 13 applies and the corresponding source of the combined work is
published here.

## What the pipeline does

| step | what happens |
|---|---|
| 0 | Pre-flight checks — file size, dimensions, aspect ratio, PDF page count |
| 0.5 | A PDF is rendered to raster. A **vector** PDF also yields its own text layer with exact coordinates, and OCR is skipped |
| 1 | Quality gate — is this actually a floor plan? |
| 2 | Preprocessing — deskew, CLAHE, normalise |
| 3 | OCR (PaddleOCR), or the PDF's own text when it has one |
| 4 · 5 | Scale/title-block detection and MEP zone detection, in parallel |
| 6 | Room segmentation, gated against the drawing's own printed labels, falling back to a language model when the gates are not met |
| 7 | Spatial reconciliation — printed dimensions win over inferred ones |
| 8 | Rules engine — plausibility checks and confidence scoring |
| 9 | Webhook callback |

## What is deliberately not here

- **The trained model weights.** A model trained *with* a framework is not a derivative work
  *of* it. The weights are the output of Ekatan's own annotation work.
- **Dataset and scoring tooling, and internal planning documents.** AGPL asks for the source
  of the combined work, not for everything that sits beside it in development.

## ⚠️ Keeping this current

**A stale copy here is not compliance.** AGPL asks for the source *corresponding to the
version being served*. `fpis_pipeline.py` is deployed with `modal deploy fpis_pipeline.py`
straight from a working copy — nothing forces a commit — so the discipline has to be
deliberate:

> **Deploy and publish in the same motion.** If you change the pipeline and deploy it, push
> the change here too, before or with the deploy.

## Licence

`fpis_pipeline.py` is published under the **GNU Affero General Public License v3.0** — see
[`LICENSE`](./LICENSE).

Ultralytics YOLO is © Ultralytics and licensed AGPL-3.0. Ekatan uses it under that licence
rather than under a commercial one; publishing this file is the consequence, and an
intentional choice rather than an oversight.
