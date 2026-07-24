"""
Background document ingestion pipeline for LandIQ.

Replaces Celery + Redis with FastAPI BackgroundTasks.
Runs in-process as a background task when a PDF is uploaded.

Flow:
    1. Load PDF with PyMuPDF → extract pages (multi-method + OCR fallback)
    2. Clean & split into text chunks → store in document_chunks (Postgres FTS)
    3. Run spaCy NLP extractor → extract project signals
    4. Geocode locations → save DevelopmentProject rows
    5. Recalculate Development Index for all affected areas via PostGIS
    6. Update zone labels and run ML predictions on impacted areas

Usage (from FastAPI route):
    from pipeline.ingest import process_document
    background_tasks.add_task(process_document, filename, file_bytes, db_url)
"""

import os
import re
import sys
import time
import unicodedata
import subprocess
import logging
from typing import Optional

import fitz                               # PyMuPDF
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from geoalchemy2.shape import from_shape
from shapely.geometry import Point

from pipeline.extractor import extract_projects
from db.models import DocumentChunk, DevelopmentProject
from db.spatial import update_development_index

# ── Logging ──────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="[ingest] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ── Chunking parameters ───────────────────────────────────────────────────── #
CHUNK_SIZE   = 1000   # characters per chunk
CHUNK_OVERLAP = 200   # overlap between consecutive chunks
MIN_CHUNK_LEN = 30    # discard chunks shorter than this
OCR_MIN_CHARS = 50    # pages with fewer chars trigger OCR fallback


# ═══════════════════════════════════════════════════════════════════════════ #
#  Session factory
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_session(database_url: Optional[str] = None):
    url = database_url or os.environ.get(
        "DATABASE_URL", "postgresql://landiq:landiq@localhost:5432/landiq"
    )
    engine = create_engine(url, pool_pre_ping=True)
    return sessionmaker(bind=engine)()


# ═══════════════════════════════════════════════════════════════════════════ #
#  OCR helpers
# ═══════════════════════════════════════════════════════════════════════════ #

def _ensure_easyocr():
    """Auto-install EasyOCR and its dependencies if not present."""
    try:
        import easyocr  # noqa: F401
    except ImportError:
        log.warning("EasyOCR not found — installing automatically …")
        for pkg in ("easyocr", "opencv-python-headless", "Pillow"):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        log.info("EasyOCR installation complete.")


_ocr_reader = None   # lazy singleton


def _get_ocr_reader():
    """Return a cached EasyOCR reader (English only)."""
    global _ocr_reader
    if _ocr_reader is None:
        _ensure_easyocr()
        import easyocr
        log.info("Initialising EasyOCR reader (first call may take a moment) …")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        log.info("EasyOCR reader ready.")
    return _ocr_reader


def _ocr_page(page: fitz.Page) -> str:
    """
    Render a PDF page to a PIL image and extract text via EasyOCR.
    Returns the joined OCR text.
    """
    try:
        reader = _get_ocr_reader()
        # Render at 2× zoom for better OCR quality
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # EasyOCR accepts raw bytes
        results = reader.readtext(img_bytes, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception as exc:
        log.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════ #
#  Text extraction (multi-method with OCR fallback)
# ═══════════════════════════════════════════════════════════════════════════ #

def _extract_page_text(page: fitz.Page) -> tuple[str, bool]:
    """
    Try multiple embedded-text extraction methods in priority order.
    Falls back to EasyOCR if all methods yield < OCR_MIN_CHARS characters.

    Returns (text, ocr_used).
    """
    # ── 1. Standard text extraction ────────────────────────────────────────
    candidates: list[str] = []

    try:
        t = page.get_text("text") or ""
        candidates.append(t)
    except Exception:
        candidates.append("")

    # ── 2. Blocks fallback ─────────────────────────────────────────────────
    if not candidates[0].strip():
        try:
            blocks = page.get_text("blocks") or []
            t = "\n".join(
                b[4] for b in blocks
                if isinstance(b, (list, tuple)) and len(b) > 4 and isinstance(b[4], str)
            )
            candidates.append(t)
        except Exception:
            candidates.append("")
    else:
        candidates.append("")

    # ── 3. Words fallback ──────────────────────────────────────────────────
    if not any(c.strip() for c in candidates):
        try:
            words = page.get_text("words") or []
            t = " ".join(
                w[4] for w in words
                if isinstance(w, (list, tuple)) and len(w) > 4 and isinstance(w[4], str)
            )
            candidates.append(t)
        except Exception:
            candidates.append("")
    else:
        candidates.append("")

    # ── 4. Dict fallback ───────────────────────────────────────────────────
    if not any(c.strip() for c in candidates):
        try:
            d = page.get_text("dict") or {}
            parts = []
            for blk in d.get("blocks", []):
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        parts.append(span.get("text", ""))
            candidates.append(" ".join(parts))
        except Exception:
            candidates.append("")
    else:
        candidates.append("")

    # Pick the longest result
    best = max(candidates, key=lambda s: len(s.strip()))

    # ── 5. OCR fallback ────────────────────────────────────────────────────
    ocr_used = False
    if len(best.strip()) < OCR_MIN_CHARS:
        log.info("  Page %d: embedded text too short (%d chars) — running OCR …",
                 page.number + 1, len(best.strip()))
        ocr_text = _ocr_page(page)
        if len(ocr_text.strip()) > len(best.strip()):
            best = ocr_text
            ocr_used = True

    return best, ocr_used


# ═══════════════════════════════════════════════════════════════════════════ #
#  Text cleaning & chunking
# ═══════════════════════════════════════════════════════════════════════════ #

# Characters we consider "garbage" (non-printable, control chars, replacement char)
_CTRL_RE  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SP = re.compile(r"[ \t]{2,}")


def _clean_text(text: str) -> str:
    """
    Normalise and clean raw extracted / OCR text:
      - Decompose unicode
      - Remove non-printable / control characters
      - Collapse repeated spaces and newlines
      - Strip leading/trailing whitespace
    """
    # Unicode normalisation (NFKC: compatibility + composed form)
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters
    text = _CTRL_RE.sub(" ", text)
    # Collapse runs of whitespace
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _make_chunks(text: str) -> list[str]:
    """
    Split cleaned text into overlapping chunks of ~CHUNK_SIZE characters.
    Skips chunks shorter than MIN_CHUNK_LEN.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        # Advance by (CHUNK_SIZE - CHUNK_OVERLAP) so we get the overlap window
        step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
        start += step
    return chunks


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main PDF-to-chunks function
# ═══════════════════════════════════════════════════════════════════════════ #

def _pdf_to_chunks(
    filename: str,
    pdf_bytes: bytes,
) -> tuple[list[tuple[int, str]], dict]:
    """
    Open PDF and extract text from every page using multi-method + OCR.

    Returns:
        chunks  : list of (global_chunk_index, chunk_text) tuples
        pdf_stats: dict with page / char / OCR counts for logging
    """
    all_chunks: list[tuple[int, str]] = []
    pdf_stats = {
        "pages": 0,
        "embedded_chars": 0,
        "ocr_pages": 0,
    }
    global_chunk_idx = 0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        pdf_stats["pages"] = total_pages
        log.info("Processing: %s", filename)
        log.info("PDF Pages: %d", total_pages)

        for page_num, page in enumerate(doc):
            page_label = page_num + 1
            try:
                page_text, ocr_used = _extract_page_text(page)

                char_count = len(page_text.strip())
                log.info(
                    "  Page %d extracted %d characters%s",
                    page_label, char_count,
                    " [OCR]" if ocr_used else "",
                )
                if char_count > 0:
                    log.info(
                        "  First 300 chars: %s",
                        page_text.strip()[:300].replace("\n", " "),
                    )
                else:
                    log.info("  No embedded text found.")

                if ocr_used:
                    pdf_stats["ocr_pages"] += 1
                pdf_stats["embedded_chars"] += char_count

                # Clean and chunk this page's text
                cleaned = _clean_text(page_text)
                page_chunks = _make_chunks(cleaned)

                for chunk_text in page_chunks:
                    all_chunks.append((global_chunk_idx, chunk_text))
                    global_chunk_idx += 1

            except Exception as exc:
                log.error(
                    "  Page %d failed — skipping. Error: %s",
                    page_label, exc,
                )
                continue   # Per-page isolation: one bad page never stops the pipeline

    log.info("Chunks Created: %d", len(all_chunks))
    return all_chunks, pdf_stats


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main ingestion entry point
# ═══════════════════════════════════════════════════════════════════════════ #

def process_document(
    filename: str,
    file_bytes: bytes,
    database_url: Optional[str] = None,
) -> dict:
    """
    Full ingestion pipeline — called as a FastAPI BackgroundTask.

    Returns a summary dict (mostly for logging; caller doesn't await it).
    """
    start_time = time.monotonic()

    db = _make_session(database_url)
    stats = {
        "filename":          filename,
        "pages":             0,
        "embedded_chars":    0,
        "ocr_pages":         0,
        "chunks_created":    0,
        "chunks_stored":     0,
        "projects_extracted": 0,
        "projects_saved":    0,
        "processing_time_s": 0.0,
    }

    try:
        # ── Step 1: Extract text from PDF ────────────────────────────────── #
        chunks, pdf_stats = _pdf_to_chunks(filename, file_bytes)
        stats.update(pdf_stats)
        stats["chunks_created"] = len(chunks)

        if not chunks:
            log.warning("No readable text extracted from %s — aborting ingestion.", filename)
            stats["processing_time_s"] = round(time.monotonic() - start_time, 2)
            _print_summary(stats)
            return stats

        # ── Step 2: Store chunks (Postgres FTS) — ALWAYS, regardless of NLP #
        stored = 0
        for idx, chunk_text in chunks:
            try:
                db.add(DocumentChunk(
                    source_document=filename,
                    chunk_index=idx,
                    content=chunk_text,
                ))
                stored += 1
            except Exception as exc:
                log.warning("Failed to store chunk %d: %s", idx, exc)

        db.commit()
        stats["chunks_stored"] = stored
        log.info("Chunks Stored: %d", stored)

        # ── Step 3: Concatenate full text for NLP ────────────────────────── #
        full_text = "\n\n".join(chunk for _, chunk in chunks)
        log.info("Total extracted text length: %d characters", len(full_text))

        # ── Step 4: NLP extraction ───────────────────────────────────────── #
        log.info("Running NLP extraction …")
        extracted = extract_projects(filename, full_text)
        stats["projects_extracted"] = len(extracted)
        log.info("Projects Extracted: %d", len(extracted))

        # ── Step 5: Save DevelopmentProject rows ─────────────────────────── #
        saved = 0
        for proj in extracted:
            try:
                location = None
                if proj.lat is not None and proj.lon is not None:
                    location = from_shape(Point(proj.lon, proj.lat), srid=4326)

                db_proj = DevelopmentProject(
                    source_document=proj.source_document,
                    project_name=proj.project_name[:255],
                    project_type=proj.project_type,
                    location_text=proj.location_text[:255],
                    location=location,
                    budget_crore=proj.budget_crore,
                    estimated_completion_year=proj.estimated_completion_year,
                    impact_radius_km=proj.impact_radius_km,
                )
                db.add(db_proj)
                saved += 1
            except Exception as exc:
                log.warning("Failed to save project '%s': %s", proj.project_name[:60], exc)

        db.commit()
        stats["projects_saved"] = saved
        log.info("Projects Saved: %d", saved)

        # ── Step 6: Recalculate Development Index ────────────────────────── #
        if saved > 0:
            try:
                update_development_index(db)
                log.info("Development Index updated.")
            except Exception as exc:
                log.warning("Development Index update failed: %s", exc)

            # ── Step 7: Rerun ML predictions ─────────────────────────────── #
            try:
                _refresh_area_predictions(db)
                log.info("Area predictions refreshed.")
            except Exception as exc:
                log.warning("Area predictions refresh failed: %s", exc)
        else:
            log.info("No projects saved — skipping index / prediction refresh.")

    except Exception as exc:
        db.rollback()
        log.error("FATAL ERROR processing %s: %s", filename, exc)
        raise
    finally:
        stats["processing_time_s"] = round(time.monotonic() - start_time, 2)
        _print_summary(stats)
        db.close()

    return stats


# ═══════════════════════════════════════════════════════════════════════════ #
#  Debug summary printer
# ═══════════════════════════════════════════════════════════════════════════ #

def _print_summary(stats: dict) -> None:
    """Print a human-readable ingestion summary to the log."""
    sep = "-" * 50
    log.info(sep)
    log.info("INGESTION SUMMARY")
    log.info(sep)
    log.info("Filename            : %s", stats.get("filename", "?"))
    log.info("Pages               : %d", stats.get("pages", 0))
    log.info("Embedded text chars : %d", stats.get("embedded_chars", 0))
    log.info("OCR pages           : %d", stats.get("ocr_pages", 0))
    log.info("Chunks created      : %d", stats.get("chunks_created", 0))
    log.info("Chunks stored       : %d", stats.get("chunks_stored", 0))
    log.info("Projects extracted  : %d", stats.get("projects_extracted", 0))
    log.info("Projects saved      : %d", stats.get("projects_saved", 0))
    log.info("Processing time     : %.1f s", stats.get("processing_time_s", 0.0))
    log.info(sep)

    # Why-did-it-fail diagnostics
    if stats.get("embedded_chars", 0) == 0 and stats.get("ocr_pages", 0) == 0:
        log.warning("No embedded text detected and OCR was not triggered.")
    if stats.get("chunks_created", 0) == 0:
        log.warning("Chunk generation produced 0 chunks — PDF may be empty or corrupt.")
    if stats.get("projects_extracted", 0) == 0 and stats.get("chunks_created", 0) > 0:
        log.info(
            "Zero projects extracted despite %d chunks — "
            "document may not contain infrastructure-related content.",
            stats["chunks_created"],
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  ML prediction refresh (unchanged logic, wrapped in per-step try/except)
# ═══════════════════════════════════════════════════════════════════════════ #

def _refresh_area_predictions(db) -> None:
    """
    Refresh ML predictions for all areas after a new document is ingested.
    Only runs if the predictor model is available.
    """
    predictor_path = os.path.join("models", "predictor.joblib")
    classifier_path = os.path.join("models", "classifier.joblib")

    if not os.path.exists(predictor_path) or not os.path.exists(classifier_path):
        return

    import joblib
    from db.models import Area, PricePrediction

    predictor  = joblib.load(predictor_path)
    classifier = joblib.load(classifier_path)

    areas = db.query(Area).all()
    for area in areas:
        area_dict = {
            "current_price_sqft":         area.current_price_sqft or 0,
            "price_cagr_3yr":             area.price_cagr_3yr or 0,
            "population_growth_rate":     area.population_growth_rate or 0,
            "distance_to_city_center_km": area.distance_to_city_center_km or 0,
            "distance_to_metro_km":       area.distance_to_metro_km or 0,
            "distance_to_highway_km":     area.distance_to_highway_km or 0,
            "development_index":          area.development_index or 0,
            "num_upcoming_projects":      area.num_upcoming_projects or 0,
        }

        # Reclassify zone
        zone = classifier.predict_one(area_dict)
        area_dict["zone_label"] = zone

        # Get predictions
        preds = predictor.predict_one(area_dict)

        # Upsert prediction row
        existing = (
            db.query(PricePrediction)
            .filter(PricePrediction.area_id == area.id)
            .first()
        )
        if existing:
            existing.predicted_appreciation_1yr = preds["1yr"]
            existing.predicted_appreciation_3yr = preds["3yr"]
            existing.predicted_appreciation_5yr = preds["5yr"]
            existing.confidence_lower           = preds["lower"]
            existing.confidence_upper           = preds["upper"]
            existing.model_version              = "xgb-v1"
        else:
            db.add(PricePrediction(
                area_id=area.id,
                predicted_appreciation_1yr=preds["1yr"],
                predicted_appreciation_3yr=preds["3yr"],
                predicted_appreciation_5yr=preds["5yr"],
                confidence_lower=preds["lower"],
                confidence_upper=preds["upper"],
                model_version="xgb-v1",
            ))

    db.commit()
