"""
spaCy NLP extractor for government planning documents — v3 (quality pass).

Note on gazetteer matching:
  Short abbreviations like 'orr' and 'nimz' use word-boundary regex to avoid
  false substring matches inside words such as 'corridor' or 'nimzone'.

Extracts structured signals from raw PDF text:
  - Location names → geocoded lat/lon
  - Project type (metro, highway, industrial, park, it_sez, etc.)
  - Budget allocation (crores)
  - Estimated completion year
  - Impact radius based on project type

Quality improvements in this version:
  - Administrative sentence blocklist — filters out non-project text
  - Stronger project validation gate (type + location OR budget OR year)
  - Gazetteer-first location resolution, then spaCy; context window (prev/next)
  - Unknown location → project SKIPPED (not saved)
  - Meaningful project name generation (noun-phrase or template)
  - Deduplication key: (project_type, location, completion_year)
  - Per-sentence accept/skip logging

Compatibility:
  - ExtractedProject dataclass unchanged
  - extract_projects() signature unchanged
  - No changes to ingest.py, DB schema, or API routes

Usage:
    from pipeline.extractor import extract_projects
    projects = extract_projects("document.pdf", text_content)
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

log = logging.getLogger("ingest.extractor")

# ── spaCy model ──────────────────────────────────────────────────────────── #
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    log.info("Downloading spaCy model en_core_web_sm …")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )
    _nlp = spacy.load("en_core_web_sm")

_geocoder = Nominatim(user_agent="landiq-nlp", timeout=10)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Constants / lookup tables
# ═══════════════════════════════════════════════════════════════════════════ #

# ── Impact radii by project type (km) ────────────────────────────────────── #
PROJECT_TYPE_RADII: dict[str, float] = {
    "metro":       5.0,
    "highway":     3.0,
    "industrial":  8.0,
    "it_sez":      6.0,
    "park":        2.0,
    "education":   2.0,
    "hospital":    2.0,
    "airport":    10.0,
    "other":       3.0,
}

# ── Keyword → project type mapping ───────────────────────────────────────── #
# NOTE: Order matters — more specific types must come before generic ones.
TYPE_KEYWORDS: dict[str, list[str]] = {
    "metro": [
        "metro", "mrts", "rapid transit", "metro rail", "metro corridor",
        "metro station", "railway", "rail corridor", "elevated rail",
        "metro extension", "metro phase",
    ],
    "it_sez": [
        "it sez", "it park", "it corridor", "information technology",
        "software park", "technopark", "cyberabad", "special economic zone",
        "smart city", "it hub",
    ],
    "industrial": [
        "industrial estate", "nimz", "pharma city", "industrial park",
        "manufacturing zone", "logistics hub", "logistics park",
        "industrial corridor", "industrial area",
    ],
    "highway": [
        "highway", "expressway", "national highway", "nh-", "nh ",
        "orr", "rrr", "ring road", "outer ring road", "flyover",
        "bypass road", "road widening", "grade separator",
    ],
    "park": [
        "eco park", "green space", "lake development", "eco zone",
        "botanical garden",
    ],
    "education": [
        "university", "iit", "iim", "education hub", "knowledge city",
    ],
    "hospital": [
        "hospital", "medical college", "health city", "aiims", "medical hub",
    ],
    "airport": [
        "airport", "aerodrome", "airstrip",
    ],
}

# ── Broader infrastructure gate ───────────────────────────────────────────── #
# A sentence must match at least one of these to be considered at all.
INFRA_GATE_KEYWORDS: list[str] = [
    # Transport
    "metro", "metro rail", "metro corridor", "railway", "airport",
    "highway", "expressway", "flyover", "ring road", "orr", "rrr",
    "national highway", "bypass", "grade separator", "mrts",
    # Industry / zones
    "industrial", "industrial park", "industrial corridor",
    "special economic zone", "sez", "it park", "it corridor", "it hub",
    "township", "smart city", "logistics", "pharma city", "nimz",
    # Social infra
    "hospital", "university", "iit", "iim", "medical college",
    # Construction verbs + nouns (only combined with above via validator)
    "construction", "corridor", "extension", "expansion", "development",
    "phase", "project", "infrastructure", "hub",
]

# ── Administrative word blocklist ─────────────────────────────────────────── #
# Sentences containing ANY of these are silently skipped.
ADMIN_BLOCKLIST: list[str] = [
    "proceedings",
    "commissioner",
    "office of",
    "layout charges",
    "file no",
    "file number",
    "memo no",
    "memo number",
    "government order",
    "g.o.",
    "go no",
    "annexure",
    "dated ",
    "subject:",
    "reference:",
    "ref:",
    "enclosure",
    "enclosed",
    "applicant",
    "approval is granted",
    "permission is granted",
    "layout permission",
    "development charges",
    "regulation",
    "penalty",
    "fee of",
    "fees of",
    "challan",
    "stamp duty",
    "registration",
    "vide",
    "as per rule",
    "the above mentioned",
    "with reference to",
    "is hereby",
    "is informed",
    "no objection",
    "noc",
    "building permission",
    "plinth area",
    "floor space index",
    "fsi",
    "setback",
    "zoning regulation",
    "sanctioned plan",
    "sanctioned layout",
    "occupancy certificate",
    "completion certificate",
]

# ── Human-readable type labels for name generation ─────────────────────────── #
_TYPE_LABELS: dict[str, str] = {
    "metro":      "Metro Rail",
    "it_sez":     "IT Park",
    "industrial": "Industrial Corridor",
    "highway":    "Road",
    "park":       "Eco Park",
    "education":  "Education Hub",
    "hospital":   "Hospital",
    "airport":    "Airport",
    "other":      "Development",
}

# ── Known Hyderabad area gazetteer ───────────────────────────────────────── #
# Short / ambiguous keys that need word-boundary matching instead of plain
# substring search (avoids 'orr' matching inside 'corridor', etc.)
_GAZETTEER_WORD_BOUNDARY_KEYS: frozenset[str] = frozenset({
    "orr", "rrr", "nimz", "sez",
})

HYDERABAD_GAZETTEER: dict[str, tuple[float, float]] = {
    "hitech city":        (17.4484, 78.3805),
    "gachibowli":         (17.4401, 78.3376),
    "kokapet":            (17.4058, 78.3242),
    "financial district": (17.4219, 78.3393),
    "madhapur":           (17.4485, 78.3785),
    "kondapur":           (17.4632, 78.3607),
    "tellapur":           (17.4671, 78.2907),
    "mokila":             (17.4793, 78.2611),
    "patancheru":         (17.5301, 78.2657),
    "kompally":           (17.5395, 78.4790),
    "nizampet":           (17.5082, 78.4025),
    "miyapur":            (17.4949, 78.3675),
    "uppal":              (17.4052, 78.5593),
    "ghatkesar":          (17.4401, 78.6877),
    "shamshabad":         (17.2457, 78.4203),
    "adibatla":           (17.2985, 78.6044),
    "tukkuguda":          (17.2531, 78.5049),
    "shadnagar":          (17.0689, 78.1919),
    "kothur":             (17.0490, 78.3143),
    "yadadri":            (17.0893, 79.0196),
    "bibinagar":          (17.4777, 78.8920),
    "sangareddy":         (17.6198, 77.9999),
    "medchal":            (17.6272, 78.5516),
    "bachupally":         (17.5343, 78.4042),
    "narsingi":           (17.3947, 78.3325),
    "rajendranagar":      (17.3196, 78.3915),
    "banjara hills":      (17.4130, 78.4392),
    "jubilee hills":      (17.4299, 78.4066),
    "kukatpally":         (17.4944, 78.4045),
    "ameerpet":           (17.4356, 78.4483),
    "secunderabad":       (17.4436, 78.4985),
    "lb nagar":           (17.3483, 78.5477),
    "hafeezpet":          (17.4801, 78.3600),
    "pocharam":           (17.5139, 78.6229),
    "yadagirigutta":      (17.5783, 79.0844),
    "hyderabad":          (17.3850, 78.4867),
    "raidurg":            (17.4291, 78.3842),
    "nanakramguda":       (17.4210, 78.3534),
    "manikonda":          (17.4050, 78.3920),
    "puppalaguda":        (17.3876, 78.3833),
    "gandipet":           (17.3680, 78.3182),
    "tolichowki":         (17.3979, 78.4243),
    "mehdipatnam":        (17.3960, 78.4330),
    "attapur":            (17.3678, 78.4289),
    "himayatnagar":       (17.4070, 78.4762),
    "dilsukhnagar":       (17.3684, 78.5260),
    "vanasthalipuram":    (17.3418, 78.5463),
    "nagole":             (17.3770, 78.5605),
    "boduppal":           (17.4166, 78.5947),
    "peerzadiguda":       (17.4299, 78.6218),
    "alwal":              (17.5004, 78.5089),
    "yapral":             (17.5142, 78.5310),
    # Corridors / zones (kept for keyword matching)
    "orr":                (17.3500, 78.4000),
    "outer ring road":    (17.3500, 78.4000),
    "pharma city":        (17.2000, 78.5000),
    "nimz":               (17.5301, 78.2657),
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  ExtractedProject dataclass  (unchanged — required for compatibility)
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ExtractedProject:
    source_document: str
    project_name: str
    project_type: str
    location_text: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    budget_crore: Optional[float] = None
    estimated_completion_year: Optional[int] = None
    impact_radius_km: float = 5.0


# ═══════════════════════════════════════════════════════════════════════════ #
#  1. Administrative blocklist filter
# ═══════════════════════════════════════════════════════════════════════════ #

def _is_administrative(text: str) -> tuple[bool, str]:
    """
    Return (True, matched_word) if the sentence looks like administrative
    government text (proceedings, layout charges, G.O., etc.).
    Return (False, '') if it is safe to process.
    """
    tl = text.lower()
    for word in ADMIN_BLOCKLIST:
        if word in tl:
            return True, word
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════ #
#  2. Infrastructure gate
# ═══════════════════════════════════════════════════════════════════════════ #

def _passes_infra_gate(text: str) -> bool:
    """Return True if the sentence contains at least one infra keyword."""
    tl = text.lower()
    return any(kw in tl for kw in INFRA_GATE_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════════ #
#  3. Project type detection
# ═══════════════════════════════════════════════════════════════════════════ #

def _detect_project_type(text: str) -> str:
    tl = text.lower()
    for ptype, keywords in TYPE_KEYWORDS.items():
        if any(kw in tl for kw in keywords):
            return ptype
    return "other"


# ═══════════════════════════════════════════════════════════════════════════ #
#  4. Budget extraction
# ═══════════════════════════════════════════════════════════════════════════ #

# Covers all common government budget formats:
#   ₹250 crore  ₹2,500 crore  Rs.1200 crore  Rs 800 Cr
#   INR 450 crore  1,250 crores  Rs.1,250 Cr  250 Cr
_BUDGET_PATTERNS: list[str] = [
    # With explicit currency prefix (most reliable)
    r"(?:₹|rs\.?\s*|inr\s*)(\d[\d,\.]*)(?:\s*(?:crores?|cr\.?))\b",
    # Bare number followed by crore/cr
    r"\b(\d[\d,\.]{1,})\s*(?:crores?|cr\.?)\b",
]


def _extract_budget(text: str) -> Optional[float]:
    """Extract rupee budget in crores from text."""
    for pattern in _BUDGET_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val_str = m.group(1).replace(",", "").replace(" ", "")
            try:
                val = float(val_str)
                if 1.0 <= val <= 10_000_000:
                    return val
            except ValueError:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════ #
#  5. Completion year extraction
# ═══════════════════════════════════════════════════════════════════════════ #

_current_year = datetime.utcnow().year
_YEAR_MIN = _current_year
_YEAR_MAX = _current_year + 25

_YEAR_PATTERNS: list[str] = [
    # Explicit phrase patterns (highest confidence)
    r"(?:target(?:ed)?\s+(?:year|by|for)"
    r"|completion\s+(?:by|in|year)"
    r"|complete[d\s]+by"
    r"|expected\s+(?:by|in|to\s+be\s+completed)"
    r"|likely\s+operational\s+by"
    r"|operational\s+by"
    r"|(?:by|before|in)\s+(?:the\s+year\s+)?)"
    r"\s*(20\d{2})\b",
    # Bare 4-digit future year
    r"\b(20[2-4]\d)\b",
]


def _extract_year(text: str) -> Optional[int]:
    """Extract completion year from text."""
    for pattern in _YEAR_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                year_str = m.group(1) if m.lastindex else m.group(0)
                year = int(year_str)
                if _YEAR_MIN <= year <= _YEAR_MAX:
                    return year
            except (ValueError, IndexError):
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════ #
#  6. Location resolution
# ═══════════════════════════════════════════════════════════════════════════ #

def _gazetteer_key_in_text(key: str, text_lower: str) -> bool:
    """
    Check whether a gazetteer key appears in text.
    Short/ambiguous keys (orr, rrr, nimz, sez) require word boundaries
    to avoid false hits inside longer words like 'corridor'.
    """
    if key in _GAZETTEER_WORD_BOUNDARY_KEYS:
        return bool(re.search(r"\b" + re.escape(key) + r"\b", text_lower))
    return key in text_lower


def _find_location_in_gazetteer(text: str) -> Optional[str]:
    """
    Scan text for any gazetteer key (longest match wins).
    Returns the matched key (title-cased) or None.
    """
    tl = text.lower()
    best_key: Optional[str] = None
    for key in HYDERABAD_GAZETTEER:
        if _gazetteer_key_in_text(key, tl):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    return best_key.title() if best_key else None


def _spacy_locations(text: str) -> list[str]:
    """Return GPE/LOC/FAC entity texts from spaCy for the given text."""
    doc = _nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]


def _resolve_location(
    sent_text: str,
    prev_sent: str,
    next_sent: str,
) -> Optional[str]:
    """
    Resolve the best location for a sentence using a 3-level strategy:

    1. Gazetteer match in current sentence          (highest confidence)
    2. Gazetteer match in previous or next sentence (context window)
    3. spaCy NER in current sentence
    4. spaCy NER in previous or next sentence

    Returns location string or None if nothing found.
    'Unknown' is never returned — callers receive None and SKIP the project.
    """
    # Level 1 — Gazetteer in current sentence
    loc = _find_location_in_gazetteer(sent_text)
    if loc:
        return loc

    # Level 2 — Gazetteer in context window
    for ctx in (prev_sent, next_sent):
        if ctx:
            loc = _find_location_in_gazetteer(ctx)
            if loc:
                return loc

    # Level 3 — spaCy NER in current sentence
    ner_locs = _spacy_locations(sent_text)
    if ner_locs:
        return ner_locs[0]

    # Level 4 — spaCy NER in context window
    for ctx in (prev_sent, next_sent):
        if ctx:
            ner_locs = _spacy_locations(ctx)
            if ner_locs:
                return ner_locs[0]

    return None   # no location found anywhere


def _geocode_location(location_text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Try gazetteer first, then Nominatim fallback.
    NEVER raises — returns (None, None) on any failure.
    """
    lt_lower = location_text.lower().strip()

    # Gazetteer lookup (longest key wins, with word-boundary guard for short keys)
    best_key: Optional[str] = None
    for key in HYDERABAD_GAZETTEER:
        if _gazetteer_key_in_text(key, lt_lower):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key:
        return HYDERABAD_GAZETTEER[best_key]

    # Nominatim fallback (rate-limited)
    try:
        query = f"{location_text}, Hyderabad, Telangana, India"
        geo = _geocoder.geocode(query, timeout=10)
        if geo:
            return geo.latitude, geo.longitude
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        log.debug("Geocoding timeout/service error for '%s': %s", location_text, exc)
    except Exception as exc:
        log.debug("Geocoding unexpected error for '%s': %s", location_text, exc)

    return None, None


# ═══════════════════════════════════════════════════════════════════════════ #
#  7. Project name generation
# ═══════════════════════════════════════════════════════════════════════════ #

# Key phrases that, when found in a sentence, suggest a natural project name
_NAME_PATTERNS: list[tuple[str, str]] = [
    # ORR / RRR
    (r"\borr\b", "ORR"),
    (r"\brrr\b", "RRR"),
    (r"outer\s+ring\s+road", "Outer Ring Road"),
    # Metro
    (r"metro\s+rail", "Metro Rail"),
    (r"metro\s+phase\s*[–\-]?\s*(\w+)", r"Metro Phase \1"),
    (r"metro\s+corridor", "Metro Corridor"),
    (r"metro\s+extension", "Metro Extension"),
    # Roads
    (r"national\s+highway\s*([\w\-]+)", r"NH\1"),
    (r"elevated\s+corridor", "Elevated Corridor"),
    (r"flyover", "Flyover"),
    (r"bypass\s+road", "Bypass Road"),
    (r"expressway", "Expressway"),
    # Industrial
    (r"pharma\s+city", "Pharma City"),
    (r"nimz", "NIMZ"),
    (r"industrial\s+park", "Industrial Park"),
    (r"industrial\s+corridor", "Industrial Corridor"),
    (r"logistics\s+hub", "Logistics Hub"),
    (r"logistics\s+park", "Logistics Park"),
    # IT / SEZ
    (r"it\s+sez", "IT SEZ"),
    (r"it\s+park", "IT Park"),
    (r"it\s+corridor", "IT Corridor"),
    (r"software\s+park", "Software Park"),
    (r"cyberabad", "Cyberabad"),
    (r"smart\s+city", "Smart City"),
    (r"special\s+economic\s+zone", "SEZ"),
    # Hospitals / Education
    (r"medical\s+college", "Medical College"),
    (r"health\s+city", "Health City"),
    (r"aiims", "AIIMS"),
    (r"university", "University"),
    # Airport
    (r"airport\s+expansion", "Airport Expansion"),
    (r"aerodrome", "Aerodrome"),
    # Township
    (r"township\s+development", "Township Development"),
    (r"township", "Township"),
]

# Action verbs that describe infrastructure activity
_ACTION_VERBS: list[str] = [
    "construction", "development", "expansion", "extension",
    "widening", "upgrade", "upgradation", "establishment",
    "setting up", "phase", "corridor",
]


def _build_project_name(sent_text: str, ptype: str, location_text: str) -> str:
    """
    Build a meaningful project name using three strategies, in priority order:

    1. Extract a known infrastructure phrase from the sentence.
    2. Try to grab a noun phrase containing an action verb via spaCy.
    3. Fall back to a template: '<TypeLabel> at <Location>'.
    """
    tl = sent_text.lower()

    # ── Strategy 1: pattern match for known infra phrases ─────────────────
    for raw_pattern, label in _NAME_PATTERNS:
        m = re.search(raw_pattern, tl, re.IGNORECASE)
        if m:
            # If label has backreferences, apply them
            try:
                matched_label = re.sub(raw_pattern, label, m.group(0), flags=re.IGNORECASE).strip()
            except Exception:
                matched_label = label
            if location_text and location_text.lower() not in ("unknown", "hyderabad"):
                return f"{location_text} {matched_label}".strip()
            return matched_label.strip()

    # ── Strategy 2: spaCy noun-phrase containing an action verb ───────────
    doc = _nlp(sent_text[:300])  # cap for performance
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower()
        if any(verb in chunk_lower for verb in _ACTION_VERBS):
            name = chunk.text.strip()
            if len(name) > 8:
                return name[:120]

    # ── Strategy 3: template fallback ─────────────────────────────────────
    type_label = _TYPE_LABELS.get(ptype, "Development")
    if location_text and location_text.lower() not in ("unknown", "hyderabad"):
        return f"{location_text} {type_label}"
    return f"{type_label} Project"


# ═══════════════════════════════════════════════════════════════════════════ #
#  8. Project validation gate
# ═══════════════════════════════════════════════════════════════════════════ #

def _is_valid_project(
    ptype: str,
    location_text: Optional[str],
    budget: Optional[float],
    year: Optional[int],
) -> tuple[bool, str]:
    """
    A candidate is a real project only if it has:
      - a concrete project type (not 'other')
      OR
      - ptype == 'other' but has BOTH a real location AND (budget OR year)

    Additionally: a real location (not None) is ALWAYS required.

    Returns (valid, reason_if_invalid).
    """
    if location_text is None:
        return False, "no location found in sentence or context window"

    has_signals = budget is not None or year is not None
    concrete_type = ptype != "other"

    if concrete_type and (location_text or has_signals):
        return True, ""

    if not concrete_type and not has_signals:
        return False, f"type='other' and no budget/year signal"

    if not concrete_type and not location_text:
        return False, "type='other' and no location"

    # concrete_type is False but has signals — accept
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main extraction function  (signature unchanged)
# ═══════════════════════════════════════════════════════════════════════════ #

def extract_projects(source_document: str, text: str) -> list[ExtractedProject]:
    """
    Main extraction function. Processes raw PDF text and returns a list
    of ExtractedProject objects with structured fields.

    Compatibility: signature and return type are unchanged.
    """
    # Cap to 1M chars to avoid memory issues
    working_text = text[:1_000_000]

    doc = _nlp(working_text)
    projects: list[ExtractedProject] = []

    # Split into sentences; skip very short ones
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 30]
    log.info("  Sentences to evaluate: %d", len(sentences))

    # Counters for summary log
    cnt_admin    = 0
    cnt_no_infra = 0
    cnt_no_loc   = 0
    cnt_invalid  = 0
    cnt_accepted = 0
    spacy_locs   = 0
    gazet_locs   = 0

    for idx, sent_text in enumerate(sentences):
        # ── Guard 1: administrative blocklist ─────────────────────────────
        is_admin, admin_word = _is_administrative(sent_text)
        if is_admin:
            log.debug("  SKIP (admin word '%s'): %s", admin_word, sent_text[:80])
            cnt_admin += 1
            continue

        # ── Guard 2: must contain at least one infra keyword ──────────────
        if not _passes_infra_gate(sent_text):
            log.debug("  SKIP (no infra keyword): %s", sent_text[:80])
            cnt_no_infra += 1
            continue

        # ── Extract signals ────────────────────────────────────────────────
        ptype  = _detect_project_type(sent_text)
        budget = _extract_budget(sent_text)
        year   = _extract_year(sent_text)

        # ── Location resolution with context window ────────────────────────
        prev_sent = sentences[idx - 1] if idx > 0 else ""
        next_sent = sentences[idx + 1] if idx < len(sentences) - 1 else ""

        location_text = _resolve_location(sent_text, prev_sent, next_sent)

        # Track how location was found (for the summary log)
        if location_text:
            if _find_location_in_gazetteer(sent_text):
                gazet_locs += 1
            else:
                spacy_locs += 1

        # ── Validation gate ────────────────────────────────────────────────
        valid, reason = _is_valid_project(ptype, location_text, budget, year)
        if not valid:
            log.debug("  SKIP (validation): %s | reason: %s", sent_text[:80], reason)
            if location_text is None:
                cnt_no_loc += 1
            else:
                cnt_invalid += 1
            continue

        # ── Build meaningful project name ──────────────────────────────────
        project_name = _build_project_name(sent_text, ptype, location_text or "")

        # ── Geocode ────────────────────────────────────────────────────────
        lat, lon = _geocode_location(location_text)  # type: ignore[arg-type]

        cnt_accepted += 1
        log.info("  ACCEPT sentence: %s", sent_text[:100])
        log.info("    Project type  : %s", ptype)
        log.info("    Project name  : %s", project_name)
        log.info("    Location      : %s (lat=%.4f, lon=%.4f)",
                 location_text,
                 lat if lat is not None else 0.0,
                 lon if lon is not None else 0.0)
        log.info("    Budget        : %s Cr", budget)
        log.info("    Completion yr : %s", year)

        projects.append(ExtractedProject(
            source_document=source_document,
            project_name=project_name,
            project_type=ptype,
            location_text=location_text,    # type: ignore[arg-type]
            lat=lat,
            lon=lon,
            budget_crore=budget,
            estimated_completion_year=year,
            impact_radius_km=PROJECT_TYPE_RADII.get(ptype, 3.0),
        ))

    # ── Summary log ────────────────────────────────────────────────────────
    log.info(
        "  Filter summary — admin: %d | no-infra: %d | no-location: %d "
        "| failed-validation: %d | accepted: %d",
        cnt_admin, cnt_no_infra, cnt_no_loc, cnt_invalid, cnt_accepted,
    )
    log.info(
        "  Location resolution — gazetteer: %d | spaCy: %d",
        gazet_locs, spacy_locs,
    )

    # ── Deduplication: (project_type, location, completion_year) ──────────
    # This is stricter than the old (location, type) key:
    # two entries for the same project at the same location and year merge.
    seen: set[tuple] = set()
    unique: list[ExtractedProject] = []
    for p in projects:
        key = (
            p.project_type,
            p.location_text.lower(),
            p.estimated_completion_year,
        )
        if key not in seen:
            seen.add(key)
            unique.append(p)

    log.info(
        "  Raw project candidates: %d  |  After deduplication: %d",
        len(projects), len(unique),
    )
    return unique
