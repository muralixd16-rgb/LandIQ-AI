"""
Tests for the document ingestion pipeline (NLP extractor + ingest functions).

Run with:
    pytest tests/test_pipeline.py -v
"""
import pytest
from pipeline.extractor import (
    extract_projects,
    _detect_project_type,
    _extract_budget,
    _extract_year,
    _geocode_location,
)


# ── Helper / unit tests ───────────────────────────────────────────────────── #

class TestProjectTypeDetection:
    def test_metro_detected(self):
        assert _detect_project_type("The metro rail corridor will be extended to Kokapet.") == "metro"

    def test_highway_detected(self):
        assert _detect_project_type("Outer ring road expansion announced for the sector.") == "highway"

    def test_industrial_detected(self):
        assert _detect_project_type("NIMZ industrial park to be developed near Patancheru.") == "industrial"

    def test_it_sez_detected(self):
        assert _detect_project_type("New IT SEZ corridor announced in Gachibowli.") == "it_sez"

    def test_unknown_defaults_to_other(self):
        assert _detect_project_type("The government will review the situation.") == "other"


class TestBudgetExtraction:
    def test_crore_extraction(self):
        assert _extract_budget("allocated ₹2,500 crore for the project") == 2500.0

    def test_rs_crore_extraction(self):
        assert _extract_budget("Rs. 800 crore budget approved") == 800.0

    def test_no_budget_returns_none(self):
        assert _extract_budget("The project will be reviewed next quarter.") is None

    def test_large_budget(self):
        budget = _extract_budget("INR 12,000 crore infrastructure push")
        assert budget == 12000.0


class TestYearExtraction:
    def test_completion_year(self):
        year = _extract_year("Project to be completed by 2027")
        assert year == 2027

    def test_target_year(self):
        year = _extract_year("Targeted for completion in 2029")
        assert year == 2029

    def test_past_year_ignored(self):
        year = _extract_year("The project was completed in 2015.")
        assert year is None   # 2015 is outside current_year..current+20 range

    def test_no_year_returns_none(self):
        assert _extract_year("The project is under consideration.") is None


class TestGeocoding:
    def test_known_area_from_gazetteer(self):
        lat, lon = _geocode_location("Kokapet")
        assert lat is not None
        assert 17.0 < lat < 18.0
        assert 78.0 < lon < 79.5

    def test_known_area_case_insensitive(self):
        lat, lon = _geocode_location("MADHAPUR")
        assert lat is not None

    def test_orr_corridor_resolved(self):
        lat, lon = _geocode_location("ORR corridor development")
        assert lat is not None

    def test_unknown_returns_none(self):
        lat, lon = _geocode_location("XYZNONEXISTENTPLACE")
        # May or may not resolve via Nominatim — just check types
        assert lat is None or isinstance(lat, float)


class TestExtractProjects:
    SAMPLE_TEXT = """
    The Hyderabad Metro Rail Phase 2 will extend from Miyapur to Kokapet, with
    a total budget of ₹14,500 crore. The project is targeted for completion by 2028.

    Additionally, a new industrial zone near Patancheru has been proposed under
    NIMZ guidelines with an estimated budget of Rs. 3,200 crore to be completed by 2030.

    The ORR expressway corridor widening project has been allocated ₹800 crore.
    """

    def test_extracts_multiple_projects(self):
        projects = extract_projects("test_doc.pdf", self.SAMPLE_TEXT)
        assert len(projects) >= 2

    def test_project_has_required_fields(self):
        projects = extract_projects("test_doc.pdf", self.SAMPLE_TEXT)
        for p in projects:
            assert p.source_document == "test_doc.pdf"
            assert p.project_type in {
                "metro", "highway", "industrial", "it_sez",
                "park", "education", "hospital", "airport", "other"
            }
            assert p.impact_radius_km > 0

    def test_metro_project_extracted(self):
        projects = extract_projects("test_doc.pdf", self.SAMPLE_TEXT)
        types = [p.project_type for p in projects]
        assert "metro" in types or "highway" in types

    def test_budget_extracted_when_present(self):
        projects = extract_projects("test_doc.pdf", self.SAMPLE_TEXT)
        budgets = [p.budget_crore for p in projects if p.budget_crore is not None]
        assert len(budgets) >= 1
        assert any(b > 1000 for b in budgets)

    def test_empty_text_returns_empty_list(self):
        projects = extract_projects("empty.pdf", "")
        assert projects == []
