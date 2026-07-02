"""Shared test fixtures."""

import unittest.mock

import fitz  # PyMuPDF
import pytest

from pdf_gantry.db import get_connection


class _StubModel:
    """Zero-cost stand-in for SentenceTransformer — returns deterministic 768-d vectors.

    Tests that verify DB-level contracts (DELETE-before-INSERT on vec0, quarantine
    exclusion, pipeline orchestration) don't care whether vector values are real
    embeddings. What they need is for the embed path to complete without crashing and
    to write correctly-serializable bytes into the vec0 tables.

    The real nomic-embed-text-v2-moe takes 11-17 s to load from disk. This stub
    takes < 1 ms. The one test that validates the ImportError signal when the model
    is genuinely absent (test_embed_documents_propagates_import_error) overrides
    this fixture locally with a function that raises ImportError, so the session-level
    stub does not shadow that contract.
    """

    DIM = 768

    def encode(self, texts, show_progress_bar=False, **kwargs):
        """Return a list of 768-element float lists (one per input string)."""
        if isinstance(texts, str):
            # embed_query passes a single string; return a flat vector.
            return [0.01 * (i % 100) for i in range(self.DIM)]
        return [[0.01 * (i % 100) for i in range(self.DIM)] for _ in texts]


@pytest.fixture(autouse=True, scope="session")
def _stub_embedding_model():
    """Patch _get_embedding_model for the entire test session.

    Without this patch the real ML model is loaded at least three times per run
    (once cold, twice from OS page cache) contributing ~43 s of dead time to a
    107-s suite. DB-level tests don't need real embeddings — they need vectors
    of the right shape that serialize without error. The _StubModel above satisfies
    that contract in < 1 ms per call.

    Tests that explicitly probe the ImportError signal (e.g.
    test_embed_documents_propagates_import_error) use function-scoped monkeypatch
    to temporarily replace this stub with a raising callable for their own scope.
    """
    patcher = unittest.mock.patch(
        "pdf_gantry.embeddings._get_embedding_model",
        return_value=_StubModel(),
    )
    patcher.start()
    yield
    patcher.stop()


@pytest.fixture
def tmp_db(tmp_path):
    """Provides a temporary database with schema initialized."""
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    return conn


@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a valid PDF with substantial text content (classified as digital)."""
    doc = fitz.open()
    page = doc.new_page()
    # Insert enough text to ensure digital classification (text area > 5% of page)
    y = 72
    lines = [
        "This is a test document about climate change and its impacts on coastal regions.",
        "The study examines adaptation strategies across multiple geographic regions.",
        "Climate adaptation requires coordinated policy responses at local and national levels.",
        "Findings suggest that early intervention can significantly reduce long-term costs.",
        "The methodology combines quantitative analysis with qualitative case studies.",
        "Results indicate that vulnerable populations face disproportionate climate risks.",
        "Infrastructure investments in resilient systems show positive returns over 20 years.",
        "Community engagement is essential for successful climate adaptation programs.",
        "The paper concludes with recommendations for integrated climate policy frameworks.",
        "Future research should examine cross-sector adaptation synergies and tradeoffs.",
        "Additional analysis of urban heat island effects supports targeted cooling strategies.",
        "Water resource management under climate uncertainty demands flexible governance.",
        "Agricultural adaptation includes crop diversification and improved irrigation systems.",
        "Coastal zone management must account for accelerating sea level rise projections.",
        "Public health preparedness for climate-related events requires enhanced surveillance.",
    ]
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 20
    pdf_path = tmp_path / "test_climate.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def scanned_pdf(tmp_path):
    """Creates a PDF that looks scanned (image-only, no text blocks)."""
    doc = fitz.open()
    page = doc.new_page()
    # Insert a full-page image rectangle (no text)
    rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
    # Draw a filled rectangle to simulate a scanned page
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    shape.commit()
    pdf_path = tmp_path / "scanned_doc.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def papers_dir(tmp_path, sample_pdf):
    """Creates a papers directory with sample PDFs."""
    papers = tmp_path / "papers"
    papers.mkdir()
    # Copy sample PDF into papers dir
    import shutil
    shutil.copy(sample_pdf, papers / "test_climate.pdf")

    # Create a second PDF with enough text for digital classification
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    ml_lines = [
        "Machine learning approaches to natural language processing and understanding.",
        "Neural networks have revolutionized the field of computational linguistics.",
        "Deep learning models achieve state-of-the-art results on many NLP benchmarks.",
        "Transformer architectures enable efficient parallel processing of sequences.",
        "Pre-trained language models transfer knowledge across downstream tasks.",
        "Attention mechanisms allow models to focus on relevant input features.",
        "Word embeddings capture semantic relationships between terms in vector space.",
        "Recurrent neural networks process sequential data for language modeling tasks.",
        "Convolutional networks extract local features from text for classification.",
        "Ensemble methods combine multiple models for improved prediction accuracy.",
        "Evaluation metrics include precision recall and F1 score for classification.",
        "Cross-validation ensures robust estimation of model generalization performance.",
        "Hyperparameter tuning optimizes model architecture and training procedures.",
        "Data augmentation techniques expand limited training datasets for NLP tasks.",
        "Transfer learning reduces the need for large task-specific labeled datasets.",
    ]
    for line in ml_lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 20
    doc.save(str(papers / "ml_nlp_paper.pdf"))
    doc.close()

    return papers


@pytest.fixture
def encrypted_pdf(tmp_path):
    """Creates a user-password-protected PDF (needs_pass=1).

    PyMuPDF's fitz.open() does *not* raise on a file like this — it opens
    successfully and reports a nonzero page_count. The ValueError only
    surfaces later, the first time code touches page content (page.rect,
    page.get_text()) without having authenticated. This is exactly the shape
    of PDF a real ~2000-file academic corpus can contain: a DRM'd publisher
    export or an accidentally-locked download.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "you cannot read this without a password")
    pdf_path = tmp_path / "locked.pdf"
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="owner",
    )
    doc.close()
    return pdf_path


@pytest.fixture
def config_dir(tmp_path):
    """Provides a temporary config directory."""
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture
def populated_db(tmp_db, papers_dir):
    """A database with papers ingested from papers_dir."""
    from pdf_gantry.ingest import ingest_directory
    ingest_directory(tmp_db, papers_dir)
    return tmp_db
