"""Shared test fixtures."""

import pytest
import fitz  # PyMuPDF

from pdf_gantry.db import get_connection


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
