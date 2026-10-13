"""Comprehensive unit and integration test suite for Task 1.2 components."""

from __future__ import annotations

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from devdocs.corpus import Chunk, load_markdown, parse_markdown_file, split_long_text
from devdocs.lsa import LsaModel
from devdocs.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from devdocs.search import Hit, SemanticIndex
from devdocs.text import tokenize
from devdocs.tfidf import TfidfVectorizer

# -----------------------------------------------------------------------------
# 1. Corpus tests
# -----------------------------------------------------------------------------


def test_chunk_dataclass_is_frozen() -> None:
    """Verify Chunk dataclass is immutable (frozen)."""
    chunk = Chunk(id="c1", source="test.md", heading="Test", text="Hello world")
    with pytest.raises(FrozenInstanceError):
        chunk.text = "New text"  # type: ignore[misc]


def test_split_long_text() -> None:
    """Verify long sections are divided into paragraph-respecting chunks."""
    short_text = "This is a short paragraph."
    assert split_long_text(short_text, max_words=300) == [short_text]

    long_para_1 = " ".join(["word"] * 200)
    long_para_2 = " ".join(["term"] * 150)
    full_text = f"{long_para_1}\n\n{long_para_2}"

    sub_chunks = split_long_text(full_text, max_words=300)
    assert len(sub_chunks) == 2
    assert sub_chunks[0] == long_para_1
    assert sub_chunks[1] == long_para_2


def test_parse_markdown_file_and_load_markdown() -> None:
    """Verify markdown files are parsed into heading-delimited Chunks."""
    sample_md = """# Controllers

A controller handles incoming requests.

## Routing

Routing controls which handler receives each request.

### Sub-routing

Further details on routing subpaths.
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        doc_file = tmp_path / "controllers.md"
        doc_file.write_text(sample_md, encoding="utf-8")

        chunks = parse_markdown_file(doc_file)
        assert len(chunks) == 3
        assert chunks[0].heading == "Controllers"
        assert chunks[1].heading == "Routing"
        assert chunks[2].heading == "Sub-routing"

        loaded = load_markdown(tmp_path)
        assert len(loaded) == 3


def test_load_markdown_missing_dir_raises() -> None:
    """Verify non-existent directory raises ValueError."""
    with pytest.raises(ValueError, match="Directory not found"):
        load_markdown("non_existent_directory_12345")


# -----------------------------------------------------------------------------
# 2. Text tokenization tests
# -----------------------------------------------------------------------------


def test_tokenize_preserves_decorators_and_expands_camelcase() -> None:
    """Verify tokenizer preserves @ decorators and expands camelCase identifiers."""
    sample = "@Injectable() export class AuthGuard { forRoot() { return canActivate(); } }"
    tokens = tokenize(sample)

    # Decorator preserved with @
    assert "@injectable" in tokens

    # camelCase identifiers preserved as single tokens and constituent subwords
    assert "authguard" in tokens
    assert "auth" in tokens
    assert "guard" in tokens

    assert "forroot" in tokens
    assert "for" in tokens
    assert "root" in tokens

    assert "canactivate" in tokens
    assert "can" in tokens
    assert "activate" in tokens


def test_tokenize_clean_punctuation() -> None:
    """Verify punctuation is stripped and lowercased."""
    tokens = tokenize("Hello, World! (Testing: 123).")
    assert tokens == ["hello", "world", "testing", "123"]


# -----------------------------------------------------------------------------
# 3. TF-IDF Vectorizer tests
# -----------------------------------------------------------------------------


def test_tfidf_vectorizer_basic_fit_transform() -> None:
    """Verify TF-IDF builds vocabulary and produces L2-normalized float32 matrix."""
    docs = [
        "NestJS controllers handle incoming requests.",
        "NestJS providers are injected into controllers via dependency injection.",
        "Guards handle authorization in NestJS.",
    ]
    vec = TfidfVectorizer(min_df=1, max_features=100)
    matrix = vec.fit_transform(docs)

    assert vec.is_fitted
    assert matrix.dtype == np.float32
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0

    # Row L2-norms must equal 1.0
    row_norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)


def test_tfidf_vectorizer_idf_formula() -> None:
    """Verify smoothed IDF formula matches: ln((1 + N) / (1 + df)) + 1."""
    docs = [
        "rare apple",
        "common apple banana",
        "common banana cherry",
    ]
    vec = TfidfVectorizer(min_df=1)
    vec.fit(docs)

    n_docs = 3
    df_apple = 2
    expected_apple_idf = np.log((1.0 + n_docs) / (1.0 + df_apple)) + 1.0

    col_apple = vec.vocabulary_["apple"]
    assert vec.idf_[col_apple] == pytest.approx(expected_apple_idf, abs=1e-5)


def test_tfidf_vectorizer_oov_produces_zero_vector() -> None:
    """Verify transform on pure out-of-vocabulary terms returns all zeros without NaNs."""
    vec = TfidfVectorizer(min_df=1)
    vec.fit(["cats dogs"])

    oov_matrix = vec.transform(["elephants giraffes"])
    assert oov_matrix.shape == (1, len(vec.vocabulary_))
    assert np.all(oov_matrix == 0.0)
    assert np.isfinite(oov_matrix).all()


def test_tfidf_vectorizer_unfitted_raises() -> None:
    """Verify transform without fit raises ValueError."""
    vec = TfidfVectorizer()
    with pytest.raises(ValueError, match="must be fitted"):
        vec.transform(["test"])


# -----------------------------------------------------------------------------
# 4. LSA Model tests
# -----------------------------------------------------------------------------


def test_lsa_fit_project_reconstruction() -> None:
    """Verify LSA projects into k dimensions and reconstruction error decreases with k."""
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((20, 50)).astype(np.float32)

    lsa_small = LsaModel(k=5).fit(X)
    lsa_large = LsaModel(k=15).fit(X)

    proj_small = lsa_small.project(X)
    assert proj_small.shape == (20, 5)
    assert proj_small.dtype == np.float32

    # L2-normalized rows in latent space
    norms = np.linalg.norm(proj_small, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Reconstruction error must decrease as k increases
    err_small = lsa_small.reconstruction_error(X)
    err_large = lsa_large.reconstruction_error(X)
    assert err_large < err_small


def test_lsa_errors() -> None:
    """Verify LSA error handling for invalid parameters and un-fitted state."""
    with pytest.raises(ValueError, match="k must be a positive integer"):
        LsaModel(k=0)

    lsa = LsaModel(k=5)
    with pytest.raises(ValueError, match="must be fitted"):
        lsa.project(np.ones((2, 10), dtype=np.float32))


# -----------------------------------------------------------------------------
# 5. Metrics tests
# -----------------------------------------------------------------------------


def test_metrics_manual_examples() -> None:
    """Verify Recall@k and MRR match exact manual hand-calculated values."""
    targets = ["chunk_3", "chunk_7"]

    # Target at rank 2
    retrieved_1 = ["chunk_1", "chunk_3", "chunk_5", "chunk_9"]
    assert recall_at_k(retrieved_1, targets, k=1) == 0.0
    assert recall_at_k(retrieved_1, targets, k=2) == 1.0
    assert recall_at_k(retrieved_1, targets, k=5) == 1.0
    assert reciprocal_rank(retrieved_1, targets) == pytest.approx(1.0 / 2.0)

    # Target at rank 4
    retrieved_2 = ["chunk_2", "chunk_4", "chunk_6", "chunk_7"]
    assert recall_at_k(retrieved_2, targets, k=3) == 0.0
    assert recall_at_k(retrieved_2, targets, k=4) == 1.0
    assert reciprocal_rank(retrieved_2, targets) == pytest.approx(1.0 / 4.0)

    # Mean Reciprocal Rank over two queries: (0.5 + 0.25) / 2 = 0.375
    mrr_val = mean_reciprocal_rank([retrieved_1, retrieved_2], [targets, targets])
    assert mrr_val == pytest.approx(0.375)

    # Edge cases
    assert recall_at_k([], targets, k=5) == 0.0
    assert recall_at_k(retrieved_1, [], k=5) == 0.0
    assert reciprocal_rank([], targets) == 0.0
    assert mean_reciprocal_rank([], []) == 0.0


# -----------------------------------------------------------------------------
# 6. SemanticIndex search and persistence tests
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide a small collection of sample documentation chunks."""
    return [
        Chunk(
            id="guards#auth",
            source="guards.md",
            heading="Guards",
            text=(
                "Guards determine whether a given request will be handled by the route handler. "
                "Authorization guard implements CanActivate."
            ),
        ),
        Chunk(
            id="pipes#validation",
            source="pipes.md",
            heading="Pipes",
            text=(
                "Pipes transform input data and handle schema validation using ValidationPipe "
                "and class-validator."
            ),
        ),
        Chunk(
            id="interceptors#logging",
            source="interceptors.md",
            heading="Interceptors",
            text=(
                "Interceptors bind extra logic before and after method execution, such as logging "
                "call duration and response mapping."
            ),
        ),
        Chunk(
            id="modules#dynamic",
            source="modules.md",
            heading="Modules",
            text=(
                "Dynamic modules allow configuring modules dynamically with forRoot "
                "and register methods."
            ),
        ),
    ]


def test_semantic_index_tfidf_and_lsa_search(sample_chunks: list[Chunk]) -> None:
    """Verify search returns correct hits for both TF-IDF and LSA."""
    index_tfidf = SemanticIndex(method="tfidf").build(sample_chunks)
    hits_tfidf = index_tfidf.search("how to validate data with pipes", top_k=2)

    assert len(hits_tfidf) > 0
    assert isinstance(hits_tfidf[0], Hit)
    assert hits_tfidf[0].chunk.id == "pipes#validation"
    assert hits_tfidf[0].score > 0.0

    index_lsa = SemanticIndex(method="lsa", k=3).build(sample_chunks)
    hits_lsa = index_lsa.search("guards authorization canActivate", top_k=2)
    assert len(hits_lsa) > 0
    assert hits_lsa[0].chunk.id == "guards#auth"


def test_semantic_index_edge_cases(sample_chunks: list[Chunk]) -> None:
    """Verify empty query raises ValueError, OOV returns [], and top_k > len is handled."""
    index = SemanticIndex(method="tfidf").build(sample_chunks)

    # Empty query raises ValueError
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        index.search("   ")

    # All OOV query returns empty list
    oov_hits = index.search("quantum teleportation superconductor")
    assert oov_hits == []

    # top_k larger than total chunk count
    all_hits = index.search("module", top_k=100)
    assert len(all_hits) <= len(sample_chunks)

    # top_k <= 0
    assert index.search("module", top_k=0) == []


def test_semantic_index_argpartition_matches_full_argsort(sample_chunks: list[Chunk]) -> None:
    """Verify argpartition optimization strictly matches full argsort ordering."""
    index = SemanticIndex(method="tfidf").build(sample_chunks)
    query = "authorization guard request"

    # Search with default argpartition top-k
    hits = index.search(query, top_k=3)
    hit_ids = [h.chunk.id for h in hits]

    # Recreate search scoring with full argsort
    q_vec = index.vectorizer.transform([query])
    from devdocs.linalg import cosine_sim_matrix

    scores = cosine_sim_matrix(q_vec, index.doc_embeddings_)[0]
    expected_full_order = np.argsort(-scores)

    # Take top 3 non-zero indices
    expected_ids: list[str] = [
        index.chunks[int(i)].id for i in expected_full_order if scores[i] > 0
    ][:3]
    assert hit_ids == expected_ids


def test_semantic_index_save_and_load(sample_chunks: list[Chunk]) -> None:
    """Verify save and load round-trip returns identical search scores and hits."""
    index = SemanticIndex(method="lsa", k=3).build(sample_chunks)
    query = "request authorization guard"
    original_hits = index.search(query, top_k=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_index"
        index.save(save_path)

        loaded_index = SemanticIndex.load(save_path)
        loaded_hits = loaded_index.search(query, top_k=3)

        assert len(loaded_hits) == len(original_hits)
        for orig, loaded in zip(original_hits, loaded_hits, strict=True):
            assert orig.chunk.id == loaded.chunk.id
            assert orig.score == pytest.approx(loaded.score, abs=1e-5)


# -----------------------------------------------------------------------------
# 7. Additional branch coverage tests
# -----------------------------------------------------------------------------


def test_lsa_additional_branches() -> None:
    """Verify remaining error and edge branches in LsaModel."""
    lsa = LsaModel(k=5)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        lsa.fit(np.ones((5, 5), dtype=np.float32), k=-1)

    with pytest.raises(ValueError, match="Expected 2D matrix"):
        lsa.fit(np.ones(5, dtype=np.float32))

    with pytest.raises(ValueError, match="Cannot fit LSA on an empty matrix"):
        lsa.fit(np.zeros((0, 5), dtype=np.float32))

    lsa.fit(np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        lsa.project(np.ones(4, dtype=np.float32))

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        lsa.project(np.ones((2, 8), dtype=np.float32))

    unfitted = LsaModel(k=2)
    with pytest.raises(ValueError, match="must be fitted"):
        unfitted.reconstruct(np.ones((2, 2), dtype=np.float32))


def test_search_additional_branches(sample_chunks: list[Chunk]) -> None:
    """Verify remaining error and edge branches in SemanticIndex."""
    with pytest.raises(ValueError, match="Unknown indexing method"):
        SemanticIndex(method="invalid")  # type: ignore[arg-type]

    idx = SemanticIndex()
    with pytest.raises(ValueError, match="Cannot build SemanticIndex on an empty"):
        idx.build([])

    with pytest.raises(ValueError, match="Unknown method"):
        idx.build(sample_chunks, method="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="k must be positive"):
        idx.build(sample_chunks, k=0)

    unbuilt = SemanticIndex()
    with pytest.raises(ValueError, match="has not been built yet"):
        unbuilt.search("test")

    with pytest.raises(ValueError, match="Cannot save an unbuilt index"):
        unbuilt.save("test_unbuilt")

    with pytest.raises(FileNotFoundError, match="Missing index files"):
        SemanticIndex.load("missing_non_existent_path")


def test_tfidf_additional_branches() -> None:
    """Verify remaining error branches in TfidfVectorizer."""
    vec = TfidfVectorizer()
    with pytest.raises(ValueError, match="Cannot fit TfidfVectorizer on an empty"):
        vec.fit([])

    # max_features truncation
    vec_limited = TfidfVectorizer(max_features=2)
    vec_limited.fit(["one two three four five"])
    assert len(vec_limited.vocabulary_) == 2

    # empty documents in transform
    empty_res = vec_limited.transform([])
    assert empty_res.shape == (0, 2)


def test_metrics_mismatched_lengths() -> None:
    """Verify mean_reciprocal_rank handles mismatched lengths gracefully."""
    assert mean_reciprocal_rank([["doc1"]], []) == 0.0
