"""
Two-stage highlight extraction pipeline.

Stage 1 — BM25 (Okapi Best Matching 25):
    A custom implementation of the BM25 ranking algorithm that scores each
    transcript sentence against the podcast title as a query. Returns the
    top N lexically relevant sentences.

Stage 2 — MSMARCO DistilBERT Re-ranking:
    Uses a pre-trained sentence-transformer model to compute dense embeddings
    and re-ranks the BM25 candidates by cosine similarity to the title.
"""

from __future__ import annotations

import math
from functools import lru_cache

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util


# ---------------------------------------------------------------------------
# Model loading (singleton — loaded once at startup, not per request)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the MSMARCO DistilBERT model once and cache it."""
    return SentenceTransformer("/models")


# ---------------------------------------------------------------------------
# Stage 1: BM25 Lexical Ranking
# ---------------------------------------------------------------------------

class BM25:
    """
    Okapi BM25 ranking algorithm.

    Scores documents (sentences) against a query using term frequency,
    inverse document frequency, and document length normalization.

    Parameters
    ----------
    k1 : float, default 1.5
        Term frequency saturation parameter. Higher values increase the
        influence of term frequency.
    b : float, default 0.75
        Document length normalization parameter (0 = no normalization,
        1 = full normalization relative to average document length).

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term frequency per document.
    df_ : dict[str, int]
        Document frequency per term (number of documents containing the term).
    idf_ : dict[str, float]
        Inverse document frequency per term.
    doc_len_ : list[int]
        Number of terms per document.
    corpus_size_ : int
        Total number of documents in the corpus.
    avg_doc_len_ : float
        Average document length across the corpus.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def fit(self, corpus: list[list[str]]) -> BM25:
        """
        Compute term statistics from the tokenized corpus.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element is a document represented as a list of terms.

        Returns
        -------
        self
        """
        tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        idf: dict[str, float] = {}
        doc_len: list[int] = []
        corpus_size = 0

        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            tf.append(frequencies)

            for term in frequencies:
                df[term] = df.get(term, 0) + 1

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query: list[str]) -> list[float]:
        """Score every document in the corpus against the query."""
        return [self._score(query, idx) for idx in range(self.corpus_size_)]

    def _score(self, query: list[str], index: int) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]

        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len_
            )
            score += numerator / denominator

        return score


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def get_sentence_tokens(transcript: str) -> list[str]:
    """Split transcript into sentences using NLTK."""
    return sent_tokenize(transcript)


def _tokenize(sentence: str) -> list[str]:
    """Whitespace tokenization for BM25 input."""
    return sentence.split()


def get_bm25_scores(query: str, corpus: list[str]) -> list[float]:
    """Score each sentence in the corpus against the query using BM25."""
    tokenized_corpus = [_tokenize(s) for s in corpus]
    query_tokens = _tokenize(query)

    bm25 = BM25()
    bm25.fit(tokenized_corpus)
    return bm25.search(query_tokens)


def get_bm25_topN(transcript: str, scores: list[float], n: int = 10) -> list[str]:
    """Return the top N sentences ranked by BM25 score."""
    sentences = get_sentence_tokens(transcript)
    ranked = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
    return [sent for _, sent in ranked[:n]]


def get_bert_topN(title: str, sentences: list[str], n: int = 5) -> list[str]:
    """Re-rank candidate sentences by semantic similarity to the title."""
    model = _load_model()

    title_embedding = model.encode(title)
    sentence_embeddings = model.encode(sentences)

    similarities = [
        util.cos_sim(emb, title_embedding).item()
        for emb in sentence_embeddings
    ]

    ranked = sorted(zip(similarities, sentences), key=lambda x: x[0], reverse=True)
    return [sent for _, sent in ranked[:n]]
