"""
Podcast Highlight Extractor API

FastAPI service that extracts the most relevant highlight sentences from
a podcast transcript using a two-stage retrieval pipeline:
  Stage 1: BM25 lexical ranking
  Stage 2: MSMARCO DistilBERT semantic re-ranking
"""

from fastapi import FastAPI, HTTPException

from schemas import PodcastInput, PodcastOutput
from model import (
    get_sentence_tokens,
    get_bm25_scores,
    get_bm25_topN,
    get_bert_topN,
)

app = FastAPI(
    title="Podcast Highlight Extractor",
    description="Two-stage NLP pipeline for extracting key highlights from podcast transcripts",
    version="1.0.0",
)


@app.post("/predict", response_model=PodcastOutput, status_code=200)
def extract_highlights(payload: PodcastInput) -> PodcastOutput:
    """Extract the top 5 highlight sentences from a podcast transcript."""
    corpus = get_sentence_tokens(payload.transcript)
    scores = get_bm25_scores(payload.title, corpus)
    candidates = get_bm25_topN(payload.transcript, scores)
    highlights = get_bert_topN(payload.title, candidates)

    if not highlights:
        raise HTTPException(status_code=400, detail="Could not extract highlights from transcript.")

    return PodcastOutput(title=payload.title, highlights=highlights)
