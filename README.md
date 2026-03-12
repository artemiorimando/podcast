# Podcast Highlight Extractor

A two-stage NLP pipeline that extracts the most relevant highlights from podcast transcripts using BM25 lexical ranking and MSMARCO DistilBERT semantic re-ranking.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          Podcast Transcript              │
                        └──────────────────┬──────────────────────┘
                                           │
                              ┌─────────────▼─────────────┐
                              │   NLTK Sentence Tokenizer  │
                              └─────────────┬─────────────┘
                                            │
                  ┌─────────────────────────▼─────────────────────────┐
                  │  Stage 1: BM25 Lexical Ranking                    │
                  │  Score each sentence against the title query      │
                  │  Select top 10 candidates                         │
                  └─────────────────────────┬─────────────────────────┘
                                            │
                  ┌─────────────────────────▼─────────────────────────┐
                  │  Stage 2: DistilBERT Semantic Re-ranking          │
                  │  Encode title + candidates as dense embeddings    │
                  │  Rank by cosine similarity                        │
                  │  Return top 5 highlights                          │
                  └─────────────────────────┬─────────────────────────┘
                                            │
                              ┌─────────────▼─────────────┐
                              │     Top 5 Highlights       │
                              └───────────────────────────┘
```

## How It Works

### Stage 1 — BM25 (Custom Implementation)

The pipeline uses a **from-scratch implementation** of the Okapi BM25 ranking algorithm — no external BM25 library is used. BM25 scores each transcript sentence against the podcast title as a query, combining:

- **Term Frequency (TF)** — how often query terms appear in each sentence
- **Inverse Document Frequency (IDF)** — downweights common terms across the corpus
- **Length normalization** — adjusts for sentence length to avoid bias toward longer sentences

The top 10 sentences by BM25 score are passed to Stage 2.

### Stage 2 — MSMARCO DistilBERT Re-ranking

The BM25 candidates are re-ranked using `msmarco-distilbert-base-tas-b`, a sentence-transformer model trained on the MS MARCO passage retrieval dataset. The model encodes the title and each candidate sentence into dense vector embeddings, then ranks by **cosine similarity** — capturing semantic meaning that BM25's lexical matching misses.

## API Reference

### `POST /predict`

**Request:**
```json
{
  "title": "invisible matters of time",
  "transcript": "This is ninety nine percent, invisible, I'm Roman Mars for the most board. We take time for granted. Maybe we don't have enough of it, but at least we know how it works..."
}
```

**Response:**
```json
{
  "title": "invisible matters of time",
  "highlights": [
    "At least you know most of the time a lot of what we think about time and and how we keep track of.",
    "I mean when you think about the history of the implementation of one time zone.",
    "It is interesting to me that the time of day actually depended on the ethnicity of who you were asking.",
    "Then I started noticing that the time that they used the numbers they used for time were two hours off that of facing time.",
    "So time is just one example of how these you know intimate parts of weaker culture being suppressed."
  ]
}
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI + Uvicorn |
| Lexical Ranking | Custom BM25 implementation |
| Semantic Re-ranking | MSMARCO DistilBERT (sentence-transformers) |
| Tokenization | NLTK sentence tokenizer |
| Containerization | Docker |
| Validation | Pydantic |

## Quick Start

```bash
# Build the Docker image (downloads the DistilBERT model at build time)
docker build -t podcast-highlights .

# Run the container
docker run -p 8000:8000 podcast-highlights

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "your podcast title", "transcript": "full transcript text..."}'
```

Interactive API docs available at `http://localhost:8000/docs`.

## Project Structure

```
podcast/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and endpoint
│   ├── model.py          # BM25 + DistilBERT pipeline
│   └── schemas.py        # Pydantic request/response models
├── Dockerfile
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## References

- Robertson, S.E. et al. (1995). *Okapi at TREC-3.* — Original BM25 paper
- [Spotify Podcast Segment Retrieval](https://trec.nist.gov/pubs/trec29/papers/Spotify.P.pdf) — TREC 2020 podcast track
- [MSMARCO DistilBERT](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) — Pre-trained retrieval model

## Future Improvements

- Train a BERT classification model that predicts whether a sentence is a highlight (binary relevance) rather than relying on cosine similarity
- Add timestamp mapping to link highlights back to specific podcast moments
- Support batch processing of multiple episodes

## License

MIT
