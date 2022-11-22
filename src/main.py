from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from model import *

app = FastAPI()

# pydantic models
class TextIn(BaseModel):
    title: str
    transcript: str

class TextOut(BaseModel):
    title: str
    summary: list

@app.post("/predict", response_model=TextOut, status_code=200)
def get_prediction(payload: TextIn):
    title = payload.title
    transcript = payload.transcript
    corpus = get_sentence_tokens(transcript)
    scores = get_bm25_scores(title, corpus)
    sentences = get_bm25_topN(transcript, scores)
    bert_topN = get_bert_topN(title, sentences)

    if not bert_topN:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {
        "title": title,
        "summary": bert_topN
    }
    return response_object