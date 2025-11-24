from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel

import spacy
import numpy as np

app = FastAPI(
    title="sps_genai",
    version="0.1.0",
    description="Simple text generation (bigram) and word embeddings API."
)

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

try:
    nlp = spacy.load("en_core_web_lg")
except OSError as e:
    raise RuntimeError(
        "spaCy model 'en_core_web_lg' not found. "
        "Install it with:\n  uv run python -m spacy download en_core_web_lg"
    ) from e

class EmbedRequest(BaseModel):
    word: str
    normalize: Optional[bool] = False  # Return a unit vector if True

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(req: TextGenerationRequest):
    start = (req.start_word or "").strip()
    if not start:
        raise HTTPException(status_code=400, detail="start_word must be non-empty")
    if req.length < 0:
        raise HTTPException(status_code=400, detail="length must be >= 0")

    generated_text = bigram_model.generate_text(start, req.length)
    return {"generated_text": generated_text}

@app.post("/embed")
def embed_word(req: EmbedRequest):
    w = (req.word or "").strip()
    if not w:
        raise HTTPException(status_code=400, detail="word must be non-empty")

    doc = nlp(w)
    
    vec = doc[0].vector if len(doc) == 1 else doc.vector

    if req.normalize:
        norm = np.linalg.norm(vec) or 1.0
        vec = vec / norm

    return {
        "word": w,
        "dim": int(vec.shape[0]),
        "vector": vec.tolist(),
        "normalized": bool(req.normalize),
    }
