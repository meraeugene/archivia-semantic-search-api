import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import faiss

# ===========================
# FastAPI Setup
# ===========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://archivia-official.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Load Environment
# ===========================
print("Initializing Supabase connection...")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing from .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===========================
# Load Theses (ONCE)
# ===========================
print("Fetching theses from database...")
response = supabase.table("theses").select("*").execute()
df = pd.DataFrame(response.data)

if df.empty:
    raise ValueError("No theses found in database")

# Replace NaN/inf
df = df.replace([np.nan, np.inf, -np.inf], "")

# Build combined text field for search
df["text"] = (
    df["title"] + " " +
    df["adviser_name"] + " " +
    df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
    df["proponents"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
    df["category"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
)

# ===========================
# Build Models (ONCE)
# ===========================
print("Building TF-IDF + SBERT models...")

# TF-IDF (optional, still precompute if needed)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

# SBERT
sbert = SentenceTransformer("all-MiniLM-L6-v2")
sbert_matrix = sbert.encode(df["text"].tolist(), convert_to_numpy=True)

# FAISS index for fast semantic search
d = sbert_matrix.shape[1]
index = faiss.IndexFlatIP(d)  # cosine similarity
faiss.normalize_L2(sbert_matrix)
index.add(sbert_matrix)

print("AI models ready!")

PAGE_SIZE = 6  # number of theses per page

# ===========================
# SEARCH API
# ===========================
@app.get("/search")
def search(query: str, page: int = Query(1, ge=1)):
    if not query.strip():
        return {"total": 0, "page": page, "page_size": PAGE_SIZE, "data": []}

    q = query.lower()

    # ===== 1. Adviser Name Match =====
    adviser_mask = df["adviser_name"].str.lower().str.contains(rf"\b{q}\b", regex=True, na=False)

    # ===== 2. Proponents Match =====
    proponents_text = df["proponents"].apply(
        lambda x: " ".join(x).lower() if isinstance(x, list) else str(x).lower()
    )
    proponents_mask = proponents_text.str.contains(rf"\b{q}\b", regex=True, na=False)

    if adviser_mask.any():
        filtered = df[adviser_mask].copy()
    elif proponents_mask.any():
        filtered = df[proponents_mask].copy()
    else:
        # ===== 3. Semantic Search (FAISS + SBERT) =====
        q_vec = sbert.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, k=len(df))  # search all
        scores = D.flatten()
        filtered = df.iloc[I[0]].copy()
        filtered["score"] = scores

        # Filter low scores
        filtered = filtered[filtered["score"] >= 0.12]

    # Pagination
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    paginated = filtered.iloc[start:end].replace([np.nan, np.inf, -np.inf], None)

    # Columns to return
    cols = list(df.columns)
    if "score" in filtered.columns:
        cols.append("score")

    return {
        "total": len(filtered),
        "page": page,
        "page_size": PAGE_SIZE,
        "data": paginated[cols].to_dict(orient="records"),
    }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np

# ===========================
# Pydantic model for new thesis
# ===========================
class ThesisUpload(BaseModel):
    title: str
    adviser_name: str
    keywords: list[str] = []
    proponents: list[str] = []
    category: list[str] = []

@app.post("/upload")
def upload_thesis(new_thesis: ThesisUpload):
    global df, sbert, index

    # Reload latest theses from Supabase
    response = supabase.table("theses").select("*").execute()
    df = pd.DataFrame(response.data)

    if df.empty:
        df = pd.DataFrame([new_thesis.dict()])
    else:
        new_row = new_thesis.dict()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Rebuild text column
    df["text"] = (
        df["title"] + " " +
        df["adviser_name"] + " " +
        df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
        df["proponents"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
        df["category"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    )

    # Rebuild FAISS index
    sbert_matrix = sbert.encode(df["text"].tolist(), convert_to_numpy=True)
    d = sbert_matrix.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(sbert_matrix)
    index.add(sbert_matrix)

    return {"message": "Thesis uploaded and fully reindexed!"}
