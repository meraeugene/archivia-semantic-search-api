import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from dotenv import load_dotenv


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
print(" Initializing Supabase connection...")
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

#  FINAL SAFETY CLEAN (prevents ALL NaN forever)
df = df.replace([np.nan, np.inf, -np.inf], "")

# ===========================
# Build Search Index 
# ===========================
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

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

sbert = SentenceTransformer("all-MiniLM-L6-v2")
sbert_matrix = sbert.encode(df["text"].tolist(), convert_to_numpy=True)

print("AI models ready!")

PAGE_SIZE = 6  # number of theses per page


# ===========================
# SEARCH API
# ===========================

@app.get("/search")
def search(query: str, page: int = Query(1, ge=1)):
    if not query.strip():
        return {
            "total": 0,
            "page": page,
            "page_size": PAGE_SIZE,
            "data": [],
        }

    # Fetch latest theses from Supabase
    response = supabase.table("theses").select("*").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        return {"total": 0, "page": page, "page_size": PAGE_SIZE, "data": []}

    df = df.replace([np.nan, np.inf, -np.inf], "")

    # Build text for search
    df["text"] = (
        df["title"] + " " +
        df["adviser_name"] + " " +
        df["keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
        df["proponents"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
        df["category"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    )

    q = query.lower()

    # ===== 1. Adviser Name Match =====
    adviser_mask = df["adviser_name"].str.lower().str.contains(rf"\b{q}\b", regex=True, na=False)

    # ===== 2. Proponents Match =====
    proponents_text = df["proponents"].apply(
        lambda x: " ".join(x).lower() if isinstance(x, list) else str(x).lower()
    )
    proponents_mask = proponents_text.str.contains(rf"\b{q}\b", regex=True, na=False)

    if adviser_mask.any():
        filtered = df[adviser_mask]
    elif proponents_mask.any():
        filtered = df[proponents_mask]
    else:
        # ===== 3. Semantic Search (TF-IDF + SBERT) =====
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["text"])

        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        sbert_matrix = sbert.encode(df["text"].tolist(), convert_to_numpy=True)

        tfidf_vec = tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix).flatten()

        sbert_vec = sbert.encode([query], convert_to_numpy=True)
        sbert_scores = cosine_similarity(sbert_vec, sbert_matrix).flatten()

        combined = 0.5 * tfidf_scores + 0.5 * sbert_scores

        df["score"] = combined
        df = df.replace([np.nan, np.inf, -np.inf], 0)
        df = df.sort_values(by="score", ascending=False)
        filtered = df[df["score"] >= 0.12]

    # Pagination
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    paginated = filtered.iloc[start:end]
    paginated = paginated.replace([np.nan, np.inf, -np.inf], None)

    cols = list(df.columns)
    if "score" in filtered.columns:
        cols.append("score")

    return {
        "total": len(filtered),
        "page": page,
        "page_size": PAGE_SIZE,
        "data": paginated[cols].to_dict(orient="records"),
    }
