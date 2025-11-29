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
print("🔐 Initializing Supabase connection...")
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL or SUPABASE_KEY is missing from .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===========================
# Load Theses (ONCE)
# ===========================
print("📥 Fetching theses from database...")
response = supabase.table("theses").select("*").execute()
df = pd.DataFrame(response.data)

if df.empty:
    raise ValueError("❌ No theses found in database")

# ===========================
# Build Search Index 
# ===========================
df["text"] = (
    df["title"].fillna("") + " " +
    df["abstract"].fillna("") + " " +
    df["adviser_name"].fillna("") + " " +
    df["keywords"].fillna("").apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
    df["proponents"].fillna("").apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " +
    df["category"].fillna("").apply(lambda x: " ".join(x) if isinstance(x, list) else "")
)

# ===========================
# Build Models (ONCE)
# ===========================
print("⚙️ Building TF-IDF + SBERT models...")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["text"])

sbert = SentenceTransformer("all-MiniLM-L6-v2")
sbert_matrix = sbert.encode(df["text"], convert_to_numpy=True)

print("✅ AI models ready!")

PAGE_SIZE = 6  # number of theses per page

# ===========================
# SEARCH API
# ===========================

@app.get("/search")
def search(query: str, page: int = Query(1, ge=1)):
    if not query.strip():
        return []

    df_copy = df.copy()

    # ===== 1️ Check if query matches adviser_name (case-insensitive, partial match) =====
    adviser_mask = df_copy["adviser_name"].str.lower().str.contains(query.lower(), na=False)
    if adviser_mask.any():
        filtered = df_copy[adviser_mask]
    else:
        # ===== 2️ Otherwise, use TF-IDF + SBERT =====
        # TF-IDF score
        tfidf_vec = tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix).flatten()

        # SBERT score
        sbert_vec = sbert.encode([query], convert_to_numpy=True)
        sbert_scores = cosine_similarity(sbert_vec, sbert_matrix).flatten()

        # Hybrid score
        combined = 0.5 * tfidf_scores + 0.5 * sbert_scores

        df_copy["score"] = combined

        # Prevent NaN JSON crash
        df_copy = df_copy.replace([np.nan, np.inf, -np.inf], 0)

        df_copy = df_copy.sort_values(by="score", ascending=False)

        # Filter by threshold
        filtered = df_copy[df_copy["score"] >= 0.12]

    # ===== Pagination =====
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    paginated = filtered.iloc[start:end]

    cols = list(df.columns)
    if "score" in filtered.columns:
        cols.append("score")

    return {
        "total": len(filtered),
        "page": page,
        "page_size": PAGE_SIZE,
        "data": paginated[cols].to_dict(orient="records"),
    }
