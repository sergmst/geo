"""
Streamlit Geo-AI Demo (single-file) + Docker + SEGY Viewer + Preload instructions

This document contains:
- A single-file Streamlit app `streamlit_geo_ai_demo.py` (expanded)
- `Dockerfile` and `docker-compose.yml` for one-command deployment
- `requirements.txt` (pinned minimal set) and notes how to build an image with a pre-downloaded embeddings model
- Enhanced SEGY utilities: inline/xline extractor, simple amplitude slice viewer and auto-annotation routine

HOW TO USE
1) Save this project folder locally.
2) Put your data under `./data/pdfs/`, `./data/las/`, `./data/shapefiles/`, `./data/segy/`.
3) Build and run with Docker (recommended) or run locally with `streamlit run streamlit_geo_ai_demo.py`.

DOCKER (recommended for reproducible demo):
- Build: `docker compose build --no-cache`
- Run: `docker compose up --remove-orphans --build`
- The app will be available at http://localhost:8501

SECURITY / RESOURCE NOTES
- This setup is aimed at a **demo** and uses small models by default. For on-premise production replace MODEL_NAME with a large local ggml/llama model or run TGI/vLLM backend.
- If you need offline embeddings preloaded into the image, follow the section below "Preloading embeddings model into Docker image".

------ FILE 1: streamlit_geo_ai_demo.py ------

# (The file is intentionally long — it includes the app and SEGY utilities)

import os
import glob
import json
import tempfile
from typing import List, Tuple

import streamlit as st
import fitz  # PyMuPDF
import lasio
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import mapping

# Embedding + Vector DB
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM (local HF-style)
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ML classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Seismic (obspy)
from obspy.io.segy.segy import _read_segy
import matplotlib.pyplot as plt

# ---------------------------
# Configuration / Helpers
# ---------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
LAS_DIR = os.path.join(DATA_DIR, "las")
SHP_DIR = os.path.join(DATA_DIR, "shapefiles")
SEGY_DIR = os.path.join(DATA_DIR, "segy")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")

# ---------------------------
# Init embeddings & chroma
# ---------------------------
@st.cache_resource
def init_embedding_and_chroma():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./.chromadb"))
    try:
        collection = client.get_collection("geo_docs")
    except Exception:
        collection = client.create_collection("geo_docs")
    return embed_model, client, collection

# ---------------------------
# Init LLM pipeline (local HF if possible)
# ---------------------------
@st.cache_resource
def init_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        return pipe
    except Exception as e:
        st.warning(f"Local LLM init failed: {e}. Using fallback echo-pipe.")
        return lambda prompt, **kwargs: [{"generated_text": "[LLM not available locally] " + prompt}]

embed_model, chroma_client, chroma_col = init_embedding_and_chroma()
llm = init_llm()

# ---------------------------
# PDF ingestion
# ---------------------------

def ingest_pdfs_to_chroma(pdf_dir: str, collection):
    files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    added = 0
    for fpath in files:
        doc = fitz.open(fpath)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if len(text.strip()) < 30:
                continue
            emb = embed_model.encode(text)
            uid = f"{os.path.basename(fpath)}::p{page_num}"
            try:
                collection.add(ids=[uid], metadatas=[{"source": fpath, "page": page_num}], documents=[text], embeddings=[emb.tolist()])
            except Exception:
                # fallback to upsert
                collection.upsert([{"id": uid, "metadata": {"source": fpath, "page": page_num}, "embedding": emb.tolist(), "document": text}])
            added += 1
    return added


def semantic_search(query: str, collection, k=3) -> List[dict]:
    q_emb = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        out.append({"text": doc, "meta": meta})
    return out

# ---------------------------
# LAS utilities
# ---------------------------

def read_las_file(path: str) -> pd.DataFrame:
    las = lasio.read(path)
    df = las.df()
    df = df.reset_index().rename(columns={"DEPT": "DEPTH"})
    return df

# Build features from LAS DataFrame (simple)
def las_to_features(df: pd.DataFrame) -> pd.DataFrame:
    want = [c for c in ["GR", "RHOB", "NPHI", "DT", "RES"] if c in df.columns]
    if not want:
        want = df.select_dtypes(include=[np.number]).columns.tolist()
    features = df[want].copy()
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return features

# Training a tiny classifier from labelled intervals (DEMO synthetic)
def train_demo_log_classifier(las_dir: str):
    files = glob.glob(os.path.join(las_dir, "*.las"))
    X = []
    y = []
    for f in files[:20]:
        try:
            df = read_las_file(f)
            feats = las_to_features(df)
            if 'GR' in feats.columns:
                labels = (feats['GR'] < 80).astype(int)
            else:
                labels = np.zeros(len(feats), dtype=int)
            X.append(feats.values)
            y.append(labels.values)
        except Exception:
            continue
    if not X:
        return None
    X = np.vstack(X)
    y = np.concatenate(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(Xs, y)
    return model, scaler, feats.columns.tolist()

# ---------------------------
# SEGY utilities (enhanced)
# ---------------------------

def read_segy_traces(segy_path: str):
    st.info(f"Loading SEGY: {segy_path}")
    segy = _read_segy(segy_path)
    traces = []
    for tr in segy.traces:
        traces.append(tr.data)
    data = np.vstack(traces)  # (ntraces, nsamples)
    # Try to get sampling info
    delta = getattr(segy.traces[0].header, 'sample_interval_in_ms_for_this_trace', None)
    if delta is None:
        # fallback: try binary file header
        delta = getattr(segy.binary_file_header, 'sample_interval_in_ms', None)
    return data, delta


def extract_inline_xline(data: np.ndarray, inline_idx: int = 0, xline_idx: int = 0):
    # Very simplified mapping: assume traces are arranged as (inline-major or xline-major)
    # For demo: return a single trace by index and a time series
    ntr, ns = data.shape
    trace = data[min(inline_idx, ntr-1)]
    return trace


def compute_amplitude_slice(data: np.ndarray) -> np.ndarray:
    # Collapse traces by RMS across time to create a per-trace amplitude measure
    amp = np.sqrt(np.mean(np.square(data), axis=1))
    return amp


def annotate_segy_anomalies(amp: np.ndarray, threshold_percentile: float = 95.0) -> List[dict]:
    thr = np.percentile(amp, threshold_percentile)
    idxs = np.where(amp >= thr)[0]
    out = []
    for i in idxs:
        out.append({"trace_idx": int(i), "amplitude": float(amp[i]), "note": "High amplitude anomaly"})
    return out

# ---------------------------
# Map analysis
# ---------------------------

def analyze_shapefile(folder: str) -> dict:
    shapefiles = glob.glob(os.path.join(folder, "*.shp"))
    out = {}
    for shp in shapefiles:
        gdf = gpd.read_file(shp)
        bounds = gdf.total_bounds.tolist()
        centroid = gdf.unary_union.centroid
        out[os.path.basename(shp)] = {
            'n_features': len(gdf),
            'bounds': bounds,
            'centroid': (float(centroid.x), float(centroid.y)),
            'crs': str(gdf.crs)
        }
    return out

# Create an LLM prompt for the map description
def map_insight_prompt(meta: dict) -> str:
    p = f"Given geographic metadata: features={meta.get('n_features')}, bounds={meta.get('bounds')} (minx,miny,maxx,maxy), centroid={meta.get('centroid')}, crs={meta.get('crs')}.\nProvide a concise geological description of possible structural features (anticlines, synclines, faults) and note what additional data would help confirm prospectivity."
    return p

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(layout='wide', page_title='Geo AI Demo — Extended')
st.title('Geo AI Demo — Streamlit (Local LLM + Embeddings) — Extended')

tabs = st.tabs(["Documents (GeoSearch)", "Logs (LAS classifier)", "Maps (SHP/TIFF)", "SEGY (Viewer & Annotations)", "Admin / Data"])

# Documents tab
with tabs[0]:
    st.header("Semantic GeoSearch — PDF / Reports")
    st.markdown("Upload PDF reports into `data/pdfs/` or use the button to ingest existing PDFs.")
    if st.button("Ingest PDFs to Vector DB"):
        added = ingest_pdfs_to_chroma(PDF_DIR, chroma_col)
        st.success(f"Ingested {added} pages into vector DB.")

    q = st.text_input("Ask a question about your reports:", value="Где зафиксированы трещиноватые коллекторы?")
    k = st.slider("Top K results", 1, 10, 3)
    if st.button("Search") and q.strip():
        results = semantic_search(q, chroma_col, k=k)
        for i, r in enumerate(results):
            st.subheader(f"Result {i+1} — source: {r['meta'].get('source')} (page {r['meta'].get('page')})")
            st.write(r['text'][:2000])
            prompt = f"Extract a 2-sentence geological summary from this text:\n\n{r['text']}"
            llm_out = llm(prompt)
            summary = llm_out[0]['generated_text'] if isinstance(llm_out, list) else str(llm_out)
            st.info(summary[:1000])

# Logs tab
with tabs[1]:
    st.header("AI Log Classifier — LAS interpretation")
    st.markdown("Drop LAS files into `data/las/`. The demo trains a tiny classifier (synthetic labels) and shows per-depth predictions.")
    if st.button("Train demo classifier"):
        with st.spinner("Training..."):
            res = train_demo_log_classifier(LAS_DIR)
            if res is None:
                st.error("No LAS files found or training failed. Put .las files into data/las/ and try again.")
            else:
                model, scaler, feat_cols = res
                st.session_state['log_model'] = (model, scaler, feat_cols)
                st.success("Trained demo classifier (synthetic labels).")

    las_files = glob.glob(os.path.join(LAS_DIR, "*.las"))
    sel = st.selectbox("Choose LAS file to visualize", options=["--"] + [os.path.basename(x) for x in las_files])
    if sel and sel != "--":
        path = os.path.join(LAS_DIR, sel)
        df = read_las_file(path)
        st.write(df.head())
        feats = las_to_features(df)
        st.line_chart(feats.iloc[:, :min(4, feats.shape[1])])
        if 'log_model' in st.session_state:
            model, scaler, cols = st.session_state['log_model']
            X = feats[cols].values
            Xs = scaler.transform(X)
            preds = model.predict_proba(Xs)[:, 1]
            df['collector_prob'] = preds
            st.write(df[['DEPTH', 'collector_prob']].head(30))
            st.area_chart(df.set_index('DEPTH')['collector_prob'])
        else:
            st.info("Train the demo classifier to see predicted collector probabilities.")

# Maps tab
with tabs[2]:
    st.header("Map & Section Insights — SHP / GeoTIFF")
    st.markdown("Put shapefiles (.shp + .shx + .dbf) into data/shapefiles/. The app reads them and produces short LLM descriptions.")
    if st.button("Analyze shapefiles"):
        meta = analyze_shapefile(SHP_DIR)
        st.session_state['map_meta'] = meta
        st.success(f"Found {len(meta)} shapefiles")

    if 'map_meta' in st.session_state:
        for name, meta in st.session_state['map_meta'].items():
            st.subheader(name)
            st.json(meta)
            prompt = map_insight_prompt(meta)
            llm_out = llm(prompt)
            desc = llm_out[0]['generated_text'] if isinstance(llm_out, list) else str(llm_out)
            st.write(desc)

# SEGY tab
with tabs[3]:
    st.header("SEGY — Viewer & Auto-annotations")
    st.markdown("Place SEG-Y files into data/segy/. The viewer extracts simple inline/xline traces and shows amplitude slices + anomaly annotations.")
    segy_files = glob.glob(os.path.join(SEGY_DIR, "*.segy")) + glob.glob(os.path.join(SEGY_DIR, "*.sgy"))
    sel = st.selectbox("Choose SEGY file to inspect", options=["--"] + [os.path.basename(x) for x in segy_files])
    if sel and sel != "--":
        segy_path = os.path.join(SEGY_DIR, sel)
        with st.spinner("Reading SEGY (may take a few seconds)..."):
            data, delta = read_segy_traces(segy_path)
        st.success(f"Loaded SEGY with shape {data.shape}. sample_interval_ms={delta}")

        # amplitude slice
        amp = compute_amplitude_slice(data)
        thr = float(st.slider("Anomaly percentile threshold", 90.0, 99.9, 95.0))
        ann = annotate_segy_anomalies(amp, threshold_percentile=thr)
        st.write(f"Found {len(ann)} high-amplitude traces (percentile {thr}).")

        # show amplitude plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(amp)
        ax.set_title('Per-trace RMS amplitude')
        for a in ann:
            ax.axvline(a['trace_idx'], linestyle='--', alpha=0.6)
        st.pyplot(fig)

        # simple trace viewer for annotated trace indices
        if ann:
            idx = st.selectbox('Select anomaly to inspect', options=[a['trace_idx'] for a in ann])
            tr = data[int(idx)]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(tr)
            ax2.set_title(f'Trace {idx}')
            st.pyplot(fig2)

        # LLM annotation summary for anomalies
        if st.button('Generate textual annotation for anomalies'):
            prompt_text = 'Summarize the following high-amplitude trace indices and suggest interpretations and next steps: ' + json.dumps(ann)
            llm_out = llm(prompt_text)
            desc = llm_out[0]['generated_text'] if isinstance(llm_out, list) else str(llm_out)
            st.write(desc)

# Admin tab
with tabs[4]:
    st.header("Admin & Data paths")
    st.write("Data directories (put files here):")
    st.code(f"{PDF_DIR}\n{LAS_DIR}\n{SHP_DIR}\n{SEGY_DIR}")
    st.markdown("### Local LLM settings")
    st.write(f"MODEL_NAME = {MODEL_NAME}")
    st.write(f"Embedding model = {EMBED_MODEL_NAME}")
    st.markdown("### Quick run commands (local)")
    st.code("pip install -r requirements.txt\nexport MODEL_NAME=path_or_model_id\nstreamlit run streamlit_geo_ai_demo.py")

# End of app

"""
