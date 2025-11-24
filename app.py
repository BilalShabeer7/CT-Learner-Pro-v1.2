"""
CT-Learner Pro V2.0 - Unified Educational AI Platform
Advanced RAG-based Auto-Grader + Critical Thinking Analyzer
0–10 Grading | Heatmaps | Paul's CT Framework | Streamlit
"""

import os
import re
import json
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# File reading
import PyPDF2
from docx import Document

# Optional: Grammar checking
try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool('en-US')
except:
    lang_tool = None

# ==================== CONFIG & STYLING ====================
st.set_page_config(
    page_title="CT-Learner Pro v2.0",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

COLORS = {
    "primary": "#2E7D5B",
    "secondary": "#4A90E2",
    "accent": "#FF6B35",
    "success": "#27AE60",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "background": "#F8F9FA",
    "text": "#2C3E50"
}

CUSTOM_CSS = """
<style>
    .main-header {font-size: 3rem !important; color: #2E7D5B; text-align: center; font-weight: 800;}
    .subtitle {font-size: 1.4rem; color: #2C3E50; text-align: center; margin-bottom: 2rem;}
    .module-card {
        background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #2E7D5B; margin: 1rem 0;
    }
    .score-box {text-align: center; padding: 1rem; border-radius: 12px; font-size: 2rem; font-weight: bold;}
    .progress-fill {height: 12px; background: linear-gradient(90deg, #2E7D5B, #4A90E2); border-radius: 6px;}
    .highlight-sentence {padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 5px solid;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==================== PAUL'S CT RUBRIC ====================
PAUL_CT_RUBRIC = {
    "Clarity": {"desc": "Clear examples and illustrations", "q": "Could you elaborate with an example?", "patterns": ["for example", "e.g.", "such as", "to illustrate"], "color": "#2E7D5B"},
    "Accuracy": {"desc": "Verifiable facts and sources", "q": "How can we verify this?", "patterns": ["cite", "according to", "study", "research", "data", "%"], "color": "#4A90E2"},
    "Relevance": {"desc": "Stays on topic", "q": "How is this connected to the question?", "patterns": ["related to", "regarding", "pertaining"], "color": "#3498DB"},
    "Significance": {"desc": "Focuses on key ideas", "q": "What is most important here?", "patterns": ["main", "key", "crucial", "essential"], "color": "#27AE60"},
    "Logic": {"desc": "Sound reasoning", "q": "Does this follow logically?", "patterns": ["therefore", "because", "thus", "hence", "however"], "color": "#FF6B35"},
    "Precision": {"desc": "Specific and exact", "q": "Could you be more specific?", "patterns": ["specifically", "exactly", "precisely"], "color": "#9B59B6"},
    "Fairness": {"desc": "Considers other views", "q": "Are we being fair to other perspectives?", "patterns": ["on the other hand", "although", "despite", "however"], "color": "#1ABC9C"},
    "Depth": {"desc": "Explores complexity", "q": "What are the complexities here?", "patterns": ["because", "although", "complex", "intricacy"], "color": "#F1C40F"},
    "Breadth": {"desc": "Multiple perspectives", "q": "Is there another way to look at this?", "patterns": ["alternatively", "another view", "in contrast"], "color": "#E67E22"}
}

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner="🧠 Loading AI model...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

def embed(texts: List[str]) -> np.ndarray:
    texts = [t.strip() or " " for t in texts]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])

# ==================== FILE READING ====================
def read_file(uploaded_file) -> str:
    if not uploaded_file:
        return ""
    bytes_data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".txt"):
            return bytes_data.decode("utf-8")
        elif name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
            os.unlink(tmp_path)
            return text
        elif name.endswith(".docx"):
            doc = Document(io.BytesIO(bytes_data))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            return bytes_data.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""

# ==================== GRAMMAR CHECK ====================
def grammar_check(text: str) -> Dict:
    if not lang_tool or not text.strip():
        return {"available": False, "count": 0}
    try:
        matches = lang_tool.check(text)
        return {"available": True, "count": len(matches), "examples": matches[:5]}
    except:
        return {"available": False, "count": 0}

# ==================== GRADING ENGINE ====================
def grade_submission(model_text: str, student_text: str, rubric: Dict = None) -> Dict:
    vec1, vec2 = embed([model_text, student_text])
    sim = cosine_sim(vec1, vec2)
    sim_score = max(0.0, min((sim + 1) / 2, 1.0)) * 10

    grammar = grammar_check(student_text)
    penalty = min(3.0, grammar["count"] * 0.15) if grammar["available"] else 0.0
    final = max(0.0, sim_score - penalty)

    return {
        "score": round(final, 2),
        "similarity": round(sim_score, 2),
        "grammar_issues": grammar["count"],
        "penalty": round(penalty, 2)
    }

# ==================== CT ANALYSIS ====================
def analyze_ct(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    lower = text.lower()
    scores = {}
    highlights = {k: [] for k in PAUL_CT_RUBRIC.keys()}

    for sent in sents:
        sent_l = sent.lower()
        for std, data in PAUL_CT_RUBRIC.items():
            if any(p in sent_l for p in data["patterns"]):
                highlights[std].append((sent, data["color"]))
                break

    # Heuristic scoring
    scores["Clarity"] = 1.0 if any(p in lower for p in ["example", "e.g.", "such as"]) else 0.5
    scores["Accuracy"] = 1.0 if any(p in lower for p in ["study", "research", "data", "cite"]) else 0.4
    scores["Relevance"] = 0.8 if len(sents) > 2 and any(w in lower.split()[:10] for w in lower.split()[10:]) else 0.5
    scores["Significance"] = 1.0 if any(p in lower for p in ["main", "key", "important"]) else 0.6
    scores["Logic"] = min(1.0, sum(1 for p in ["because", "therefore", "thus", "however"] if p in lower) * 0.3)
    scores["Precision"] = 0.9 if "specifically" in lower else 0.6
    scores["Fairness"] = 1.0 if any(p in lower for p in ["although", "however", "despite"]) else 0.4
    scores["Depth"] = 0.9 if "because" in lower and len(text) > 200 else 0.5
    scores["Breadth"] = 1.0 if "alternatively" in lower else 0.4

    return scores, highlights

# ==================== VISUALIZATIONS ====================
def radar_chart(scores: Dict, title: str):
    cats = list(scores.keys())
    vals = list(scores.values()) + [list(scores.values())[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=vals, theta=cats + [cats[0]], fill='toself',
        line_color=COLORS["primary"], fillcolor=f"{COLORS['primary']}30"
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=400, title=title)
    return fig

# ==================== MAIN APP ====================
def main():
    st.markdown('<div class="main-header">🧠 CT-Learner Pro v2.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Grading • Critical Thinking • Analytics</div>', unsafe_allow_html=True)

    if "grading_results" not in st.session_state:
        st.session_state.grading_results = []
    if "ct_results" not in st.session_state:
        st.session_state.ct_results = []

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Dashboard", "Auto-Grading", "CT Analysis", "Resources"], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("## Quick Actions")
        if st.button("Clear All Data"):
            st.session_state.grading_results = []
            st.session_state.ct_results = []
            st.success("Cleared!")
            st.rerun()

    if page == "Dashboard":
        st.header("Dashboard")
        if st.session_state.grading_results or st.session_state.ct_results:
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.grading_results:
                    avg = np.mean([r["score"] for r in st.session_state.grading_results])
                    st.metric("Average Grade", f"{avg:.2f}/10")
            with col2:
                if st.session_state.ct_results:
                    ct_avg = np.mean([np.mean(list(r["scores"].values())) for r in st.session_state.ct_results])
                    st.metric("Avg CT Score", f"{ct_avg:.2f}")
        else:
            st.info("Upload data in other modules to see analytics!")

        if st.button("Generate Sample Data"):
            st.session_state.grading_results = [
                {"name": f"Student {i}", "score": round(np.random.uniform(5, 9.5), 2)} for i in range(1, 9)
            ]
            st.session_state.ct_results = [
                {"name": f"Student {i}", "scores": {k: round(np.random.uniform(0.4, 0.95), 2) for k in PAUL_CT_RUBRIC}}
                for i in range(1, 9)
            ]
            st.success("Sample data generated!")

    elif page == "Auto-Grading":
        st.header("Auto-Grading (0–10 Scale)")
        col1, col2 = st.columns(2)
        with col1:
            model_file = st.file_uploader("Model Answer", type=["txt", "pdf", "docx"])
            model_text = st.text_area("Or paste model answer", height=150)
        with col2:
            student_files = st.file_uploader("Student Submissions", accept_multiple_files=True, type=["txt", "pdf", "docx"])

        if st.button("Start Grading", type="primary", use_container_width=True):
            model_content = model_text or read_file(model_file)
            if not model_content.strip():
                st.error("Please provide a model answer.")
                return

            submissions = []
            for f in student_files:
                text = read_file(f)
                if text.strip():
                    submissions.append({"name": f.name, "text": text})

            if not submissions:
                st.error("No valid student submissions.")
                return

            progress = st.progress(0)
            results = []
            for i, sub in enumerate(submissions):
                grade = grade_submission(model_content, sub["text"])
                results.append({"name": sub["name"], **grade})
                progress.progress((i + 1) / len(submissions))

            st.session_state.grading_results = results
            st.success(f"Graded {len(results)} submissions!")
            st.rerun()

        if st.session_state.grading_results:
            for r in st.session_state.grading_results:
                with st.expander(f"**{r['name']}** → {r['score']}/10", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.progress(r["score"] / 10)
                    with col2:
                        st.write(f"Similarity: {r['similarity']}/10 | Grammar Issues: {r['grammar_issues']}")

    elif page == "CT Analysis":
        st.header("Critical Thinking Analysis")
        files = st.file_uploader("Upload Student Responses", accept_multiple_files=True, type=["txt", "pdf", "docx"])

        if files and st.button("Analyze CT Skills", type="primary"):
            results = []
            prog = st.progress(0)
            for i, f in enumerate(files):
                text = read_file(f)
                if text.strip():
                    scores, highlights = analyze_ct(text)
                    results.append({"name": f.name, "text": text, "scores": scores, "highlights": highlights})
                prog.progress((i + 1) / len(files))
            st.session_state.ct_results = results
            st.success("Analysis Complete!")

        for res in st.session_state.ct_results:
            with st.expander(f"🧠 {res['name']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    for std, items in res["highlights"].items():
                        if items:
                            for sent, color in items[:3]:
                                st.markdown(f'<div class="highlight-sentence" style="border-left-color:{color}"><strong>{std}:</strong> {sent}</div>', unsafe_allow_html=True)
                with col2:
                    st.plotly_chart(radar_chart(res["scores"], "CT Profile"), use_container_width=True)

    elif page == "Resources":
        st.header("Learning Resources")
        st.info("Coming soon: Interactive CT exercises, videos, and adaptive learning paths!")

if __name__ == "__main__":
    main()
