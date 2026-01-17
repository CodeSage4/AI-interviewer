# app.py  — CLEANED VERSION (demo-safe, concept-consistent)

import streamlit as st
import numpy as np
import cv2
import easyocr
import ollama
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Screen Interviewer",
    layout="wide"
)

st.title("🎤 AI Screen-Aware Interviewer (Ollama + EasyOCR)")

# -------------------------------
# LOAD OCR
# -------------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

ocr = load_ocr()

# -------------------------------
# SESSION STATE
# -------------------------------
defaults = {
    "history": [],
    "answers": [],
    "screen_text": "",
    "screen_summary": "",
    "current_question": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------
# OCR FUNCTION (STRUCTURED OUTPUT)
# -------------------------------
def extract_screen_text(image_bytes: bytes) -> str:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = ocr.readtext(img)

    lines = [
        f"- {text} (conf={round(conf,2)})"
        for (_, text, conf) in result
        if conf > 0.25
    ]

    return "\n".join(lines) if lines else "(No reliable text detected)"

# -------------------------------
# SCREEN UNDERSTANDING LAYER
# -------------------------------

def classify_screen(screen_text: str) -> str:
    """Detect what kind of presentation is on screen."""
    if "def " in screen_text or "{" in screen_text or "}" in screen_text:
        return "CODE"
    elif any(k in screen_text for k in ["API", "DB", "Service", "Gateway", "Cache"]):
        return "ARCHITECTURE"
    else:
        return "SLIDES"

def analyze_code(screen_text: str):
    """Very light reasoning over visible code."""
    issues = []
    if "(left + right)//2" in screen_text:
        issues.append("Possible integer overflow")
    if "left <= right" in screen_text:
        issues.append("Loop boundary seems correct")
    if "return -1" in screen_text:
        issues.append("Explicit not-found case handled")
    return issues

def extract_components(screen_text: str):
    """Pull visible components from diagrams."""
    keywords = ["API", "Gateway", "FastAPI", "DB", "PostgreSQL", "Cache", "Service"]
    return [k for k in keywords if k in screen_text]



# -------------------------------
# OLLAMA QUESTION GENERATOR
# -------------------------------
def ask_ollama(screen_text: str, screen_summary: str, history: list, last_answer: str | None = None):

    if last_answer:
        history.append(f"Student: {last_answer}")

    screen_type = classify_screen(screen_text)

    # Pre-analysis for context
    code_issues = analyze_code(screen_text) if screen_type == "CODE" else []
    components = extract_components(screen_text) if screen_type == "ARCHITECTURE" else []

    prompt = f"""
You are a technical interviewer.

SCREEN TYPE: {screen_type}

VISIBLE TEXT:
{screen_text}

HUMAN SUMMARY:
{screen_summary}

PRE-ANALYSIS:
Code issues detected: {code_issues}
Visible components: {components}

CONVERSATION HISTORY:
{' '.join(history)}

RULES:
- Ask exactly ONE question.
- The question MUST reference a specific line or component visible on screen.
- If screen is CODE: focus on bugs, edge cases, or correctness.
- If screen is ARCHITECTURE: focus on failures, scaling, or reliability.
- If screen is SLIDES: challenge one claim or assumption.
- No explanations. No thanks. Output ONLY the question.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    question = response["message"]["content"].strip()

    if not question.endswith("?"):
        question = f"Based on the visible content, what is the most critical risk here?"

    history.append(f"AI: {question}")
    return question


# -------------------------------
# MORE CREDIBLE SCORING
# -------------------------------
def compute_scores(answers):
    combined = " ".join(answers).lower()

    tech_concepts = ["overflow","complexity","index","bounds","latency","scale","cache","retries"]
    clarity_signals = [".", "because", "for example"]
    originality_signals = ["instead", "better", "alternative", "i would change"]
    implementation_signals = ["bug", "error", "fail", "edge case", "bottleneck"]

    # ---- Metrics aligned with your requirements ----

    technical_depth = min(10, len(set(tech_concepts) & set(combined.split())))
    clarity = min(10, sum(1 for s in clarity_signals if s in combined) + 2)
    originality = min(10, sum(1 for s in originality_signals if s in combined) + 1)
    implementation = min(10, sum(1 for s in implementation_signals if s in combined) + 2)
    confidence = min(10, len(answers))

    return {
        "Technical Depth": technical_depth,
        "Clarity": clarity,
        "Originality": originality,
        "Implementation": implementation,
        "Confidence": confidence,
    }


# -------------------------------
# RADAR CHART (SAFE VERSION)
# -------------------------------
def plot_radar(scores):
    plt.close("all")   # IMPORTANT

    labels = list(scores.keys())
    values = list(scores.values())

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values += values[:1]
    angles = np.append(angles, angles[0])

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([2,4,6,8,10])

    st.pyplot(fig)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("How this works")
st.sidebar.write("""
1) Show your screen
2) Upload screenshot
3) Write a 1-line summary of what we see
4) Answer ONE question
5) Repeat 2 turns → get a score + radar chart
""")

st.sidebar.subheader("Interview so far")
st.sidebar.write("\n".join(st.session_state.history))


st.sidebar.info("EasyOCR + Ollama (Mistral) — offline, turn-based.")

# -------------------------------
# STEP 1 — CAPTURE SCREEN (EVERY TURN)
# -------------------------------
st.divider()
st.subheader("📸 Step 1 — Capture what you're presenting")

image_input = st.file_uploader(
    "Upload screenshot of your screen",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{len(st.session_state.answers)}"
)

if image_input:
    with st.spinner("Running OCR on screen..."):
        screen_text = extract_screen_text(image_input.getvalue())
        st.session_state.screen_text = screen_text

    st.success("Extracted screen text:")
    st.code(screen_text)

    # Force human summary (important)
    summary = st.text_input(
        "1-line summary of what is shown (e.g., '3-tier backend diagram')",
        key=f"summary_{len(st.session_state.answers)}"
    )
    st.session_state.screen_summary = summary

    # Generate first question if none exists
    if not st.session_state.current_question:
        with st.spinner("Generating first question..."):
            q = ask_ollama(
                st.session_state.screen_text,
                st.session_state.screen_summary,
                st.session_state.history
            )
            st.session_state.current_question = q

# -------------------------------
# STEP 2 — AI QUESTION
# -------------------------------
st.divider()

if st.session_state.current_question:
    st.subheader("🤖 Step 2 — AI Question")
    st.write(st.session_state.current_question)

# -------------------------------
# STEP 3 — STUDENT ANSWER
# -------------------------------
answer = st.text_area("Step 3 — Your answer (type here)")

if st.button("Submit Answer"):

    if not st.session_state.screen_text:
        st.warning("Upload a screenshot first.")
        st.stop()

    st.session_state.answers.append(answer)
    turns_done = len(st.session_state.answers)
    st.info(f"Turn {turns_done} / 4 completed")

    # ---- FINISH AFTER 4 TURNS ----
    if turns_done >= 4:
        st.success("🎯 Interview Complete — Generating Evaluation")

        scores = compute_scores(st.session_state.answers)
        st.json(scores)

        st.subheader("📊 Performance Radar Chart")
        plot_radar(scores)

        st.stop()

    # ---- OTHERWISE: RESET SCREEN FOR NEXT TURN ----
    st.session_state.screen_text = ""
    st.session_state.screen_summary = ""
    st.session_state.current_question = None

    st.success(
        "✅ Answer saved. "
        "Please upload your NEXT screen to continue."
    )

    st.rerun()
