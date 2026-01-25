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
    "current_question": None,
    "last_intent": None,
    "screen_type": None,
    "image_turn": 0,
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
    """Detect what kind of screen is shown."""
    if "def " in screen_text or "{" in screen_text or "}" in screen_text:
        return "CODE"
    elif any(k in screen_text for k in ["API", "DB", "Service", "Gateway", "Cache"]):
        return "ARCHITECTURE"
    elif "Upload" in screen_text or "Button" in screen_text or "Click" in screen_text or "Estimate" in screen_text:
        return "UI"
    else:
        return "SLIDES"

def analyze_code(screen_text: str):
    issues = []
    if "(left + right)//2" in screen_text:
        issues.append("Possible integer overflow")
    if "return -1" in screen_text:
        issues.append("Explicit not-found case handled")
    return issues

def extract_components(screen_text: str):
    keywords = ["API", "Gateway", "FastAPI", "PostgreSQL", "Cache", "Service"]
    return [k for k in keywords if k in screen_text]

def analyze_ui(screen_text: str):
    risks = []
    if "A4" in screen_text:
        risks.append("Scale calibration risk")
    if "Upload" in screen_text:
        risks.append("Image quality risk")
    return risks


# -------------------------------
# OLLAMA QUESTION GENERATOR
# -------------------------------
def ask_ollama(screen_text, screen_summary, history, last_answer=None):

    if last_answer:
        history.append(f"Student: {last_answer}")

    # ---- Decide intent (UNDERSTAND → PROBE cycle) ----
    if st.session_state.image_turn == 0:
        intent = "UNDERSTAND"
    elif st.session_state.image_turn == 1:
        intent = "PROBE"
    else:
        intent = "UNDERSTAND"   # should rarely happen

    st.session_state.last_intent = intent

    # ---- Screen analysis ----
    screen_type = classify_screen(screen_text)
    st.session_state.screen_type = screen_type

    clean_text = "\n".join(
        line.split(" (conf=")[0] for line in screen_text.splitlines()
    )

    code_issues = analyze_code(clean_text) if screen_type == "CODE" else []
    components = extract_components(clean_text) if screen_type == "ARCHITECTURE" else []
    ui_risks = analyze_ui(clean_text) if screen_type == "UI" else []

    prompt = f"""
You are a technical interviewer.

IMPORTANT: Your question must be based primarily on the CURRENT SCREEN,
even if the conversation history discusses other images or topics.

SCREEN TYPE: {screen_type}

VISIBLE TEXT:
{clean_text}

HUMAN SUMMARY:
{screen_summary}

PRE-ANALYSIS:
Code issues: {code_issues}
Components: {components}
UI risks: {ui_risks}

CONVERSATION HISTORY:
{' '.join(history)}

CURRENT INTENT: {intent}

IF intent is UNDERSTAND:
Ask ONE high-level question to understand what this screen does.

IF intent is PROBE:
Pick ONE specific element visible on the screen and ask:
- either about a failure mode, 
- or an edge case, 
- or a concrete limitation.
You must reference that element explicitly in the question.

RULES:
- Ask exactly ONE question.
- Reference something visible on the screen.
- No explanations. No meta talk. Only the question.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    question = response["message"]["content"].strip()

    history.append(f"AI: {question}")
    st.session_state.image_turn += 1
    return question


# -------------------------------
# MORE CREDIBLE SCORING
# -------------------------------
def compute_scores(answers):
    combined = " ".join(answers).lower()

    tech_concepts = ["overflow","complexity","latency","scale","index","bounds","segmentation","calibration"]
    originality_terms = ["instead","better","alternative","i would change"]
    implementation_terms = ["bug","error","edge case","fail","bottleneck","distortion"]

    technical_depth = min(10, len(set(tech_concepts) & set(combined.split())) + 2)
    clarity = min(10, combined.count(".") + 2)
    originality = min(10, sum(t in combined for t in originality_terms) + 1)
    implementation = min(10, sum(t in combined for t in implementation_terms) + 2)
    confidence = min(10, len(answers)//2)  # images, not verbosity

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
st.subheader("📸 Show what you're presenting")

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

    

    
    
    if image_input:

        with st.form(key=f"summary_form_{len(st.session_state.answers)}"):

            summary = st.text_input(
                "Explain what is shown on this screen"
            )

            submitted = st.form_submit_button("Ask interviewer")

            if submitted:
                st.session_state.screen_summary = summary
                st.session_state.image_turn = 0

                with st.spinner("Thinking..."):
                    q = ask_ollama(
                        st.session_state.screen_text,
                        st.session_state.screen_summary,
                        st.session_state.history
                    )

                st.session_state.current_question = q
                st.rerun()




# -------------------------------
# STEP 3 — STUDENT ANSWER
# -------------------------------
if st.session_state.current_question:

    st.subheader("👨‍💼 Interview Question")
    st.write(st.session_state.current_question)

    answer = st.text_area("Your answer")

    if st.button("Submit Answer"):

        st.session_state.answers.append(answer)
        turns_done = len(st.session_state.answers)

        # ---- CASE 1: Ask second question on SAME IMAGE ----
        if st.session_state.last_intent == "UNDERSTAND":
            next_q = ask_ollama(
                st.session_state.screen_text,
                st.session_state.screen_summary,
                st.session_state.history,
                last_answer=answer
            )
            st.session_state.current_question = next_q
            st.rerun()

        # ---- CASE 2: Finish interview after 4 answers ----
        if turns_done >= 4:
            scores = compute_scores(st.session_state.answers)
            st.subheader("📊 Scoring & Feedback")
            st.json(scores)
            plot_radar(scores)
            st.stop()

        # ---- CASE 3: Move to next image ----
        st.session_state.screen_text = ""
        st.session_state.screen_summary = ""
        st.session_state.current_question = None
        st.session_state.last_intent = None
        st.session_state.image_turn = 0

        st.info("Show the next screen when you're ready.")
        st.rerun()

# if st.button("Submit Answer"):

#     if not st.session_state.screen_text:
#         st.warning("Upload a screenshot first.")
#         st.stop()

#     st.session_state.answers.append(answer)
#     turns_done = len(st.session_state.answers)

#     # ---- CASE 1: Ask second question on SAME IMAGE ----
#     if st.session_state.last_intent == "UNDERSTAND":
#         next_q = ask_ollama(
#             st.session_state.screen_text,
#             st.session_state.screen_summary,
#             st.session_state.history,
#             last_answer=answer
#         )
#         st.session_state.current_question = next_q
#         st.rerun()

#     # ---- CASE 2: Finish interview after 4 answers ----
#     if turns_done >= 4:
#         with st.spinner("Computing scores..."):
#             scores = compute_scores(st.session_state.answers)

#         st.subheader("📊 Scoring & Feedback")
#         st.json(scores)
#         plot_radar(scores)
#         st.stop()

#     # ---- CASE 3: Move to next image (after Q2 answered) ----
#     st.session_state.screen_text = ""
#     st.session_state.screen_summary = ""
#     st.session_state.current_question = None
#     st.session_state.last_intent = None   # RESET for next image
#     st.session_state.image_turn = 0

#     st.success("Upload next screenshot")
#     st.rerun()
