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

# =========================================================
# LAYER 1 — PERCEPTION (OCR)
# =========================================================
# Purpose:
# Convert pixels → text with confidence scores.
# No reasoning. No interpretation.
# =========================================================

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

# =========================================================
# LAYER 2 — SCREEN UNDERSTANDING & ANALYSIS (DETERMINISTIC)
# =========================================================
# Purpose:
# Convert raw OCR text into structured meaning
# - Screen type classification (UI / CODE / ARCHITECTURE / SLIDES)
# - Risk & edge-case extraction
# - Component identification
#
# This layer performs NO LLM calls.
# =========================================================


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


# =========================================================
# LAYER 3 — STRUCTURED INTERVIEW ENGINE (LLM-GUIDED)
# =========================================================
# Purpose:
# - Maintain interview intent (UNDERSTAND → PROBE)
# - Ground questions in current screen only
# - Prevent context leakage across images
# =========================================================
# Interview intent is explicit and stateful.
# Each screen is discussed in two phases.

INTERVIEW_INTENTS = ["UNDERSTAND", "PROBE"]

INTENT_DESCRIPTIONS = {
    "UNDERSTAND": "High-level comprehension of what is shown",
    "PROBE": "Failure modes, edge cases, and design tradeoffs"
}


def ask_ollama(screen_text, screen_summary, history, last_answer=None):

    if last_answer:
        history.append(f"Student: {last_answer}")

    # ---- 1. EXTRACT LAST QUESTION (THE ANCHOR) ----
    last_question = "None (First turn)"
    for msg in reversed(history):
        if msg.startswith("AI:"):
            last_question = msg.replace("AI:", "").strip()
            break

    # ---- Decide intent (UNDERSTAND → PROBE cycle) ----
    if st.session_state.image_turn == 0:
        intent = "UNDERSTAND"
    elif st.session_state.image_turn == 1:
        intent = "PROBE"
    else:
        intent = "UNDERSTAND" 

    st.session_state.last_intent = intent

    # ---- Screen analysis & Logic ----
    # (Assuming you are using the robust classification we discussed earlier)
    # For now, keeping your structure to minimize rewrite noise:
    
    screen_type = classify_screen(screen_text)
    st.session_state.screen_type = screen_type
    
    clean_text = "\n".join(line.split(" (conf=")[0] for line in screen_text.splitlines())
    
    # Pre-analysis context
    code_issues = analyze_code(clean_text) if screen_type == "CODE" else []
    components = extract_components(clean_text) if screen_type == "ARCHITECTURE" else []
    ui_risks = analyze_ui(clean_text) if screen_type == "UI" else []

    # ---- 2. THE HARD CONSTRAINT PROMPT ----
    prompt = f"""
You are a technical interviewer.

IMPORTANT: Your question must be based primarily on the CURRENT SCREEN.

SCREEN TYPE: {screen_type}
VISIBLE TEXT:
{clean_text}

HUMAN SUMMARY:
{screen_summary}

PRE-ANALYSIS (Use this for PROBE phase only):
Code issues: {code_issues}
Components: {components}
UI risks: {ui_risks}

PREVIOUS QUESTION (DO NOT REPEAT OR REPHRASE):
"{last_question}"

CONVERSATION HISTORY:
{' '.join(history)}

CURRENT INTENT: {intent}

IF intent is UNDERSTAND:
Ask ONE high-level question to understand what this screen does.
Constraint: Do not ask about bugs, risks, or edge cases yet.

IF intent is PROBE:
Pick ONE specific element visible on the screen and ask about a failure mode, edge case, or limitation.
Constraint: You MUST change the dimension. Do not ask a variation of the Previous Question.

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
import json

def evaluate_interview(history):
    transcript = "\n".join(history)
    
    eval_prompt = f"""
    Review this technical interview transcript. 
    Be logical, straightforward, and do not sugarcoat.
    
    TRANSCRIPT:
    {transcript}
    
    Return ONLY a JSON object:
    {{
        "scores": {{
            "Technical Depth": 1-10,
            "Resilience": 1-10,
            "Communication": 1-10,
            "Edge Case Awareness": 1-10
        }},
        "feedback": {{
            "strengths": "Short list",
            "gaps": "Critical analysis of what they missed",
            "verdict": "Hired/No Hire based on senior standards"
        }}
    }}
    """

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        # Strip potential markdown backticks from LLM response
        raw_content = response["message"]["content"].strip().replace("```json", "").replace("```", "")
        data = json.loads(raw_content)
        return data
    except Exception as e:
        # Fallback if the LLM hallucinating JSON or fails
        return {
            "scores": {"Technical Depth": 5, "Resilience": 5, "Communication": 5, "Practicality": 5},
            "summary": f"Evaluation error: {str(e)}"
        }
        


# -------------------------------
# RADAR CHART (SAFE VERSION)
# -------------------------------
def plot_radar(scores):
    plt.close("all") 
    
    labels = list(scores.keys())
    values = list(scores.values())

    # Complete the loop for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=7)
    plt.ylim(0, 10)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    # Center the plot in Streamlit
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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
# INTERVIEW — QUESTION & ANSWER
# -------------------------------
if st.session_state.current_question:
    intent = st.session_state.last_intent

    st.subheader(
        "🧭 Understanding Phase"
        if intent == "UNDERSTAND"
        else "🔍 Probing Phase"
    )

    st.caption(INTENT_DESCRIPTIONS[intent])

    st.write(st.session_state.current_question)

    answer = st.text_area("Your answer")

    if st.button("Submit Answer"):

        st.session_state.answers.append(answer)
        turns_done = len(st.session_state.answers)

        # ---- STATE MACHINE: SAME IMAGE ----
        if intent == "UNDERSTAND":
            next_q = ask_ollama(
                st.session_state.screen_text,
                st.session_state.screen_summary,
                st.session_state.history,
                last_answer=answer
            )
            st.session_state.current_question = next_q
            st.rerun()

        # ---- STATE MACHINE: END INTERVIEW ----
        # Inside the "Submit Answer" button logic:
        if turns_done >= 4:
            with st.spinner("Finalizing evaluation..."):
                # CALL THE NEW ISOLATED FUNCTION
                report = evaluate_interview(st.session_state.history)
                
                st.subheader("📊 Final Assessment")
                # FIX: Check for 'summary' OR 'feedback' keys, or show a default message
                summary_text = report.get("summary") or report.get("feedback", {}).get("verdict") or "Evaluation complete."
                st.write(f"**Verdict:** {summary_text}")
                
                # Display the Radar Chart
                plot_radar(report.get("scores", {}))
                
                # Show Gaps (if using the honest prompt)
                if "feedback" in report and "gaps" in report["feedback"]:
                    st.error(f"**Critical Gaps:** {report['feedback']['gaps']}")
                    
                st.stop()

        # ---- STATE MACHINE: NEXT IMAGE ----
        st.session_state.screen_text = ""
        st.session_state.screen_summary = ""
        st.session_state.current_question = None
        st.session_state.last_intent = None
        st.session_state.image_turn = 0

        st.info("Show the next screen when you're ready.")
        st.rerun()
