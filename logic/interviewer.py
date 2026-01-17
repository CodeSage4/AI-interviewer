import streamlit as st
import numpy as np
import cv2
import easyocr

st.set_page_config(
    page_title="AI Screen Interviewer",
    layout="wide"
)

st.title("🎤 AI Screen-Aware Interviewer")

# -------------------------------
# LOAD OCR (EASYOCR - STABLE)
# -------------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

ocr = load_ocr()

# -------------------------------
# SESSION STATE
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "screen_text" not in st.session_state:
    st.session_state.screen_text = ""

# -------------------------------
# OCR FUNCTION (FIXED)
# -------------------------------
def extract_screen_text(image_bytes: bytes) -> str:
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = ocr.readtext(img)

    lines = [text for (_, text, conf) in result if conf > 0.3]
    return "\n".join(lines)

# -------------------------------
# SIMPLE (RELIABLE) QUESTION GENERATOR
# -------------------------------
def ask_question(screen_text: str, history: list, last_answer: str | None = None):

    if last_answer:
        history.append(f"Student: {last_answer}")

    text = screen_text.lower()

    if "api" in text or "endpoint" in text:
        q = "How do you handle authentication and rate limiting for this API?"
    elif "react" in text or "ui" in text or "button" in text:
        q = "How does your UI handle errors and loading states?"
    elif "database" in text or "sql" in text or "table" in text:
        q = "How do you prevent data inconsistency in concurrent requests?"
    elif "model" in text or "accuracy" in text:
        q = "How did you validate this model and avoid overfitting?"
    elif "docker" in text or "cloud" in text:
        q = "How do you monitor failures in production?"
    else:
        q = "What was the hardest technical decision in building this system?"

    history.append(f"AI: {q}")
    return q

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("How this works")
st.sidebar.write("""
1) Show your screen (slides, code, or diagram)
2) Upload a screenshot OR take a photo
3) Read AI question
4) Type your answer
5) Repeat
""")

st.sidebar.info("Uses EasyOCR + rule-based questioning (reliable for demos).")

# -------------------------------
# STEP 1 — CAPTURE SCREEN
# -------------------------------
st.divider()
st.subheader("📸 Capture what you're presenting")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "Upload screenshot of your screen",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    camera = st.camera_input("Or take a photo of your screen")

image_input = uploaded or camera

if image_input:
    with st.spinner("Running OCR on screen..."):
        screen_text = extract_screen_text(image_input.getvalue())
        st.session_state.screen_text = screen_text

    st.success("Extracted screen text:")
    st.code(screen_text)

    if not st.session_state.current_question:
        with st.spinner("Generating first question..."):
            q = ask_question(
                st.session_state.screen_text,
                st.session_state.history
            )
            st.session_state.current_question = q

# -------------------------------
# STEP 2 — AI QUESTION
# -------------------------------
st.divider()

if st.session_state.current_question:
    st.subheader("🤖 AI Question")
    st.write(st.session_state.current_question)

# -------------------------------
# STEP 3 — STUDENT ANSWER
# -------------------------------
answer = st.text_area("Your answer (type here)")

if st.button("Submit Answer"):
    if not st.session_state.screen_text:
        st.warning("Capture your screen first!")
    else:
        with st.spinner("Thinking..."):
            next_q = ask_question(
                st.session_state.screen_text,
                st.session_state.history,
                last_answer=answer
            )

        st.session_state.current_question = next
