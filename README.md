# 🎤 AI Screen-Aware Technical Interviewer

A turn-based, screen-grounded AI interviewer that reasons from what is *currently visible on screen* — not just uploaded documents — and conducts a structured technical interview.

This system is designed to evaluate a presenter’s understanding of their own work by analyzing screenshots of slides, UI, diagrams, or code, and generating targeted, context-aware questions.

---

## 🎯 Problem Statement

Traditional AI interview tools rely heavily on pre-uploaded PDFs or documents.  
This approach fails when:

- The presenter explains live diagrams  
- Code is shown dynamically  
- UI screens change in real time  
- Important context is visible *only on screen*

My goal was to build an interviewer that:
- Grounds every question in **what is currently visible**
- Maintains a **coherent interview flow**
- Separates **understanding from probing**
- Provides **structured evaluation**, not vibes.

---

## 🧠 Core Design Philosophy
The system uses a single-loop interviewer interaction rather than a step-by-step wizard, allowing questions to adapt naturally as the presenter progresses.

I deliberately **did NOT** build a real-time video system.

Instead, I chose a **high-quality turn-based loop**:

1. Presenter shows a screen and uploads a screenshot  
2. OCR extracts visible text  
3. Lightweight analysis classifies the screen  
4. Presenter briefly explains what is shown  
5. AI asks two structured questions (understand → probe)  
6. Process repeats across multiple screens  
7. Final rubric-based evaluation + radar chart

This avoids noisy streaming, improves reliability, and keeps the interview focused.

---

## 🏗️ System Architecture

This pipeline has **three explicit layers**:


---

## 🔍 How Each Component Works

### 1️⃣ Screen Capture
The presenter uploads a screenshot of what they are currently showing.

I intentionally avoid live video because:
- OCR is more reliable on static images  
- Noise is reduced  
- The system becomes deterministic and debuggable  

---

### 2️⃣ OCR Layer (EasyOCR)

For every image I:
- Run EasyOCR  
- Extract visible text  
- Keep confidence scores  
- Clean and structure output for downstream reasoning  

This allows the system to reason about:
- Code snippets  
- UI labels  
- Diagram text  
- Slide bullets  

---

### 3️⃣ Screen Understanding Layer

Instead of dumping raw OCR into an LLM, I perform lightweight analysis:

I classify the screen as one of:
- **CODE** → analyze variables, control flow, and potential bugs  
- **ARCHITECTURE** → extract components like API, DB, Cache, Services  
- **UI** → identify interaction risks (blur, scale, alignment, lighting)  
- **SLIDES** → extract key claims and assumptions  

This is what makes the system *screen-aware*, not just text-aware.

---

### 4️⃣ The questions are asked sequentially within a single interaction loop, not as separate UI steps.

For each screenshot, the AI follows a deliberate pattern:

#### **Q1 — Understanding (breadth)**
Goal: Align on what the screen represents.

Example:
> “At a high level, how does this ‘Upload Foot Image on A4 paper’ step fit into your pipeline?”

#### **Q2 — Probing (depth)**
Goal: Explore risks, edge cases, and design tradeoffs.

Example:
> “If the A4 sheet is rotated or partially cropped, how does your system recover the correct scale?”

This creates a **coherent conversation**, not disconnected Q&A.

---

### 5️⃣ Stateful Memory

The system retains:
- Previous questions  
- Student answers  
- Current intent (understand vs probe)  

This ensures continuity across turns rather than random questioning.

---

## 📊 Evaluation & Scoring

At the end of the interview, the system generates a radar chart based on four axes:

### ✔ Technical Depth  
Measures how many real technical concepts the student used.

### ✔ Clarity  
Rewards structured, well-reasoned explanations.

### ✔ Originality  
Detects whether the student proposed alternatives or improvements.

### ✔ Implementation Understanding  
Checks whether the student identified bugs, risks, or edge cases.

Scores are heuristic but **grounded in reasoning signals**, not word count.

---

## ✅ Why This Design is Strong

I intentionally chose:
- **Turn-based capture** over streaming  
- **OCR + human summary** over pure vision models  
- **Light analysis before LLM** instead of raw text dumping  
- **Two-question structure** instead of single-shot queries  

These are **engineering decisions, not shortcuts.**

---

## ⚠️ Limitations

I am transparent about what this system does *not* do:

- No true multimodal vision reasoning  
- Diagrams without text still need a human summary  
- OCR can struggle with handwriting or low-quality images  
- LLM questions may still be imperfect  

### Future Work:
- Add GPT-4o / vision models  
- Better diagram parsing  
- Automatic diagram summarization  
- Live speech support  

---

## 🎯 Final Goal

This project is not about perfect AI — it is about:

> A reliable, explainable, screen-grounded interview loop that reasons from what is visible, rather than relying solely on uploaded documents.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **OCR:** EasyOCR  
- **LLM:** Mistral via Ollama (offline)  
- **Processing:** OpenCV + NumPy  
- **Visualization:** Matplotlib (radar chart)

---
