# logic/ocr_utils.py

import cv2
import numpy as np
import easyocr

# Initialize once (cached reader)
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'])
    return _reader

def preprocess_image(img):
    """
    Light preprocessing that meaningfully improves OCR on slides/screens.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    return gray

def extract_screen_text(image_bytes: bytes) -> str:
    """
    Run OCR on a screenshot or camera image using EasyOCR.
    Returns structured, confidence-aware text.
    """

    # Convert bytes → OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return "(ERROR: Could not decode image)"

    # Preprocess before OCR (important)
    img = preprocess_image(img)

    reader = get_reader()
    result = reader.readtext(img)

    if not result:
        return "(No reliable text detected)"

    # Format output so the LLM can actually use it
    formatted_lines = []
    for (_, text, conf) in result:
        if conf > 0.25:  # slightly lower threshold = better recall
            formatted_lines.append(f"- {text} (conf={round(conf,2)})")

    return "\n".join(formatted_lines) if formatted_lines else "(Low-confidence image)"
