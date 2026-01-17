import streamlit as st
from paddleocr import PaddleOCR

ocr = PaddleOCR()

st.write("OCR loaded successfully")
