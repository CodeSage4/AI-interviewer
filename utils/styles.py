# utils/styles.py
import streamlit as st

def load_custom_css():
    st.markdown("""
        <style>
            /* 1. Import a Google Font (Inter is the standard for SaaS) */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

            html, body, [class*="css"]  {
                font-family: 'Inter', sans-serif;
            }

            /* 2. Remove the default top padding (Streamlit adds massive whitespace at the top) */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            /* 3. Modernize Buttons: No borders, slight shadow */
            .stButton>button {
                width: 100%;
                border-radius: 8px;
                border: none;
                background-color: #F63366; /* Match your primaryColor */
                color: white;
                font-weight: 600;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: all 0.2s ease;
            }
            .stButton>button:hover {
                background-color: #D92050;
                box-shadow: 0 6px 8px rgba(0,0,0,0.15);
                transform: translateY(-1px);
            }

            /* 4. CENTER & SHRINK THE CHAT INPUT */
            .stChatInput {
                max-width: 850px; /* Adjust this to make it wider/narrower */
                margin: 0 auto;   /* Centers it horizontally */
            }
            /* Optional: Add a subtle shadow to the input box to make it pop */
            .stChatInput > div {
                box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
                border-radius: 20px; /* Rounder corners */
            }
            
            /* Make chat input bigger */
            div[data-testid="stChatInput"] textarea {
                min-height: 80px !important;
            }

            /* Make assistant messages wider */
            div[data-testid="stChatMessage"] {
                max-width: 900px;
                margin: auto;
            }

            
            /* 5. Chat Message Styling (Make it look like ChatGPT/WhatsApp) */
            [data-testid="stChatMessage"] {
                background-color: #262730;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 0.5rem;
                border: 1px solid #363945;
            }
            
            /* 6. Hide the Deploy Button & Menu */
            .stDeployButton {display:none;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)