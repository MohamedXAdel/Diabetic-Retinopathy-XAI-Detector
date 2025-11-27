import google.generativeai as genai
import io
import streamlit as st
import numpy as np
from PIL import Image
import cv2


def img_to_part(img):
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = (np.clip(img, 0, 255)).astype(np.uint8)
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img)
                    else:
                        img = img.convert("RGB")
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=95)
                    return {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": buffer.getvalue()
                        }
                    }


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

prompt = """
You are an expert ophthalmologist AI assistant. 
Analyze the uploaded retinal fundus photograph and the two explainability maps (Grad-CAM and LIME).

The deep-learning model classified this image as: {prediction}.

Generate a concise, structured clinical explanation for the referring doctor containing:

1. Diagnosis (repeat the model's prediction)
2. Key Findings (what pathological signs are visible and where)
3. Clinical Implications & recommended next steps
4. Model Confidence and brief note on what the red/yellow areas in the heatmaps represent

Write in clear medical English, bullet-point format, maximum 12–15 lines.
Do NOT hallucinate findings that are not supported by the heatmaps.
"""
