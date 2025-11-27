import google.generativeai as genai
import io
import base64
import numpy as np
from PIL import Image
import cv2

def pil_to_base64(img):
    """
    Converts PIL Image, numpy array (BGR/RGB, uint8/float), or OpenCV image
    into base64 string suitable for sending to Gemini/Vision models.
    """
    # 1. If it's already a PIL Image
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGB")  # Remove alpha if exists
    
    # 2. If it's a numpy array (most common from Grad-CAM, LIME, etc.)
    elif isinstance(img, np.ndarray):
        if img.dtype == np.float64 or img.dtype == np.float32 or img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        if img.shape[-1] == 3:  # HWC
            if len(img.shape) == 3 and img.shape[2] == 3:
                # OpenCV uses BGR, PIL expects RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[-1] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(img)
            pil_img = pil_img.convert("RGB")  # Remove alpha
            img = np.array(pil_img)
        
        pil_img = Image.fromarray(img.astype(np.uint8))
    
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    # Save as JPEG in memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")


genai.configure(api_key="GEMINI_KEY")

model = genai.GenerativeModel('gemini-1.5-pro-vision-latest') 

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
