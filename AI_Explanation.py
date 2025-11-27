import google.generativeai as genai
import io
import base64



def pil_to_base64(img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


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