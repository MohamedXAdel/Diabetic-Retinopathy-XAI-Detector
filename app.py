import os
import numpy as np
import streamlit as st
import torch
from PIL import Image as PILImage
from torchvision import models,transforms
import torch.nn as nn
from src.preprocess import preprocess_image
from Grad_Cam_XAI import GradCAM, overlay_heatmap
from LIME_XAI import LIME_Explainer
import google.generativeai as genai
from AI_Explanation import img_to_part,prompt

# -----------------------------------------------------------
# Device Selection
# -----------------------------------------------------------
def get_device(provided_device=None):
    if provided_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return provided_device


# -----------------------------------------------------------
# Load Models
# -----------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
def load_models(binary_path, severity_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Binary model ----
    binary_model = models.efficientnet_b0(weights=None)
    num_features = binary_model.classifier[1].in_features
    binary_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 1)
    )
    binary_model.load_state_dict(torch.load(binary_path, map_location=device))
    binary_model.to(device)
    binary_model.eval()

    # ---- Severity model ----
    severity_model = models.efficientnet_b0(weights=None)
    num_features = severity_model.classifier[1].in_features
    severity_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 4)
    )
    severity_model.load_state_dict(torch.load(severity_path, map_location=device))
    severity_model.to(device)
    severity_model.eval()

    return binary_model, severity_model, device


# -----------------------------------------------------------
# Two-stage Classification Logic
# -----------------------------------------------------------
def get_two_stage_prediction(binary_model, severity_model, image_tensor, threshold=0.5):
    binary_model.eval()
    severity_model.eval()
    
    with torch.no_grad():
        bin_logits = binary_model(image_tensor)
        bin_prob = torch.sigmoid(bin_logits).item()
        
        if bin_prob < threshold:
            return 0
        
        sev_logits = severity_model(image_tensor)
        _, sev_pred_idx = torch.max(sev_logits, 1)
        
        final_pred = sev_pred_idx.item() + 1
        return final_pred


# -----------------------------------------------------------
# Prediction Wrapper
# -----------------------------------------------------------
def predict_single_image(image_path, binary_model, sev_model):
    pil_img = preprocess_image(image_path, sigmaX=10)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    tensor_img = transform(pil_img).unsqueeze(0).to(device)

    prediction = get_two_stage_prediction(binary_model, sev_model, tensor_img)
    
    labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
    return prediction, labels[prediction], tensor_img


# -----------------------------------------------------------
# Streamlit UI Configuration
# -----------------------------------------------------------
# UI CSS 
def local_css(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder where app.py is
    css_path = os.path.join(base_dir, file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS from assets folder
local_css("assets/style.css")



# -----------------------------------------------------------
# Header
# -----------------------------------------------------------
st.set_page_config(
    page_title="DR Detector",
    layout="wide",
)
st.markdown('<p class="title">🔬 Diabetic Retinopathy Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered retinal analysis with explainable visualizations</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Session state init
# ---------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "dr_chat_history" not in st.session_state:
    st.session_state.dr_chat_history = []


# -----------------------------------------------------------
# Load Models Once
# -----------------------------------------------------------
@st.cache_resource
def load_model_once():
    return load_models(
        binary_path = os.path.join(base_dir, "models", "final_binary_model.pth"),
        severity_path = os.path.join(base_dir, "models", "final_severity_model.pth")
    )

binary_model, severity_model, device = load_model_once()
lime_explainer = LIME_Explainer(severity_model, device=device)

# -----------------------------------------------------------
# Uploader
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg","jpeg","png"])
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

if uploaded_file:
    pil_image = PILImage.open(uploaded_file).convert("RGB")

    # Centered small preview
    st.markdown("### Image Preview")
    st.write("")
    st.image(pil_image, width=300)

    st.write("")

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing image..."):
            temp_path = f"temp_{uploaded_file.name}"
            pil_image.save(temp_path)

            pred_id, pred_text, image_tensor = predict_single_image(
                temp_path, binary_model, severity_model
            )

            # -------------------- XAI --------------------
            if pred_id > 0:
                target_layer = severity_model.features[-1]
                gradcam = GradCAM(severity_model, target_layer)
            else:
                target_layer = binary_model.features[-1]
                gradcam = GradCAM(binary_model, target_layer)

            heatmap, _ = gradcam(image_tensor)
            orig, overlay = overlay_heatmap(temp_path, heatmap)
            img_np = np.array(pil_image.resize((224,224)))
            lime_img, _ = lime_explainer.explain(img_np)

        # -----------------------------------------------------------
        # Result Card
        # -----------------------------------------------------------
        class_map = {
            0: "no-dr",
            1: "mild",
            2: "moderate",
            3: "severe",
            4: "proliferative",
        }

        st.markdown(
            f'<div class="result-card {class_map[pred_id]}">'
            f"{pred_text}</div>",
            unsafe_allow_html=True
        )

        st.write("")

        # -----------------------------------------------------------
        # XAI Display (3 columns)
        # -----------------------------------------------------------
        st.markdown("### Model Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original Image**")
            st.image(orig, width=260)
            st.markdown('<p class="xai-caption">Original retinal scan for reference</p>', unsafe_allow_html=True)

        with col2:
            st.markdown("**Grad-CAM Visualization**")
            st.image(overlay, width=260)
            st.markdown('<p class="xai-caption">Yellow: highly relevant areas for prediction<br>Blue: less important regions</p>', unsafe_allow_html=True)


        with col3:
            st.markdown("**LIME Visualization**")
            st.image(lime_img, width=260)
            st.markdown('<p class="xai-caption">Green: positively contributes to predicted class<br>Red: negatively affects prediction</p>', unsafe_allow_html=True)



        # -----------------------------------------------------------
        # AI Explnation 
        # -----------------------------------------------------------
        st.markdown("### AI Clinical Explanation")
        with st.spinner("Generating medical explanation..."):
            try:
                contents = [
                    prompt.format(prediction=pred_text),
                    img_to_part(pil_image),
                    img_to_part(overlay),
                    img_to_part(lime_img)
                ]

                response = model.generate_content(contents)
                ai_explanation_text = response.text

            except Exception as e:
                ai_explanation_text = f"Explanation temporarily unavailable.\nError: {str(e)}"

            st.write(ai_explanation_text)


# Chat Panel 
# Initialize session state
# Initialize chat state
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# Use a single placeholder at the very end to avoid layout shift
chat_placeholder = st.empty()

with chat_placeholder.container():
    # Floating Button (always visible unless chat is open)
    if not st.session_state.chat_open:
        st.markdown("""
        <div class="chat-fab" title="Ask Medical Assistant">
            💬
        </div>
        """, unsafe_allow_html=True)

        # Invisible real button behind the FAB
        if st.button("Open Chat", key="open_fab_real", use_container_width=False):
            st.session_state.chat_open = True
            st.rerun()

    # Chat Modal
    if st.session_state.chat_open:
        st.markdown("""
        <div class="chat-modal">
            <div class="chat-box">
                <button class="close-btn" title="Close">✕</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Real close button (hidden but works)
        col_close1, col_close2 = st.columns([8, 1])
        with col_close2:
            if st.button("✕", key="close_modal_btn", help="Close chat"):
                st.session_state.chat_open = False
                st.rerun()

        # Import and run your chatbot
        from Chatbot import init_dr_chatbot
        init_dr_chatbot()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Powered by EfficientNet • Explainability AI • APTOS 2019</p>",
    unsafe_allow_html=True
)
