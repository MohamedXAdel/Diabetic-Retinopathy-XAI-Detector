import glob
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image as PILImage
from torchvision import models,transforms
import torch.nn as nn
from src.preprocess import preprocess_image




def get_device(provided_device=None):
    if provided_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return provided_device

def load_models(binary_path, severity_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Binary model ----
    binary_model = models.efficientnet_b0(weights=None)  # Do not load pretrained weights
    num_features = binary_model.classifier[1].in_features
    binary_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 1)
    )
    binary_model.load_state_dict(torch.load(binary_path, map_location=device))
    binary_model.to(device)
    binary_model.eval()

    # ---- Severity model (example: same structure, adjust if you trained differently) ----
    severity_model = models.efficientnet_b0(weights=None)
    num_features = severity_model.classifier[1].in_features
    severity_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 4)  # 4 classes for severity
    )
    severity_model.load_state_dict(torch.load(severity_path, map_location=device))
    severity_model.to(device)
    severity_model.eval()

    return binary_model, severity_model, device


def get_two_stage_prediction(binary_model, severity_model, image_tensor, threshold=0.5):
    # Ensure models are in eval mode
    binary_model.eval()
    severity_model.eval()
    
    with torch.no_grad():
        #Stage 1: Binary
        bin_logits = binary_model(image_tensor)
        bin_prob = torch.sigmoid(bin_logits).item()
        
        if bin_prob < threshold:
            return 0  # No DR
        
        # Stage 2: Severity
        # If we are here, the model thinks it IS DR.
        # Pass the SAME image to the severity model.
        sev_logits = severity_model(image_tensor)
        _, sev_pred_idx = torch.max(sev_logits, 1) # Returns 0, 1, 2, or 3
        
        final_pred = sev_pred_idx.item() + 1 # Convert back to 1, 2, 3, 4
        
        return final_pred


def predict_single_image(image_path,binary_model,sev_model):
    """
    Reads an image file, preprocesses it, and runs the full 2-stage pipeline.
    """
    try:
        # Load and Preprocess 
        pil_img = preprocess_image(image_path, sigmaX=10)
        
        # Transform (Resize -> Tensor -> Normalize)
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        
    except Exception as e:
        return None, f"Error: {e}"

    # Run Pipeline
    prediction = get_two_stage_prediction(binary_model, sev_model, tensor_img, threshold=0.5)
    
    # Map to Text
    labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}
    return prediction, labels[prediction]

#----------------------------------------------------------------------------------------------------------------------

# Page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .big-font { font-size: 52px !important; font-weight: bold; text-align: center; }
    .result-card { padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; }
    .no-dr { background: linear-gradient(90deg, #00ff00, #33ff33); color: black; }
    .mild { background: linear-gradient(90deg, #99cc00, #ccff33); color: black; }
    .moderate { background: linear-gradient(90deg, #ffcc00, #ffeb3b); color: black; }
    .severe { background: linear-gradient(90deg, #ff6600, #ff9800); color: white; }
    .proliferative { background: linear-gradient(90deg, #ff0000, #ff4444); color: white; }
    .confidence { font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<p class="big-font">Diabetic Retinopathy Detector</p>', unsafe_allow_html=True)
st.markdown("Upload a retinopathy image and get an instant AI diagnosis using a multi classification deep learning model.", unsafe_allow_html=True)

# Load models once
@st.cache_resource
def load_model_once():
    with st.spinner("Loading AI models..."):
        binary_model, severity_model, used_device = load_models(
            binary_path = os.path.join("models", "final_binary_model.pth"),
            severity_path = os.path.join("models", "final_severity_model.pth"),
            device=None
        )
    return binary_model, severity_model, used_device

binary_model, severity_model, device = load_model_once()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a retinopathy image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)


if uploaded_file is not None:

    pil_image = PILImage.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Retinal Image", use_column_width=True)

    if st.button("Analyze Image for Diabetic Retinopathy", type="primary"):
        with st.spinner("Analyzing image... This may take a few seconds."):

            # Save uploaded file temporarily because preprocess_image expects a path
            temp_path = f"temp_uploaded_{uploaded_file.name}"
            pil_image.save(temp_path)

            try:
                # existing prediction function 
                pred_id, pred_text = predict_single_image(
                    temp_path, binary_model, severity_model
                )

                # Clean up temp file
                os.remove(temp_path)

            except Exception as e:
                os.remove(temp_path)
                st.error(f"Error during prediction: {e}")
                st.stop()

        # ------------------------------------------------------------------
        # Beautiful result card
        # ------------------------------------------------------------------
        severity_class = pred_text.lower().replace(" ", "-").replace("dr", "")
        if pred_id is not None:
            st.markdown(f"""
            <div class="result-card {severity_class}">
                <h2>Diagnosis Result</h2>
                <p class="confidence">{pred_text}</p>
                <p>Severity Level: {pred_id if pred_id > 0 else 0} 
                   {"" if pred_id == 0 else f"(Class {pred_id})"}</p>
            </div>
            """, unsafe_allow_html=True)

            # Optional: Add medical advice
            if pred_id == 0:
                st.success("Great news! No signs of diabetic retinopathy detected.")
            elif pred_id == 1:
                st.warning("Mild NPDR detected. Regular screening recommended.")
            elif pred_id == 2:
                st.warning("Moderate NPDR detected. Please consult an ophthalmologist soon.")
            elif pred_id == 3:
                st.error("Severe NPDR detected. Urgent specialist referral recommended!")
            elif pred_id == 4:
                st.error("Proliferative DR detected. Immediate medical attention required!")
        else:
            st.error("Prediction failed. Please try another image.")
# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Two-stage Deep Learning Model • APTOS 2019 Dataset • Not for clinical use"
    "</p>",
    unsafe_allow_html=True
)
