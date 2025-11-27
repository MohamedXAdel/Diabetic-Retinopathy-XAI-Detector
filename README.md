# Diabetic Retinopathy XAI Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-EfficientNet-orange)

A deep-learning powered **Diabetic Retinopathy (DR) Detection System** integrated with **Explainable AI (XAI)** support. This project utilizes **EfficientNet** for high-accuracy classification and provides model interpretability using **LIME** and **Grad-CAM**, allowing medical professionals to understand *why* a specific diagnosis was made.

The application is accessible via a user-friendly **Streamlit** interface for real-time medical image analysis.

🔗 **Live Demo:** [Click here to view the App](https://diabetic-retinopathy-xai-detector.streamlit.app/)

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model & XAI Details](#model--xai-details)
- [Contact](#contact)

---

## 📖 About the Project

Diabetic Retinopathy is a leading cause of blindness. Early detection is crucial, but deep learning models often act as "black boxes." This project bridges the gap between accuracy and trust by combining state-of-the-art classification with visual explanations.

The system accepts retinal fundus images, classifies them into varying stages of Diabetic Retinopathy, and generates heatmaps and superpixel explanations to highlight the regions of the eye contributing to the model's decision.

---

## ✨ Key Features

* **High-Accuracy Detection**: Utilizes **EfficientNet** architecture for robust image classification.
* **Real-time Analysis**: Instant prediction results through a web-based UI.
* **Explainable AI (XAI)**:
    * **Grad-CAM** (Gradient-weighted Class Activation Mapping): Visualizes the regions of interest as a heatmap.
    * **LIME** (Local Interpretable Model-agnostic Explanations): Highlights specific superpixels that influenced the prediction.
* **Interactive Interface**: Built with **Streamlit** for easy image uploading and visualization.

---

## 🛠 Technologies Used

* **Python**: Core programming language.
* **Streamlit**: Web framework for the user interface.
* **EfficientNet**: Convolutional Neural Network (CNN) architecture for feature extraction.
* **TensorFlow / Keras**: Deep learning framework.
* **LIME**: For local model interpretability.
* **OpenCV**: For image processing and heatmap generation.

---

## 📂 Project Structure

```bash
Diabetic-Retinopathy-XAI-Detector/
├── assets/                 # Static assets (css file)
├── data/                   # Dataset directory 
├── model_training/         # Scripts used for training the model
├── models/                 # Saved model weights 
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code for helper functions
├── AI_Explanation.py       # Wrapper for generating XAI explanations
├── Grad_Cam_XAI.py         # Implementation of Grad-CAM logic
├── LIME_XAI.py             # Implementation of LIME logic
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
