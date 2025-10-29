import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, SwinForImageClassification, AutoImageProcessor
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Ishihara Card Classifier", layout="wide")
st.title("Ishihara Card Classifier with Grad-CAM")
st.write("Upload an Ishihara card image, select a model, and visualize predictions with Grad-CAM.")

# Load models
@st.cache_resource
def load_vit():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model.load_state_dict(torch.load("best_vit_model.pth", map_location="cpu"))
    model.eval()
    return model, processor

@st.cache_resource
def load_swin():
    model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
    processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    model.load_state_dict(torch.load("best_swin_model.pth", map_location="cpu"))
    model.eval()
    return model, processor

# Model selection
model_choice = st.selectbox("Select model:", ["Vision Transformer (ViT)", "Swin Transformer"])
if model_choice == "Vision Transformer (ViT)":
    model, processor = load_vit()
else:
    model, processor = load_swin()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Image upload
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Inference
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = probs[0, pred_class].item()
        top3_prob, top3_idx = torch.topk(probs, 3, dim=1)

    st.subheader("Predictions")
    st.write(f"Predicted class: {pred_class}")
    st.write(f"Confidence: {confidence:.2%}")

    top3_table = {
        "Class": top3_idx[0].cpu().numpy(),
        "Confidence": [f"{p:.2%}" for p in top3_prob[0].cpu().numpy()]
    }
    st.table(top3_table)

    # Grad-CAM
    def generate_gradcam(model, inputs, target_class):
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        logits[:, target_class].backward()

        try:
            activations = model.vit.encoder.layer[-1].layernorm_before.weight.grad
        except AttributeError:
            try:
                activations = model.swin.encoder.layers[-1].blocks[-1].norm1.weight.grad
            except Exception:
                return None

        if activations is None:
            return None

        grads = activations.mean(dim=0).detach().cpu().numpy()
        cam = np.maximum(grads, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam

    cam = generate_gradcam(model, inputs, pred_class)
    if cam is not None:
        image_resized = image.resize((224, 224))
        original = np.array(image_resized)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        st.subheader("Grad-CAM Visualization")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="Original", use_container_width=True)
        with col2:
            st.image(heatmap, caption="Heatmap", use_container_width=True)
        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)
    else:
        st.warning("Grad-CAM could not be generated for this model.")
