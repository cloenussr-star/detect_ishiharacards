import os
import requests
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt

# Streamlit setup
st.set_page_config(page_title="Ishihara Card Classifier", layout="wide")
st.title("Ishihara Card Classifier with Grad-CAM")
st.write("Upload an Ishihara card image, and the app will classify it using a Vision Transformer model fine-tuned for Ishihara tests.")

# Model loader
@st.cache_resource
def load_vit():
    model_path = "best_vit_model.pth"
    url = "https://huggingface.co/akmal2222/ishihara-vit-model/resolve/main/best_vit_model.pth"

    # Download model from Hugging Face if not present
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Hugging Face..."):
            response = requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully.")

    # Load model and processor
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    return model, processor

# Load model
model, processor = load_vit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Upload image
uploaded_file = st.file_uploader("Upload an Ishihara card image", type=["jpg", "jpeg", "png"])

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
            return None

        if activations is None:
            return None

        grads = activations.mean(dim=0).detach().cpu().numpy()
        cam = np.maximum(grads, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.resize(cam, (224, 224))
        return cam

    cam = generate_gradcam(model, inputs, pred_class)
    if cam is not None:
        image_resized = image.resize((224, 224))
        img_arr = np.array(image_resized) / 255.0
        plt.figure(figsize=(5, 5))
        plt.imshow(img_arr)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.axis("off")
        st.subheader("Grad-CAM Visualization")
        st.pyplot(plt)
    else:
        st.warning("Grad-CAM could not be generated for this model.")
