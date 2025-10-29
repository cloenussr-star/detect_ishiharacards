import os
import requests
import torch
import numpy as np
from PIL import Image
import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="Ishihara Card Classifier", layout="wide")
st.title("Ishihara Card Classifier")
st.write("Upload an Ishihara card image, and the app will classify it using a Vision Transformer model fine-tuned for Ishihara tests.")

# ------------------------------------------------------------
# Load fine-tuned model (with flexible detection)
# ------------------------------------------------------------
@st.cache_resource
def load_vit():
    import os, requests, torch
    from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig

    model_path = "best_vit_model.pth"
    url = "https://huggingface.co/akmal2222/ishihara-vit-model/resolve/main/best_vit_model.pth"

    # Download model file if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Hugging Face..."):
            r = requests.get(url)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)
            st.success("Model downloaded successfully.")

    # Try loading
    raw = torch.load(model_path, map_location="cpu")

    # Case 1: full model object
    if isinstance(raw, torch.nn.Module):
        st.info("Loaded full model object from checkpoint.")
        model = raw
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model.eval()
        return model, processor

    # Case 2: dict of state_dict or wrapped dict
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]

    # Try multiple possible num_labels
    for num_labels in [2, 3, 4, 5]:
        try:
            config = ViTConfig.from_pretrained(
                "google/vit-base-patch16-224-in21k", num_labels=num_labels
            )
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", config=config
            )
            model.load_state_dict(raw, strict=False)
            st.success(f"Loaded model successfully with num_labels={num_labels}.")
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model.eval()
            return model, processor
        except Exception:
            continue

    st.error("Failed to load model: architecture mismatch.")
    raise RuntimeError("Checkpoint incompatible with architecture.")


# ------------------------------------------------------------
# Initialize model
# ------------------------------------------------------------
model, processor = load_vit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------------------------------------------
# Upload Image
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an Ishihara card image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and run inference
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = probs[0, pred_class].item()

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Predicted class index: {pred_class}")
    st.write(f"Confidence: {confidence:.2%}")

    k = min(probs.shape[1], 3)
    topk_prob, topk_idx = torch.topk(probs, k, dim=1)
    st.table({
        "Class Index": topk_idx[0].cpu().numpy(),
        "Confidence": [f"{p:.2%}" for p in topk_prob[0].cpu().numpy()]
    })

    # --------------------------------------------------------
    # Grad-CAM visualization
    # --------------------------------------------------------
    def generate_gradcam(model, inputs, target_class):
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        logits[:, target_class].backward()

        try:
            grads = model.vit.encoder.layer[-1].layernorm_before.weight.grad
        except AttributeError:
            return None

        if grads is None:
            return None

        grads = grads.mean(dim=0).detach().cpu().numpy()
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
