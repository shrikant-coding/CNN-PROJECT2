from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "animal_cnn.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = (160, 160)

st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 Animal Image Classifier")
st.write(
    "Upload a picture of one of the 10 animal classes. "
    "The CNN will predict the most likely class."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def load_classes():
    if not CLASS_PATH.exists():
        return []
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
classes = load_classes()

if model is None or not classes:
    st.error(
        "The trained model files are missing. Run `python train.py` first. "
        "The models folder must contain animal_cnn.keras and class_names.json."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        resized = image.resize(IMAGE_SIZE)
        image_array = np.asarray(resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Classifying image..."):
            probabilities = model.predict(image_array, verbose=0)[0]

        if len(probabilities) != len(classes):
            st.error("Model output and class-label file do not match.")
            st.stop()

        best_index = int(np.argmax(probabilities))
        best_class = classes[best_index]["label"]
        confidence = float(probabilities[best_index]) * 100

        st.subheader(f"Prediction: {best_class}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.write("Top predictions")
        top_indices = np.argsort(probabilities)[::-1][:min(5, len(classes))]
        for index in top_indices:
            label = classes[int(index)]["label"]
            score = float(probabilities[int(index)])
            st.write(f"**{label}** — {score * 100:.2f}%")
            st.progress(score)

        st.caption(
            "Confidence is the model's estimated probability, not a guarantee "
            "that the prediction is correct. Results may be weaker for images "
            "that differ from the training dataset."
        )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
