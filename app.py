# app.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU (prevents CUDA DLL issues)

import json
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini basic only
from dotenv import load_dotenv

load_dotenv()
# ------------ Config ------------
DEVICE = "cpu"
MODEL_PATH = r"C:/Users/Sanjeet SIngh/Desktop/Projects/PlantDisease/Plant Disease Detection/plant-disease-model-complete.pth"
# full model file
LABELS_JSON = "class_labels.json"            # optional: list of class names
LLM_MODEL = "gemini-2.0-flash"
# ---------------------------------

# Fallback labels (use your own order if not using JSON)
class_labels = [
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Potato___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Tomato___Early_blight",
    "Tomato___Septoria_leaf_spot",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Strawberry___Leaf_scorch",
    "Peach___healthy",
    "Apple___Apple_scab",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Apple___Black_rot",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Apple___Cedar_apple_rust",
    "Tomato___Target_Spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Late_blight",
    "Tomato___Tomato_mosaic_virus",
    "Strawberry___healthy",
    "Apple___healthy",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___Esca_(Black_Measles)",
    "Raspberry___healthy",
    "Tomato___Leaf_Mold",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Pepper,_bell___Bacterial_spot",
    "Corn_(maize)___healthy"
]


# Load labels from JSON if present
if os.path.exists(LABELS_JSON):
    try:
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            class_labels = json.load(f)
    except Exception as e:
        st.warning(f"Could not read {LABELS_JSON}: {e}. Using inline class_labels list.")

def parse_label(label: str):
    plant, status = label.split("___")
    return plant, status.replace("_", " ")

# --- Gemini: basic explanation (no Wikipedia) ---
def explain_disease(label: str) -> str:
    plant, status = parse_label(label)
    prompt = (
        f"You are a plant pathology assistant. Be concise, helpful, and non-prescriptive.\n"
        f"Explain '{status}' in {plant} leaves in 6–8 bullet points:\n"
        f"- what it is\n- visible leaf symptoms\n"
        f"- typical conditions\n- basic prevention/management\n"
        f"- when to consult an expert\n"
        f"Avoid chemical names and diagnostic claims."
    )
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    return llm.invoke(prompt).content


import torch.nn as nn
import torch.nn.functional as F

# Shared helper
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = (out.argmax(dim=1) == labels).float().mean()
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

# Trained model architecture
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)



@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval().to(DEVICE)
    return model

def preprocess(img):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),  # <-- not 224, no CenterCrop
        transforms.ToTensor(),          # <-- you trained without Normalize
    ])
    return tfms(img.convert("RGB"))


def predict(model, img: Image.Image, topk=3):
    if not class_labels:
        raise RuntimeError("class_labels is empty. Provide class_labels.json or fill the list.")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idxs = np.argsort(-probs)[:min(topk, len(class_labels))]
    return [(class_labels[i], float(probs[i])) for i in idxs]

# -------- Streamlit UI --------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Disease Classifier")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    model = load_model()
    try:
        top = predict(model, img, topk=3)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.subheader("Prediction:")
    best_label, best_prob = top[0]
    plant, status = parse_label(best_label)

    if status.lower() == "healthy":
        st.success(f"{plant}: **Healthy** ({best_prob:.1%})")
    else:
        st.error(f"{plant}: **{status}** ({best_prob:.1%})")

    st.write("Top results:")
    for label, p in top:
        p_plant, p_status = parse_label(label)
        st.write(f"- **{p_plant} — {p_status}** — {p:.1%}")

    st.divider()
    if st.button("Explain Disease"):
        with st.spinner("Generating explanation..."):
            st.write(explain_disease(best_label))
else:
    st.info("Upload a clear photo of a single leaf to begin.")
