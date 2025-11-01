# app.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU (prevents CUDA DLL issues)

import json
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import torch.nn as nn
import torch.nn.functional as F
import json


import pandas as pd
from weather import get_15day_forecast as get_14day_forecast, flatten_forecast, WeatherError

load_dotenv()

# after imports + load_dotenv + st.set_page_config
from weatherforecast_page import render as render_weather

# --- Router ---
qp = st.query_params
if qp.get("view", "main") == "weather":
    render_weather()   # call every run
    st.stop()


top_cols = st.columns([8, 2])
with top_cols[1]:
    if st.button("☁️ Weather", use_container_width=True):
        st.query_params["view"] = "weather"
        st.rerun()



DEVICE = "cpu"
MODEL_PATH = r"C:/Users/Sanjeet SIngh/Desktop/Projects/PlantDisease/Plant Disease Detection/plant-disease-model-complete.pth"
LABELS_JSON = "class_labels.json"
LLM_MODEL = "gemini-2.0-flash"

# ---------- Labels ----------
if not os.path.exists(LABELS_JSON):
    st.error("class_labels.json not found. Please place it next to app.py.")
    st.stop()

# ---------- Utils ----------
def parse_label(label: str):
    plant, status = label.split("___")
    return plant, status.replace("_", " ")

with open(LABELS_JSON, "r", encoding="utf-8") as f:
    class_labels = json.load(f)

class_to_plant = [parse_label(label)[0] for label in class_labels]
unique_plants = sorted({p for p in class_to_plant})



# class_to_plant = [parse_label(label)[0] for label in class_labels] converts :
# [
#   "Tomato___Early_blight",
#   "Tomato___Late_blight",
#   "Potato___Healthy",
#   "Corn___Northern_Leaf_Blight"
# ]
# to
# class_to_plant = ["Tomato", "Tomato", "Potato", "Corn"]


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
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
            nn.Linear(512, num_diseases),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)
#----------------------------------Weather Data-----------------------------
@st.cache_data(ttl=3600)
def fetch_weather_rows(place: str) -> list[dict]:
    data = get_14day_forecast(place)
    return flatten_forecast(data)  # list of dicts (14 rows)

def format_weather_text(rows: list[dict], place: str, days: int = 7) -> str:
    """
    Compact text for prompts: Date, condition, Tmin–Tmax °C, rain%.
    Keep days small (5–7) to save tokens.
    """
    rows = rows[:days]
    lines = []
    for r in rows:
        d   = r.get("Date")
        cnd = r.get("Condition")
        tmin = r.get("Min (°C)")
        tmax = r.get("Max (°C)")
        rain = r.get("Rain chance (%)")
        lines.append(f"- {d}: {cnd}, {tmin}–{tmax}°C, rain {rain}%")
    return f"Weather forecast for {place} (next {len(rows)} days):\n" + "\n".join(lines)

def format_weather_json(rows: list[dict], place: str, days: int = 14) -> str:
    """
    JSON string if you prefer structured context in the prompt.
    """
    cols = ["Date","Condition","Min (°C)","Max (°C)","Rain chance (%)","Max wind (kph)","Avg humidity (%)","Sunrise","Sunset"]
    slim = [{k: r.get(k) for k in cols} for r in rows[:days]]
    return json.dumps({"location": place, "forecast": slim}, ensure_ascii=False)



#--------------------------Loading Model------------------------------------
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval().to(DEVICE)
    return model

def preprocess(img: Image.Image):
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # match training (no normalization)
    ])
    return tfms(img.convert("RGB"))


def predict(model, img: Image.Image, topk=10, selected_plant="Auto"):
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)  # raw scores, all plants

        # Apply masking here
        if selected_plant != "Auto":
            mask = torch.tensor(
                [p == selected_plant for p in class_to_plant],
                device=logits.device
            )
            masked_logits = logits.clone()
            masked_logits[:, ~mask] = float('-inf')
        else:
            masked_logits = logits

        # Softmax AFTER masking
        probs = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()

    idxs = np.argsort(-probs)[:min(topk, len(class_labels))]
    return [(class_labels[i], float(probs[i])) for i in idxs]


# ---------- UI ----------
st.title("🌿 Plant Health Assistant")
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

# User chooses whether to filter prediction
plant_choice = st.selectbox(
        "Filter predictions to a plant (optional):",
        ["Auto"] + unique_plants,
        index=0
)


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    model = load_model()
    try:
        top = predict(model, img, topk=3, selected_plant=plant_choice)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.subheader("Prediction")
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

    # =========================
    # Chat Assistant
    # =========================
    # Reset chat when context changes
    ctx_new = {"plant": plant, "status": status, "best_prob": float(best_prob)}
    if "ctx" not in st.session_state or st.session_state.ctx != ctx_new:
        st.session_state.ctx = ctx_new
        st.session_state.chat_messages = []

    st.subheader("🤖 Ask the Plant Assistant")
    st.caption(
        f"Context → Plant: **{plant}**, Status: **{status}**, Confidence: **{best_prob:.1%}**"
    )

    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Clear Chat"):
            st.session_state.chat_messages = []

    # Rendering history
    for msg in st.session_state.get("chat_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_msg = st.chat_input("Ask about causes, prevention, care, signs to monitor, etc.")
    if user_msg:
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Build prompt with context + brief history
        system_instructions = (
            "You are a helpful plant health assistant. "
            "Be concise (5–8 bullets), practical, and non-prescriptive. "
            "Avoid recommending specific chemicals or products. "
            "If the leaf is healthy, give general care & monitoring tips. "
            "Always include when to consult an expert.\n\n"
            f"Detected context:\n- Plant: {plant}\n- Status: {status}\n- Model confidence: {best_prob:.1%}\n"
        )
        history_text = ""
        for m in st.session_state.chat_messages:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n"

        # decide a place; you can also store user's last input in session_state
        weather_place = st.session_state.get("weather_place", "New Delhi, IN")

        try:
            w_rows = fetch_weather_rows(weather_place)
            weather_ctx = format_weather_text(w_rows, weather_place, days=7)   # concise text
            # Or use JSON:
            # weather_ctx = "WEATHER_JSON:\n" + format_weather_json(w_rows, weather_place, days=14)
        except Exception as _:
            weather_ctx = "Weather data unavailable."

        system_instructions = (
            "You are a helpful plant health assistant. "
            "Be concise (5–8 bullets), practical, and non-prescriptive. "
            "Avoid recommending specific chemicals or products. "
            "If the leaf is healthy, give general care & monitoring tips. "
            "Always include when to consult an expert.\n\n"
            f"Detected context:\n- Plant: {plant}\n- Status: {status}\n- Model confidence: {best_prob:.1%}\n"
            f"\nAdditional context:\n{weather_ctx}\n"   # <-- inject weather here
            )

        
        
        full_prompt = (
            system_instructions
            + "\nConversation so far:\n"
            + history_text
            + "\nNow answer the last user question clearly and helpfully."
        )

        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                temperature=0.2,
                google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            )
            answer = llm.invoke(full_prompt).content.strip()
        except Exception as e:
            answer = (
                "I couldn't generate a response (LLM error). "
                "Please check your API key or internet connection.\n\n"
                f"Details: {e}"
            )

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload a clear photo of a single leaf to begin.")
