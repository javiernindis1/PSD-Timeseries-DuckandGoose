import streamlit as st
import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
import os
import tempfile

# =============================
# CONFIG
# =============================
MODEL_DIR = "model_svm_mfcc"

st.set_page_config(
    page_title="Duck & Goose Classifier",
    layout="centered"
)

# =============================
# LOAD ARTIFACTS
# =============================
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "svm_mfcc_model.pkl"), "rb") as f:
        svm = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "mfcc_config.pkl"), "rb") as f:
        cfg = pickle.load(f)

    return svm, scaler, le, cfg


svm, scaler, le, cfg = load_artifacts()

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(ts_1d):
    # Z-score normalization (same as training)
    ts = (ts_1d - ts_1d.mean()) / (ts_1d.std() if ts_1d.std() != 0 else 1.0)

    mfcc = librosa.feature.mfcc(
        y=ts.astype(np.float32),
        sr=cfg["SR"],          # dummy sample rate (konsisten)
        n_mfcc=cfg["N_MFCC"]
    )

    feat = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc.min(axis=1),
        mfcc.max(axis=1),
    ]).reshape(1, -1)

    feat_scaled = scaler.transform(feat)
    return feat_scaled, ts


def predict(ts_1d):
    feat_scaled, ts_norm = extract_features(ts_1d)
    probs = svm.predict_proba(feat_scaled)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label, probs, ts_norm

# =============================
# LOAD MP3
# =============================
def load_mp3(uploaded_file):
    # simpan sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # load audio → mono
    wav, _ = librosa.load(
        tmp_path,
        sr=None,     # sample rate asli (tidak digunakan absolut)
        mono=True
    )

    os.remove(tmp_path)
    return wav.astype(np.float32)

# =============================
# UI - MAIN PAGE
# =============================
st.title("🦆 Duck & Goose Classifier")

st.write("""
Aplikasi ini menerima **file audio (.mp3)** atau **file sinyal (.npy)**
dan memprediksi **jenis burung (duck & goose)**
menggunakan pendekatan **MFCC + Support Vector Machine (SVM)**.
""")

# =============================
# SIDEBAR - INFO DATA AUDIO
# =============================
st.sidebar.markdown("""
## 🎧 Sumber Data Audio Uji

Sebagai data uji tambahan, tersedia kumpulan audio suara **duck dan goose**
yang disimpan pada Google Drive berikut:

🔗 https://drive.google.com/drive/folders/12P8EAjT3BD-pahCghr9bzeCd11xVn0D3

Folder tersebut berisi:
- Rekaman suara duck dan goose
- Audio diunduh dari platform **YouTube**
- Format file **.mp3**

Audio pada folder tersebut dapat diunduh dan
**diunggah ke aplikasi Streamlit ini**
untuk dilakukan proses klasifikasi otomatis.
""")

st.divider()

# =============================
# FILE UPLOADER
# =============================
uploaded = st.file_uploader(
    "Upload file (.mp3 atau .npy)",
    type=["mp3", "npy"]
)

if uploaded is not None:

    # =============================
    # LOAD INPUT
    # =============================
    if uploaded.name.endswith(".mp3"):
        ts = load_mp3(uploaded)
        st.info("Input MP3 berhasil dimuat")
        st.audio(uploaded)
    else:
        ts = np.load(uploaded)
        st.info("Input NPY berhasil dimuat")

    # =============================
    # VALIDATION
    # =============================
    if ts.ndim != 1:
        st.error("Input harus berupa sinyal 1D.")
    else:
        pred_label, probs, ts_norm = predict(ts)

        # =============================
        # RESULT
        # =============================
        st.subheader("✅ Hasil Prediksi")
        st.markdown(f"### **{pred_label}**")

        # =============================
        # CONFIDENCE
        # =============================
        st.subheader("📊 Confidence Prediksi")
        fig1 = plt.figure(figsize=(6,3))
        plt.bar(le.classes_, probs)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Probability")
        plt.tight_layout()
        st.pyplot(fig1)

        # =============================
        # WAVEFORM
        # =============================
        st.subheader("📈 Waveform (Normalized)")
        fig2 = plt.figure(figsize=(8,3))
        plt.plot(ts_norm)
        plt.xlabel("Time")
        plt.ylabel("Amplitude (Z-score)")
        plt.tight_layout()
        st.pyplot(fig2)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("MFCC + SVM | CRISP-DM | PSD UAS")
