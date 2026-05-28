import streamlit as st
import numpy as np
import json
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

st.set_page_config(
    page_title="Klasifikasi Ketan Putih dan Ketan Hitam",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = (224, 224)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background-color: #0F2D24 !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.75) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #9FE1CB !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-top: 1.2rem;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 13px;
    color: rgba(255,255,255,0.65) !important;
    line-height: 1.7;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* ---- METHOD CARDS SIDEBAR ---- */
.method-card {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.1);
}
.method-card .method-name {
    font-size: 12px;
    font-weight: 600;
    color: #9FE1CB !important;
    margin-bottom: 4px;
}
.method-card .method-desc {
    font-size: 11px;
    color: rgba(255,255,255,0.5) !important;
    line-height: 1.5;
}

/* ---- PAGE HEADER ---- */
.page-header {
    border-bottom: 1px solid rgba(128,128,128,0.3);
    padding-bottom: 1.2rem;
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #1D9E75;
    margin: 0;
}
.page-sub {
    font-size: 14px;
    color: rgba(128,128,128,0.9);
    margin-top: 4px;
}
.badge {
    display: inline-block;
    background: #1D9E75;
    color: #ffffff;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-top: 8px;
}

/* ---- UPLOAD HINT ---- */
.upload-hint {
    border: 1.5px dashed rgba(29,158,117,0.5);
    border-radius: 12px;
    padding: 1.2rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    background: rgba(29,158,117,0.05);
}
.upload-hint p {
    font-size: 13px;
    color: rgba(128,128,128,0.85) !important;
    margin: 0;
}

/* ---- RESULT PILLS ---- */
.result-label-putih {
    display: inline-block;
    background: #1D9E75;
    color: #ffffff !important;
    font-size: 14px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 20px;
}
.result-label-hitam {
    display: inline-block;
    background: #534AB7;
    color: #ffffff !important;
    font-size: 14px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 20px;
}

/* ---- METHOD RESULT BLOCKS ---- */
.method-block {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 12px;
    background: rgba(128,128,128,0.04);
}
.method-header-bar {
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(128,128,128,0.06);
}
.method-header-title {
    font-size: 13px;
    font-weight: 600;
    color: #1D9E75 !important;
    margin: 0;
}
.method-header-sub {
    font-size: 11px;
    color: rgba(128,128,128,0.8) !important;
    margin: 0;
}
.method-body {
    padding: 14px 16px;
}

/* ---- K-MEANS BOX ---- */
.kmeans-box {
    background: rgba(186,117,23,0.12);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    color: #BA7517 !important;
    line-height: 1.7;
    border: 1px solid rgba(186,117,23,0.3);
    font-weight: 500;
}
.kmeans-warning {
    font-size: 11px;
    color: rgba(128,128,128,0.75) !important;
    margin-top: 8px;
    line-height: 1.6;
}

/* ---- SECTION LABEL ---- */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: rgba(128,128,128,0.75) !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 10px;
}

/* ---- CHIPS ---- */
.chip-putih {
    display: inline-block;
    background: rgba(29,158,117,0.15);
    color: #1D9E75 !important;
    border: 1px solid rgba(29,158,117,0.4);
    font-size: 12px;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 20px;
    margin: 4px;
}
.chip-hitam {
    display: inline-block;
    background: rgba(83,74,183,0.15);
    color: #534AB7 !important;
    border: 1px solid rgba(83,74,183,0.4);
    font-size: 12px;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 20px;
    margin: 4px;
}

/* ---- EMPTY STATE ---- */
.empty-state {
    text-align: center;
    padding: 2rem 1rem;
    color: rgba(128,128,128,0.75) !important;
}
.empty-title {
    font-size: 15px;
    font-weight: 500;
    color: rgba(128,128,128,0.9) !important;
    margin-bottom: 6px;
}
.empty-sub {
    font-size: 13px;
    line-height: 1.7;
    color: rgba(128,128,128,0.75) !important;
}

/* ---- IMG CAPTION ---- */
.img-caption {
    font-size: 11px;
    color: rgba(128,128,128,0.75) !important;
    text-align: center;
    margin-top: 6px;
}

/* ---- PROGRESS BAR ---- */
div[data-testid="stProgress"] > div > div {
    background-color: #1D9E75 !important;
}
div[data-testid="stProgress"] > div {
    background-color: rgba(128,128,128,0.2) !important;
}

/* ---- METRIC ---- */
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 14px 16px;
    background: rgba(128,128,128,0.04);
}
[data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    color: rgba(128,128,128,0.8) !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1D9E75 !important;
}
[data-testid="stMetricDelta"] svg {
    display: none !important;
}
[data-testid="stMetricDelta"] {
    color: rgba(128,128,128,0.8) !important;
    font-size: 11px !important;
}

/* ---- CAPTION ---- */
[data-testid="stCaptionContainer"] p {
    color: rgba(128,128,128,0.8) !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD SEMUA MODEL
# ============================================================
@st.cache_resource
def load_all_models():
    mobilenet   = load_model('models/model_mobilenetv2.keras')
    effnet      = load_model('models/efficientnet_extractor.keras')
    knn         = joblib.load('models/knn_model.pkl')
    scaler      = joblib.load('models/scaler.pkl')
    pca         = joblib.load('models/pca_model.pkl')
    kmeans      = joblib.load('models/kmeans_model.pkl')
    class_names = joblib.load('models/class_names.pkl')
    with open('models/model_info.json', 'r') as f:
        info    = json.load(f)
    return mobilenet, effnet, knn, scaler, pca, kmeans, class_names, info

mobilenet_model, effnet_extractor, knn, scaler, pca, kmeans, CLASS_NAMES, MODEL_INFO = load_all_models()


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_image_mobilenet(img):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = mobilenet_preprocess(arr)
    return arr

def preprocess_image_efficientnet(img):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = efficientnet_preprocess(arr)
    return arr


# ============================================================
# PREDIKSI
# ============================================================
def predict_mobilenet(img):
    arr  = preprocess_image_mobilenet(img)
    pred = mobilenet_model.predict(arr, verbose=0)[0][0]
    if pred > 0.5:
        label      = 'Ketan Putih'
        confidence = float(pred)
    else:
        label      = 'Ketan Hitam'
        confidence = float(1 - pred)
    return label, confidence

def predict_knn(img):
    arr      = preprocess_image_efficientnet(img)
    features = effnet_extractor.predict(arr, verbose=0)
    pred     = knn.predict(features)[0]
    proba    = knn.predict_proba(features)[0]
    label    = CLASS_NAMES[pred]
    label    = 'Ketan Putih' if 'putih' in label else 'Ketan Hitam'
    confidence = float(max(proba))
    return label, confidence

def predict_kmeans(img):
    arr      = preprocess_image_efficientnet(img)
    features = effnet_extractor.predict(arr, verbose=0)
    scaled   = scaler.transform(features)
    reduced  = pca.transform(scaled)
    cluster  = kmeans.predict(reduced)[0]
    sil      = MODEL_INFO['sil_score']
    metode   = MODEL_INFO['metode_label']
    return cluster, sil, metode


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom:1.2rem;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:1rem">
        <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#E1F5EE;line-height:1.4">
            Sistem Klasifikasi Ketan
        </div>
        <div style="font-size:11px;color:rgba(255,255,255,0.35);margin-top:4px">
            Deep Learning + Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Metode")
    st.markdown(f"""
    <div class="method-card">
        <div class="method-name">MobileNetV2</div>
        <div class="method-desc">CNN dengan transfer learning dari ImageNet. Supervised learning.</div>
    </div>
    <div class="method-card">
        <div class="method-name">KNN (k=3)</div>
        <div class="method-desc">Fitur diekstraksi menggunakan EfficientNetB0. Supervised learning.</div>
    </div>
    <div class="method-card">
        <div class="method-name">K-Means</div>
        <div class="method-desc">Clustering unsupervised. Silhouette Score: {MODEL_INFO['sil_score']:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Dataset")
    st.markdown("""
    <div style="font-size:13px;color:rgba(255,255,255,0.65);line-height:2">
        Total gambar &nbsp;&nbsp;<strong style="color:#9FE1CB">824</strong><br>
        Per kelas &nbsp;&nbsp;<strong style="color:#9FE1CB">412</strong><br>
        Train / Val / Test &nbsp;&nbsp;<strong style="color:#9FE1CB">70% / 20% / 10%</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:rgba(255,255,255,0.35);line-height:1.9">
        <strong style="color:rgba(255,255,255,0.6)">Nadjwa Tasya Safira</strong><br>
        2315061024<br>
        Teknik Informatika<br>
        Universitas Lampung
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HALAMAN UTAMA
# ============================================================
st.markdown("""
<div class="page-header">
    <div class="page-title">Klasifikasi Ketan Putih dan Ketan Hitam</div>
    <div class="page-sub">Upload gambar ketan untuk dianalisis menggunakan tiga metode: MobileNetV2, KNN, dan K-Means.</div>
    <div class="badge">3 Model Aktif</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload gambar ketan",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is not None:
    img = Image.open(uploaded)

    col_img, col_hasil = st.columns([1, 1.6], gap="large")

    with col_img:
        st.image(img, use_column_width=True)
        st.markdown(
            f'<div class="img-caption">{uploaded.name} &nbsp;|&nbsp; {img.size[0]} x {img.size[1]} px</div>',
            unsafe_allow_html=True
        )

    with col_hasil:
        with st.spinner("Menganalisis gambar..."):
            label_mn,  conf_mn            = predict_mobilenet(img)
            label_knn, conf_knn           = predict_knn(img)
            cluster_km, sil_km, metode_km = predict_kmeans(img)

        # ---- MobileNetV2 ----
        pill_mn = "result-label-putih" if label_mn == "Ketan Putih" else "result-label-hitam"
        st.markdown(f"""
        <div class="method-block">
            <div class="method-header-bar">
                <div style="width:22px;height:22px;border-radius:6px;background:#1D9E75;color:#fff;
                            font-size:10px;font-weight:700;display:flex;align-items:center;
                            justify-content:center;flex-shrink:0;">1</div>
                <div>
                    <div class="method-header-title">MobileNetV2</div>
                    <div class="method-header-sub">CNN Utama — Transfer Learning</div>
                </div>
            </div>
            <div class="method-body">
                <span class="{pill_mn}">{label_mn}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf_mn)
        st.caption(f"Confidence: {conf_mn*100:.2f}%")

        # ---- KNN ----
        pill_knn = "result-label-putih" if label_knn == "Ketan Putih" else "result-label-hitam"
        st.markdown(f"""
        <div class="method-block" style="margin-top:16px">
            <div class="method-header-bar">
                <div style="width:22px;height:22px;border-radius:6px;background:#1D9E75;color:#fff;
                            font-size:10px;font-weight:700;display:flex;align-items:center;
                            justify-content:center;flex-shrink:0;">2</div>
                <div>
                    <div class="method-header-title">KNN — K-Nearest Neighbor</div>
                    <div class="method-header-sub">Fitur dari EfficientNetB0</div>
                </div>
            </div>
            <div class="method-body">
                <span class="{pill_knn}">{label_knn}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf_knn)
        st.caption(f"Confidence: {conf_knn*100:.2f}%")

        # ---- K-Means ----
        st.markdown(f"""
        <div class="method-block" style="margin-top:16px">
            <div class="method-header-bar">
                <div style="width:22px;height:22px;border-radius:6px;background:#1D9E75;color:#fff;
                            font-size:10px;font-weight:700;display:flex;align-items:center;
                            justify-content:center;flex-shrink:0;">3</div>
                <div>
                    <div class="method-header-title">K-Means Clustering</div>
                    <div class="method-header-sub">Unsupervised — {metode_km}</div>
                </div>
            </div>
            <div class="method-body">
                <div class="kmeans-box">
                    Gambar masuk ke <strong>Cluster {cluster_km}</strong>
                    &nbsp;&mdash;&nbsp; Silhouette Score: <strong>{sil_km:.4f}</strong>
                </div>
                <div class="kmeans-warning">
                    Catatan: K-Means bersifat unsupervised sehingga tidak menghasilkan label kelas
                    secara langsung. Hasil berupa nomor cluster berdasarkan kemiripan fitur gambar.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Ringkasan Hasil</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("MobileNetV2", label_mn, f"{conf_mn*100:.1f}% confidence")
    with col_b:
        st.metric("KNN", label_knn, f"{conf_knn*100:.1f}% confidence")
    with col_c:
        st.metric("K-Means", f"Cluster {cluster_km}", f"Silhouette: {sil_km:.4f}")

else:
    st.markdown("""
    <div class="upload-hint">
        <p>Drag & drop gambar ketan di sini, atau klik tombol Browse files di atas.</p>
        <p style="margin-top:10px">
            <span class="chip-putih">Ketan Putih</span>
            <span class="chip-hitam">Ketan Hitam</span>
        </p>
    </div>
    <div class="empty-state">
        <div class="empty-title">Belum ada gambar yang dianalisis</div>
        <div class="empty-sub">
            Upload gambar ketan putih atau ketan hitam untuk memulai klasifikasi<br>
            dengan tiga model machine learning sekaligus.
        </div>
    </div>
    """, unsafe_allow_html=True)