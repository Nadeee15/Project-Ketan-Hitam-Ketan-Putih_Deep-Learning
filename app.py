import streamlit as st
import numpy as np
import json
import joblib
from PIL import Image
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=Syne:wght@600;700&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background-color: #0a0a0a !important;
}
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1100px !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background-color: #0F2D24 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] > div {
    padding: 1.8rem 1.4rem !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.7) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 1rem 0 !important;
}

/* ===== HIDE DEFAULT STREAMLIT CHROME ===== */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(29,158,117,0.06) !important;
    border: 1.5px dashed rgba(29,158,117,0.4) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: rgba(160,160,160,0.85) !important;
    font-size: 13px !important;
}
[data-testid="stBaseButton-secondary"] {
    background: #1D9E75 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #16805f !important;
}

/* ===== PROGRESS BAR ===== */
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 4px !important;
    height: 5px !important;
}
[data-testid="stProgress"] > div > div {
    background: #1D9E75 !important;
    border-radius: 4px !important;
}

/* ===== CAPTION ===== */
[data-testid="stCaptionContainer"] p {
    color: rgba(160,160,160,0.75) !important;
    font-size: 11px !important;
}

/* ===== SPINNER ===== */
[data-testid="stSpinner"] * {
    color: rgba(160,160,160,0.7) !important;
    font-size: 13px !important;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: #161616 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: rgba(160,160,160,0.6) !important;
}
[data-testid="stMetricValue"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    color: #1D9E75 !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ===== IMAGE ===== */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ===== CUSTOM COMPONENTS ===== */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
}
.page-sub {
    font-size: 13px;
    color: rgba(160,160,160,0.75);
    margin: 0;
}
.top-bar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 1.2rem;
    margin-bottom: 1.5rem;
}
.badge {
    background: rgba(29,158,117,0.18);
    color: #1D9E75;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(29,158,117,0.35);
    white-space: nowrap;
    margin-top: 4px;
}
.method-block {
    background: #161616;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 10px;
}
.method-header {
    background: #1c1c1c;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.method-num {
    width: 22px;
    height: 22px;
    min-width: 22px;
    border-radius: 6px;
    background: #1D9E75;
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.method-name {
    font-size: 13px;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
    line-height: 1.3;
}
.method-sub {
    font-size: 11px;
    color: rgba(160,160,160,0.65);
    margin: 0;
    line-height: 1.3;
}
.method-body {
    padding: 12px 14px;
}
.pill-putih {
    display: inline-block;
    background: rgba(29,158,117,0.18);
    color: #1D9E75;
    border: 1px solid rgba(29,158,117,0.4);
    font-size: 13px;
    font-weight: 600;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 8px;
}
.pill-hitam {
    display: inline-block;
    background: rgba(83,74,183,0.18);
    color: #8B85E8;
    border: 1px solid rgba(83,74,183,0.4);
    font-size: 13px;
    font-weight: 600;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 8px;
}
.conf-label {
    font-size: 11px;
    color: rgba(160,160,160,0.65);
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
}
.conf-val {
    font-weight: 600;
    color: #ffffff;
}
.kmeans-box {
    background: rgba(186,117,23,0.1);
    border: 1px solid rgba(186,117,23,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    color: #EF9F27;
    font-size: 13px;
    line-height: 1.7;
    font-weight: 500;
}
.kmeans-note {
    font-size: 11px;
    color: rgba(160,160,160,0.55);
    margin-top: 8px;
    line-height: 1.6;
}
.section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: rgba(160,160,160,0.5);
    margin-bottom: 10px;
}
.img-meta {
    font-size: 11px;
    color: rgba(160,160,160,0.55);
    text-align: center;
    margin-top: 6px;
}
.empty-wrap {
    background: rgba(29,158,117,0.05);
    border: 1.5px dashed rgba(29,158,117,0.2);
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1rem;
}
.empty-title {
    font-size: 15px;
    font-weight: 500;
    color: rgba(255,255,255,0.65);
    margin-bottom: 6px;
}
.empty-sub {
    font-size: 13px;
    color: rgba(160,160,160,0.55);
    line-height: 1.7;
}
.chip-putih {
    display: inline-block;
    background: rgba(29,158,117,0.12);
    color: #1D9E75;
    border: 1px solid rgba(29,158,117,0.35);
    font-size: 12px;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 20px;
    margin: 6px 4px 0;
}
.chip-hitam {
    display: inline-block;
    background: rgba(83,74,183,0.12);
    color: #8B85E8;
    border: 1px solid rgba(83,74,183,0.35);
    font-size: 12px;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 20px;
    margin: 6px 4px 0;
}
.sidebar-logo {
    padding-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1.2rem;
}
.sidebar-logo-title {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #E1F5EE !important;
    line-height: 1.4;
}
.sidebar-logo-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.3) !important;
    margin-top: 3px;
}
.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3) !important;
    margin-bottom: 8px;
    margin-top: 1.2rem;
}
.mcard {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.mcard-name {
    font-size: 12px;
    font-weight: 600;
    color: #9FE1CB !important;
    margin-bottom: 3px;
}
.mcard-desc {
    font-size: 11px;
    color: rgba(255,255,255,0.45) !important;
    line-height: 1.5;
}
.drow {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.drow:last-child { border-bottom: none; }
.dkey { color: rgba(255,255,255,0.4) !important; }
.dval { color: #9FE1CB !important; font-weight: 600; }
.credit {
    font-size: 11px;
    color: rgba(255,255,255,0.3) !important;
    line-height: 1.9;
}
.credit strong { color: rgba(255,255,255,0.55) !important; }
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
    return mobilenet_preprocess(arr)

def preprocess_image_efficientnet(img):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return efficientnet_preprocess(arr)


# ============================================================
# PREDIKSI
# ============================================================
def predict_mobilenet(img):
    pred = mobilenet_model.predict(preprocess_image_mobilenet(img), verbose=0)[0][0]
    if pred > 0.5:
        return 'Ketan Putih', float(pred)
    return 'Ketan Hitam', float(1 - pred)

def predict_knn(img):
    features = effnet_extractor.predict(preprocess_image_efficientnet(img), verbose=0)
    pred     = knn.predict(features)[0]
    proba    = knn.predict_proba(features)[0]
    label    = 'Ketan Putih' if 'putih' in CLASS_NAMES[pred] else 'Ketan Hitam'
    return label, float(max(proba))

def predict_kmeans(img):
    features = effnet_extractor.predict(preprocess_image_efficientnet(img), verbose=0)
    reduced  = pca.transform(scaler.transform(features))
    cluster  = kmeans.predict(reduced)[0]
    return cluster, MODEL_INFO['sil_score'], MODEL_INFO['metode_label']


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-title">Sistem Klasifikasi Ketan</div>
        <div class="sidebar-logo-sub">Deep Learning + Machine Learning</div>
    </div>

    <div class="sidebar-section-label">Metode</div>
    <div class="mcard">
        <div class="mcard-name">MobileNetV2</div>
        <div class="mcard-desc">CNN dengan transfer learning dari ImageNet. Supervised learning.</div>
    </div>
    <div class="mcard">
        <div class="mcard-name">KNN (k=3)</div>
        <div class="mcard-desc">Fitur diekstraksi menggunakan EfficientNetB0. Supervised learning.</div>
    </div>
    <div class="mcard">
        <div class="mcard-name">K-Means</div>
        <div class="mcard-desc">Clustering unsupervised. Silhouette Score: {MODEL_INFO['sil_score']:.4f}</div>
    </div>

    <div class="sidebar-section-label" style="margin-top:1.4rem">Dataset</div>
    <div class="drow"><span class="dkey">Total gambar</span><span class="dval">824</span></div>
    <div class="drow"><span class="dkey">Per kelas</span><span class="dval">412</span></div>
    <div class="drow"><span class="dkey">Train</span><span class="dval">70%</span></div>
    <div class="drow"><span class="dkey">Validasi</span><span class="dval">20%</span></div>
    <div class="drow"><span class="dkey">Test</span><span class="dval">10%</span></div>

    <div style="margin-top:1.4rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.08)">
        <div class="credit">
            <strong>Nadjwa Tasya Safira</strong><br>
            2315061024<br>
            Teknik Informatika<br>
            Universitas Lampung
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HALAMAN UTAMA
# ============================================================
st.markdown("""
<div class="top-bar">
    <div>
        <div class="page-title">Klasifikasi Ketan</div>
        <div class="page-sub">Upload gambar untuk dianalisis dengan 3 metode ML</div>
    </div>
    <div class="badge">3 Model Aktif</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is not None:
    img = Image.open(uploaded)

    col_img, col_hasil = st.columns([1, 1.6], gap="large")

    with col_img:
        st.image(img, use_column_width=True)
        st.markdown(
            f'<div class="img-meta">{uploaded.name} &nbsp;|&nbsp; {img.size[0]} x {img.size[1]} px'
            f' &nbsp;|&nbsp; {uploaded.size // 1024} KB</div>',
            unsafe_allow_html=True
        )

    with col_hasil:
        with st.spinner("Menganalisis gambar..."):
            label_mn,  conf_mn            = predict_mobilenet(img)
            label_knn, conf_knn           = predict_knn(img)
            cluster_km, sil_km, metode_km = predict_kmeans(img)

        # ---- MobileNetV2 ----
        pill_mn = "pill-putih" if label_mn == "Ketan Putih" else "pill-hitam"
        st.markdown(f"""
        <div class="method-block">
            <div class="method-header">
                <div class="method-num">1</div>
                <div>
                    <div class="method-name">MobileNetV2</div>
                    <div class="method-sub">CNN Utama &mdash; Transfer Learning</div>
                </div>
            </div>
            <div class="method-body">
                <span class="{pill_mn}">{label_mn}</span>
                <div class="conf-label">Confidence <span class="conf-val">{conf_mn*100:.2f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf_mn)

        # ---- KNN ----
        pill_knn = "pill-putih" if label_knn == "Ketan Putih" else "pill-hitam"
        st.markdown(f"""
        <div class="method-block" style="margin-top:10px">
            <div class="method-header">
                <div class="method-num">2</div>
                <div>
                    <div class="method-name">KNN &mdash; K-Nearest Neighbor</div>
                    <div class="method-sub">Fitur dari EfficientNetB0</div>
                </div>
            </div>
            <div class="method-body">
                <span class="{pill_knn}">{label_knn}</span>
                <div class="conf-label">Confidence <span class="conf-val">{conf_knn*100:.2f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf_knn)

        # ---- K-Means ----
        st.markdown(f"""
        <div class="method-block" style="margin-top:10px">
            <div class="method-header">
                <div class="method-num">3</div>
                <div>
                    <div class="method-name">K-Means Clustering</div>
                    <div class="method-sub">Unsupervised &mdash; {metode_km}</div>
                </div>
            </div>
            <div class="method-body">
                <div class="kmeans-box">
                    Gambar masuk ke <strong>Cluster {cluster_km}</strong>
                    &nbsp;&mdash;&nbsp; Silhouette Score: <strong>{sil_km:.4f}</strong>
                </div>
                <div class="kmeans-note">
                    K-Means bersifat unsupervised sehingga tidak menghasilkan label kelas secara langsung.
                    Hasil berupa nomor cluster berdasarkan kemiripan fitur gambar.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
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
    <div class="empty-wrap">
        <div class="empty-title">Belum ada gambar yang dianalisis</div>
        <div class="empty-sub">
            Upload gambar ketan putih atau ketan hitam untuk memulai klasifikasi<br>
            dengan tiga model machine learning sekaligus.
        </div>
        <div style="margin-top:4px">
            <span class="chip-putih">Ketan Putih</span>
            <span class="chip-hitam">Ketan Hitam</span>
        </div>
    </div>
    """, unsafe_allow_html=True)