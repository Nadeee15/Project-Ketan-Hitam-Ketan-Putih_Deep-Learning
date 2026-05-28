import streamlit as st
import numpy as np
import json
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Ketan Putih dan Ketan Hitam",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = (224, 224)

# ============================================================
# LOAD SEMUA MODEL (CACHE AGAR TIDAK RELOAD TIAP INTERAKSI)
# ============================================================
@st.cache_resource
def load_all_models():
    mobilenet  = load_model('models/model_mobilenetv2.keras')
    effnet     = load_model('models/efficientnet_extractor.keras')
    knn        = joblib.load('models/knn_model.pkl')
    scaler     = joblib.load('models/scaler.pkl')
    pca        = joblib.load('models/pca_model.pkl')
    kmeans     = joblib.load('models/kmeans_model.pkl')
    class_names= joblib.load('models/class_names.pkl')
    with open('models/model_info.json', 'r') as f:
        info   = json.load(f)
    return mobilenet, effnet, knn, scaler, pca, kmeans, class_names, info

mobilenet_model, effnet_extractor, knn, scaler, pca, kmeans, CLASS_NAMES, MODEL_INFO = load_all_models()

# ============================================================
# FUNGSI PREPROCESSING GAMBAR
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
# FUNGSI PREDIKSI
# ============================================================
def predict_mobilenet(img):
    arr  = preprocess_image_mobilenet(img)
    pred = mobilenet_model.predict(arr, verbose=0)[0][0]
    # ketan_hitam=0, ketan_putih=1
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
    st.title("Informasi Sistem")
    st.markdown("---")

    st.subheader("Tentang Penelitian")
    st.write(
        "Sistem klasifikasi ketan putih dan ketan hitam "
        "berbasis citra digital menggunakan Deep Learning "
        "dan Machine Learning."
    )
    st.markdown("---")

    st.subheader("Metode yang Digunakan")

    st.markdown("**1. MobileNetV2**")
    st.write("Model CNN utama dengan transfer learning dari ImageNet. Supervised learning.")

    st.markdown("**2. KNN (k=3)**")
    st.write(
        "K-Nearest Neighbor menggunakan fitur yang "
        "diekstraksi dari EfficientNetB0. Supervised learning."
    )

    st.markdown("**3. K-Means**")
    st.write(
        f"Clustering unsupervised menggunakan "
        f"{MODEL_INFO['metode_label'].replace('K-Means + ', '')}. "
        f"Silhouette Score: {MODEL_INFO['sil_score']:.4f}."
    )
    st.markdown("---")

    st.subheader("Dataset")
    st.write("412 gambar per kelas, total 824 gambar mandiri.")
    st.write("Split: Train 70%, Validation 20%, Test 10%.")
    st.markdown("---")

    st.subheader("Dibuat oleh")
    st.write("Nadjwa Tasya Safira")
    st.write("2315061024")
    st.write("Teknik Informatika, Universitas Lampung")

# ============================================================
# HALAMAN UTAMA
# ============================================================
st.title("Klasifikasi Ketan Putih dan Ketan Hitam")
st.write(
    "Upload gambar ketan untuk diklasifikasikan menggunakan "
    "tiga metode: MobileNetV2, KNN, dan K-Means."
)
st.markdown("---")

# Upload gambar
uploaded = st.file_uploader(
    "Upload gambar ketan (JPG, PNG, HEIC)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    img = Image.open(uploaded)

    col_img, col_hasil = st.columns([1, 2])

    with col_img:
        st.subheader("Gambar yang Diupload")
        st.image(img, use_column_width=True)
        st.caption(f"Ukuran: {img.size[0]} x {img.size[1]} px")

    with col_hasil:
        st.subheader("Hasil Klasifikasi")

        with st.spinner("Menganalisis gambar..."):

            # ---- MobileNetV2 ----
            label_mn, conf_mn = predict_mobilenet(img)

            # ---- KNN ----
            label_knn, conf_knn = predict_knn(img)

            # ---- K-Means ----
            cluster_km, sil_km, metode_km = predict_kmeans(img)

        # Tampilkan hasil MobileNetV2
        st.markdown("#### 1. MobileNetV2 (CNN Utama)")
        col1, col2 = st.columns(2)
        with col1:
            if label_mn == 'Ketan Putih':
                st.success(f"Hasil: **{label_mn}**")
            else:
                st.error(f"Hasil: **{label_mn}**")
        with col2:
            st.metric("Confidence Score", f"{conf_mn*100:.2f}%")
        st.progress(conf_mn)

        st.markdown("---")

        # Tampilkan hasil KNN
        st.markdown("#### 2. KNN — K-Nearest Neighbor")
        st.caption("Fitur diekstraksi menggunakan EfficientNetB0")
        col3, col4 = st.columns(2)
        with col3:
            if label_knn == 'Ketan Putih':
                st.success(f"Hasil: **{label_knn}**")
            else:
                st.error(f"Hasil: **{label_knn}**")
        with col4:
            st.metric("Confidence Score", f"{conf_knn*100:.2f}%")
        st.progress(conf_knn)

        st.markdown("---")

        # Tampilkan hasil K-Means
        st.markdown("#### 3. K-Means Clustering (Unsupervised)")
        st.caption(f"Metode: {metode_km}")
        st.info(
            f"Gambar masuk ke **Cluster {cluster_km}**\n\n"
            f"Silhouette Score model: **{sil_km:.4f}**"
        )
        st.warning(
            "Catatan: K-Means adalah metode unsupervised sehingga "
            "tidak menghasilkan label kelas secara langsung. "
            "Hasil berupa nomor cluster yang terbentuk berdasarkan "
            "kemiripan fitur gambar."
        )

    st.markdown("---")

    # Perbandingan ringkas
    st.subheader("Ringkasan Hasil")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            label="MobileNetV2",
            value=label_mn,
            delta=f"Confidence: {conf_mn*100:.1f}%"
        )

    with col_b:
        st.metric(
            label="KNN",
            value=label_knn,
            delta=f"Confidence: {conf_knn*100:.1f}%"
        )

    with col_c:
        st.metric(
            label="K-Means",
            value=f"Cluster {cluster_km}",
            delta=f"Silhouette: {sil_km:.4f}"
        )

else:
    # Tampilan awal sebelum upload
    st.info(
        "Silakan upload gambar ketan putih atau ketan hitam "
        "untuk memulai klasifikasi."
    )

    st.markdown("### Contoh gambar yang bisa diupload:")
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown(
            "**Ketan Putih** — biji beras ketan berwarna putih "
            "dengan tekstur opaque."
        )
    with col_y:
        st.markdown(
            "**Ketan Hitam** — biji beras ketan berwarna hitam "
            "atau ungu gelap."
        )
