# Klasifikasi Ketan Putih dan Ketan Hitam

Aplikasi web klasifikasi ketan putih dan ketan hitam berbasis citra digital
menggunakan MobileNetV2, KNN, dan K-Means Clustering.

## Tentang Penelitian

- Nama   : Nadjwa Tasya Safira
- NIM    : 2315061024
- Prodi  : Teknik Informatika, Universitas Lampung
- Tahun  : 2026

## Metode yang Digunakan

| Metode | Peran | Tipe |
|---|---|---|
| MobileNetV2 | Model CNN klasifikasi utama | Supervised |
| EfficientNetB0 | Feature extractor untuk KNN dan K-Means | Feature Extractor |
| KNN (k=3) | Klasifikasi berbasis fitur | Supervised |
| K-Means + PCA | Clustering | Unsupervised |

## Struktur Repository

```
├── app.py                          # Aplikasi Streamlit utama
├── requirements.txt                # Library yang dibutuhkan
├── models/
│   ├── model_mobilenetv2.keras     # Model CNN utama
│   ├── efficientnet_extractor.keras# Feature extractor
│   ├── knn_model.pkl               # Model KNN
│   ├── scaler.pkl                  # StandardScaler
│   ├── pca_model.pkl               # PCA model
│   ├── kmeans_model.pkl            # K-Means model
│   ├── class_names.pkl             # Nama kelas
│   └── model_info.json             # Info metode K-Means
└── README.md
```

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Cloud

1. Push semua file ke GitHub
2. Buka streamlit.io
3. New app - pilih repo ini
4. Main file: app.py
5. Deploy

## Catatan

Model .keras berukuran besar. Jika melebihi batas GitHub (100MB),
gunakan Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/
git commit -m "Add models with LFS"
git push
```
