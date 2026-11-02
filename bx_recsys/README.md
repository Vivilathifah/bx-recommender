# Dashboard Sistem Rekomendasi Buku — Collaborative Filtering (Book-Crossing)

## Struktur
```
bx_recsys/
├── preprocess.py      # Tahap 1: data cleaning  -> data/
├── train.py           # Tahap 2: model CF + evaluasi -> models/
├── app.py             # Tahap 3: dashboard Streamlit
├── data/              # hasil cleaning (parquet) — sudah disertakan
├── models/            # model terlatih + metrik — sudah disertakan
└── requirements.txt
```

## Menjalankan dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```
Folder `data/` dan `models/` sudah ada, jadi dashboard langsung bisa dibuka.

## Mengulang pipeline dari data mentah
```bash
python preprocess.py <folder_berisi_BX-*.csv> data
python train.py data models      # ±2 menit
```

## Ringkasan proses
1. **Cleaning** — usia tidak valid (<5 / >100) dikosongkan; lokasi dipecah jadi kota/provinsi/negara;
   baris buku dengan tanda kutip ter-escape dibaca benar (`escapechar="\\"`); tahun terbit di luar
   1900–2005 dikosongkan; rating 0 (implisit) dipisah dari rating eksplisit 1–10; edisi berbeda dari
   buku yang sama (judul + penulis) digabung; filter k-core ≥5 rating per user dan per buku.
2. **Model** — Baseline bias, Item-based KNN (cosine pada residual + shrinkage), Matrix Factorization
   (Funk-SVD berbias, SGD), dan rekomendasi Populer sebagai pembanding.
3. **Evaluasi** — hold-out 20% rating per user; RMSE/MAE dan Precision/Recall/HitRate@10.
4. **Dashboard** — Ringkasan, Eksplorasi Data, Rekomendasi (buku mirip, user yang ada, user baru), Evaluasi.

## Deploy gratis
Push folder ini ke GitHub, lalu buat app baru di Streamlit Community Cloud dengan `app.py` sebagai entry point.
