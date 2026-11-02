# Dashboard Sistem Rekomendasi Buku — R Shiny (Book-Crossing)

## Struktur
```
bx_shiny/
├── app.R                # dashboard interaktif (Dashboard, Rekomendasi, Peta Buku, Evaluasi Model, Tentang Data)
├── data/
│   ├── app_data.rds     # data bersih + model CF + metrik dalam 1 file (dibaca dashboard, 17 MB)
│   └── csv/             # isi yang sama dalam 18 file CSV + kamus data (README.md) https://drive.google.com/drive/folders/1Za0bR4mRFiC8p71X5iZv7w5WeXxZbVpe?usp=sharing
├── manifest.json        # daftar paket R untuk deploy ke Posit Connect Cloud
└── prep/                # skrip pembuat app_data.rds (build_rds.R) dan CSV (export_csv.R)
```

## Fitur interaktif
- **Dashboard**: filter jenis rating, negara, kelompok usia, tahun terbit → semua KPI & grafik ikut berubah.
  Klik bar negara/usia untuk memfilter (klik lagi untuk melepas), klik buku → langsung ke rekomendasinya,
  klik penulis → daftar bukunya.
- **Tampilan web bersampul**: rekomendasi tampil sebagai kartu buku dengan sampul — diambil dari URL gambar
  pada kolom `Image-URL-M/L` dataset (BX-Books.csv), cadangan Open Library Covers API (per ISBN), lalu sampul
  pengganti berwarna, halaman detail buku, rak buku teratas di Dashboard,
  dan rak buku tiap user. Bisa diganti ke mode "Tabel & grafik".
- **Rekomendasi**: buku mirip (klik bar/baris untuk menjelajah, ada jejak penjelajahan), rekomendasi per user
  (MF / Item-KNN / Populer, tombol user acak), dan simulasi user baru (geser rating → hasil langsung berubah).
- **Peta Buku**: sebaran buku dari faktor laten MF; zoom, hover, sorot judul, klik titik untuk detail.
- **Evaluasi Model**: RMSE/MAE, metrik ranking Top-10, kurva pelatihan MF, interpretasi.

## Menjalankan di komputer (RStudio)
```r
install.packages(c("shiny", "bslib", "dplyr", "plotly", "DT"))
shiny::runApp()        # dari folder yang berisi app.R
```

## Deploy gratis ke Posit Connect Cloud
1. Upload isi folder ini ke repo GitHub **publik** (app.R dan manifest.json di root repo).
2. Login di connect.posit.cloud dengan akun GitHub.
3. **Publish → Shiny →** pilih repo → branch `main` → Primary file `app.R` → **Publish**.

## Asal data & model
Cleaning dan pelatihan model dilakukan di pipeline Python (`preprocess.py`, `train.py`):
Baseline bias, Item-based KNN (cosine + shrinkage), Matrix Factorization (Funk-SVD).
Hasilnya diekspor ke R dengan:
```bash
python prep/export_for_r.py <folder_bx_recsys> <folder_csv>
Rscript prep/build_rds.R <folder_csv> .
```

Kalau `manifest.json` perlu dibuat ulang (mis. setelah menambah paket):
```r
rsconnect::writeManifest(appFiles = c("app.R", "data/app_data.rds"))
```
