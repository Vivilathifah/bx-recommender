# Dashboard Sistem Rekomendasi Buku — R Shiny (Book-Crossing)

## Struktur
```
bx_shiny/
├── app.R                # dashboard Shiny (Ringkasan, Eksplorasi Data, Rekomendasi, Evaluasi Model)
├── data/app_data.rds    # data bersih + model CF terlatih + metrik (4,3 MB)
├── manifest.json        # daftar paket R untuk deploy ke Posit Connect Cloud
└── prep/                # skrip pembuat app_data.rds dari hasil pipeline Python
```

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
