"""
Dashboard Sistem Rekomendasi Buku (Collaborative Filtering) - Book-Crossing dataset.
Jalankan:  streamlit run app.py
(butuh folder data/ dan models/ hasil preprocess.py dan train.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
DATA, MODELS = BASE / "data", BASE / "models"
ACCENT = "#2E6FD8"
SEQ = px.colors.sequential.Blues

st.set_page_config(page_title="BX Book Recommender", page_icon="📚", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.block-container {padding-top: 2rem;}
.small-note {color: #6b7280; font-size: 0.85rem;}
</style>""", unsafe_allow_html=True)


# ============================================================ LOAD
@st.cache_data(show_spinner="Memuat data...")
def load_data():
    users = pd.read_parquet(DATA / "users.parquet")
    books = pd.read_parquet(DATA / "books.parquet")
    r_all = pd.read_parquet(DATA / "ratings_all.parquet")
    r_cf = pd.read_parquet(DATA / "ratings_cf.parquet")
    report = json.loads((DATA / "cleaning_report.json").read_text())
    return users, books, r_all, r_cf, report


@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    mf = dict(np.load(MODELS / "mf_model.npz", allow_pickle=False))
    nb = pd.read_parquet(MODELS / "item_neighbors.parquet")
    metrics = json.loads((MODELS / "metrics.json").read_text())
    return mf, nb, metrics


users, books, r_all, r_cf, report = load_data()
mf, neighbors, metrics = load_models()
book_info = books.set_index("isbn")[["title", "author", "year", "publisher"]]


@st.cache_data
def cf_book_stats():
    s = r_cf.groupby("isbn")["rating"].agg(n_rating="size", avg_rating="mean").reset_index()
    s = s.join(book_info, on="isbn")
    s["label"] = s["title"].str.slice(0, 80) + " — " + s["author"]
    return s.sort_values("n_rating", ascending=False).reset_index(drop=True)


bstats = cf_book_stats()
bstats_idx = bstats.set_index("isbn")
mf_isbn = mf["isbns"].astype(str)
mf_user_pos = {int(u): i for i, u in enumerate(mf["user_ids"])}


def fmt(n): return f"{n:,.0f}".replace(",", ".")


def book_table(df, extra_cols):
    out = df.join(bstats_idx[["title", "author", "year", "avg_rating", "n_rating"]], on="isbn")
    out = out[["title", "author", "year"] + extra_cols + ["avg_rating", "n_rating"]]
    out.index = np.arange(1, len(out) + 1)
    return out.rename(columns={"title": "Judul", "author": "Penulis", "year": "Tahun",
                               "avg_rating": "Rata-rata rating", "n_rating": "Jumlah rating"})


COLCFG = {"Rata-rata rating": st.column_config.NumberColumn(format="%.2f"),
          "Skor": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
          "Prediksi rating": st.column_config.NumberColumn(format="%.2f"),
          "Kemiripan": st.column_config.NumberColumn(format="%.4f")}


# ============================================================ SIDEBAR
st.sidebar.title("📚 BX Recommender")
page = st.sidebar.radio("Halaman", ["Ringkasan", "Eksplorasi Data", "Rekomendasi", "Evaluasi Model"])
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data CF: **{fmt(report['cf_users'])}** user × **{fmt(report['cf_books'])}** buku, "
    f"**{fmt(report['cf_ratings'])}** rating eksplisit (densitas {report['cf_density_pct']:.3f}%).")
st.sidebar.caption("Sumber: Book-Crossing dataset (Ziegler et al., 2005).")


# ============================================================ RINGKASAN
if page == "Ringkasan":
    st.title("Sistem Rekomendasi Buku — Collaborative Filtering")
    st.markdown("Dashboard ini merangkum alur kerja: **data cleaning → eksplorasi → pemodelan "
                "collaborative filtering → evaluasi → rekomendasi**, pada dataset Book-Crossing.")

    c = st.columns(4)
    c[0].metric("User", fmt(report["users_raw"]))
    c[1].metric("Buku", fmt(report["books_raw"]))
    c[2].metric("Rating", fmt(report["ratings_raw"]))
    c[3].metric("Rating eksplisit (1–10)", fmt(report["ratings_explicit"]),
                f"{100*report['ratings_explicit']/report['ratings_matched_to_books']:.1f}% dari rating valid",
                delta_color="off")

    st.subheader("Alur data: dari mentah sampai matriks CF")
    funnel = pd.DataFrame({
        "Tahap": ["Rating mentah", "Cocok dgn katalog buku", "Rating eksplisit",
                  "Setelah gabung edisi", f"k-core (≥{report['min_ratings_per_user']} per user & buku)"],
        "Jumlah": [report["ratings_raw"], report["ratings_matched_to_books"], report["ratings_explicit"],
                   report["cf_ratings_after_edition_merge"], report["cf_ratings"]]})
    fig = go.Figure(go.Funnel(y=funnel["Tahap"], x=funnel["Jumlah"], marker_color=ACCENT,
                              textinfo="value+percent initial"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    with l:
        st.subheader("Langkah cleaning")
        st.markdown(f"""
- **Users** — usia < 5 atau > 100 tahun dianggap tidak valid ({fmt(report['age_outlier_set_nan'])} nilai → kosong);
  kolom *Location* dipecah menjadi kota, provinsi/negara bagian, dan negara.
- **Books** — {fmt(report['books_raw'])} baris dibaca utuh (tanda kutip ter-*escape* di judul ditangani);
  entitas HTML (`&amp;`) dibersihkan; tahun terbit di luar 1900–2005 → kosong ({fmt(report['year_invalid_set_nan'])} buku).
- **Ratings** — hanya rating yang ISBN-nya ada di katalog; rating **0 = implisit** (interaksi tanpa nilai),
  sehingga model CF memakai rating **1–10** saja.
- **Gabung edisi** — {fmt(report['books_merged_into_other_edition'])} ISBN dengan judul + penulis sama
  digabung ke satu ISBN kanonik agar satu buku tidak terpecah.
- **k-core filter** — user dan buku dengan < {report['min_ratings_per_user']} rating dibuang berulang sampai stabil.
""")
    with r:
        st.subheader("Model yang dibangun")
        st.markdown("""
| Model | Ide |
|---|---|
| **Baseline (bias)** | rata-rata global + kecenderungan user + kualitas buku |
| **Item-based KNN** | buku dinilai mirip bila dirating serupa oleh user yang sama (cosine + shrinkage) |
| **Matrix Factorization** | user & buku dipetakan ke 16 faktor laten (Funk-SVD, SGD) |
| **Populer** | pembanding: buku dengan rating terbanyak |
""")
        st.info("Buka halaman **Rekomendasi** untuk mencoba: cari buku mirip, rekomendasi untuk "
                "user yang ada, atau rekomendasi untuk user baru.")


# ============================================================ EDA
elif page == "Eksplorasi Data":
    st.title("Eksplorasi Data")
    t1, t2, t3 = st.tabs(["Rating", "Buku", "User"])

    with t1:
        a, b = st.columns(2)
        dist = r_all["rating"].value_counts().sort_index().reset_index()
        dist["Jenis"] = np.where(dist["rating"] == 0, "Implisit (0)", "Eksplisit (1–10)")
        fig = px.bar(dist, x="rating", y="count", color="Jenis",
                     color_discrete_map={"Implisit (0)": "#B8C4D6", "Eksplisit (1–10)": ACCENT},
                     labels={"rating": "Rating", "count": "Jumlah"}, title="Distribusi semua rating")
        fig.update_layout(xaxis=dict(dtick=1), legend_title=None)
        a.plotly_chart(fig, width="stretch")
        ex = r_all[r_all["rating"] > 0]["rating"].value_counts().sort_index().reset_index()
        fig = px.bar(ex, x="rating", y="count", labels={"rating": "Rating", "count": "Jumlah"},
                     title="Rating eksplisit saja", color_discrete_sequence=[ACCENT])
        fig.update_layout(xaxis=dict(dtick=1))
        b.plotly_chart(fig, width="stretch")
        st.caption("Sekitar dua pertiga rating bernilai 0 (implisit). Rating eksplisit condong ke "
                   "nilai tinggi (7–10): user cenderung hanya menilai buku yang disukai.")

        per_user = r_cf.groupby("user_id").size()
        per_book = r_cf.groupby("isbn").size()
        a, b = st.columns(2)
        fig = px.histogram(per_user, nbins=60, log_y=True, color_discrete_sequence=[ACCENT],
                           labels={"value": "Rating per user"}, title="Rating per user (data CF, skala log)")
        fig.update_layout(showlegend=False, yaxis_title="Jumlah user")
        a.plotly_chart(fig, width="stretch")
        fig = px.histogram(per_book, nbins=60, log_y=True, color_discrete_sequence=[ACCENT],
                           labels={"value": "Rating per buku"}, title="Rating per buku (data CF, skala log)")
        fig.update_layout(showlegend=False, yaxis_title="Jumlah buku")
        b.plotly_chart(fig, width="stretch")
        st.caption(f"Distribusi *long-tail*: median {per_user.median():.0f} rating per user dan "
                   f"{per_book.median():.0f} per buku — inilah sebab matriks sangat jarang "
                   f"(densitas {report['cf_density_pct']:.3f}%).")

    with t2:
        min_n = st.slider("Minimal jumlah rating eksplisit", 5, 200, 50, 5)
        top = bstats[bstats["n_rating"] >= min_n]
        a, b = st.columns(2)
        d = top.head(15).iloc[::-1]
        fig = px.bar(d, x="n_rating", y="title", orientation="h", title="15 buku paling banyak dirating",
                     labels={"n_rating": "Jumlah rating", "title": ""}, color_discrete_sequence=[ACCENT],
                     hover_data={"author": True})
        fig.update_layout(height=520)
        a.plotly_chart(fig, width="stretch")
        d = top.sort_values("avg_rating", ascending=False).head(15).iloc[::-1]
        fig = px.bar(d, x="avg_rating", y="title", orientation="h",
                     title=f"15 buku rating tertinggi (≥ {min_n} rating)",
                     labels={"avg_rating": "Rata-rata rating", "title": ""}, color_discrete_sequence=[ACCENT],
                     hover_data={"author": True, "n_rating": True})
        fig.update_layout(height=520, xaxis_range=[0, 10])
        b.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        auth = bstats.groupby("author")["n_rating"].sum().nlargest(15).iloc[::-1].reset_index()
        fig = px.bar(auth, x="n_rating", y="author", orientation="h", title="Penulis paling banyak dirating",
                     labels={"n_rating": "Jumlah rating", "author": ""}, color_discrete_sequence=[ACCENT])
        a.plotly_chart(fig, width="stretch")
        yrs = books.dropna(subset=["year"])
        yrs = yrs[yrs["year"] >= 1950]["year"].astype(int).value_counts().sort_index().reset_index()
        fig = px.area(yrs, x="year", y="count", title="Jumlah buku per tahun terbit (katalog, ≥1950)",
                      labels={"year": "Tahun", "count": "Jumlah buku"}, color_discrete_sequence=[ACCENT])
        b.plotly_chart(fig, width="stretch")

    with t3:
        a, b = st.columns(2)
        ages = users["age"].dropna()
        fig = px.histogram(ages, nbins=48, color_discrete_sequence=[ACCENT], title="Distribusi usia user",
                           labels={"value": "Usia"})
        fig.update_layout(showlegend=False, yaxis_title="Jumlah user")
        a.plotly_chart(fig, width="stretch")
        a.caption(f"{100*users['age'].isna().mean():.0f}% user tidak mengisi usia (atau usianya tidak valid). "
                  f"Median usia {ages.median():.0f} tahun.")
        ctry = users["country"].value_counts().head(15).iloc[::-1].reset_index()
        fig = px.bar(ctry, x="count", y="country", orientation="h", title="15 negara asal user terbanyak",
                     labels={"count": "Jumlah user", "country": ""}, color_discrete_sequence=[ACCENT])
        b.plotly_chart(fig, width="stretch")

        ra = r_all[r_all["rating"] > 0].merge(users[["user_id", "age_group"]], on="user_id")
        ag = ra.dropna(subset=["age_group"]).groupby("age_group")["rating"].agg(["mean", "size"]).reset_index()
        order = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        fig = px.bar(ag, x="age_group", y="mean", text_auto=".2f", color_discrete_sequence=[ACCENT],
                     category_orders={"age_group": order}, hover_data={"size": True},
                     labels={"age_group": "Kelompok usia", "mean": "Rata-rata rating eksplisit", "size": "Jumlah rating"},
                     title="Rata-rata rating eksplisit per kelompok usia")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, width="stretch")


# ============================================================ REKOMENDASI
elif page == "Rekomendasi":
    st.title("Rekomendasi Buku")
    t1, t2, t3 = st.tabs(["🔎 Buku mirip", "👤 Untuk user yang ada", "✨ User baru"])

    # ---------------- buku mirip (item-based KNN)
    with t1:
        st.markdown("Pilih sebuah buku; sistem mencari buku lain yang **dirating serupa oleh pembaca "
                    "yang sama** (item-based KNN).")
        labels = bstats["label"].tolist()
        default = next((i for i, l in enumerate(labels) if l.startswith("Harry Potter and the Sorcerer")), 0)
        choice = st.selectbox("Buku", labels, index=default)
        n = st.slider("Jumlah rekomendasi", 5, 20, 10, key="n_sim")
        isbn = bstats.loc[bstats["label"] == choice, "isbn"].iloc[0]
        info = bstats_idx.loc[isbn]
        c = st.columns(4)
        c[0].metric("Penulis", info["author"])
        c[1].metric("Tahun", "-" if pd.isna(info["year"]) else int(info["year"]))
        c[2].metric("Rata-rata rating", f"{info['avg_rating']:.2f}")
        c[3].metric("Jumlah rating", int(info["n_rating"]))
        res = neighbors[neighbors["isbn"] == isbn].nsmallest(n, "rank")[["neighbor_isbn", "similarity"]]
        res = res.rename(columns={"neighbor_isbn": "isbn", "similarity": "Kemiripan"})
        if res.empty:
            st.warning("Buku ini belum punya cukup pembaca bersama untuk dicari kemiripannya.")
        else:
            st.dataframe(book_table(res.set_index(res["isbn"])[["Kemiripan"]].reset_index(), ["Kemiripan"]),
                         column_config=COLCFG, width="stretch")

    # ---------------- user yang ada
    with t2:
        ucount = r_cf.groupby("user_id").size().sort_values(ascending=False)
        a, b = st.columns([2, 1])
        uid = a.selectbox("User-ID (diurutkan dari yang paling banyak memberi rating)", ucount.index.tolist(),
                          format_func=lambda u: f"{u}  ({ucount[u]} rating)", index=min(10, len(ucount) - 1))
        model = b.radio("Model", ["Matrix Factorization", "Item-based KNN", "Populer"], horizontal=False)
        n = st.slider("Jumlah rekomendasi", 5, 20, 10, key="n_user")

        prof = users.set_index("user_id").loc[uid]
        hist = r_cf[r_cf["user_id"] == uid].sort_values("rating", ascending=False)
        c = st.columns(4)
        c[0].metric("Lokasi", (prof["country"] or "-").title() if isinstance(prof["country"], str) else "-")
        c[1].metric("Usia", "-" if pd.isna(prof["age"]) else int(prof["age"]))
        c[2].metric("Jumlah rating", len(hist))
        c[3].metric("Rata-rata rating user", f"{hist['rating'].mean():.2f}")

        seen = set(hist["isbn"])
        if model == "Matrix Factorization":
            u = mf_user_pos[int(uid)]
            pred = mf["mu"] + mf["bu"][u] + mf["bi"] + mf["Q"] @ mf["P"][u]
            rec = pd.DataFrame({"isbn": mf_isbn, "Prediksi rating": np.clip(pred, 1, 10)})
            rec = rec[~rec["isbn"].isin(seen)].nlargest(n, "Prediksi rating")
            cols = ["Prediksi rating"]
        elif model == "Item-based KNN":
            m = neighbors[neighbors["isbn"].isin(seen)].merge(hist[["isbn", "rating"]], on="isbn")
            m = m[~m["neighbor_isbn"].isin(seen)]
            m["w"] = m["similarity"] * (m["rating"] - hist["rating"].mean() + 1)
            m = m[m["w"] > 0]
            agg = m.groupby("neighbor_isbn").agg(score=("w", "sum")).reset_index()
            why = (m.sort_values("w", ascending=False).drop_duplicates("neighbor_isbn")
                     .set_index("neighbor_isbn")["isbn"].map(bstats_idx["title"]))
            agg["Karena Anda menyukai"] = agg["neighbor_isbn"].map(why).str.slice(0, 60)
            rec = agg.nlargest(n, "score").rename(columns={"neighbor_isbn": "isbn"})
            rec["Skor"] = rec["score"] / rec["score"].max() if len(rec) else rec["score"]
            cols = ["Skor", "Karena Anda menyukai"]
        else:
            rec = bstats[~bstats["isbn"].isin(seen)].head(n)[["isbn"]]
            cols = []
        a, b = st.columns([3, 2])
        with a:
            st.subheader(f"Rekomendasi — {model}")
            if rec.empty:
                st.warning("Tidak ada kandidat rekomendasi untuk user ini dengan model tersebut.")
            else:
                st.dataframe(book_table(rec[["isbn"] + cols].reset_index(drop=True), cols),
                             column_config=COLCFG, width="stretch")
        with b:
            st.subheader("Riwayat rating user")
            h = hist.head(15).join(bstats_idx[["title", "author"]], on="isbn")[["title", "author", "rating"]]
            h.index = np.arange(1, len(h) + 1)
            st.dataframe(h.rename(columns={"title": "Judul", "author": "Penulis", "rating": "Rating"}),
                         width="stretch")

    # ---------------- user baru (cold start)
    with t3:
        st.markdown("Simulasi **user baru**: pilih beberapa buku yang pernah dibaca dan beri rating; "
                    "rekomendasi dihitung dari kemiripan antar-buku (item-based KNN). Bila belum ada "
                    "buku yang dipilih, sistem menampilkan buku populer.")
        picks = st.multiselect("Buku yang pernah dibaca", bstats["label"].tolist(), max_selections=10,
                               placeholder="Ketik judul, mis. Harry Potter, Da Vinci, Tolkien...")
        given = {}
        if picks:
            cols_ = st.columns(min(len(picks), 4))
            for k, p in enumerate(picks):
                isbn_p = bstats.loc[bstats["label"] == p, "isbn"].iloc[0]
                given[isbn_p] = cols_[k % len(cols_)].slider(p[:45], 1, 10, 8, key=f"r_{isbn_p}")
        n = st.slider("Jumlah rekomendasi", 5, 20, 10, key="n_new")
        if not given:
            rec = bstats.head(n)[["isbn"]]
            st.dataframe(book_table(rec.reset_index(drop=True), []), column_config=COLCFG, width="stretch")
        else:
            g = pd.DataFrame({"isbn": list(given), "rating": list(given.values())})
            mean_r = g["rating"].mean()
            m = neighbors[neighbors["isbn"].isin(g["isbn"])].merge(g, on="isbn")
            m = m[~m["neighbor_isbn"].isin(g["isbn"])]
            m["w"] = m["similarity"] * (m["rating"] - mean_r + 1)
            m = m[m["w"] > 0]
            if m.empty:
                st.info("Belum ada buku yang mirip dengan pilihan Anda; menampilkan buku populer.")
                rec = bstats[~bstats["isbn"].isin(g["isbn"])].head(n)[["isbn"]]
                cols = []
            else:
                agg = m.groupby("neighbor_isbn")["w"].sum().reset_index(name="score")
                why = (m.sort_values("w", ascending=False).drop_duplicates("neighbor_isbn")
                         .set_index("neighbor_isbn")["isbn"].map(bstats_idx["title"]))
                agg["Karena Anda menyukai"] = agg["neighbor_isbn"].map(why).str.slice(0, 60)
                rec = agg.nlargest(n, "score").rename(columns={"neighbor_isbn": "isbn"})
                rec["Skor"] = rec["score"] / rec["score"].max()
                cols = ["Skor", "Karena Anda menyukai"]
            st.dataframe(book_table(rec[["isbn"] + cols].reset_index(drop=True), cols),
                         column_config=COLCFG, width="stretch")


# ============================================================ EVALUASI
else:
    st.title("Evaluasi Model")
    sp = metrics["split"]
    st.markdown(f"Data dibagi per user: **{int(sp['test_frac']*100)}%** rating tiap user disisihkan sebagai "
                f"data uji ({fmt(sp['train'])} latih / {fmt(sp['test'])} uji, seed {sp['seed']}).")
    res = pd.DataFrame(metrics["results"]).T.drop(columns=["catatan_ranking", "n_users_eval"], errors="ignore")
    res.index = ["Global mean / Populer" if i == "Global mean" else i for i in res.index]
    res = res.astype(float)

    st.subheader("1. Akurasi prediksi rating (semakin kecil semakin baik)")
    a, b = st.columns([3, 2])
    long = res[["rmse", "mae"]].reset_index().melt(id_vars="index", var_name="Metrik", value_name="Nilai")
    long["Metrik"] = long["Metrik"].str.upper()
    fig = px.bar(long, x="Nilai", y="index", color="Metrik", barmode="group", orientation="h", text_auto=".3f",
                 color_discrete_sequence=[ACCENT, "#9DB7E6"], labels={"index": ""})
    fig.update_layout(height=330, legend_title=None, margin=dict(t=10))
    a.plotly_chart(fig, width="stretch")
    b.dataframe(res[["rmse", "mae"]].rename(columns=str.upper).style.format("{:.4f}")
                .highlight_min(color="#d6e4fb"), width="stretch")

    st.subheader("2. Kualitas ranking Top-10 (semakin besar semakin baik)")
    rel = metrics["params"]["relevant_min_rating"]
    st.caption(f"Buku di data uji dengan rating ≥ {rel} dianggap relevan. Untuk tiap user, model meranking "
               f"semua buku yang belum dibaca lalu dihitung berapa buku relevan yang masuk 10 teratas.")
    rk = res[["precision@10", "recall@10", "hitrate@10"]]
    long = rk.reset_index().melt(id_vars="index", var_name="Metrik", value_name="Nilai")
    fig = px.bar(long, x="Metrik", y="Nilai", color="index", barmode="group", text_auto=".3f",
                 color_discrete_sequence=["#B8C4D6", "#7FA3E0", ACCENT, "#1B3F80"], labels={"index": ""})
    fig.update_layout(height=360, legend_title=None, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("3. Kurva pelatihan Matrix Factorization")
    cv = pd.DataFrame(metrics["mf_curve"]).melt(id_vars="epoch", var_name="Data", value_name="RMSE")
    cv["Data"] = cv["Data"].map({"train_rmse": "Latih", "test_rmse": "Uji"})
    fig = px.line(cv, x="epoch", y="RMSE", color="Data", markers=True,
                  color_discrete_sequence=[ACCENT, "#E07A3F"], labels={"epoch": "Epoch"})
    fig.update_layout(height=320, legend_title=None, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Interpretasi")
    best_rmse = res["rmse"].idxmin()
    best_rank = res["hitrate@10"].idxmax()
    st.markdown(f"""
- **Prediksi rating:** {best_rmse} memberi RMSE terendah ({res.loc[best_rmse, 'rmse']:.3f}).
  Matrix Factorization hampir menyamai baseline, artinya pada data sejarang ini (densitas
  {report['cf_density_pct']:.3f}%) sebagian besar sinyal yang bisa dipelajari adalah *bias* —
  ada user yang royal memberi nilai, ada buku yang memang disukai banyak orang.
- **Ranking:** {best_rank} memiliki HitRate@10 tertinggi. Rekomendasi buku populer menjadi pembanding yang
  sulit dikalahkan pada data Book-Crossing, karena buku laris memang paling sering dibaca. Di antara model
  personal, **Item-based KNN** unggul dalam ranking dan sekaligus mudah dijelaskan ("karena Anda menyukai X").
- **Kurva MF:** error data uji berhenti turun di sekitar epoch ke-{int(pd.DataFrame(metrics['mf_curve'])['test_rmse'].idxmin()) + 1};
  regularisasi (λ = {metrics['params']['mf']['reg']}) dan jumlah faktor ({metrics['params']['mf']['factors']})
  dipilih lewat grid search untuk menahan *overfitting*.
- **Saran pengembangan:** memanfaatkan rating implisit (mis. ALS implisit / BPR), model hybrid
  CF + konten (penulis, penerbit), dan memperketat filter k-core bila mengutamakan akurasi di atas cakupan.
""")
    with st.expander("Parameter model"):
        st.json(metrics["params"])
