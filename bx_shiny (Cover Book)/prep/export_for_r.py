"""
Ekspor hasil cleaning & model (Python) menjadi CSV yang lalu diubah ke .rds oleh build_rds.R.
Pemakaian:  python export_for_r.py <folder_bx_recsys> <folder_csv_output>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

users = pd.read_parquet(SRC / "data/users.parquet")
books = pd.read_parquet(SRC / "data/books.parquet")
r_all = pd.read_parquet(SRC / "data/ratings_all.parquet")
r_cf = pd.read_parquet(SRC / "data/ratings_cf.parquet")
nb = pd.read_parquet(SRC / "models/item_neighbors.parquet")
mf = np.load(SRC / "models/mf_model.npz")

# --- tabel untuk rekomendasi
bstats = (r_cf.groupby("isbn")["rating"].agg(n_rating="size", avg_rating="mean").reset_index()
          .merge(books[["isbn", "title", "author", "year", "publisher"]], on="isbn", how="left")
          .sort_values("n_rating", ascending=False))
bstats.to_csv(OUT / "bstats.csv", index=False)
r_cf.to_csv(OUT / "ratings_cf.csv", index=False)
users[users["user_id"].isin(r_cf["user_id"])][["user_id", "country", "age"]].to_csv(OUT / "users_cf.csv", index=False)
nb.to_csv(OUT / "neighbors.csv", index=False)

pd.DataFrame({"user_id": mf["user_ids"], "bu": mf["bu"]}).to_csv(OUT / "mf_users.csv", index=False)
pd.DataFrame({"isbn": mf["isbns"], "bi": mf["bi"]}).to_csv(OUT / "mf_items.csv", index=False)
pd.DataFrame(mf["P"]).to_csv(OUT / "mf_P.csv", index=False)
pd.DataFrame(mf["Q"]).to_csv(OUT / "mf_Q.csv", index=False)
(OUT / "mf_mu.txt").write_text(str(float(mf["mu"])))

# --- ringkasan EDA (diolah di sini agar app R ringan)
rd = r_all["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="n")
rd.to_csv(OUT / "eda_rating_dist.csv", index=False)
ages = users["age"].dropna().astype(int).value_counts().sort_index().rename_axis("age").reset_index(name="n")
ages.to_csv(OUT / "eda_age.csv", index=False)
users["country"].value_counts().head(15).rename_axis("country").reset_index(name="n").to_csv(
    OUT / "eda_country.csv", index=False)
ex = r_all[r_all["rating"] > 0].merge(users[["user_id", "age_group"]], on="user_id").dropna(subset=["age_group"])
ex.groupby("age_group")["rating"].agg(mean="mean", n="size").reset_index().to_csv(OUT / "eda_age_group.csv", index=False)
y = books["year"].dropna().astype(int)
y[y >= 1950].value_counts().sort_index().rename_axis("year").reset_index(name="n").to_csv(OUT / "eda_year.csv", index=False)
pd.DataFrame({"missing_age_pct": [100 * users["age"].isna().mean()],
              "median_age": [users["age"].median()]}).to_csv(OUT / "eda_users_summary.csv", index=False)

# --- data level-baris untuk filter interaktif di dashboard
work = books.set_index("isbn")["work_isbn"]
rows = r_all[["user_id", "isbn", "rating"]].copy()
rows["isbn"] = rows["isbn"].map(work)
rows.to_csv(OUT / "rows_ratings.csv", index=False)
u = users[users["user_id"].isin(rows["user_id"])][["user_id", "country", "age", "age_group"]]
u.to_csv(OUT / "rows_users.csv", index=False)
bk = books[books["isbn"].isin(rows["isbn"].unique())][["isbn", "title", "author", "year", "publisher"]]
bk.to_csv(OUT / "rows_books.csv", index=False)

for f in ["data/cleaning_report.json", "models/metrics.json"]:
    (OUT / Path(f).name).write_text((SRC / f).read_text())
print("selesai:", sorted(p.name for p in OUT.iterdir()))
