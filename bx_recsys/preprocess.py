"""
Tahap 1 - Data cleaning Book-Crossing (BX) dataset.

Pemakaian:
    python preprocess.py <folder_berisi_csv_BX> [folder_output]

Output (default ./data):
    users.parquet, books.parquet, ratings_all.parquet, ratings_cf.parquet,
    cleaning_report.json
"""
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("raw")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
OUT.mkdir(parents=True, exist_ok=True)

# Ambang filter untuk matriks collaborative filtering
MIN_RATINGS_PER_USER = 5    # user minimal memberi 5 rating eksplisit
MIN_RATINGS_PER_BOOK = 5    # buku minimal menerima 5 rating eksplisit

READ = dict(sep=";", encoding="latin-1", escapechar="\\", low_memory=False)
report = {}

# ---------------------------------------------------------------- USERS
users = pd.read_csv(RAW / "BX-Users.csv", **READ)
users.columns = ["user_id", "location", "age"]
report["users_raw"] = len(users)

users["age"] = pd.to_numeric(users["age"], errors="coerce")
report["age_missing_raw"] = int(users["age"].isna().sum())
outlier = (users["age"] < 5) | (users["age"] > 100)
report["age_outlier_set_nan"] = int(outlier.sum())
users.loc[outlier, "age"] = np.nan

parts = users["location"].astype(str).str.split(",")
def _clean_loc(s):
    s = s.strip().strip('"').strip().lower()
    return np.nan if s in {"", "n/a", "na", "-", "?", "none", "null", "."} else s
users["city"] = parts.str[0].map(_clean_loc)
users["state"] = parts.map(lambda p: _clean_loc(p[1]) if len(p) > 2 else np.nan)
users["country"] = parts.str[-1].map(_clean_loc)

bins = [0, 17, 24, 34, 44, 54, 64, 100]
labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
users["age_group"] = pd.cut(users["age"], bins=bins, labels=labels).astype("string")

# ---------------------------------------------------------------- BOOKS
books = pd.read_csv(RAW / "BX-Books.csv", **READ)
books.columns = ["isbn", "title", "author", "year", "publisher",
                 "img_s", "img_m", "img_l"]
report["books_raw"] = len(books)

books["isbn"] = books["isbn"].astype(str).str.strip().str.upper()
for c in ["title", "author", "publisher"]:
    books[c] = books[c].fillna("Unknown").astype(str).map(html.unescape).str.strip()
books["year"] = pd.to_numeric(books["year"], errors="coerce")
bad_year = (books["year"] < 1900) | (books["year"] > 2005)  # data dikumpulkan 2004
report["year_invalid_set_nan"] = int(bad_year.sum())
books.loc[bad_year, "year"] = np.nan
books["year"] = books["year"].astype("Int64")
books = books.drop(columns=["img_s", "img_l"]).drop_duplicates("isbn")

# ---------------------------------------------------------------- RATINGS
ratings = pd.read_csv(RAW / "BX-Book-Ratings.csv", **READ)
ratings.columns = ["user_id", "isbn", "rating"]
report["ratings_raw"] = len(ratings)

ratings["isbn"] = ratings["isbn"].astype(str).str.strip().str.upper()
ratings = ratings.drop_duplicates(["user_id", "isbn"])
ratings = ratings[ratings["isbn"].isin(books["isbn"]) & ratings["user_id"].isin(users["user_id"])]
report["ratings_matched_to_books"] = len(ratings)
ratings["type"] = np.where(ratings["rating"] == 0, "implisit", "eksplisit")
ratings.to_parquet(OUT / "ratings_all.parquet", index=False, compression="zstd", compression_level=19)

# --------------------------------------------- RATINGS UNTUK CF (eksplisit)
cf = ratings[ratings["rating"] > 0][["user_id", "isbn", "rating"]].copy()
report["ratings_explicit"] = len(cf)

# Gabungkan edisi: ISBN berbeda dengan judul + penulis sama = satu buku.
# ISBN kanonik = edisi dengan rating eksplisit terbanyak.
def _norm_title(s):
    s = s.str.lower().str.replace(r"\(.*?\)|\[.*?\]", "", regex=True)
    s = s.str.split(":").str[0]
    return s.str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")
books["work_key"] = (_norm_title(books["title"]) + "|" +
                     books["author"].str.lower().str.replace(r"[^a-z]", "", regex=True))
cnt = cf["isbn"].value_counts()
books["_n"] = books["isbn"].map(cnt).fillna(0)
canon = (books.sort_values("_n", ascending=False)
              .drop_duplicates("work_key")[["work_key", "isbn"]]
              .rename(columns={"isbn": "work_isbn"}))
books = books.merge(canon, on="work_key").drop(columns=["_n"])
cf["isbn"] = cf["isbn"].map(books.set_index("isbn")["work_isbn"])
cf = cf.groupby(["user_id", "isbn"], as_index=False)["rating"].mean()
report["cf_ratings_after_edition_merge"] = len(cf)
report["books_merged_into_other_edition"] = int((books["isbn"] != books["work_isbn"]).sum())

# filter iteratif sampai stabil (k-core)
while True:
    n0 = len(cf)
    bc = cf["isbn"].value_counts()
    cf = cf[cf["isbn"].isin(bc[bc >= MIN_RATINGS_PER_BOOK].index)]
    uc = cf["user_id"].value_counts()
    cf = cf[cf["user_id"].isin(uc[uc >= MIN_RATINGS_PER_USER].index)]
    if len(cf) == n0:
        break
report["cf_ratings"] = len(cf)
report["cf_users"] = int(cf["user_id"].nunique())
report["cf_books"] = int(cf["isbn"].nunique())
report["cf_density_pct"] = round(100 * len(cf) / (report["cf_users"] * report["cf_books"]), 4)
report["min_ratings_per_user"] = MIN_RATINGS_PER_USER
report["min_ratings_per_book"] = MIN_RATINGS_PER_BOOK
cf.to_parquet(OUT / "ratings_cf.parquet", index=False, compression="zstd", compression_level=19)

# ---------------------------------------------------------------- SIMPAN
users.to_parquet(OUT / "users.parquet", index=False, compression="zstd", compression_level=19)
books.to_parquet(OUT / "books.parquet", index=False, compression="zstd", compression_level=19)
report["users_clean"] = len(users)
report["books_clean"] = len(books)
(OUT / "cleaning_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
