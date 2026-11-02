"""
Tahap 2 - Model collaborative filtering + evaluasi.

Model yang dibandingkan:
  1. Global mean            : prediksi = rata-rata semua rating
  2. Baseline (bias)        : mu + b_user + b_item (regularized)
  3. Item-based KNN         : baseline + rata-rata tertimbang residual dari
                              k buku paling mirip (cosine, dengan shrinkage)
  4. Matrix Factorization   : Funk-SVD berbias, dilatih dengan SGD

Evaluasi: hold-out 20% rating tiap user -> RMSE, MAE (prediksi rating)
          dan Precision@10, Recall@10, HitRate@10 (kualitas ranking).

Pemakaian:  python train.py [folder_data] [folder_model]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
MODELS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRAC = 0.2
K_NEIGHBORS = 20         # tetangga untuk prediksi item-KNN
N_NEIGHBORS_SAVE = 20    # tetangga yang disimpan untuk dashboard
SHRINK = 100             # shrinkage kemiripan (penalti item dengan sedikit co-rater)
MF = dict(factors=16, epochs=20, lr=0.007, reg=0.2, reg_bias=0.02)
TOPN = 10
RELEVANT_MIN = 7         # rating >= 7 dianggap "relevan" untuk metrik ranking
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ data
df = pd.read_parquet(DATA / "ratings_cf.parquet")
users = np.sort(df["user_id"].unique())
items = np.sort(df["isbn"].unique())
uidx = {u: i for i, u in enumerate(users)}
iidx = {b: i for i, b in enumerate(items)}
df["u"] = df["user_id"].map(uidx).astype(np.int32)
df["i"] = df["isbn"].map(iidx).astype(np.int32)
df["r"] = df["rating"].astype(np.float32)
nU, nI = len(users), len(items)

# split per user: 20% rating tiap user ke test (min 1)
df["rnd"] = rng.random(len(df))
df["rank"] = df.groupby("u")["rnd"].rank(method="first")
df["n"] = df.groupby("u")["u"].transform("size")
df["is_test"] = df["rank"] <= np.maximum(1, np.floor(df["n"] * TEST_FRAC))
train, test = df[~df["is_test"]], df[df["is_test"]]
print(f"train={len(train):,} test={len(test):,} users={nU:,} items={nI:,}")


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def clip(p): return np.clip(p, 1, 10)


# ------------------------------------------------------------------ model
def fit_baseline(tr, reg_u=5.0, reg_i=5.0, iters=10):
    mu = tr["r"].mean()
    bu, bi = np.zeros(nU), np.zeros(nI)
    u, i, r = tr["u"].values, tr["i"].values, tr["r"].values
    for _ in range(iters):
        res = r - mu - bu[u]
        bi = np.bincount(i, res, nI) / (np.bincount(i, minlength=nI) + reg_i)
        res = r - mu - bi[i]
        bu = np.bincount(u, res, nU) / (np.bincount(u, minlength=nU) + reg_u)
    return mu, bu, bi


def item_similarity(tr, mu, bu, bi, k):
    """Cosine kemiripan antar-buku pada residual baseline, + shrinkage, top-k per buku."""
    res = tr["r"].values - mu - bu[tr["u"].values] - bi[tr["i"].values]
    R = sp.csr_matrix((res, (tr["i"].values, tr["u"].values)), shape=(nI, nU))
    B = sp.csr_matrix((np.ones(len(tr)), (tr["i"].values, tr["u"].values)), shape=(nI, nU))
    norm = np.sqrt(R.multiply(R).sum(axis=1)).A1
    norm[norm == 0] = 1
    Rn = sp.diags(1 / norm) @ R
    S = (Rn @ Rn.T).tocsr()
    C = (B @ B.T).tocsr()                               # jumlah co-rater
    C.data = C.data / (C.data + SHRINK)                 # shrinkage: n/(n+SHRINK)
    S = S.multiply(C).tocsr()
    S.setdiag(0); S.eliminate_zeros()
    # ambil top-k per baris
    rows, cols, vals = [], [], []
    for r_ in range(nI):
        s, e = S.indptr[r_], S.indptr[r_ + 1]
        if e == s:
            continue
        c, v = S.indices[s:e], S.data[s:e]
        top = np.argsort(-v)[:k]
        top = top[v[top] > 0]
        rows.append(np.full(len(top), r_)); cols.append(c[top]); vals.append(v[top])
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    return sp.csr_matrix((vals, (rows, cols)), shape=(nI, nI)), R, B


def predict_itemknn(Sk, R, B, mu, bu, bi, u, i):
    num = (Sk @ R).tocsr()            # items x users
    den = (abs(Sk) @ B).tocsr()
    n = np.asarray(num[i, u]).ravel()
    d = np.asarray(den[i, u]).ravel()
    adj = np.divide(n, d, out=np.zeros_like(n), where=d > 0)
    return mu + bu[u] + bi[i] + adj


def fit_mf(tr, te=None, factors=32, epochs=40, lr=0.007, reg=0.05, reg_bias=0.02):
    g = np.random.default_rng(SEED)
    mu = tr["r"].mean()
    P = g.normal(0, 0.1, (nU, factors)); Q = g.normal(0, 0.1, (nI, factors))
    bu, bi = np.zeros(nU), np.zeros(nI)
    u_all, i_all, r_all = tr["u"].values, tr["i"].values, tr["r"].values.astype(float)
    curve = []
    for ep in range(epochs):
        order = g.permutation(len(tr))
        for idx in order:
            u, i, r = u_all[idx], i_all[idx], r_all[idx]
            pu, qi = P[u], Q[i]
            e = r - (mu + bu[u] + bi[i] + pu @ qi)
            bu[u] += lr * (e - reg_bias * bu[u])
            bi[i] += lr * (e - reg_bias * bi[i])
            P[u] = pu + lr * (e * qi - reg * pu)
            Q[i] = qi + lr * (e * pu - reg * qi)
        tr_p = clip(mu + bu[u_all] + bi[i_all] + np.sum(P[u_all] * Q[i_all], 1))
        row = {"epoch": ep + 1, "train_rmse": rmse(r_all, tr_p)}
        if te is not None:
            tu, ti = te["u"].values, te["i"].values
            te_p = clip(mu + bu[tu] + bi[ti] + np.sum(P[tu] * Q[ti], 1))
            row["test_rmse"] = rmse(te["r"].values, te_p)
        curve.append(row)
    return mu, bu, bi, P, Q, curve


def ranking_metrics(score_fn, tr, te, n=TOPN):
    """score_fn(u) -> skor untuk semua item. Item di train di-mask."""
    seen = tr.groupby("u")["i"].apply(np.array).to_dict()
    rel = te[te["r"] >= RELEVANT_MIN].groupby("u")["i"].apply(set).to_dict()
    prec, rec, hit = [], [], []
    for u, rel_items in rel.items():
        s = score_fn(u).astype(float).copy()
        s[seen.get(u, [])] = -np.inf
        top = np.argpartition(-s, n)[:n]
        h = len(rel_items.intersection(top))
        prec.append(h / n); rec.append(h / len(rel_items)); hit.append(h > 0)
    return {"precision@10": float(np.mean(prec)), "recall@10": float(np.mean(rec)),
            "hitrate@10": float(np.mean(hit)), "n_users_eval": len(prec)}


# ------------------------------------------------------------- evaluasi
y = test["r"].values
tu, ti = test["u"].values, test["i"].values
results = {}

t = time.time()
mu0 = train["r"].mean()
results["Global mean"] = {"rmse": rmse(y, np.full_like(y, mu0)), "mae": mae(y, np.full_like(y, mu0))}

mu, bu, bi = fit_baseline(train)
p = clip(mu + bu[tu] + bi[ti])
results["Baseline (bias)"] = {"rmse": rmse(y, p), "mae": mae(y, p)}

Sk, R, B = item_similarity(train, mu, bu, bi, K_NEIGHBORS)
p = clip(predict_itemknn(Sk, R, B, mu, bu, bi, tu, ti))
results["Item-based KNN"] = {"rmse": rmse(y, p), "mae": mae(y, p)}
print(f"baseline & item-KNN selesai ({time.time()-t:.1f}s)")

t = time.time()
mu_m, bu_m, bi_m, P, Q, curve = fit_mf(train, test, **MF)
p = clip(mu_m + bu_m[tu] + bi_m[ti] + np.sum(P[tu] * Q[ti], 1))
results["Matrix Factorization (SVD)"] = {"rmse": rmse(y, p), "mae": mae(y, p)}
print(f"MF selesai ({time.time()-t:.1f}s)")

# metrik ranking
pop = np.bincount(train["i"].values, minlength=nI).astype(float)
results["Global mean"].update(ranking_metrics(lambda u: pop, train, test))  # = rekomendasi populer
results["Baseline (bias)"].update(ranking_metrics(lambda u: bi, train, test))
SkR = (Sk @ R).T.tocsr()                 # users x items (numerator)
SkB = (abs(Sk) @ B).T.tocsr()
def knn_scores(u):
    n_ = SkR[u].toarray().ravel(); d_ = SkB[u].toarray().ravel()
    return mu + bu[u] + bi + np.divide(n_, d_, out=np.zeros_like(n_), where=d_ > 0)
results["Item-based KNN"].update(ranking_metrics(knn_scores, train, test))
results["Matrix Factorization (SVD)"].update(
    ranking_metrics(lambda u: mu_m + bu_m[u] + bi_m + Q @ P[u], train, test))
results["Global mean"]["catatan_ranking"] = "ranking memakai popularitas (jumlah rating)"

for k, v in results.items():
    print(f"{k:28s} " + " ".join(f"{m}={v[m]:.4f}" for m in ["rmse", "mae", "precision@10", "recall@10", "hitrate@10"]))

# --------------------------------------------- latih ulang pada semua data
full = df
mu, bu, bi = fit_baseline(full)
Sk_full, _, _ = item_similarity(full, mu, bu, bi, N_NEIGHBORS_SAVE)
coo = Sk_full.tocoo()
nb = pd.DataFrame({"isbn": items[coo.row], "neighbor_isbn": items[coo.col], "similarity": coo.data})
nb["rank"] = nb.groupby("isbn")["similarity"].rank(ascending=False, method="first").astype(int)
nb.sort_values(["isbn", "rank"]).to_parquet(MODELS / "item_neighbors.parquet", index=False)

mu_m, bu_m, bi_m, P, Q, _ = fit_mf(full, None, **MF)
np.savez_compressed(MODELS / "mf_model.npz", user_ids=users, isbns=items.astype(str),
                    P=P.astype(np.float32), Q=Q.astype(np.float32),
                    bu=bu_m.astype(np.float32), bi=bi_m.astype(np.float32), mu=np.float32(mu_m))

meta = {"split": {"train": len(train), "test": len(test), "test_frac": TEST_FRAC, "seed": SEED},
        "params": {"k_neighbors": K_NEIGHBORS, "shrink": SHRINK, "mf": MF,
                   "relevant_min_rating": RELEVANT_MIN, "top_n": TOPN},
        "results": results, "mf_curve": curve}
(MODELS / "metrics.json").write_text(json.dumps(meta, indent=2))
print("artefak model tersimpan di", MODELS)
