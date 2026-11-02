# Ekspor isi data/app_data.rds menjadi file CSV di data/csv/ (untuk dibaca di Excel / dilampirkan di laporan).
# Setiap file dijaga < 20 MB agar bisa di-upload lewat web GitHub (batas 25 MB per file);
# tabel yang lebih besar dipecah menjadi beberapa bagian (_part1, _part2, ...).
# Pemakaian (dari folder yang berisi app.R):  Rscript prep/export_csv.R
d   <- readRDS("data/app_data.rds")
out <- file.path("data", "csv")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
unlink(list.files(out, pattern = "\\.csv$", full.names = TRUE))

MAX_MB <- 20
BOM <- "﻿"   # agar Excel membaca huruf khusus (UTF-8) dengan benar

write_one <- function(df, path) {
  con <- file(path, open = "w", encoding = "UTF-8")
  writeLines(BOM, con, sep = "")
  write.csv(df, con, row.names = FALSE, na = "", fileEncoding = "UTF-8")
  close(con)
}

save_csv <- function(df, name) {
  path <- file.path(out, paste0(name, ".csv"))
  write_one(df, path)
  mb <- file.size(path) / 1e6
  if (mb <= MAX_MB) { cat(sprintf("%-32s %9d baris  %6.2f MB\n", basename(path), nrow(df), mb)); return(invisible()) }
  unlink(path)
  k <- ceiling(mb / MAX_MB) + 1
  idx <- split(seq_len(nrow(df)), cut(seq_len(nrow(df)), k, labels = FALSE))
  for (i in seq_along(idx)) {
    p <- file.path(out, sprintf("%s_part%d.csv", name, i))
    write_one(df[idx[[i]], , drop = FALSE], p)
    cat(sprintf("%-32s %9d baris  %6.2f MB\n", basename(p), length(idx[[i]]), file.size(p) / 1e6))
  }
}

# ---- data rating (fact): ganti indeks buku dengan ISBN agar bisa dihubungkan ke books.csv
fact <- d$fact
fact$isbn <- d$books$isbn[fact$book]
fact <- fact[, c("user_id", "isbn", "rating", "country", "age_group", "age", "year")]
fact$country   <- as.character(fact$country)
fact$age_group <- as.character(fact$age_group)
save_csv(fact, "ratings_all")

save_csv(d$books[, c("isbn", "title", "author", "year", "publisher", "img_s", "img_m", "img_l")], "books")
save_csv(d$bstats[, c("isbn", "title", "author", "year", "publisher", "n_rating", "avg_rating", "img_s", "img_m", "img_l")], "cf_books")
save_csv(d$ratings, "cf_ratings")
save_csv(d$users, "cf_users")
save_csv(d$neighbors, "item_neighbors")

# ---- model Matrix Factorization
P <- as.data.frame(d$mf$P); names(P) <- paste0("f", seq_len(ncol(P)))
Q <- as.data.frame(d$mf$Q); names(Q) <- paste0("f", seq_len(ncol(Q)))
save_csv(data.frame(user_id = d$mf$user_ids, bias = d$mf$bu, P), "mf_user_factors")
save_csv(data.frame(isbn = d$mf$isbns, bias = d$mf$bi, Q), "mf_book_factors")
save_csv(data.frame(parameter = "global_mean_mu", nilai = d$mf$mu), "mf_global")

# ---- hasil cleaning & evaluasi
save_csv(data.frame(keterangan = names(d$report), nilai = unlist(d$report), row.names = NULL), "cleaning_report")
res <- d$metrics$results
save_csv(do.call(rbind, lapply(names(res), function(m) data.frame(
  model = m, rmse = res[[m]]$rmse, mae = res[[m]]$mae, precision_at_10 = res[[m]]$`precision@10`,
  recall_at_10 = res[[m]]$`recall@10`, hitrate_at_10 = res[[m]]$`hitrate@10`,
  n_user_dievaluasi = res[[m]]$n_users_eval))), "evaluation_metrics")
save_csv(d$metrics$mf_curve, "mf_training_curve")
prm <- d$metrics$params; sp <- d$metrics$split
save_csv(data.frame(
  parameter = c("test_frac", "seed", "n_train", "n_test", "k_neighbors", "shrink", "relevant_min_rating", "top_n",
                paste0("mf_", names(prm$mf))),
  nilai = c(sp$test_frac, sp$seed, sp$train, sp$test, prm$k_neighbors, prm$shrink, prm$relevant_min_rating,
            prm$top_n, unlist(prm$mf))), "model_parameters")

cat("\nTotal:", length(list.files(out)), "file CSV di", out, "\n")
