# Ubah CSV hasil export_for_r.py menjadi satu file data/app_data.rds untuk app Shiny.
# Pemakaian:  Rscript build_rds.R <folder_csv> <folder_app>
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; app <- args[2]
rd <- function(f, ...) read.csv(file.path(src, f), stringsAsFactors = FALSE, ...)

bstats <- rd("bstats.csv", colClasses = c(isbn = "character"))
bstats$label <- paste0(substr(bstats$title, 1, 80), " — ", bstats$author)

mf_users <- rd("mf_users.csv")
mf_items <- rd("mf_items.csv", colClasses = c(isbn = "character"))

# tabel fakta level-baris (1 baris = 1 rating) untuk filter interaktif
rr <- rd("rows_ratings.csv", colClasses = c(isbn = "character"))
ru <- rd("rows_users.csv")
rb <- rd("rows_books.csv", colClasses = c(isbn = "character"))
ru$country[ru$country == ""] <- NA
rb$year[rb$year == 0] <- NA
age_lv <- c("<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+")
ui_ <- match(rr$user_id, ru$user_id)
bi_ <- match(rr$isbn, rb$isbn)
fact <- data.frame(
  user_id   = rr$user_id,
  book      = bi_,                                   # indeks baris di tabel `books`
  rating    = as.integer(rr$rating),
  country   = factor(ru$country[ui_]),
  age_group = factor(ru$age_group[ui_], levels = age_lv),
  age       = ru$age[ui_],
  year      = as.integer(rb$year[bi_])
)
rb$label <- paste0(substr(rb$title, 1, 80), " — ", rb$author)

app_data <- list(
  fact      = fact,
  books     = rb,
  bstats    = bstats,
  ratings   = rd("ratings_cf.csv", colClasses = c(isbn = "character")),
  users     = rd("users_cf.csv"),
  neighbors = rd("neighbors.csv", colClasses = c(isbn = "character", neighbor_isbn = "character")),
  mf = list(
    user_ids = mf_users$user_id, bu = mf_users$bu,
    isbns = mf_items$isbn, bi = mf_items$bi,
    P = as.matrix(rd("mf_P.csv")), Q = as.matrix(rd("mf_Q.csv")),
    mu = as.numeric(readLines(file.path(src, "mf_mu.txt")))
  ),
  eda = list(
    rating_dist = rd("eda_rating_dist.csv"),
    age         = rd("eda_age.csv"),
    country     = rd("eda_country.csv"),
    age_group   = rd("eda_age_group.csv"),
    year        = rd("eda_year.csv"),
    users_sum   = rd("eda_users_summary.csv")
  ),
  report  = jsonlite::fromJSON(file.path(src, "cleaning_report.json")),
  metrics = jsonlite::fromJSON(file.path(src, "metrics.json"))
)
dir.create(file.path(app, "data"), showWarnings = FALSE)
saveRDS(app_data, file.path(app, "data", "app_data.rds"), compress = "xz")
cat("tersimpan:", file.path(app, "data", "app_data.rds"),
    round(file.size(file.path(app, "data", "app_data.rds")) / 1e6, 2), "MB\n")
