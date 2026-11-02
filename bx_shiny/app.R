# =====================================================================
# Dashboard Sistem Rekomendasi Buku (Collaborative Filtering)
# Book-Crossing dataset — R Shiny
# Jalankan:  shiny::runApp()   (butuh file data/app_data.rds)
# =====================================================================
library(shiny)
library(bslib)
library(dplyr)
library(plotly)
library(DT)

d <- readRDS("data/app_data.rds")
bstats  <- d$bstats
ratings <- d$ratings
users   <- d$users
nbr     <- d$neighbors
mf      <- d$mf
eda     <- d$eda
rep     <- d$report
met     <- d$metrics

ACCENT <- "#2E6FD8"
LIGHT  <- "#B8C4D6"
`%||%` <- function(a, b) if (is.null(a)) b else a
fmt <- function(x) format(round(x), big.mark = ".", decimal.mark = ",", scientific = FALSE)
user_counts <- ratings |> count(user_id, name = "n") |> arrange(desc(n))
mf_upos <- setNames(seq_along(mf$user_ids), mf$user_ids)

# ---------- helper tabel buku
book_table <- function(df, extra = character(0)) {
  out <- df |>
    left_join(bstats |> select(isbn, title, author, year, avg_rating, n_rating), by = "isbn") |>
    select(Judul = title, Penulis = author, Tahun = year, all_of(extra),
           `Rata-rata rating` = avg_rating, `Jumlah rating` = n_rating)
  dt <- datatable(out, rownames = seq_len(nrow(out)),
                  options = list(dom = "t", pageLength = 25, ordering = FALSE),
                  class = "compact stripe hover")
  dt <- formatRound(dt, "Rata-rata rating", 2)
  for (col in intersect(extra, c("Kemiripan", "Skor", "Prediksi rating")))
    dt <- formatRound(dt, col, if (col == "Kemiripan") 4 else 3)
  dt
}

# ---------- item-KNN dari daftar (isbn, rating) -> skor kandidat
knn_recommend <- function(hist, n) {
  mean_r <- mean(hist$rating)
  m <- nbr |>
    filter(isbn %in% hist$isbn) |>
    inner_join(hist, by = "isbn") |>
    filter(!neighbor_isbn %in% hist$isbn) |>
    mutate(w = similarity * (rating - mean_r + 1)) |>
    filter(w > 0)
  if (nrow(m) == 0) return(NULL)
  why <- m |> group_by(neighbor_isbn) |> slice_max(w, n = 1, with_ties = FALSE) |>
    ungroup() |> left_join(bstats |> select(isbn, title), by = "isbn") |>
    select(neighbor_isbn, why = title)
  m |> group_by(neighbor_isbn) |> summarise(score = sum(w), .groups = "drop") |>
    slice_max(score, n = n, with_ties = FALSE) |>
    left_join(why, by = "neighbor_isbn") |>
    transmute(isbn = neighbor_isbn, Skor = score / max(score),
              `Karena Anda menyukai` = substr(why, 1, 60))
}

bar_h <- function(df, x, y, title, xlab) {
  plot_ly(df, x = df[[x]], y = factor(df[[y]], levels = df[[y]]), type = "bar",
          orientation = "h", marker = list(color = ACCENT)) |>
    layout(title = title, xaxis = list(title = xlab), yaxis = list(title = ""),
           margin = list(l = 10))
}

value_box2 <- function(title, value, sub = NULL)
  value_box(title = title, value = value, p(sub, class = "small"),
            theme = value_box_theme(bg = "#EEF3FC", fg = "#1B3F80"))

# =====================================================================
# UI
# =====================================================================
ui <- page_navbar(
  title = "📚 BX Book Recommender",
  theme = bs_theme(version = 5, primary = ACCENT),
  sidebar = sidebar(
    width = 260, open = "desktop",
    p(HTML(sprintf("Data CF: <b>%s</b> user × <b>%s</b> buku, <b>%s</b> rating eksplisit (densitas %.3f%%).",
                   fmt(rep$cf_users), fmt(rep$cf_books), fmt(rep$cf_ratings), rep$cf_density_pct))),
    p("Sumber: Book-Crossing dataset (Ziegler et al., 2005).", class = "text-muted small")
  ),

  # ------------------------------------------------ RINGKASAN
  nav_panel("Ringkasan",
    h3("Sistem Rekomendasi Buku — Collaborative Filtering"),
    p("Alur kerja: data cleaning → eksplorasi → pemodelan collaborative filtering → evaluasi → rekomendasi."),
    layout_columns(
      value_box2("User", fmt(rep$users_raw)),
      value_box2("Buku", fmt(rep$books_raw)),
      value_box2("Rating", fmt(rep$ratings_raw)),
      value_box2("Rating eksplisit (1–10)", fmt(rep$ratings_explicit),
                 sprintf("%.1f%% dari rating valid", 100 * rep$ratings_explicit / rep$ratings_matched_to_books))
    ),
    card(card_header("Alur data: dari mentah sampai matriks CF"), plotlyOutput("funnel", height = 320)),
    layout_columns(
      card(card_header("Langkah cleaning"), uiOutput("cleaning_text")),
      card(card_header("Model yang dibangun"), tableOutput("model_table"))
    )
  ),

  # ------------------------------------------------ EDA
  nav_panel("Eksplorasi Data",
    navset_card_tab(
      nav_panel("Rating",
        layout_columns(plotlyOutput("p_rating_all"), plotlyOutput("p_rating_ex")),
        p("Sekitar dua pertiga rating bernilai 0 (implisit). Rating eksplisit condong ke nilai tinggi (7–10): user cenderung hanya menilai buku yang disukai.", class = "text-muted"),
        layout_columns(plotlyOutput("p_per_user"), plotlyOutput("p_per_book")),
        textOutput("longtail_text")
      ),
      nav_panel("Buku",
        sliderInput("min_n", "Minimal jumlah rating eksplisit", 5, 200, 50, step = 5),
        layout_columns(plotlyOutput("p_top_count", height = 520), plotlyOutput("p_top_avg", height = 520)),
        layout_columns(plotlyOutput("p_authors"), plotlyOutput("p_year"))
      ),
      nav_panel("User",
        layout_columns(plotlyOutput("p_age"), plotlyOutput("p_country")),
        textOutput("age_text"),
        plotlyOutput("p_age_group")
      )
    )
  ),

  # ------------------------------------------------ REKOMENDASI
  nav_panel("Rekomendasi",
    navset_card_tab(
      nav_panel("🔎 Buku mirip",
        p("Pilih sebuah buku; sistem mencari buku lain yang dirating serupa oleh pembaca yang sama (item-based KNN)."),
        layout_columns(col_widths = c(8, 4),
          selectizeInput("book", "Buku", choices = NULL, width = "100%"),
          sliderInput("n_sim", "Jumlah rekomendasi", 5, 20, 10)),
        uiOutput("book_info"),
        DTOutput("tbl_similar")
      ),
      nav_panel("👤 Untuk user yang ada",
        layout_columns(col_widths = c(5, 4, 3),
          selectizeInput("uid", "User-ID (urut dari yang paling banyak memberi rating)", choices = NULL),
          radioButtons("model", "Model", c("Matrix Factorization", "Item-based KNN", "Populer"), inline = TRUE),
          sliderInput("n_user", "Jumlah rekomendasi", 5, 20, 10)),
        uiOutput("user_info"),
        layout_columns(col_widths = c(7, 5),
          card(card_header(textOutput("rec_title")), DTOutput("tbl_user")),
          card(card_header("Riwayat rating user"), DTOutput("tbl_hist")))
      ),
      nav_panel("✨ User baru",
        p("Simulasi user baru: pilih beberapa buku yang pernah dibaca dan beri rating. Rekomendasi dihitung dari kemiripan antar-buku; bila belum ada pilihan, sistem menampilkan buku populer."),
        selectizeInput("picks", "Buku yang pernah dibaca (maks. 10)", choices = NULL, multiple = TRUE,
                       width = "100%", options = list(maxItems = 10, placeholder = "Ketik judul, mis. Harry Potter, Da Vinci...")),
        uiOutput("pick_sliders"),
        sliderInput("n_new", "Jumlah rekomendasi", 5, 20, 10),
        uiOutput("new_note"),
        DTOutput("tbl_new")
      )
    )
  ),

  # ------------------------------------------------ EVALUASI
  nav_panel("Evaluasi Model",
    uiOutput("split_text"),
    card(card_header("1. Akurasi prediksi rating (semakin kecil semakin baik)"),
         layout_columns(col_widths = c(7, 5), plotlyOutput("p_err", height = 330), tableOutput("tbl_err"))),
    card(card_header("2. Kualitas ranking Top-10 (semakin besar semakin baik)"),
         p(sprintf("Buku di data uji dengan rating ≥ %d dianggap relevan. Untuk tiap user, model meranking semua buku yang belum dibaca lalu dihitung berapa buku relevan yang masuk 10 teratas.",
                   met$params$relevant_min_rating), class = "text-muted small"),
         plotlyOutput("p_rank", height = 360)),
    card(card_header("3. Kurva pelatihan Matrix Factorization"), plotlyOutput("p_curve", height = 320)),
    card(card_header("Interpretasi"), uiOutput("interpretasi"))
  )
)

# =====================================================================
# SERVER
# =====================================================================
server <- function(input, output, session) {

  # ------------------------------------------------ RINGKASAN
  output$funnel <- renderPlotly({
    stages <- c("Rating mentah", "Cocok dgn katalog buku", "Rating eksplisit", "Setelah gabung edisi",
                sprintf("k-core (≥%d per user & buku)", rep$min_ratings_per_user))
    vals <- c(rep$ratings_raw, rep$ratings_matched_to_books, rep$ratings_explicit,
              rep$cf_ratings_after_edition_merge, rep$cf_ratings)
    plot_ly(type = "funnel", y = stages, x = vals, marker = list(color = ACCENT),
            textinfo = "value+percent initial") |> layout(margin = list(l = 10, t = 10))
  })
  output$cleaning_text <- renderUI(HTML(sprintf("<ul>
    <li><b>Users</b> — usia &lt; 5 atau &gt; 100 tahun dianggap tidak valid (%s nilai → kosong); lokasi dipecah jadi kota, provinsi, negara.</li>
    <li><b>Books</b> — %s baris dibaca utuh (tanda kutip ter-escape ditangani); entitas HTML dibersihkan; tahun terbit di luar 1900–2005 → kosong (%s buku).</li>
    <li><b>Ratings</b> — hanya ISBN yang ada di katalog; rating <b>0 = implisit</b>, sehingga model CF memakai rating <b>1–10</b>.</li>
    <li><b>Gabung edisi</b> — %s ISBN dengan judul + penulis sama digabung ke satu ISBN kanonik.</li>
    <li><b>k-core filter</b> — user dan buku dengan &lt; %d rating dibuang berulang sampai stabil.</li></ul>",
    fmt(rep$age_outlier_set_nan), fmt(rep$books_raw), fmt(rep$year_invalid_set_nan),
    fmt(rep$books_merged_into_other_edition), rep$min_ratings_per_user)))
  output$model_table <- renderTable(data.frame(
    Model = c("Baseline (bias)", "Item-based KNN", "Matrix Factorization", "Populer"),
    Ide = c("rata-rata global + kecenderungan user + kualitas buku",
            "buku mirip bila dirating serupa oleh user yang sama (cosine + shrinkage)",
            "user & buku dipetakan ke 16 faktor laten (Funk-SVD, SGD)",
            "pembanding: buku dengan rating terbanyak")), striped = TRUE, width = "100%")

  # ------------------------------------------------ EDA
  output$p_rating_all <- renderPlotly({
    r <- eda$rating_dist
    plot_ly(r, x = ~rating, y = ~n, type = "bar",
            marker = list(color = ifelse(r$rating == 0, LIGHT, ACCENT))) |>
      layout(title = "Distribusi semua rating (abu = implisit)", xaxis = list(title = "Rating", dtick = 1),
             yaxis = list(title = "Jumlah"))
  })
  output$p_rating_ex <- renderPlotly({
    r <- eda$rating_dist |> filter(rating > 0)
    plot_ly(r, x = ~rating, y = ~n, type = "bar", marker = list(color = ACCENT)) |>
      layout(title = "Rating eksplisit saja", xaxis = list(title = "Rating", dtick = 1), yaxis = list(title = "Jumlah"))
  })
  per_user <- user_counts$n
  per_book <- bstats$n_rating
  output$p_per_user <- renderPlotly(
    plot_ly(x = per_user, type = "histogram", nbinsx = 60, marker = list(color = ACCENT)) |>
      layout(title = "Rating per user (data CF, skala log)", xaxis = list(title = "Rating per user"),
             yaxis = list(title = "Jumlah user", type = "log")))
  output$p_per_book <- renderPlotly(
    plot_ly(x = per_book, type = "histogram", nbinsx = 60, marker = list(color = ACCENT)) |>
      layout(title = "Rating per buku (data CF, skala log)", xaxis = list(title = "Rating per buku"),
             yaxis = list(title = "Jumlah buku", type = "log")))
  output$longtail_text <- renderText(sprintf(
    "Distribusi long-tail: median %d rating per user dan %d per buku — inilah sebab matriks sangat jarang (densitas %.3f%%).",
    as.integer(median(per_user)), as.integer(median(per_book)), rep$cf_density_pct))

  top_books <- reactive(bstats |> filter(n_rating >= input$min_n))
  output$p_top_count <- renderPlotly({
    t <- top_books() |> slice_max(n_rating, n = 15, with_ties = FALSE) |> arrange(n_rating)
    bar_h(t, "n_rating", "title", "15 buku paling banyak dirating", "Jumlah rating")
  })
  output$p_top_avg <- renderPlotly({
    t <- top_books() |> slice_max(avg_rating, n = 15, with_ties = FALSE) |> arrange(avg_rating)
    bar_h(t, "avg_rating", "title", sprintf("15 buku rating tertinggi (≥ %d rating)", input$min_n),
          "Rata-rata rating") |> layout(xaxis = list(range = c(0, 10)))
  })
  output$p_authors <- renderPlotly({
    a <- bstats |> group_by(author) |> summarise(n = sum(n_rating)) |> slice_max(n, n = 15) |> arrange(n)
    bar_h(a, "n", "author", "Penulis paling banyak dirating", "Jumlah rating")
  })
  output$p_year <- renderPlotly(
    plot_ly(eda$year, x = ~year, y = ~n, type = "scatter", mode = "lines", fill = "tozeroy",
            line = list(color = ACCENT)) |>
      layout(title = "Jumlah buku per tahun terbit (katalog, ≥1950)", xaxis = list(title = "Tahun"),
             yaxis = list(title = "Jumlah buku")))

  output$p_age <- renderPlotly(
    plot_ly(eda$age, x = ~age, y = ~n, type = "bar", marker = list(color = ACCENT)) |>
      layout(title = "Distribusi usia user", xaxis = list(title = "Usia"), yaxis = list(title = "Jumlah user")))
  output$p_country <- renderPlotly({
    c <- eda$country |> arrange(n)
    bar_h(c, "n", "country", "15 negara asal user terbanyak", "Jumlah user")
  })
  output$age_text <- renderText(sprintf(
    "%.0f%% user tidak mengisi usia (atau usianya tidak valid). Median usia %.0f tahun.",
    eda$users_sum$missing_age_pct, eda$users_sum$median_age))
  output$p_age_group <- renderPlotly({
    lv <- c("<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+")
    g <- eda$age_group |> mutate(age_group = factor(age_group, levels = lv)) |> arrange(age_group)
    plot_ly(g, x = ~age_group, y = ~mean, type = "bar", marker = list(color = ACCENT),
            text = ~sprintf("%.2f", mean), textposition = "outside",
            hovertext = ~paste("Jumlah rating:", fmt(n))) |>
      layout(title = "Rata-rata rating eksplisit per kelompok usia", xaxis = list(title = "Kelompok usia"),
             yaxis = list(title = "Rata-rata rating", range = c(0, 10)))
  })

  # ------------------------------------------------ REKOMENDASI: buku mirip
  book_choices <- setNames(bstats$isbn, bstats$label)
  hp <- bstats$isbn[startsWith(bstats$title, "Harry Potter and the Sorcerer")][1]
  updateSelectizeInput(session, "book", choices = book_choices, selected = hp, server = TRUE)
  updateSelectizeInput(session, "picks", choices = book_choices, server = TRUE)
  updateSelectizeInput(session, "uid",
                       choices = setNames(user_counts$user_id, sprintf("%d  (%d rating)", user_counts$user_id, user_counts$n)),
                       selected = user_counts$user_id[11], server = TRUE)

  output$book_info <- renderUI({
    req(input$book)
    b <- bstats |> filter(isbn == input$book)
    layout_columns(
      value_box2("Penulis", b$author), value_box2("Tahun", ifelse(is.na(b$year), "-", b$year)),
      value_box2("Rata-rata rating", sprintf("%.2f", b$avg_rating)), value_box2("Jumlah rating", b$n_rating))
  })
  output$tbl_similar <- renderDT({
    req(input$book)
    res <- nbr |> filter(isbn == input$book) |> arrange(rank) |> head(input$n_sim) |>
      transmute(isbn = neighbor_isbn, Kemiripan = similarity)
    validate(need(nrow(res) > 0, "Buku ini belum punya cukup pembaca bersama untuk dicari kemiripannya."))
    book_table(res, "Kemiripan")
  })

  # ------------------------------------------------ REKOMENDASI: user yang ada
  hist <- reactive({
    req(input$uid)
    ratings |> filter(user_id == as.integer(input$uid)) |> arrange(desc(rating))
  })
  output$user_info <- renderUI({
    u <- users |> filter(user_id == as.integer(input$uid))
    h <- hist()
    ctry <- if (nrow(u) == 0 || is.na(u$country) || u$country == "") "-" else tools::toTitleCase(u$country)
    layout_columns(
      value_box2("Lokasi", ctry), value_box2("Usia", ifelse(nrow(u) == 0 || is.na(u$age), "-", u$age)),
      value_box2("Jumlah rating", nrow(h)), value_box2("Rata-rata rating user", sprintf("%.2f", mean(h$rating))))
  })
  output$rec_title <- renderText(paste("Rekomendasi —", input$model))
  output$tbl_user <- renderDT({
    h <- hist(); n <- input$n_user
    if (input$model == "Matrix Factorization") {
      u <- mf_upos[[as.character(input$uid)]]
      pred <- mf$mu + mf$bu[u] + mf$bi + as.vector(mf$Q %*% mf$P[u, ])
      rec <- data.frame(isbn = mf$isbns, `Prediksi rating` = pmin(pmax(pred, 1), 10), check.names = FALSE) |>
        filter(!isbn %in% h$isbn) |> slice_max(`Prediksi rating`, n = n, with_ties = FALSE)
      book_table(rec, "Prediksi rating")
    } else if (input$model == "Item-based KNN") {
      rec <- knn_recommend(h |> select(isbn, rating), n)
      validate(need(!is.null(rec), "Tidak ada kandidat rekomendasi untuk user ini dengan model tersebut."))
      book_table(rec, c("Skor", "Karena Anda menyukai"))
    } else {
      rec <- bstats |> filter(!isbn %in% h$isbn) |> head(n) |> select(isbn)
      book_table(rec)
    }
  })
  output$tbl_hist <- renderDT({
    h <- hist() |> head(15) |> left_join(bstats |> select(isbn, title, author), by = "isbn") |>
      select(Judul = title, Penulis = author, Rating = rating)
    datatable(h, options = list(dom = "t", pageLength = 15, ordering = FALSE), class = "compact stripe")
  })

  # ------------------------------------------------ REKOMENDASI: user baru
  output$pick_sliders <- renderUI({
    req(input$picks)
    sl <- lapply(input$picks, function(i) {
      lbl <- substr(bstats$title[bstats$isbn == i], 1, 45)
      sliderInput(paste0("r_", i), lbl, 1, 10, isolate(input[[paste0("r_", i)]]) %||% 8)
    })
    do.call(layout_columns, c(sl, list(col_widths = 3)))
  })
  new_rec <- reactive({
    n <- input$n_new
    if (length(input$picks) == 0)
      return(list(tbl = book_table(bstats |> head(n) |> select(isbn)), note = NULL))
    g <- data.frame(isbn = input$picks,
                    rating = vapply(input$picks, function(i) input[[paste0("r_", i)]] %||% 8, numeric(1)))
    rec <- knn_recommend(g, n)
    if (is.null(rec))
      return(list(tbl = book_table(bstats |> filter(!isbn %in% g$isbn) |> head(n) |> select(isbn)),
                  note = "Belum ada buku yang mirip dengan pilihan Anda; menampilkan buku populer."))
    list(tbl = book_table(rec, c("Skor", "Karena Anda menyukai")), note = NULL)
  })
  output$new_note <- renderUI({ nt <- new_rec()$note; if (!is.null(nt)) div(class = "alert alert-info", nt) })
  output$tbl_new <- renderDT(new_rec()$tbl)

  # ------------------------------------------------ EVALUASI
  res <- do.call(rbind, lapply(names(met$results), function(k) {
    r <- met$results[[k]]
    data.frame(Model = ifelse(k == "Global mean", "Global mean / Populer", k),
               RMSE = r$rmse, MAE = r$mae, `Precision@10` = r$`precision@10`,
               `Recall@10` = r$`recall@10`, `HitRate@10` = r$`hitrate@10`, check.names = FALSE)
  }))
  output$split_text <- renderUI(p(HTML(sprintf(
    "Data dibagi per user: <b>%d%%</b> rating tiap user disisihkan sebagai data uji (%s latih / %s uji, seed %d).",
    as.integer(met$split$test_frac * 100), fmt(met$split$train), fmt(met$split$test), met$split$seed))))
  output$p_err <- renderPlotly(
    plot_ly(res, y = ~Model, x = ~RMSE, type = "bar", orientation = "h", name = "RMSE",
            marker = list(color = ACCENT), text = ~sprintf("%.3f", RMSE), textposition = "auto") |>
      add_trace(x = ~MAE, name = "MAE", marker = list(color = "#9DB7E6"), text = ~sprintf("%.3f", MAE)) |>
      layout(barmode = "group", yaxis = list(title = ""), xaxis = list(title = "Nilai")))
  output$tbl_err <- renderTable(res |> select(Model, RMSE, MAE), digits = 4, striped = TRUE)
  output$p_rank <- renderPlotly({
    cols <- c("#B8C4D6", "#7FA3E0", ACCENT, "#1B3F80")
    p <- plot_ly()
    for (i in seq_len(nrow(res)))
      p <- add_trace(p, type = "bar", name = res$Model[i], marker = list(color = cols[i]),
                     x = c("Precision@10", "Recall@10", "HitRate@10"),
                     y = unlist(res[i, c("Precision@10", "Recall@10", "HitRate@10")]))
    p |> layout(barmode = "group", yaxis = list(title = "Nilai"))
  })
  output$p_curve <- renderPlotly({
    cv <- met$mf_curve
    plot_ly(cv, x = ~epoch) |>
      add_trace(y = ~train_rmse, name = "Latih", type = "scatter", mode = "lines+markers", line = list(color = ACCENT)) |>
      add_trace(y = ~test_rmse, name = "Uji", type = "scatter", mode = "lines+markers", line = list(color = "#E07A3F")) |>
      layout(xaxis = list(title = "Epoch"), yaxis = list(title = "RMSE"))
  })
  output$interpretasi <- renderUI({
    best_rmse <- res$Model[which.min(res$RMSE)]
    best_rank <- res$Model[which.max(res$`HitRate@10`)]
    best_ep <- met$mf_curve$epoch[which.min(met$mf_curve$test_rmse)]
    HTML(sprintf("<ul>
      <li><b>Prediksi rating:</b> %s memberi RMSE terendah (%.3f). Matrix Factorization hampir menyamai baseline, artinya pada data sejarang ini (densitas %.3f%%) sebagian besar sinyal yang bisa dipelajari adalah <i>bias</i> — ada user yang royal memberi nilai, ada buku yang memang disukai banyak orang.</li>
      <li><b>Ranking:</b> %s memiliki HitRate@10 tertinggi. Rekomendasi buku populer menjadi pembanding yang sulit dikalahkan pada data Book-Crossing. Di antara model personal, <b>Item-based KNN</b> unggul dalam ranking dan mudah dijelaskan (\"karena Anda menyukai X\").</li>
      <li><b>Kurva MF:</b> error data uji berhenti turun di sekitar epoch ke-%d; regularisasi (λ = %s) dan jumlah faktor (%d) dipilih lewat grid search untuk menahan overfitting.</li>
      <li><b>Saran pengembangan:</b> memanfaatkan rating implisit (ALS implisit / BPR), model hybrid CF + konten (penulis, penerbit), dan memperketat filter k-core bila mengutamakan akurasi.</li></ul>",
      best_rmse, min(res$RMSE), rep$cf_density_pct, best_rank, best_ep,
      met$params$mf$reg, met$params$mf$factors))
  })
}

shinyApp(ui, server)
