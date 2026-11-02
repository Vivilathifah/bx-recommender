# =====================================================================
# Dashboard Interaktif Sistem Rekomendasi Buku (Collaborative Filtering)
# Book-Crossing dataset — R Shiny
# Jalankan:  shiny::runApp()   (butuh file data/app_data.rds)
# =====================================================================
library(shiny)
library(bslib)
library(dplyr)
library(plotly)
library(DT)

`%||%` <- function(a, b) if (is.null(a)) b else a
click <- function(src) suppressWarnings(event_data("plotly_click", source = src))

d       <- readRDS("data/app_data.rds")
fact    <- d$fact        # 1 baris = 1 rating (semua rating, termasuk implisit)
books   <- d$books       # katalog buku yang pernah dirating
bstats  <- d$bstats      # buku dalam matriks CF
ratings <- d$ratings     # rating eksplisit dalam matriks CF
users   <- d$users
nbr     <- d$neighbors
mf      <- d$mf
rep     <- d$report
met     <- d$metrics

# ---- palet warna
ACCENT <- "#6366F1"; DARK <- "#312E81"; LIGHT <- "#CBD5E1"; ORANGE <- "#F59E0B"
PINK <- "#EC4899"; TEAL <- "#14B8A6"; VIOLET <- "#8B5CF6"
GRAD  <- list(c(0, "#818CF8"), c(0.5, "#A855F7"), c(1, "#EC4899"))   # indigo -> ungu -> pink
GRAD2 <- list(c(0, "#5EEAD4"), c(0.5, "#14B8A6"), c(1, "#4F46E5"))   # mint -> teal -> indigo
MAPSC <- list(c(0, "#312E81"), c(0.35, "#7C3AED"), c(0.65, "#EC4899"), c(1, "#FDE047"))
rating_col <- function(r) ifelse(r == 0, LIGHT,
  colorRampPalette(c("#F43F5E", "#F97316", "#FACC15", "#4ADE80", "#10B981"))(10)[pmax(r, 1)])
AGE_COL <- setNames(c("#38BDF8", "#6366F1", "#8B5CF6", "#A855F7", "#D946EF", "#EC4899", "#F43F5E"),
                    c("<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"))
MODEL_COL <- c("Global mean / Populer" = "#F59E0B", "Baseline (bias)" = "#94A3B8",
               "Item-based KNN" = "#14B8A6", "Matrix Factorization (SVD)" = "#8B5CF6")
grad_bar <- function(v, scale = GRAD) list(color = v, colorscale = scale, cmin = min(v, na.rm = TRUE) * 0.9,
                                          cmax = max(v, na.rm = TRUE), line = list(width = 0))
AGE_LV <- levels(fact$age_group)
fmt  <- function(x) format(round(x), big.mark = ".", decimal.mark = ",", scientific = FALSE)
nice_country <- function(x) ifelse(nchar(x) <= 3, toupper(x), tools::toTitleCase(x))
fmt2 <- function(x) formatC(x, format = "f", digits = 2, decimal.mark = ",")

user_counts <- ratings |> count(user_id, name = "n") |> arrange(desc(n))
mf_upos     <- setNames(seq_along(mf$user_ids), mf$user_ids)
cf_isbn     <- bstats$isbn
book_choices <- setNames(bstats$isbn, bstats$label)
top_countries <- names(sort(table(fact$country), decreasing = TRUE))[1:40]

# --- peta buku: PCA dari faktor laten MF (buku berdekatan = disukai profil pembaca serupa)
pca <- prcomp(mf$Q, center = TRUE, scale. = FALSE)
book_map <- data.frame(isbn = mf$isbns, PC1 = pca$x[, 1], PC2 = pca$x[, 2]) |>
  inner_join(bstats |> select(isbn, title, author, n_rating, avg_rating, label), by = "isbn")

# ---------------------------------------------------------------- helper
plt <- function(p, source = NULL) {
  p <- p |> layout(margin = list(l = 10, r = 10, t = 10, b = 10), font = list(size = 12),
                   paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)") |>
    config(displaylogo = FALSE, modeBarButtonsToRemove = c("lasso2d", "select2d", "autoScale2d"))
  # grafik yang bisa diklik: sembunyikan toolbar agar tidak menutupi bar paling atas
  if (!is.null(source)) p <- event_register(p, "plotly_click") |> config(displayModeBar = FALSE)
  p
}

book_table <- function(df, extra = character(0), selectable = FALSE) {
  out <- df |>
    left_join(bstats |> select(isbn, title, author, year, avg_rating, n_rating), by = "isbn") |>
    select(Judul = title, Penulis = author, Tahun = year, all_of(extra),
           `Rata-rata` = avg_rating, `Jml rating` = n_rating)
  dt <- datatable(out, rownames = seq_len(nrow(out)),
                  selection = if (selectable) "single" else "none",
                  options = list(dom = "t", pageLength = 25, ordering = FALSE),
                  class = "compact stripe hover")
  dt <- formatRound(dt, "Rata-rata", 2)
  if ("Kemiripan" %in% extra) dt <- formatRound(dt, "Kemiripan", 4)
  if ("Prediksi rating" %in% extra) dt <- formatRound(dt, "Prediksi rating", 2)
  if ("Skor" %in% extra) dt <- dt |> formatRound("Skor", 3) |>
    formatStyle("Skor", background = styleColorBar(c(0, 1), "#DDD6FE"),
                backgroundSize = "95% 70%", backgroundRepeat = "no-repeat", backgroundPosition = "center")
  dt
}

knn_recommend <- function(hist, n) {
  mean_r <- mean(hist$rating)
  m <- nbr |> filter(isbn %in% hist$isbn) |> inner_join(hist, by = "isbn") |>
    filter(!neighbor_isbn %in% hist$isbn) |>
    mutate(w = similarity * (rating - mean_r + 1)) |> filter(w > 0)
  if (nrow(m) == 0) return(NULL)
  why <- m |> group_by(neighbor_isbn) |> slice_max(w, n = 1, with_ties = FALSE) |> ungroup() |>
    left_join(bstats |> select(isbn, title), by = "isbn") |> select(neighbor_isbn, why = title)
  m |> group_by(neighbor_isbn) |> summarise(score = sum(w), .groups = "drop") |>
    slice_max(score, n = n, with_ties = FALSE) |> left_join(why, by = "neighbor_isbn") |>
    transmute(isbn = neighbor_isbn, Skor = score / max(score), `Karena Anda menyukai` = substr(why, 1, 60))
}

kpi <- function(id, title, icon_chr, k)
  value_box(title = title, value = textOutput(id, inline = TRUE), class = paste0("kpi kpi-", k),
            showcase = span(icon_chr, class = "kpi-icon"),
            theme = value_box_theme(bg = c("#6366F1", "#14B8A6", "#F59E0B", "#EC4899")[k], fg = "white"), height = "112px")
info_box <- function(t, v, k)
  value_box(title = t, value = v, class = paste0("info info-", k), height = "92px",
            theme = value_box_theme(bg = c("#EEF2FF", "#F0FDFA", "#FFFBEB", "#FDF2F8")[k],
                                    fg = c("#3730A3", "#0F766E", "#B45309", "#BE185D")[k]))

# garis rata-rata rating di sumbu kanan; dilewati bila semua NA (mis. filter rating implisit)
add_avg_line <- function(p, t, hover, markers = FALSE) {
  if (all(is.na(t$avg))) return(p)
  add_trace(p, data = t, x = t[[1]], y = ~avg, type = "scatter", mode = if (markers) "lines+markers" else "lines",
            fill = "none", name = "Rata-rata rating", yaxis = "y2", inherit = FALSE,
            line = list(color = if (markers) PINK else ORANGE, width = 2.5, dash = if (markers) "solid" else "dot"),
            marker = if (markers) list(color = PINK, size = 9, line = list(color = "white", width = 2)) else NULL,
            hovertemplate = hover)
}

# ---- sampul buku: Open Library (per ISBN) -> cadangan Amazon -> sampul pengganti berwarna
PH_GRAD <- c("linear-gradient(150deg,#6366F1,#8B5CF6)", "linear-gradient(150deg,#14B8A6,#0EA5E9)",
             "linear-gradient(150deg,#F59E0B,#F97316)", "linear-gradient(150deg,#EC4899,#F43F5E)",
             "linear-gradient(150deg,#8B5CF6,#EC4899)", "linear-gradient(150deg,#0EA5E9,#6366F1)")
ph_grad <- function(isbn) PH_GRAD[(sum(utf8ToInt(isbn)) %% length(PH_GRAD)) + 1]
ol_cover <- function(isbn, size = "M") sprintf("https://covers.openlibrary.org/b/isbn/%s-%s.jpg?default=false", isbn, size)
cover <- function(isbn, title, author = NULL, size = "M", cls = "")
  div(class = paste("cover", cls), style = paste0("background:", ph_grad(isbn)),
      div(class = "cover-ph", div(class = "ph-title", substr(title, 1, 70)),
          if (!is.null(author)) div(class = "ph-author", author)),
      tags$img(src = ol_cover(isbn, size), `data-isbn` = isbn, loading = "lazy", alt = "",
               onerror = "bxCoverFail(this)", onload = "bxCoverLoad(this)"))

stars <- function(x) if (is.na(x)) "" else { k <- round(x / 2); paste0(strrep("\u2605", k), strrep("\u2606", 5 - k)) }
click_js <- function(isbn) sprintf("Shiny.setInputValue('card_click', '%s', {priority: 'event'})", isbn)

book_card <- function(isbn, title, author, avg, n, badge = NULL, reason = NULL, size = "M")
  div(class = "book-card", onclick = click_js(isbn), title = title,
      div(class = "cover-wrap", cover(isbn, title, author, size),
          if (!is.null(badge)) span(class = "badge-score", badge)),
      div(class = "bc-body",
          div(class = "bc-title", title), div(class = "bc-author", author),
          div(class = "bc-meta", title = sprintf("rata-rata %s dari %s rating", if (is.na(avg)) "-" else fmt2(avg), n),
              span(class = "stars", stars(avg)), span(class = "ms-1 fw-semibold", if (is.na(avg)) "-" else fmt2(avg)),
              span(class = "text-muted", sprintf(" (%s)", n))),
          if (!is.null(reason) && !is.na(reason)) div(class = "bc-reason", "Karena Anda menyukai ", tags$b(reason))))

# kartu untuk daftar ISBN dalam matriks CF; badge/reason: vektor sepanjang df
card_grid <- function(isbns, badge = NULL, reason = NULL) {
  i <- match(isbns, bstats$isbn)
  div(class = "book-grid", lapply(seq_along(isbns), function(k)
    book_card(isbns[k], bstats$title[i[k]], bstats$author[i[k]], bstats$avg_rating[i[k]], bstats$n_rating[i[k]],
              badge = badge[k], reason = reason[k])))
}

view_switch <- function(card_ui, table_ui)
  tagList(conditionalPanel("input.view != 'table'", card_ui), conditionalPanel("input.view == 'table'", table_ui))

hint <- function(txt) span(txt, class = "text-muted small fw-normal ms-2")

# =====================================================================
# UI
# =====================================================================
ui <- page_navbar(
  id = "nav",
  title = "📚 BX Book Recommender",
  bg = DARK, inverse = TRUE,
  fillable = FALSE,
  theme = bs_theme(version = 5, primary = ACCENT, secondary = PINK, success = TEAL,
                   bg = "#F5F6FC", fg = "#1E1B4B",
                   base_font = font_collection(font_google("Plus Jakarta Sans", local = FALSE),
                                               "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"),
                   heading_font = font_collection(font_google("Plus Jakarta Sans", local = FALSE),
                                                  "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif")),
  header = tagList(tags$script(HTML("
    function bxCoverFail(img) {
      if (!img.dataset.stage) {            // 1) Open Library gagal -> coba Amazon
        img.dataset.stage = 'amz';
        img.src = 'https://images-na.ssl-images-amazon.com/images/P/' + img.dataset.isbn + '.01.MZZZZZZZ.jpg';
      } else { img.style.display = 'none'; }   // 2) semua gagal -> sampul pengganti
    }
    function bxCoverLoad(img) {             // Amazon memberi gambar 1x1 bila sampul tidak ada
      if (img.naturalWidth < 10) bxCoverFail(img); else img.classList.add('ok');
    }")),
  tags$style(HTML("
    /* latar: gradasi lembut indigo-pink-teal + pola titik halus */
    body {
      background-color: #F4F2FF !important;
      background-image:
        radial-gradient(circle at 8% 12%, rgba(99,102,241,.38), transparent 40%),
        radial-gradient(circle at 92% 15%, rgba(236,72,153,.32), transparent 42%),
        radial-gradient(circle at 78% 80%, rgba(20,184,166,.30), transparent 45%),
        radial-gradient(circle at 18% 88%, rgba(245,158,11,.24), transparent 42%),
        radial-gradient(rgba(79,70,229,.16) 1.2px, transparent 1.6px) !important;
      background-size: auto, auto, auto, auto, 22px 22px !important;
      background-attachment: fixed !important;
    }
    .bslib-page-navbar > .container-fluid, .bslib-page-navbar .tab-content, .tab-pane {background: transparent !important;}
    .bslib-sidebar-layout {--_main-bg: transparent; --_sidebar-bg: transparent;
                           background: transparent !important; box-shadow: none !important; border: 0 !important;}
    .bslib-sidebar-layout > .main {background: transparent !important;}
    .card {background: rgba(255,255,255,.88) !important; backdrop-filter: blur(10px);}
    .card-header {background: transparent !important;}
    .navbar {background: linear-gradient(90deg, #312E81 0%, #5B21B6 55%, #9D174D 100%) !important;
             box-shadow: 0 4px 20px rgba(49,46,129,.25);}
    .navbar .navbar-brand {font-weight: 800; letter-spacing: .2px;}
    .navbar .nav-link {color: rgba(255,255,255,.75) !important; font-weight: 500;}
    .navbar .navbar-nav .nav-link.active, .navbar .navbar-nav .nav-link.active:focus
      {color: #fff !important; border-bottom: 3px solid #F9A8D4; background: rgba(255,255,255,.08); border-radius: 6px 6px 0 0;}
    .card {border: 0; border-radius: 16px; box-shadow: 0 4px 22px rgba(49,46,129,.07);}
    .card-header {font-weight: 700; border-bottom: 1px solid #EEF0FA;
                  border-radius: 16px 16px 0 0 !important;}
    .card-header::before {content: ''; display: inline-block; width: 10px; height: 10px; margin-right: 8px;
                          border-radius: 3px; background: linear-gradient(135deg, #6366F1, #EC4899);}
    .bslib-sidebar-layout > .sidebar {background: rgba(255,255,255,.72) !important; backdrop-filter: blur(12px);
                                      border-radius: 16px; border: 1px solid rgba(255,255,255,.8) !important;
                                      box-shadow: 0 4px 22px rgba(49,46,129,.10); margin: 4px;}
    .sidebar-title {color: #4338CA; font-weight: 800;}
    .card-header:has(.nav)::before {display: none;}
    .bslib-value-box {border: 0; border-radius: 16px;}
    .bslib-value-box .value-box-value {font-size: 1.7rem; font-weight: 800;}
    .kpi {box-shadow: 0 8px 24px rgba(99,102,241,.25);}
    .kpi-1 {background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;}
    .kpi-2 {background: linear-gradient(135deg, #14B8A6, #0EA5E9) !important;}
    .kpi-3 {background: linear-gradient(135deg, #F59E0B, #F97316) !important;}
    .kpi-4 {background: linear-gradient(135deg, #EC4899, #F43F5E) !important;}
    .kpi .value-box-title {opacity: .9; font-weight: 600;}
    .kpi-icon {font-size: 2.2rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,.2));}
    .info {box-shadow: none; border: 1px solid rgba(99,102,241,.12);}
    .nav-underline .nav-link.active, .nav-tabs .nav-link.active {color: #4F46E5 !important; font-weight: 700;}
    .btn-primary {background: linear-gradient(135deg, #6366F1, #8B5CF6); border: 0;}
    .btn-outline-primary {border-color: #A5B4FC; color: #4F46E5;}
    table.dataTable tbody tr.selected > * {box-shadow: inset 0 0 0 9999px rgba(99,102,241,.18) !important; color: #1E1B4B !important;}
    table.dataTable thead th {color: #4338CA;}
    .modal-content {border-radius: 16px; border: 0;}
    .modal-header {background: linear-gradient(90deg, #EEF2FF, #FDF2F8);}
    /* ---------- sampul & kartu buku */
    .cover {position: relative; width: 100%; aspect-ratio: 2 / 3; border-radius: 10px; overflow: hidden;
            box-shadow: 0 6px 16px rgba(30,27,75,.22);}
    .cover img {position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0;
                transition: opacity .4s;}
    .cover img.ok {opacity: 1;}
    .cover-ph {position: absolute; inset: 0; padding: 38px 10px 12px; color: #fff; display: flex; flex-direction: column;
               justify-content: space-between; text-shadow: 0 1px 3px rgba(0,0,0,.25);}
    .cover-ph::before {content: ''; position: absolute; left: 7px; top: 0; bottom: 0; width: 2px;
                       background: rgba(255,255,255,.35);}
    .ph-title {font-weight: 800; font-size: .82rem; line-height: 1.2; padding-left: 6px;
               display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical; overflow: hidden;}
    .ph-author {font-size: .7rem; opacity: .9; padding-left: 6px;}
    .book-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 22px 18px;
                padding: 4px 2px 8px;}
    .book-card {cursor: pointer; border-radius: 14px; padding: 10px; background: rgba(255,255,255,.75);
                border: 1px solid rgba(99,102,241,.10); transition: transform .18s, box-shadow .18s;}
    .book-card:hover {transform: translateY(-5px); box-shadow: 0 14px 30px rgba(79,70,229,.20);
                      border-color: rgba(236,72,153,.35);}
    .cover-wrap {position: relative;}
    .badge-score {position: absolute; top: 8px; right: -6px; background: linear-gradient(135deg,#EC4899,#8B5CF6);
                  color: #fff; font-weight: 700; font-size: .72rem; padding: 3px 9px; border-radius: 10px 0 0 10px;
                  box-shadow: 0 3px 8px rgba(236,72,153,.4);}
    .bc-body {padding-top: 10px;}
    .bc-title {font-weight: 700; font-size: .88rem; line-height: 1.25; color: #1E1B4B;
               display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
    .bc-author {font-size: .78rem; color: #6B7280; margin-top: 2px; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis;}
    .bc-meta {font-size: .74rem; color: #374151; margin-top: 4px; white-space: nowrap; overflow: hidden;
              text-overflow: ellipsis;}
    .stars {color: #F59E0B; letter-spacing: 1px;}
    .bc-reason {font-size: .72rem; color: #7C3AED; background: #F5F3FF; border-radius: 8px; padding: 4px 7px;
                margin-top: 6px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                overflow: hidden;}
    .shelf {display: flex; gap: 16px; overflow-x: auto; padding: 6px 4px 14px; scroll-snap-type: x mandatory;}
    .shelf .book-card {flex: 0 0 140px; scroll-snap-align: start;}
    .shelf-sm .book-card {flex: 0 0 110px; padding: 6px;}
    .shelf-sm .bc-title {font-size: .76rem;}
    .hero {display: flex; gap: 26px; align-items: flex-start; padding: 18px; border-radius: 18px;
           background: linear-gradient(120deg, rgba(238,242,255,.9), rgba(253,242,248,.9));
           border: 1px solid rgba(99,102,241,.12); margin-bottom: 18px;}
    .hero .cover {width: 150px; flex: 0 0 150px;}
    .hero h3 {font-weight: 800; color: #1E1B4B; margin-bottom: 4px;}
    .hero-author {color: #6D28D9; font-weight: 600; margin-bottom: 10px;}
    .hero-stat {display: inline-block; background: #fff; border-radius: 12px; padding: 8px 14px; margin: 4px 8px 4px 0;
                box-shadow: 0 2px 8px rgba(49,46,129,.08);}
    .hero-stat b {display: block; font-size: 1.25rem; color: #4338CA;}
    .hero-stat span {font-size: .75rem; color: #6B7280;}
    .hero .stars {font-size: 1.1rem;}
    .view-toggle .btn {border-radius: 999px !important;}
    @media (max-width: 576px) {.hero {flex-direction: column;} .book-grid {grid-template-columns: repeat(2, 1fr);}}

    .filter-chip {display:inline-block; background: linear-gradient(135deg, #EEF2FF, #FCE7F3); color:#4338CA;
                  border:1px solid #DDD6FE; border-radius:14px; padding:2px 12px; margin:2px;
                  font-size:.8rem; font-weight:600;}"))),

  # ============================================== DASHBOARD
  nav_panel("Dashboard", value = "dash",
    layout_sidebar(
      sidebar = sidebar(
        width = 290, title = "Filter",
        radioButtons("f_type", "Jenis rating",
                     c("Eksplisit (1–10)" = "ex", "Implisit (0)" = "im", "Semua" = "all"), selected = "ex"),
        selectizeInput("f_country", "Negara (kosong = semua)", choices = top_countries, multiple = TRUE,
                       options = list(placeholder = "Klik bar negara atau pilih di sini")),
        checkboxGroupInput("f_age", "Kelompok usia", AGE_LV, selected = AGE_LV, inline = TRUE),
        checkboxInput("f_age_na", "Sertakan usia tidak diketahui", TRUE),
        sliderInput("f_year", "Tahun terbit", 1950, 2005, c(1950, 2005), sep = ""),
        checkboxInput("f_year_na", "Sertakan tahun tidak diketahui / < 1950", TRUE),
        sliderInput("f_min", "Min. rating per buku (untuk peringkat)", 5, 200, 30, step = 5),
        actionButton("f_reset", "Reset filter", icon = icon("rotate-left"), class = "btn-outline-primary btn-sm")
      ),
      uiOutput("active_filters"),
      layout_columns(fill = FALSE,
        kpi("k_rating", "Rating", "⭐", 1), kpi("k_user", "User aktif", "👤", 2),
        kpi("k_book", "Buku", "📖", 3), kpi("k_avg", "Rata-rata rating eksplisit", "📊", 4)),
      layout_columns(col_widths = c(6, 6),
        card(full_screen = TRUE, card_header("Distribusi rating"), plotlyOutput("d_rating", height = 280)),
        card(full_screen = TRUE, card_header("Rating per kelompok usia", hint("klik bar untuk memfilter")),
             plotlyOutput("d_age", height = 280))),
      layout_columns(col_widths = c(5, 7),
        card(full_screen = TRUE, card_header("Negara teratas", hint("klik: filter / lepas")),
             plotlyOutput("d_country", height = 380)),
        card(full_screen = TRUE, card_header("Aktivitas per tahun terbit buku"),
             plotlyOutput("d_year", height = 380))),
      card(card_header("Rak buku teratas", hint("sesuai filter · klik sampul → lihat rekomendasinya")),
           uiOutput("d_shelf")),
      layout_columns(col_widths = c(7, 5),
        card(full_screen = TRUE, card_header("Buku teratas", hint("klik bar → lihat rekomendasinya")),
             radioButtons("d_rank_by", NULL, c("Paling banyak dirating" = "n", "Rating tertinggi" = "avg"), inline = TRUE),
             plotlyOutput("d_books", height = 460)),
        card(full_screen = TRUE, card_header("Penulis teratas", hint("klik bar → daftar bukunya")),
             plotlyOutput("d_authors", height = 500))),
      card(full_screen = TRUE, card_header("Tabel buku (sesuai filter)", hint("klik baris → lihat rekomendasinya")),
           DTOutput("d_table", fill = FALSE))
    )
  ),

  # ============================================== REKOMENDASI
  nav_panel("Rekomendasi", value = "rec",
    div(class = "d-flex justify-content-end mb-2 view-toggle",
        radioButtons("view", NULL, c("\U0001F5BC\uFE0F Kartu bersampul" = "card", "\U0001F4CB Tabel & grafik" = "table"),
                     inline = TRUE)),
    navset_card_underline(id = "rec_tabs",
      nav_panel("🔎 Buku mirip", value = "similar",
        layout_columns(col_widths = c(8, 4),
          selectizeInput("book", "Pilih buku", choices = NULL, width = "100%"),
          sliderInput("n_sim", "Jumlah rekomendasi", 5, 20, 10)),
        uiOutput("book_info"),
        view_switch(
          div(h5("Pembaca buku ini juga menyukai", class = "fw-bold mb-3"), uiOutput("cards_similar")),
          layout_columns(col_widths = c(5, 7),
            card(card_header("Skor kemiripan", hint("klik bar untuk menjelajah")), plotlyOutput("sim_plot", height = 420)),
            card(card_header("Buku mirip", hint("klik baris untuk menjelajah")), DTOutput("tbl_similar", fill = FALSE)))),
        uiOutput("trail")
      ),
      nav_panel("👤 Untuk user yang ada", value = "user",
        layout_columns(col_widths = c(4, 2, 4, 2),
          selectizeInput("uid", "User-ID (urut dari yang paling banyak memberi rating)", choices = NULL),
          div(style = "padding-top:32px", actionButton("rand_user", "🎲 User acak", class = "btn-sm btn-outline-primary")),
          radioButtons("model", "Model", c("Matrix Factorization", "Item-based KNN", "Populer"), inline = TRUE),
          sliderInput("n_user", "Jumlah", 5, 20, 10)),
        uiOutput("user_info"),
        view_switch(
          tagList(
            card(card_header("Rak buku user ini", hint("buku yang dirating paling tinggi")), uiOutput("shelf_hist")),
            h5(textOutput("rec_title2", inline = TRUE), class = "fw-bold my-3"), uiOutput("cards_user")),
          layout_columns(col_widths = c(7, 5),
            card(card_header(textOutput("rec_title", inline = TRUE)), DTOutput("tbl_user", fill = FALSE)),
            card(card_header("Profil selera user"), plotlyOutput("user_hist_plot", height = 200), DTOutput("tbl_hist", fill = FALSE))))
      ),
      nav_panel("✨ User baru", value = "new",
        p("Simulasi user baru: pilih buku yang pernah dibaca dan geser rating-nya — rekomendasi langsung berubah."),
        selectizeInput("picks", "Buku yang pernah dibaca (maks. 10)", choices = NULL, multiple = TRUE, width = "100%",
                       options = list(maxItems = 10, placeholder = "Ketik judul, mis. Harry Potter, Da Vinci, Tolkien...")),
        uiOutput("pick_sliders"),
        sliderInput("n_new", "Jumlah rekomendasi", 5, 20, 10),
        uiOutput("new_note"),
        view_switch(uiOutput("cards_new"), DTOutput("tbl_new", fill = FALSE))
      )
    )
  ),

  # ============================================== PETA BUKU
  nav_panel("Peta Buku", value = "map",
    layout_sidebar(
      sidebar = sidebar(width = 290,
        p("Setiap titik adalah buku, diposisikan dari faktor laten model Matrix Factorization (diproyeksikan dengan PCA).
           Buku yang berdekatan cenderung disukai oleh profil pembaca yang serupa.", class = "small"),
        sliderInput("m_min", "Min. jumlah rating", 5, 100, 20, step = 5),
        selectizeInput("m_find", "Sorot buku", choices = NULL,
                       options = list(placeholder = "Cari judul...")),
        radioButtons("m_color", "Warna titik", c("Rata-rata rating" = "avg", "Jumlah rating" = "n")),
        hr(), uiOutput("m_info")
      ),
      card(full_screen = TRUE, card_header("Peta selera buku", hint("zoom, hover, klik titik")),
           plotlyOutput("map_plot", height = 640))
    )
  ),

  # ============================================== EVALUASI
  nav_panel("Evaluasi Model", value = "eval",
    uiOutput("split_text"),
    layout_columns(col_widths = c(6, 6),
      card(card_header("Akurasi prediksi rating", hint("semakin kecil semakin baik")), plotlyOutput("p_err", height = 300)),
      card(card_header("Tabel metrik"), div(class = "small", tableOutput("tbl_err")))),
    layout_columns(col_widths = c(7, 5),
      card(card_header("Kualitas ranking Top-10", hint("semakin besar semakin baik")),
           radioButtons("rank_metric", NULL, c("HitRate@10", "Precision@10", "Recall@10"), inline = TRUE),
           plotlyOutput("p_rank", height = 300)),
      card(card_header("Kurva pelatihan Matrix Factorization"), plotlyOutput("p_curve", height = 340))),
    card(card_header("Interpretasi"), uiOutput("interpretasi"))
  ),

  # ============================================== TENTANG DATA
  nav_panel("Tentang Data", value = "about",
    card(card_header("Alur data: dari mentah sampai matriks CF"), plotlyOutput("funnel", height = 320)),
    layout_columns(
      card(card_header("Langkah cleaning"), uiOutput("cleaning_text")),
      card(card_header("Model yang dibangun"), tableOutput("model_table"))),
    p(class = "small text-muted mt-2", "Sampul buku dimuat dari ",
      tags$a(href = "https://openlibrary.org", target = "_blank", "Open Library Covers API"),
      " (cadangan: gambar Amazon dari dataset Book-Crossing). Bila tidak tersedia, ditampilkan sampul pengganti.")
  ),
  nav_spacer(),
  nav_item(span(class = "text-white-50 small",
                sprintf("CF: %s user × %s buku", fmt(rep$cf_users), fmt(rep$cf_books))))
)

# =====================================================================
# SERVER
# =====================================================================
server <- function(input, output, session) {

  # ------------------------------------------------ data terfilter
  # semua filter KECUALI negara (dipakai grafik negara agar negara lain tetap bisa diklik)
  filt_base <- reactive({
    f <- fact
    f <- switch(input$f_type, ex = f[f$rating > 0, ], im = f[f$rating == 0, ], f)
    keep_age <- f$age_group %in% input$f_age | (input$f_age_na & is.na(f$age_group))
    f <- f[keep_age, ]
    in_rng <- !is.na(f$year) & f$year >= input$f_year[1] & f$year <= input$f_year[2]
    out_rng <- is.na(f$year) | f$year < 1950
    f[in_rng | (input$f_year_na & out_rng), ]
  })
  filt <- reactive({
    f <- filt_base()
    if (length(input$f_country)) f <- f[!is.na(f$country) & f$country %in% input$f_country, ]
    f
  })

  book_agg <- reactive({
    f <- filt()
    f |> group_by(book) |>
      summarise(n = n(), avg = if (any(rating > 0)) mean(rating[rating > 0]) else NA_real_, .groups = "drop") |>
      mutate(isbn = books$isbn[book], title = books$title[book], author = books$author[book],
             year = books$year[book])
  })

  observeEvent(input$f_reset, {
    updateRadioButtons(session, "f_type", selected = "ex")
    updateSelectizeInput(session, "f_country", selected = character(0))
    updateCheckboxGroupInput(session, "f_age", selected = AGE_LV)
    updateCheckboxInput(session, "f_age_na", value = TRUE)
    updateSliderInput(session, "f_year", value = c(1950, 2005))
    updateCheckboxInput(session, "f_year_na", value = TRUE)
  })

  output$active_filters <- renderUI({
    chips <- c(switch(input$f_type, ex = "Rating eksplisit", im = "Rating implisit", all = "Semua rating"),
               if (length(input$f_country)) paste("Negara:", paste(nice_country(input$f_country), collapse = ", ")),
               if (length(input$f_age) < length(AGE_LV)) paste("Usia:", paste(input$f_age, collapse = ", ")),
               if (any(input$f_year != c(1950, 2005)))
                 sprintf("Tahun %d–%d", input$f_year[1], input$f_year[2]))
    div(class = "mb-2", span("Filter aktif:", class = "small text-muted me-1"),
        lapply(chips, function(x) span(x, class = "filter-chip")))
  })

  # ------------------------------------------------ KPI
  output$k_rating <- renderText(fmt(nrow(filt())))
  output$k_user   <- renderText(fmt(length(unique(filt()$user_id))))
  output$k_book   <- renderText(fmt(length(unique(filt()$book))))
  output$k_avg    <- renderText({ r <- filt()$rating; r <- r[r > 0]; if (length(r)) fmt2(mean(r)) else "–" })

  # ------------------------------------------------ grafik dashboard
  output$d_rating <- renderPlotly({
    t <- filt() |> count(rating)
    plot_ly(t, x = ~rating, y = ~n, type = "bar",
            marker = list(color = rating_col(t$rating), line = list(width = 0)),
            hovertemplate = "Rating %{x}<br>%{y:,} rating<extra></extra>") |>
      layout(xaxis = list(title = "Rating", dtick = 1), yaxis = list(title = "Jumlah")) |> plt()
  })

  output$d_age <- renderPlotly({
    t <- filt() |> filter(!is.na(age_group)) |> group_by(age_group) |>
      summarise(n = n(), avg = if (any(rating > 0)) mean(rating[rating > 0]) else NA_real_, .groups = "drop")
    validate(need(nrow(t) > 0, "Tidak ada data usia untuk filter ini."))
    plot_ly(t, x = ~age_group, y = ~n, type = "bar", name = "Jumlah rating", customdata = ~as.character(age_group),
            marker = list(color = unname(AGE_COL[as.character(t$age_group)])), source = "age",
            hovertemplate = "%{x}<br>%{y:,} rating<extra></extra>") |>
      add_avg_line(t, "%{x}<br>rata-rata %{y:.2f}<extra></extra>", markers = TRUE) |>
      layout(xaxis = list(title = "Kelompok usia"), yaxis = list(title = "Jumlah rating"),
             yaxis2 = list(title = "Rata-rata", overlaying = "y", side = "right", range = c(0, 10), showgrid = FALSE),
             legend = list(orientation = "h", y = 1.12)) |> plt("age") |> layout(margin = list(r = 45))
  })
  observeEvent(click("age"), {
    g <- click("age")$customdata
    if (!is.null(g)) {
      sel <- if (identical(input$f_age, g)) AGE_LV else g
      updateCheckboxGroupInput(session, "f_age", selected = sel)
    }
  })

  output$d_country <- renderPlotly({
    t <- filt_base() |> filter(!is.na(country)) |> count(country) |>
      filter(rank(-n, ties.method = "first") <= 15 | country %in% input$f_country) |> arrange(n)
    validate(need(nrow(t) > 0, "Tidak ada data negara untuk filter ini."))
    t$lbl <- nice_country(as.character(t$country))
    sel <- as.character(t$country) %in% input$f_country
    plot_ly(t, x = ~n, y = ~factor(lbl, levels = lbl), type = "bar", orientation = "h",
            customdata = ~as.character(country), source = "country",
            marker = if (any(sel)) list(color = ifelse(sel, PINK, "#C7D2FE")) else grad_bar(t$n, GRAD2),
            hovertemplate = "%{y}<br>%{x:,} rating<extra></extra>") |>
      layout(xaxis = list(title = "Jumlah rating"), yaxis = list(title = "")) |> plt("country")
  })
  observeEvent(click("country"), {
    c <- click("country")$customdata
    if (!is.null(c)) {
      cur <- input$f_country %||% character(0)
      new <- if (c %in% cur) setdiff(cur, c) else c(cur, c)
      updateSelectizeInput(session, "f_country", choices = union(top_countries, new), selected = new)
    }
  })

  output$d_year <- renderPlotly({
    t <- filt() |> filter(!is.na(year), year >= 1950) |> group_by(year) |>
      summarise(n = n(), avg = if (any(rating > 0)) mean(rating[rating > 0]) else NA_real_, .groups = "drop")
    validate(need(nrow(t) > 0, "Tidak ada data tahun untuk filter ini."))
    plot_ly(t, x = ~year, y = ~n, type = "scatter", mode = "lines", fill = "tozeroy", name = "Jumlah rating",
            line = list(color = TEAL, width = 2.5, shape = "spline"), fillcolor = "rgba(20,184,166,.18)",
            hovertemplate = "%{x}<br>%{y:,} rating<extra></extra>") |>
      add_avg_line(t, "%{x}<br>rata-rata %{y:.2f}<extra></extra>") |>
      layout(xaxis = list(title = "Tahun terbit", rangeslider = list(visible = TRUE, thickness = 0.08)),
             yaxis = list(title = "Jumlah rating"),
             yaxis2 = list(title = "Rata-rata", overlaying = "y", side = "right", range = c(0, 10), showgrid = FALSE),
             legend = list(orientation = "h", y = 1.12)) |> plt() |> layout(margin = list(r = 45))
  })

  top_books <- reactive({
    b <- book_agg()
    if (input$d_rank_by == "avg" && input$f_type != "im")
      b |> filter(n >= input$f_min, !is.na(avg)) |> slice_max(avg, n = 15, with_ties = FALSE) |> arrange(avg)
    else
      b |> slice_max(n, n = 15, with_ties = FALSE) |> arrange(n)
  })
  output$d_books <- renderPlotly({
    t <- top_books()
    validate(need(nrow(t) > 0, "Belum ada buku yang memenuhi filter. Coba turunkan 'Min. rating per buku'."))
    use_avg <- input$d_rank_by == "avg" && input$f_type != "im"
    x <- if (use_avg) t$avg else t$n
    lbl <- make.unique(substr(t$title, 1, 55), sep = " ")
    plot_ly(x = x, y = factor(lbl, levels = lbl), type = "bar", orientation = "h", source = "books",
            customdata = t$isbn,
            marker = list(color = x, colorscale = GRAD, cmin = min(x) * 0.9, cmax = max(x), opacity = ifelse(t$isbn %in% cf_isbn, 1, 0.45)),
            text = sprintf("%s · %s rating · rata-rata %s", t$author, fmt(t$n), fmt2(t$avg)),
            textposition = "none", hovertemplate = "<b>%{y}</b><br>%{text}<extra></extra>") |>
      layout(xaxis = list(title = if (use_avg) "Rata-rata rating" else "Jumlah rating",
                          range = if (use_avg) c(0, 10) else NULL), yaxis = list(title = "")) |> plt("books")
  })

  output$d_shelf <- renderUI({
    t <- top_books() |> arrange(desc(if (input$d_rank_by == "avg" && input$f_type != "im") avg else n)) |> head(15)
    validate(need(nrow(t) > 0, "Belum ada buku yang memenuhi filter."))
    div(class = "shelf", lapply(seq_len(nrow(t)), function(k)
      book_card(t$isbn[k], t$title[k], t$author[k], t$avg[k], t$n[k], badge = paste0("#", k))))
  })

  go_to_book <- function(isbn) {
    if (is.null(isbn) || !isbn %in% cf_isbn) {
      showNotification("Buku ini belum punya cukup rating untuk model rekomendasi (minimal 5 rating eksplisit).",
                       type = "warning")
      return(invisible())
    }
    updateSelectizeInput(session, "book", choices = book_choices, selected = isbn, server = TRUE)
    nav_select("nav", "rec"); nav_select("rec_tabs", "similar")
  }
  observeEvent(input$card_click, go_to_book(input$card_click))
  observeEvent(click("books"),
               go_to_book(click("books")$customdata))

  output$d_authors <- renderPlotly({
    t <- book_agg() |> group_by(author) |> summarise(n = sum(n), .groups = "drop") |>
      filter(!author %in% c("Unknown", "")) |> slice_max(n, n = 15, with_ties = FALSE) |> arrange(n)
    validate(need(nrow(t) > 0, "Tidak ada data."))
    plot_ly(t, x = ~n, y = ~factor(author, levels = author), type = "bar", orientation = "h",
            customdata = ~author, source = "authors", marker = grad_bar(t$n, GRAD2),
            hovertemplate = "%{y}<br>%{x:,} rating<extra></extra>") |>
      layout(xaxis = list(title = "Jumlah rating"), yaxis = list(title = "")) |> plt("authors")
  })
  observeEvent(click("authors"), {
    a <- click("authors")$customdata
    req(a)
    t <- book_agg() |> filter(author == a) |> arrange(desc(n)) |> head(25)
    showModal(modalDialog(
      title = paste("Buku karya", a), size = "l", easyClose = TRUE, footer = modalButton("Tutup"),
      p(sprintf("%d judul dengan total %s rating (sesuai filter). Klik baris untuk melihat rekomendasi.",
                nrow(t), fmt(sum(t$n))), class = "small text-muted"),
      DTOutput("author_tbl", fill = FALSE)))
    output$author_tbl <- renderDT(
      datatable(t |> transmute(isbn, Judul = title, Tahun = year, `Jml rating` = n, `Rata-rata` = round(avg, 2)),
                selection = "single", rownames = FALSE, class = "compact stripe hover",
                options = list(pageLength = 10, dom = "tp", columnDefs = list(list(visible = FALSE, targets = 0)))))
  })
  observeEvent(input$author_tbl_rows_selected, {
    a <- click("authors")$customdata
    t <- book_agg() |> filter(author == a) |> arrange(desc(n)) |> head(25)
    removeModal(); go_to_book(t$isbn[input$author_tbl_rows_selected])
  })

  d_table_data <- reactive(book_agg() |> filter(n >= input$f_min) |> arrange(desc(n)) |> head(500))
  output$d_table <- renderDT({
    t <- d_table_data() |> transmute(Judul = title, Penulis = author, Tahun = year, `Jml rating` = n,
                                     `Rata-rata` = round(avg, 2),
                                     `Bisa direkomendasikan` = ifelse(isbn %in% cf_isbn, "✔", ""))
    datatable(t, selection = "single", rownames = FALSE, class = "compact stripe hover",
              options = list(pageLength = 10, scrollX = TRUE))
  })
  observeEvent(input$d_table_rows_selected,
               go_to_book(d_table_data()$isbn[input$d_table_rows_selected]))

  # ------------------------------------------------ REKOMENDASI: buku mirip
  hp <- bstats$isbn[startsWith(bstats$title, "Harry Potter and the Sorcerer")][1]
  updateSelectizeInput(session, "book", choices = book_choices, selected = hp, server = TRUE)
  updateSelectizeInput(session, "picks", choices = book_choices, server = TRUE)
  updateSelectizeInput(session, "m_find", choices = c("(tidak ada)" = "", book_choices), server = TRUE)
  updateSelectizeInput(session, "uid",
                       choices = setNames(user_counts$user_id, sprintf("%d  (%d rating)", user_counts$user_id, user_counts$n)),
                       selected = user_counts$user_id[11], server = TRUE)

  trail <- reactiveVal(character(0))
  observeEvent(input$book, {
    req(input$book)
    tr <- trail(); if (!length(tr) || tail(tr, 1) != input$book) trail(tail(c(tr, input$book), 6))
  })
  output$trail <- renderUI({
    tr <- trail(); req(length(tr) > 1)
    ttl <- substr(bstats$title[match(tr, bstats$isbn)], 1, 35)
    div(class = "small text-muted mt-2", "Jejak penjelajahan: ",
        lapply(seq_along(tr), function(i) tagList(
          actionLink(paste0("trail_", i), ttl[i], onclick = sprintf(
            "Shiny.setInputValue('trail_click', '%s', {priority: 'event'})", tr[i])),
          if (i < length(tr)) " → ")))
  })
  observeEvent(input$trail_click,
               updateSelectizeInput(session, "book", choices = book_choices, selected = input$trail_click, server = TRUE))

  output$book_info <- renderUI({
    req(input$book)
    b <- bstats |> filter(isbn == input$book)
    hs <- function(v, l) div(class = "hero-stat", tags$b(v), span(l))
    div(class = "hero",
        cover(b$isbn, b$title, b$author, size = "L"),
        div(h3(b$title), div(class = "hero-author", b$author),
            div(span(class = "stars", stars(b$avg_rating)), span(class = "ms-2 text-muted small",
                sprintf("%s dari 10 \u00b7 %s pembaca", fmt2(b$avg_rating), b$n_rating))),
            div(class = "mt-2", hs(ifelse(is.na(b$year), "-", b$year), "Tahun terbit"),
                hs(fmt2(b$avg_rating), "Rata-rata rating"), hs(b$n_rating, "Jumlah rating"),
                hs(if (is.na(b$publisher)) "-" else substr(b$publisher, 1, 28), "Penerbit")),
            tags$a(href = paste0("https://openlibrary.org/isbn/", b$isbn), target = "_blank", class = "small",
                   "Lihat di Open Library \u2197")))
  })
  similar <- reactive({
    req(input$book)
    nbr |> filter(isbn == input$book) |> arrange(rank) |> head(input$n_sim) |>
      transmute(isbn = neighbor_isbn, Kemiripan = similarity)
  })
  output$cards_similar <- renderUI({
    s <- similar()
    validate(need(nrow(s) > 0, "Buku ini belum punya cukup pembaca bersama untuk dicari kemiripannya."))
    card_grid(s$isbn, badge = sprintf("mirip %.0f%%", 100 * s$Kemiripan / max(s$Kemiripan)))
  })
  output$tbl_similar <- renderDT({
    validate(need(nrow(similar()) > 0, "Buku ini belum punya cukup pembaca bersama untuk dicari kemiripannya."))
    book_table(similar(), "Kemiripan", selectable = TRUE)
  })
  output$sim_plot <- renderPlotly({
    s <- similar() |> left_join(bstats |> select(isbn, title), by = "isbn") |> arrange(Kemiripan)
    validate(need(nrow(s) > 0, ""))
    lbl <- make.unique(substr(s$title, 1, 40), sep = " ")
    plot_ly(s, x = ~Kemiripan, y = factor(lbl, levels = lbl), type = "bar", orientation = "h",
            customdata = ~isbn, source = "sim", marker = grad_bar(s$Kemiripan),
            hovertemplate = "%{y}<br>kemiripan %{x:.4f}<extra></extra>") |>
      layout(xaxis = list(title = "Kemiripan"), yaxis = list(title = "")) |> plt("sim")
  })
  observeEvent(click("sim"), {
    i <- click("sim")$customdata
    if (!is.null(i)) updateSelectizeInput(session, "book", choices = book_choices, selected = i, server = TRUE)
  })
  observeEvent(input$tbl_similar_rows_selected, {
    i <- similar()$isbn[input$tbl_similar_rows_selected]
    updateSelectizeInput(session, "book", choices = book_choices, selected = i, server = TRUE)
  })

  # ------------------------------------------------ REKOMENDASI: user yang ada
  observeEvent(input$rand_user, {
    u <- sample(user_counts$user_id, 1)
    updateSelectizeInput(session, "uid",
                         choices = setNames(user_counts$user_id, sprintf("%d  (%d rating)", user_counts$user_id, user_counts$n)),
                         selected = u, server = TRUE)
  })
  hist <- reactive({ req(input$uid); ratings |> filter(user_id == as.integer(input$uid)) |> arrange(desc(rating)) })
  output$user_info <- renderUI({
    u <- users |> filter(user_id == as.integer(input$uid)); h <- hist()
    ctry <- if (nrow(u) == 0 || is.na(u$country) || u$country == "") "-" else nice_country(u$country)
    layout_columns(fill = FALSE, info_box("Negara", ctry, 1), info_box("Usia", ifelse(nrow(u) == 0 || is.na(u$age), "-", u$age), 2),
                   info_box("Jumlah rating", nrow(h), 3), info_box("Rata-rata rating user", fmt2(mean(h$rating)), 4))
  })
  output$user_hist_plot <- renderPlotly({
    h <- hist() |> count(rating)
    plot_ly(h, x = ~rating, y = ~n, type = "bar", marker = list(color = rating_col(h$rating)),
            hovertemplate = "Rating %{x}: %{y} buku<extra></extra>") |>
      layout(xaxis = list(title = "Rating yang diberikan", dtick = 1, range = c(0.5, 10.5)),
             yaxis = list(title = "Buku")) |> plt()
  })
  output$rec_title <- renderText(paste("Rekomendasi —", input$model))
  output$rec_title2 <- renderText(paste("\u2728 Rekomendasi untuk user ini \u2014", input$model))
  output$shelf_hist <- renderUI({
    h <- hist() |> head(12)
    i <- match(h$isbn, bstats$isbn)
    div(class = "shelf shelf-sm", lapply(seq_len(nrow(h)), function(k)
      book_card(h$isbn[k], bstats$title[i[k]], bstats$author[i[k]], bstats$avg_rating[i[k]], bstats$n_rating[i[k]],
                badge = paste0("\u2605 ", h$rating[k]))))
  })
  output$cards_user <- renderUI({
    r <- user_rec()
    validate(need(!is.null(r$df) && nrow(r$df) > 0, "Tidak ada kandidat rekomendasi untuk user ini dengan model tersebut."))
    d <- r$df
    badge <- if ("Prediksi rating" %in% names(d)) sprintf("prediksi %.1f", d$`Prediksi rating`)
             else if ("Skor" %in% names(d)) sprintf("cocok %.0f%%", 100 * d$Skor)
             else paste0("#", seq_len(nrow(d)), " populer")
    card_grid(d$isbn, badge = badge, reason = d$`Karena Anda menyukai`)
  })
  user_rec <- reactive({
    h <- hist(); n <- input$n_user
    if (input$model == "Matrix Factorization") {
      u <- mf_upos[[as.character(input$uid)]]
      pred <- mf$mu + mf$bu[u] + mf$bi + as.vector(mf$Q %*% mf$P[u, ])
      list(df = data.frame(isbn = mf$isbns, `Prediksi rating` = pmin(pmax(pred, 1), 10), check.names = FALSE) |>
             filter(!isbn %in% h$isbn) |> slice_max(`Prediksi rating`, n = n, with_ties = FALSE),
           extra = "Prediksi rating")
    } else if (input$model == "Item-based KNN") {
      list(df = knn_recommend(h |> select(isbn, rating), n), extra = c("Skor", "Karena Anda menyukai"))
    } else list(df = bstats |> filter(!isbn %in% h$isbn) |> head(n) |> select(isbn), extra = character(0))
  })
  output$tbl_user <- renderDT({
    r <- user_rec()
    validate(need(!is.null(r$df) && nrow(r$df) > 0, "Tidak ada kandidat rekomendasi untuk user ini dengan model tersebut."))
    book_table(r$df, r$extra, selectable = TRUE)
  })
  observeEvent(input$tbl_user_rows_selected, go_to_book(user_rec()$df$isbn[input$tbl_user_rows_selected]))
  output$tbl_hist <- renderDT({
    h <- hist() |> head(50) |> left_join(bstats |> select(isbn, title, author), by = "isbn") |>
      select(Judul = title, Penulis = author, Rating = rating)
    datatable(h, rownames = FALSE, class = "compact stripe", options = list(dom = "tp", pageLength = 8))
  })

  # ------------------------------------------------ REKOMENDASI: user baru
  output$pick_sliders <- renderUI({
    req(input$picks)
    sl <- lapply(input$picks, function(i)
      sliderInput(paste0("r_", i), substr(bstats$title[bstats$isbn == i], 1, 45), 1, 10,
                  isolate(input[[paste0("r_", i)]]) %||% 8))
    do.call(layout_columns, c(sl, list(col_widths = 3)))
  })
  new_rec <- reactive({
    n <- input$n_new
    if (length(input$picks) == 0)
      return(list(tbl = book_table(bstats |> head(n) |> select(isbn)), df = bstats |> head(n) |> select(isbn),
                  note = "Belum ada buku yang dipilih — menampilkan buku populer."))
    g <- data.frame(isbn = input$picks,
                    rating = vapply(input$picks, function(i) input[[paste0("r_", i)]] %||% 8, numeric(1)))
    rec <- knn_recommend(g, n)
    if (is.null(rec))
      return(list(tbl = book_table(bstats |> filter(!isbn %in% g$isbn) |> head(n) |> select(isbn)),
                  df = bstats |> filter(!isbn %in% g$isbn) |> head(n) |> select(isbn),
                  note = "Belum ada buku yang mirip dengan pilihan Anda; menampilkan buku populer."))
    list(tbl = book_table(rec, c("Skor", "Karena Anda menyukai")), df = rec, note = NULL)
  })
  output$new_note <- renderUI({ nt <- new_rec()$note; if (!is.null(nt)) div(class = "alert alert-info py-2", nt) })
  output$tbl_new <- renderDT(new_rec()$tbl)
  output$cards_new <- renderUI({
    d <- new_rec()$df
    if ("Skor" %in% names(d)) card_grid(d$isbn, badge = sprintf("cocok %.0f%%", 100 * d$Skor), reason = d$`Karena Anda menyukai`)
    else card_grid(d$isbn, badge = paste0("#", seq_len(nrow(d)), " populer"))
  })

  # ------------------------------------------------ PETA BUKU
  map_data <- reactive(book_map |> filter(n_rating >= input$m_min))
  output$map_plot <- renderPlotly({
    m <- map_data()
    col <- if (input$m_color == "avg") m$avg_rating else log10(m$n_rating)
    p <- plot_ly(m, x = ~PC1, y = ~PC2, type = "scattergl", mode = "markers", source = "map",
                 customdata = ~isbn, text = ~sprintf("<b>%s</b><br>%s<br>%d rating · rata-rata %.2f",
                                                    substr(title, 1, 60), author, n_rating, avg_rating),
                 hovertemplate = "%{text}<extra></extra>",
                 marker = list(size = pmin(4 + sqrt(m$n_rating), 22), color = col,
                               colorscale = MAPSC, opacity = 0.8, showscale = TRUE,
                               line = list(color = "white", width = 0.5),
                               colorbar = list(title = if (input$m_color == "avg") "Rata-rata" else "log10(n)")))
    if (isTruthy(input$m_find) && input$m_find %in% m$isbn) {
      s <- m |> filter(isbn == input$m_find)
      p <- p |> add_trace(data = s, x = ~PC1, y = ~PC2, type = "scatter", mode = "markers+text",
                          text = ~substr(title, 1, 40), textposition = "top center", hoverinfo = "skip",
                          marker = list(size = 22, color = "rgba(0,0,0,0)", line = list(color = "#06B6D4", width = 4)),
                          inherit = FALSE, showlegend = FALSE)
    }
    p |> layout(xaxis = list(title = "Dimensi 1", zeroline = FALSE),
                yaxis = list(title = "Dimensi 2", zeroline = FALSE), showlegend = FALSE) |> plt("map")
  })
  map_sel <- reactiveVal(NULL)
  observeEvent(click("map"), map_sel(click("map")$customdata))
  observeEvent(input$m_find, if (isTruthy(input$m_find)) map_sel(input$m_find))
  output$m_info <- renderUI({
    i <- map_sel()
    if (is.null(i)) return(p("Klik sebuah titik untuk melihat detail buku.", class = "small text-muted"))
    b <- bstats |> filter(isbn == i)
    nb <- nbr |> filter(isbn == i) |> arrange(rank) |> head(5) |>
      left_join(bstats |> select(isbn, title), by = c("neighbor_isbn" = "isbn"))
    tagList(div(style = "width:120px", class = "mb-2", cover(b$isbn, b$title, b$author)),
            h6(b$title), p(class = "small", b$author, br(),
                           sprintf("%d rating · rata-rata %s", b$n_rating, fmt2(b$avg_rating))),
            p(class = "small fw-bold mb-1", "Paling mirip:"),
            tags$ol(class = "small ps-3", lapply(substr(nb$title, 1, 45), tags$li)),
            actionButton("m_go", "Lihat rekomendasi →", class = "btn-sm btn-primary"))
  })
  observeEvent(input$m_go, go_to_book(map_sel()))

  # ------------------------------------------------ EVALUASI
  res <- do.call(rbind, lapply(names(met$results), function(k) {
    r <- met$results[[k]]
    data.frame(Model = ifelse(k == "Global mean", "Global mean / Populer", k), RMSE = r$rmse, MAE = r$mae,
               `Precision@10` = r$`precision@10`, `Recall@10` = r$`recall@10`, `HitRate@10` = r$`hitrate@10`,
               check.names = FALSE)
  }))
  output$split_text <- renderUI(p(HTML(sprintf(
    "Data dibagi per user: <b>%d%%</b> rating tiap user disisihkan sebagai data uji (%s latih / %s uji, seed %d). Buku dengan rating ≥ %d di data uji dianggap relevan untuk metrik ranking.",
    as.integer(met$split$test_frac * 100), fmt(met$split$train), fmt(met$split$test), met$split$seed,
    met$params$relevant_min_rating))))
  output$p_err <- renderPlotly(
    plot_ly(res, y = ~Model, x = ~RMSE, type = "bar", orientation = "h", name = "RMSE", marker = list(color = "#6366F1"),
            text = ~sprintf("%.3f", RMSE), textposition = "auto") |>
      add_trace(x = ~MAE, name = "MAE", marker = list(color = "#F9A8D4"), text = ~sprintf("%.3f", MAE)) |>
      layout(barmode = "group", yaxis = list(title = ""), xaxis = list(title = "Nilai"),
             legend = list(orientation = "h", y = 1.1)) |> plt())
  output$tbl_err <- renderTable(res, digits = 4, striped = TRUE, width = "100%")
  output$p_rank <- renderPlotly({
    m <- input$rank_metric
    plot_ly(res, x = ~Model, y = res[[m]], type = "bar",
            marker = list(color = unname(MODEL_COL[res$Model]), line = list(color = ifelse(res[[m]] == max(res[[m]]), DARK, "white"), width = 2)),
            text = sprintf("%.4f", res[[m]]), textposition = "outside",
            hovertemplate = paste0("%{x}<br>", m, " = %{y:.4f}<extra></extra>")) |>
      layout(xaxis = list(title = ""), yaxis = list(title = m, range = c(0, max(res[[m]]) * 1.18))) |> plt()
  })
  output$p_curve <- renderPlotly({
    cv <- met$mf_curve
    plot_ly(cv, x = ~epoch) |>
      add_trace(y = ~train_rmse, name = "Latih", type = "scatter", mode = "lines+markers", line = list(color = "#6366F1", width = 3, shape = "spline"),
                marker = list(color = "#6366F1", size = 7)) |>
      add_trace(y = ~test_rmse, name = "Uji", type = "scatter", mode = "lines+markers", line = list(color = PINK, width = 3, shape = "spline"),
                marker = list(color = PINK, size = 7)) |>
      layout(xaxis = list(title = "Epoch"), yaxis = list(title = "RMSE"), legend = list(orientation = "h", y = 1.1)) |> plt()
  })
  output$interpretasi <- renderUI({
    best_rmse <- res$Model[which.min(res$RMSE)]; best_rank <- res$Model[which.max(res$`HitRate@10`)]
    best_ep <- met$mf_curve$epoch[which.min(met$mf_curve$test_rmse)]
    HTML(sprintf("<ul>
      <li><b>Prediksi rating:</b> %s memberi RMSE terendah (%.3f). Matrix Factorization hampir menyamai baseline, artinya pada data sejarang ini (densitas %.3f%%) sebagian besar sinyal yang bisa dipelajari adalah <i>bias</i> — ada user yang royal memberi nilai, ada buku yang memang disukai banyak orang.</li>
      <li><b>Ranking:</b> %s memiliki HitRate@10 tertinggi. Rekomendasi buku populer menjadi pembanding yang sulit dikalahkan pada data Book-Crossing. Di antara model personal, <b>Item-based KNN</b> unggul dalam ranking dan mudah dijelaskan (\"karena Anda menyukai X\").</li>
      <li><b>Kurva MF:</b> error data uji berhenti turun di sekitar epoch ke-%d; regularisasi (λ = %s) dan jumlah faktor (%d) dipilih lewat grid search untuk menahan overfitting.</li>
      <li><b>Saran pengembangan:</b> memanfaatkan rating implisit (ALS implisit / BPR), model hybrid CF + konten (penulis, penerbit), dan memperketat filter k-core bila mengutamakan akurasi.</li></ul>",
      best_rmse, min(res$RMSE), rep$cf_density_pct, best_rank, best_ep, met$params$mf$reg, met$params$mf$factors))
  })

  # ------------------------------------------------ TENTANG DATA
  output$funnel <- renderPlotly({
    stages <- c("Rating mentah", "Cocok dgn katalog buku", "Rating eksplisit", "Setelah gabung edisi",
                sprintf("k-core (≥%d per user & buku)", rep$min_ratings_per_user))
    vals <- c(rep$ratings_raw, rep$ratings_matched_to_books, rep$ratings_explicit,
              rep$cf_ratings_after_edition_merge, rep$cf_ratings)
    plot_ly(type = "funnel", y = stages, x = vals, textinfo = "value+percent initial",
            marker = list(color = c("#6366F1", "#8B5CF6", "#A855F7", "#D946EF", "#EC4899")),
            connector = list(fillcolor = "#EDE9FE")) |> plt()
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
}

shinyApp(ui, server)
