# 📖 Data Dictionary — Menstrual Health Companion

## Dataset Overview

| Item | Detail |
|------|--------|
| **Nama Dataset** | Menstrual Cycle Tracking Dataset |
| **Sumber** | Data sintetis berbasis pola medis menstrual cycle |
| **Jumlah Record** | 95 records |
| **Jumlah Pengguna** | 8 pengguna unik |
| **Periode Data** | Januari 2025 – Desember 2025 |
| **Format File** | CSV (Comma Separated Values) |
| **Lokasi** | `ml-service/data/sample_data.csv` |

---

## Kolom / Variabel

### 1. Fitur Asli (Raw Features)

| No | Kolom | Tipe Data | Range/Nilai | Deskripsi |
|----|-------|-----------|-------------|-----------|
| 1 | `user_id` | Integer | 1–8 | ID unik pengguna. Setiap pengguna memiliki 10–12 record siklus. |
| 2 | `cycle_start` | Date (String) | YYYY-MM-DD | Tanggal awal siklus menstruasi. |
| 3 | `cycle_length` | Integer | 24–35 hari | Panjang total siklus menstruasi (dari hari pertama haid hingga hari pertama haid berikutnya). Range normal: 21–35 hari. |
| 4 | `period_length` | Integer | 4–7 hari | Durasi menstruasi (hari dengan pendarahan aktif). Range normal: 3–7 hari. |
| 5 | `avg_sleep` | Float | 5.0–8.5 jam | Rata-rata durasi tidur per hari selama siklus tersebut. |
| 6 | `avg_stress` | Float | 1.0–5.0 | Rata-rata tingkat stres selama siklus (skala 1=rendah, 5=tinggi). |
| 7 | `fasting_days` | Integer | 0–3 hari | Jumlah hari puasa (Ramadan/intermittent fasting) selama siklus. |
| 8 | `next_cycle_length` | Integer | 24–35 hari | **TARGET VARIABLE** — Panjang siklus berikutnya yang akan diprediksi. |

### 2. Fitur Hasil Engineering (Engineered Features)

Fitur-fitur ini dihasilkan oleh `preprocess.py` saat preprocessing:

| No | Kolom | Tipe Data | Formula | Deskripsi |
|----|-------|-----------|---------|-----------|
| 9 | `cycle_regularity` | Float | `abs(cycle_length - mean_cycle_per_user)` | Mengukur seberapa jauh siklus saat ini dari rata-rata siklus pengguna. Nilai tinggi = siklus tidak teratur. |
| 10 | `sleep_stress_ratio` | Float | `avg_sleep / (avg_stress + 0.1)` | Rasio kualitas tidur terhadap stres. Nilai tinggi = kondisi baik (tidur cukup, stres rendah). |
| 11 | `is_irregular` | Binary (0/1) | `1 if cycle_length < 21 or > 35 else 0` | Flag biner menandakan apakah siklus di luar range normal menurut standar medis WHO. |

---

## Statistik Deskriptif

| Variabel | Mean | Std | Min | Max | Median |
|----------|------|-----|-----|-----|--------|
| cycle_length | 28.6 | 3.1 | 24 | 35 | 28 |
| period_length | 5.2 | 1.1 | 4 | 7 | 5 |
| avg_sleep | 6.8 | 1.0 | 5.0 | 8.5 | 7.0 |
| avg_stress | 2.8 | 1.1 | 1.0 | 5.0 | 2.8 |
| fasting_days | 0.7 | 1.0 | 0 | 3 | 0 |
| next_cycle_length | 28.7 | 3.0 | 24 | 35 | 28 |

---

## Catatan Kualitas Data

- ✅ **Tidak ada missing values** — Semua kolom terisi lengkap
- ✅ **Tidak ada duplikat** — Setiap record unik berdasarkan (user_id, cycle_start)
- ✅ **Tipe data konsisten** — Semua kolom numerik memiliki tipe yang sesuai
- ✅ **Range valid** — Semua nilai berada dalam range medis yang wajar
- ⚠️ **Data sintetis** — Dataset ini dibuat berdasarkan pola medis, bukan data pasien nyata

---

## Hubungan antar Variabel

```
user_id ──────┐
cycle_start ──┤
cycle_length ─┤
period_length─┤──→ [LSTM Model] ──→ next_cycle_length (TARGET)
avg_sleep ────┤
avg_stress ───┤
fasting_days ─┘
```

## Penggunaan dalam Model

| Tahap | Fitur yang Digunakan |
|-------|---------------------|
| **Input Model** | cycle_length, period_length, avg_sleep, avg_stress, fasting_days, cycle_regularity, sleep_stress_ratio, is_irregular |
| **Target Model** | next_cycle_length |
| **Sequence Length** | 3 timesteps (3 siklus berturut-turut) |
| **Scaling** | MinMaxScaler (range 0–1) untuk semua fitur dan target |
