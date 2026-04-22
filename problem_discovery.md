# 🔍 Problem Discovery — Menstrual Health Companion

## 1. Latar Belakang Masalah

### Konteks
Kesehatan menstruasi (*menstrual health*) adalah aspek fundamental dari kesehatan reproduksi wanita. Siklus menstruasi yang teratur merupakan indikator penting kesehatan secara keseluruhan, namun banyak wanita kesulitan memprediksi kapan siklus berikutnya akan dimulai.

### Permasalahan yang Ditemukan

| No | Permasalahan | Dampak |
|----|-------------|--------|
| 1 | **Ketidakpastian siklus** | Wanita sering tidak tahu kapan menstruasi berikutnya, menyebabkan ketidaknyamanan dan kurangnya persiapan |
| 2 | **Kurangnya alat pelacakan berbasis data** | Kebanyakan aplikasi hanya menggunakan rata-rata sederhana (28 hari), tidak memperhitungkan faktor individual |
| 3 | **Faktor gaya hidup diabaikan** | Tidur, stres, dan puasa mempengaruhi siklus namun jarang diperhitungkan dalam prediksi |
| 4 | **Tidak ada personalisasi** | Setiap wanita memiliki pola siklus unik yang perlu dipelajari secara individual |
| 5 | **Data kesehatan tersebar** | Informasi siklus, mood, dan gejala sering hanya dicatat secara manual tanpa analisis |

### Analisis Permasalahan

Kami menganalisis 5 permasalahan di atas dan menemukan bahwa akar masalahnya adalah:
> **Tidak adanya sistem yang mampu mempelajari pola individual setiap wanita dan memberikan prediksi yang dipersonalisasi berdasarkan data multifaktor (siklus, gaya hidup, dan kondisi harian).**

---

## 2. Solusi yang Dipilih

### Solusi Utama
**Membangun aplikasi Menstrual Health Companion** — sebuah platform web fullstack yang mengintegrasikan:
1. **Pelacakan siklus & log harian** berbasis web
2. **Model prediksi berbasis Deep Learning (LSTM)** untuk memprediksi siklus berikutnya
3. **Dashboard interaktif** untuk menampilkan insight dari data kesehatan pengguna

### Mengapa LSTM?
Long Short-Term Memory (LSTM) dipilih karena:
- Dirancang untuk data **sekuensial/time-series** — cocok untuk pola siklus menstruasi
- Mampu menangkap **dependensi jangka panjang** antar siklus
- Bisa memperhitungkan **multiple features** secara bersamaan (tidur, stres, puasa)
- Terbukti efektif untuk prediksi pola temporal pada bidang kesehatan

---

## 3. Pertanyaan Bisnis (Measurable Business Questions)

| No | Pertanyaan Bisnis | Metrik Pengukuran |
|----|-------------------|-------------------|
| **BQ1** | Seberapa akurat model LSTM dalam memprediksi panjang siklus berikutnya? | MAE (Mean Absolute Error) dalam satuan hari, target ≤ 2 hari |
| **BQ2** | Apakah faktor gaya hidup (tidur, stres, puasa) berpengaruh signifikan terhadap panjang siklus? | Korelasi Pearson antara setiap faktor dan panjang siklus |
| **BQ3** | Seberapa besar variasi panjang siklus antar pengguna? | Standard Deviation panjang siklus per pengguna |
| **BQ4** | Apakah pola siklus cenderung stabil (regular) atau berubah-ubah (irregular) untuk setiap pengguna? | Coefficient of Variation (CV) per pengguna |
| **BQ5** | Bagaimana distribusi panjang siklus secara keseluruhan? | Histogram, skewness, kurtosis |

---

## 4. Arsitektur Solusi

```
┌──────────────────────────────────────────────────────────────────┐
│                    MENSTRUAL HEALTH COMPANION                     │
├──────────────┬──────────────────┬─────────────────────────────────┤
│  FRONTEND    │    BACKEND       │    ML SERVICE                   │
│  (React+Vite)│    (Express)     │    (Flask+TensorFlow)           │
│              │                  │                                 │
│  • Landing   │  • Auth API      │  • LSTM Model (Functional API)  │
│  • Dashboard │  • Cycles CRUD   │  • Custom Attention Layer       │
│  • Calendar  │  • DailyLogs     │  • Custom Loss Function         │
│  • Forms     │  • Predictions   │  • Preprocessing Pipeline       │
│  • Profile   │  • Feedback      │  • REST API (/predict)          │
│              │  • MySQL DB      │  • TensorBoard Integration      │
├──────────────┴──────────────────┴─────────────────────────────────┤
│                    DATA SCIENCE                                    │
│  • EDA & Visualisasi  • Streamlit Dashboard  • Data Dictionary    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Dataset yang Digunakan

- **Nama**: Menstrual Cycle Tracking Dataset
- **Jumlah Data**: 95 records dari 8 pengguna
- **Fitur**: 7 kolom asli + 3 fitur engineered = 10 total
- **Target**: `next_cycle_length` (panjang siklus berikutnya dalam hari)
- **Detail lengkap**: Lihat `data_dictionary.md`

---

## 6. Metodologi

### Data Science Pipeline
1. **Gathering** → Mengumpulkan data siklus menstruasi
2. **Assessing** → Mengevaluasi kualitas dan struktur data
3. **Cleaning** → Membersihkan dan mempersiapkan data
4. **EDA** → Exploratory Data Analysis untuk insight
5. **Feature Engineering** → Membuat fitur tambahan yang informatif
6. **Modeling** → Membangun LSTM model dengan TF Functional API
7. **Evaluation** → Mengukur performa model (MAE, Loss)
8. **Deployment** → Serving model via Flask REST API

### Tech Stack
| Layer | Teknologi |
|-------|-----------|
| Frontend | React 19, Vite 8, Axios, React Router |
| Backend | Express.js 4, MySQL, JWT, bcrypt |
| ML Service | Python, Flask, TensorFlow 2.16, LSTM |
| Data Science | Pandas, Matplotlib, Seaborn, Streamlit |
| Module Bundler | Vite |
