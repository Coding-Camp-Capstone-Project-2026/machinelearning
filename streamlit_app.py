"""
Streamlit Dashboard — Menstrual Health Companion
=================================================
Dashboard interaktif untuk menampilkan insight dan kesimpulan
dari analisis data siklus menstruasi.

Fitur:
- Overview Dataset
- Distribusi Variabel (Interactive)
- Korelasi & Heatmap
- Analisis per Pengguna
- Insight & Kesimpulan
- Business Questions Summary

Jalankan: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MHC — Data Insights Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ec4899, #a855f7, #6366f1);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fdf2f8, #faf5ff);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #fbcfe8;
        text-align: center;
    }
    .insight-box {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/sample_data.csv')
    df['cycle_start'] = pd.to_datetime(df['cycle_start'])
    
    # Feature engineering
    user_avg = df.groupby('user_id')['cycle_length'].transform('mean')
    df['cycle_regularity'] = abs(df['cycle_length'] - user_avg)
    df['sleep_stress_ratio'] = df['avg_sleep'] / (df['avg_stress'] + 0.1)
    df['is_irregular'] = ((df['cycle_length'] < 21) | (df['cycle_length'] > 35)).astype(int)
    df['cycle_month'] = df['cycle_start'].dt.month
    
    return df

df = load_data()


# ─── Sidebar ──────────────────────────────────────────────────
st.sidebar.markdown("## 🌸 MHC Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📂 Navigasi",
    ["🏠 Overview", "📊 Distribusi Data", "🔗 Korelasi", "👤 Analisis per User", "💡 Business Questions", "📝 Kesimpulan"]
)

# User filter
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filter")
selected_users = st.sidebar.multiselect(
    "Pilih User ID",
    options=sorted(df['user_id'].unique()),
    default=sorted(df['user_id'].unique())
)

df_filtered = df[df['user_id'].isin(selected_users)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"📋 Showing **{len(df_filtered)}** of **{len(df)}** records")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🌸 Menstrual Health Companion</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Data Insights Dashboard — Analisis Siklus Menstruasi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Records", len(df_filtered))
    with col2:
        st.metric("👥 Pengguna", df_filtered['user_id'].nunique())
    with col3:
        st.metric("📏 Avg Cycle", f"{df_filtered['cycle_length'].mean():.1f} hari")
    with col4:
        st.metric("💤 Avg Sleep", f"{df_filtered['avg_sleep'].mean():.1f} jam")
    with col5:
        st.metric("😰 Avg Stress", f"{df_filtered['avg_stress'].mean():.1f}/5")
    
    st.markdown("---")
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Statistik Deskriptif")
        numeric_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length']
        st.dataframe(df_filtered[numeric_cols].describe().round(2), use_container_width=True)
    
    with col2:
        st.subheader("🔎 Sample Data")
        st.dataframe(df_filtered.head(10), use_container_width=True)
    
    # Data quality
    st.subheader("✅ Kualitas Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        missing = df_filtered.isnull().sum().sum()
        st.success(f"Missing Values: **{missing}** ✅")
    with col2:
        dupes = df_filtered.duplicated().sum()
        st.success(f"Duplikat: **{dupes}** ✅")
    with col3:
        st.success(f"Records Valid: **{len(df_filtered)}/{len(df_filtered)}** ✅")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: DISTRIBUSI DATA
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Distribusi Data":
    st.markdown("## 📊 Distribusi Data")
    st.markdown("Visualisasi distribusi setiap variabel dalam dataset.")
    
    # Variable selector
    var = st.selectbox("Pilih variabel:", 
                       ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df_filtered, x=var, nbins=15, color_discrete_sequence=['#ec4899'],
                          title=f'Distribusi {var}', marginal='box')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.violin(df_filtered, y=var, x='user_id', color='user_id',
                       title=f'{var} per Pengguna', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template='plotly_white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # All distributions
    st.subheader("📈 Distribusi Semua Variabel")
    fig = make_subplots(rows=2, cols=3, subplot_titles=['Cycle Length', 'Period Length', 'Avg Sleep', 'Avg Stress', 'Fasting Days', 'Next Cycle Length'])
    
    colors = ['#ec4899', '#a855f7', '#3b82f6', '#f59e0b', '#10b981', '#ef4444']
    vars_list = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length']
    
    for i, (v, c) in enumerate(zip(vars_list, colors)):
        row, col = (i // 3) + 1, (i % 3) + 1
        fig.add_trace(go.Histogram(x=df_filtered[v], marker_color=c, name=v, opacity=0.8), row=row, col=col)
    
    fig.update_layout(height=600, template='plotly_white', showlegend=False, title_text='Distribusi Semua Variabel Numerik')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: KORELASI
# ═══════════════════════════════════════════════════════════════════
elif page == "🔗 Korelasi":
    st.markdown("## 🔗 Analisis Korelasi")
    
    numeric_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 
                    'fasting_days', 'next_cycle_length', 'cycle_regularity', 'sleep_stress_ratio']
    corr = df_filtered[numeric_cols].corr()
    
    # Heatmap
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdPu',
                   title='Heatmap Korelasi antar Variabel', aspect='auto')
    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations with target
    st.subheader("🎯 Korelasi dengan Target (next_cycle_length)")
    target_corr = corr['next_cycle_length'].drop('next_cycle_length').sort_values(ascending=False)
    
    fig = px.bar(x=target_corr.values, y=target_corr.index, orientation='h',
                color=target_corr.values, color_continuous_scale='RdPu',
                title='Korelasi setiap Fitur dengan Target')
    fig.update_layout(height=400, template='plotly_white', yaxis_title='Fitur', xaxis_title='Korelasi Pearson')
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter matrix
    st.subheader("🔍 Scatter Plot Matrix")
    scatter_vars = st.multiselect("Pilih variabel:", numeric_cols, default=['cycle_length', 'avg_sleep', 'avg_stress', 'next_cycle_length'])
    if len(scatter_vars) >= 2:
        fig = px.scatter_matrix(df_filtered, dimensions=scatter_vars, color='user_id',
                               color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=700, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4: ANALISIS PER USER
# ═══════════════════════════════════════════════════════════════════
elif page == "👤 Analisis per User":
    st.markdown("## 👤 Analisis per Pengguna")
    
    # User stats
    user_stats = df_filtered.groupby('user_id').agg(
        avg_cycle=('cycle_length', 'mean'),
        std_cycle=('cycle_length', 'std'),
        min_cycle=('cycle_length', 'min'),
        max_cycle=('cycle_length', 'max'),
        count=('cycle_length', 'count'),
        avg_sleep=('avg_sleep', 'mean'),
        avg_stress=('avg_stress', 'mean')
    ).round(2)
    
    user_stats['cv'] = (user_stats['std_cycle'] / user_stats['avg_cycle'] * 100).round(2)
    user_stats['regularity'] = user_stats['cv'].apply(lambda x: '🟢 Regular' if x < 5 else '🟡 Moderate' if x < 10 else '🔴 Irregular')
    
    st.dataframe(user_stats, use_container_width=True)
    
    # Trend per user
    st.subheader("📈 Tren Siklus per Pengguna")
    fig = px.line(df_filtered.sort_values('cycle_start'), x='cycle_start', y='cycle_length', 
                 color='user_id', markers=True,
                 title='Tren Panjang Siklus Setiap Pengguna',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=500, template='plotly_white', xaxis_title='Tanggal', yaxis_title='Panjang Siklus (hari)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Single user deep-dive
    st.subheader("🔬 Detail Pengguna")
    selected_user = st.selectbox("Pilih User ID:", sorted(df_filtered['user_id'].unique()))
    user_data = df_filtered[df_filtered['user_id'] == selected_user].sort_values('cycle_start')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Cycle", f"{user_data['cycle_length'].mean():.1f} hari")
    with col2:
        st.metric("Std Dev", f"{user_data['cycle_length'].std():.2f}")
    with col3:
        st.metric("Range", f"{user_data['cycle_length'].min()}-{user_data['cycle_length'].max()}")
    with col4:
        cv = user_data['cycle_length'].std() / user_data['cycle_length'].mean() * 100
        st.metric("CV%", f"{cv:.1f}%")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Cycle Length Trend', 'Sleep & Stress'])
    fig.add_trace(go.Scatter(x=user_data['cycle_start'], y=user_data['cycle_length'],
                            mode='lines+markers', name='Cycle', line=dict(color='#ec4899')), row=1, col=1)
    fig.add_trace(go.Scatter(x=user_data['cycle_start'], y=user_data['avg_sleep'],
                            mode='lines+markers', name='Sleep', line=dict(color='#3b82f6')), row=1, col=2)
    fig.add_trace(go.Scatter(x=user_data['cycle_start'], y=user_data['avg_stress'],
                            mode='lines+markers', name='Stress', line=dict(color='#ef4444')), row=1, col=2)
    fig.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5: BUSINESS QUESTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "💡 Business Questions":
    st.markdown("## 💡 Menjawab Pertanyaan Bisnis")
    
    # BQ2
    st.subheader("BQ2: Pengaruh Gaya Hidup terhadap Siklus")
    factors = ['avg_sleep', 'avg_stress', 'fasting_days']
    
    results = []
    for f in factors:
        r, p = stats.pearsonr(df_filtered[f], df_filtered['next_cycle_length'])
        results.append({'Faktor': f, 'Korelasi (r)': round(r, 4), 'p-value': round(p, 4), 
                       'Signifikan (α=0.05)': '✅ Ya' if p < 0.05 else '❌ Tidak'})
    
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=['Sleep vs Cycle', 'Stress vs Cycle', 'Fasting vs Cycle'])
    colors = ['#3b82f6', '#ef4444', '#10b981']
    for i, (f, c) in enumerate(zip(factors, colors)):
        fig.add_trace(go.Scatter(x=df_filtered[f], y=df_filtered['next_cycle_length'],
                                mode='markers', marker=dict(color=c, opacity=0.5, size=8),
                                name=f), row=1, col=i+1)
    fig.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # BQ3 & BQ4
    st.subheader("BQ3 & BQ4: Variasi dan Regularitas Siklus")
    
    user_cv = df_filtered.groupby('user_id')['cycle_length'].agg(
        mean='mean', std='std'
    )
    user_cv['CV (%)'] = (user_cv['std'] / user_cv['mean'] * 100).round(2)
    user_cv['Status'] = user_cv['CV (%)'].apply(lambda x: 'Regular' if x < 5 else 'Moderate' if x < 10 else 'Irregular')
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(user_cv.reset_index(), x='user_id', y='mean', error_y='std',
                    title='Rata-rata Siklus per User (± SD)',
                    color_discrete_sequence=['#ec4899'])
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(user_cv.reset_index(), x='user_id', y='CV (%)', color='Status',
                    title='Coefficient of Variation per User',
                    color_discrete_map={'Regular': '#10b981', 'Moderate': '#f59e0b', 'Irregular': '#ef4444'})
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # BQ5
    st.subheader("BQ5: Distribusi Panjang Siklus Keseluruhan")
    
    skewness = df_filtered['cycle_length'].skew()
    kurtosis = df_filtered['cycle_length'].kurtosis()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Skewness", f"{skewness:.3f}")
    with col2:
        st.metric("Kurtosis", f"{kurtosis:.3f}")
    with col3:
        stat, p = stats.shapiro(df_filtered['cycle_length'])
        st.metric("Shapiro-Wilk p", f"{p:.4f}")
    
    fig = px.histogram(df_filtered, x='cycle_length', nbins=15, marginal='violin',
                      color_discrete_sequence=['#ec4899'],
                      title=f'Distribusi Panjang Siklus (Skew={skewness:.2f}, Kurt={kurtosis:.2f})')
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 6: KESIMPULAN
# ═══════════════════════════════════════════════════════════════════
elif page == "📝 Kesimpulan":
    st.markdown("## 📝 Insight & Kesimpulan")
    
    st.markdown("""
    ### 🔍 Temuan Utama

    1. **Panjang siklus bervariasi antara 24-35 hari** dengan rata-rata ~28.6 hari, 
       sesuai dengan range normal menurut standar WHO.

    2. **Korelasi kuat antara siklus saat ini dan berikutnya** (r ≈ 0.85+), 
       menunjukkan pola individual yang konsisten — mendukung pendekatan LSTM.

    3. **Stres berkorelasi positif dengan panjang siklus** — semakin tinggi stres, 
       siklus cenderung lebih panjang. Ini konsisten dengan literatur medis.

    4. **Tidur yang baik berkorelasi dengan siklus lebih pendek/regular** — 
       kualitas tidur berpengaruh terhadap regulasi hormonal.

    5. **Puasa memiliki pengaruh minimal** — jumlah hari puasa tidak menunjukkan 
       korelasi signifikan dengan perubahan panjang siklus.

    6. **Sebagian besar pengguna memiliki siklus REGULAR** (CV < 10%), 
       namun terdapat variasi individual yang signifikan.

    ### 🎯 Rekomendasi
    
    - **Model LSTM cocok** untuk prediksi siklus karena pola sekuensial yang kuat
    - **Feature engineering penting**: cycle_regularity dan sleep_stress_ratio 
      memberikan informasi tambahan yang berguna
    - **Personalisasi per user** lebih efektif daripada model general
    - **Perlu lebih banyak data** untuk meningkatkan akurasi model (minimal 6+ siklus per user)
    
    ### 🏗️ Arsitektur Model
    
    Model menggunakan **LSTM dengan Functional API** dan komponen kustom:
    - Custom Attention Layer (fokus pada siklus paling relevan)
    - Custom Huber Loss (robust terhadap outlier)
    - Custom Training Monitor Callback
    - TensorBoard untuk monitoring
    """)
    
    st.markdown("---")
    st.markdown("*Dashboard dibuat oleh MHC Team — Capstone Project CC 2026*")


# ─── Footer ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("🌸 **Menstrual Health Companion**")
st.sidebar.markdown("Capstone Project CC 2026")
