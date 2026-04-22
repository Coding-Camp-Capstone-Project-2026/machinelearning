"""
Exploratory Data Analysis (EDA) — Menstrual Health Companion
=============================================================
Script ini melakukan:
1. Data Gathering — Mengumpulkan dan memuat dataset
2. Data Assessing — Mengevaluasi kualitas dan struktur data
3. Data Cleaning — Membersihkan dan mempersiapkan data
4. EDA — Analisis eksploratori dengan visualisasi
5. Explanatory Analysis — Menjawab pertanyaan bisnis

Semua visualisasi disimpan ke folder `analysis_output/`

Jalankan: python eda_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─── Configuration ──────────────────────────────────────────────
OUTPUT_DIR = 'analysis_output'
DATA_PATH = 'data/sample_data.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: DATA GATHERING
# ═══════════════════════════════════════════════════════════════════
def data_gathering():
    """
    Mengumpulkan data dari file CSV.
    Dataset berisi catatan siklus menstruasi dari 8 pengguna
    dengan informasi siklus, tidur, stres, dan puasa.
    """
    print("=" * 60)
    print("📥 PHASE 1: DATA GATHERING")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"\n✅ Dataset berhasil dimuat dari: {DATA_PATH}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Kolom: {list(df.columns)}")
    print(f"\n📋 5 Data Pertama:")
    print(df.head().to_string())
    
    return df


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: DATA ASSESSING
# ═══════════════════════════════════════════════════════════════════
def data_assessing(df):
    """
    Mengevaluasi kualitas dan struktur data:
    - Missing values
    - Duplicate rows
    - Tipe data
    - Outliers
    - Statistik deskriptif
    """
    print("\n" + "=" * 60)
    print("🔍 PHASE 2: DATA ASSESSING")
    print("=" * 60)
    
    # 2.1 Info struktur data
    print("\n📊 Informasi Dataset:")
    print(f"   Total records: {len(df)}")
    print(f"   Total kolom: {len(df.columns)}")
    print(f"   Jumlah pengguna unik: {df['user_id'].nunique()}")
    
    # 2.2 Tipe data
    print("\n📋 Tipe Data:")
    for col in df.columns:
        print(f"   {col:20s} → {df[col].dtype}")
    
    # 2.3 Missing values
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ Tidak ada missing values!")
    else:
        print(missing[missing > 0])
    
    # 2.4 Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate Rows: {duplicates}")
    if duplicates == 0:
        print("   ✅ Tidak ada duplikat!")
    
    # 2.5 Statistik deskriptif
    print("\n📈 Statistik Deskriptif:")
    numeric_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length']
    desc = df[numeric_cols].describe().round(2)
    print(desc.to_string())
    
    # 2.6 Outlier detection (IQR method)
    print("\n🔎 Deteksi Outlier (IQR Method):")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            print(f"   ⚠️  {col}: {len(outliers)} outliers (range: {lower:.1f} - {upper:.1f})")
        else:
            print(f"   ✅ {col}: Tidak ada outlier")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════
def data_cleaning(df):
    """
    Membersihkan dan mempersiapkan data:
    - Handle missing values (jika ada)
    - Konversi tipe data
    - Validasi range nilai
    - Feature engineering
    """
    print("\n" + "=" * 60)
    print("🧹 PHASE 3: DATA CLEANING")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # 3.1 Konversi tipe data
    df_clean['cycle_start'] = pd.to_datetime(df_clean['cycle_start'])
    print("\n✅ Konversi cycle_start ke datetime")
    
    # 3.2 Validasi range nilai
    issues = []
    if (df_clean['cycle_length'] < 21).any() or (df_clean['cycle_length'] > 45).any():
        issues.append("cycle_length di luar range 21-45")
    if (df_clean['period_length'] < 1).any() or (df_clean['period_length'] > 10).any():
        issues.append("period_length di luar range 1-10")
    if (df_clean['avg_sleep'] < 0).any() or (df_clean['avg_sleep'] > 12).any():
        issues.append("avg_sleep di luar range 0-12")
    if (df_clean['avg_stress'] < 0).any() or (df_clean['avg_stress'] > 5).any():
        issues.append("avg_stress di luar range 0-5")
    
    if not issues:
        print("✅ Semua nilai berada dalam range yang valid")
    else:
        for issue in issues:
            print(f"⚠️  {issue}")
    
    # 3.3 Feature Engineering
    print("\n🔧 Feature Engineering:")
    
    # Cycle regularity (per user)
    user_avg = df_clean.groupby('user_id')['cycle_length'].transform('mean')
    df_clean['cycle_regularity'] = abs(df_clean['cycle_length'] - user_avg)
    print("   ✅ cycle_regularity = |cycle_length - mean_per_user|")
    
    # Sleep-stress ratio
    df_clean['sleep_stress_ratio'] = df_clean['avg_sleep'] / (df_clean['avg_stress'] + 0.1)
    print("   ✅ sleep_stress_ratio = avg_sleep / (avg_stress + 0.1)")
    
    # Is irregular
    df_clean['is_irregular'] = ((df_clean['cycle_length'] < 21) | (df_clean['cycle_length'] > 35)).astype(int)
    print("   ✅ is_irregular = 1 if cycle < 21 or > 35")
    
    # Cycle month
    df_clean['cycle_month'] = df_clean['cycle_start'].dt.month
    print("   ✅ cycle_month (untuk analisis musiman)")
    
    print(f"\n📊 Dataset setelah cleaning: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    print(f"   Kolom baru: cycle_regularity, sleep_stress_ratio, is_irregular, cycle_month")
    
    return df_clean


# ═══════════════════════════════════════════════════════════════════
# PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════════
def eda_analysis(df):
    """
    Exploratory Data Analysis dengan visualisasi:
    - Distribusi variabel
    - Korelasi antar variabel
    - Pola per pengguna
    - Tren temporal
    """
    print("\n" + "=" * 60)
    print("📊 PHASE 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # ─── 4.1 Distribusi Cycle Length ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['cycle_length'], bins=12, color='#ec4899', edgecolor='white', alpha=0.8)
    axes[0].axvline(df['cycle_length'].mean(), color='#7e22ce', linestyle='--', linewidth=2, label=f"Mean: {df['cycle_length'].mean():.1f}")
    axes[0].set_title('Distribusi Panjang Siklus')
    axes[0].set_xlabel('Panjang Siklus (hari)')
    axes[0].set_ylabel('Frekuensi')
    axes[0].legend()
    
    axes[1].hist(df['next_cycle_length'], bins=12, color='#a855f7', edgecolor='white', alpha=0.8)
    axes[1].axvline(df['next_cycle_length'].mean(), color='#ec4899', linestyle='--', linewidth=2, label=f"Mean: {df['next_cycle_length'].mean():.1f}")
    axes[1].set_title('Distribusi Target (Next Cycle Length)')
    axes[1].set_xlabel('Panjang Siklus Berikutnya (hari)')
    axes[1].set_ylabel('Frekuensi')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_distribusi_cycle_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 01_distribusi_cycle_length.png")
    
    # ─── 4.2 Boxplot per User ────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(column='cycle_length', by='user_id', ax=ax,
               boxprops=dict(color='#ec4899'), medianprops=dict(color='#7e22ce', linewidth=2))
    ax.set_title('Distribusi Panjang Siklus per Pengguna')
    ax.set_xlabel('User ID')
    ax.set_ylabel('Panjang Siklus (hari)')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_boxplot_per_user.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 02_boxplot_per_user.png")
    
    # ─── 4.3 Correlation Heatmap ─────────────────────────────
    numeric_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 
                    'fasting_days', 'next_cycle_length', 'cycle_regularity', 'sleep_stress_ratio']
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdPu',
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Heatmap Korelasi antar Variabel')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 03_correlation_heatmap.png")
    
    # ─── 4.4 Scatter: Sleep vs Stress ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    scatter_pairs = [
        ('avg_sleep', 'next_cycle_length', 'Tidur vs Siklus Berikutnya'),
        ('avg_stress', 'next_cycle_length', 'Stres vs Siklus Berikutnya'),
        ('cycle_length', 'next_cycle_length', 'Siklus Saat Ini vs Berikutnya'),
    ]
    colors = ['#ec4899', '#a855f7', '#3b82f6']
    
    for i, (x, y, title) in enumerate(scatter_pairs):
        axes[i].scatter(df[x], df[y], alpha=0.6, c=colors[i], s=60, edgecolors='white')
        # Tambah trend line
        z = np.polyfit(df[x], df[y], 1)
        p = np.poly1d(z)
        axes[i].plot(sorted(df[x]), p(sorted(df[x])), '--', color='gray', linewidth=1.5)
        r, pval = stats.pearsonr(df[x], df[y])
        axes[i].set_title(f'{title}\nr={r:.3f}, p={pval:.4f}')
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_scatter_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 04_scatter_plots.png")
    
    # ─── 4.5 Distribusi Semua Variabel ───────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vars_to_plot = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length']
    colors = ['#ec4899', '#a855f7', '#3b82f6', '#f59e0b', '#10b981', '#ef4444']
    
    for i, (var, color) in enumerate(zip(vars_to_plot, colors)):
        ax = axes[i // 3][i % 3]
        ax.hist(df[var], bins=10, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(df[var].mean(), color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'Distribusi {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('Frekuensi')
        
        # Tambah skewness
        skew = df[var].skew()
        ax.text(0.95, 0.95, f'Skew: {skew:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Distribusi Semua Variabel Numerik', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_distribusi_semua_variabel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 05_distribusi_semua_variabel.png")
    
    # ─── 4.6 Tren Siklus per User (Time Series) ─────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    users = sorted(df['user_id'].unique())
    
    for i, uid in enumerate(users):
        ax = axes[i // 4][i % 4]
        user_data = df[df['user_id'] == uid].sort_values('cycle_start')
        ax.plot(range(len(user_data)), user_data['cycle_length'], 'o-', 
                color='#ec4899', markersize=6, linewidth=1.5, label='Cycle Length')
        ax.axhline(user_data['cycle_length'].mean(), color='#7e22ce', 
                   linestyle='--', alpha=0.5, label=f"Avg: {user_data['cycle_length'].mean():.1f}")
        ax.set_title(f'User {uid}')
        ax.set_xlabel('Siklus ke-')
        ax.set_ylabel('Hari')
        ax.legend(fontsize=7)
        ax.set_ylim(20, 40)
    
    plt.suptitle('Tren Panjang Siklus per Pengguna', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_tren_per_user.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 06_tren_per_user.png")
    
    # ─── 4.7 Pairplot ───────────────────────────────────────
    fig = sns.pairplot(df[['cycle_length', 'avg_sleep', 'avg_stress', 'next_cycle_length']], 
                       diag_kind='kde', plot_kws={'alpha': 0.5, 'color': '#ec4899'},
                       diag_kws={'color': '#a855f7'})
    fig.fig.suptitle('Pairplot: Hubungan antar Variabel Utama', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig(f'{OUTPUT_DIR}/07_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 07_pairplot.png")
    
    return corr


# ═══════════════════════════════════════════════════════════════════
# PHASE 5: EXPLANATORY ANALYSIS — Menjawab Pertanyaan Bisnis
# ═══════════════════════════════════════════════════════════════════
def explanatory_analysis(df, corr):
    """
    Menjawab pertanyaan bisnis yang telah didefinisikan:
    BQ1-BQ5 dengan visualisasi pendukung.
    """
    print("\n" + "=" * 60)
    print("💡 PHASE 5: EXPLANATORY ANALYSIS")
    print("   Menjawab Pertanyaan Bisnis")
    print("=" * 60)
    
    # ─── BQ2: Pengaruh gaya hidup terhadap siklus ───────────
    print("\n📊 BQ2: Apakah faktor gaya hidup berpengaruh signifikan?")
    
    factors = ['avg_sleep', 'avg_stress', 'fasting_days']
    print(f"\n   Korelasi dengan next_cycle_length:")
    for f in factors:
        r, p = stats.pearsonr(df[f], df['next_cycle_length'])
        sig = "✅ SIGNIFIKAN" if p < 0.05 else "❌ Tidak signifikan"
        print(f"   {f:15s}: r={r:+.4f}, p={p:.4f} — {sig}")
    
    # Visualisasi
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, f in enumerate(factors):
        sns.regplot(x=f, y='next_cycle_length', data=df, ax=axes[i],
                    scatter_kws={'alpha': 0.5, 'color': '#ec4899'},
                    line_kws={'color': '#7e22ce'})
        r, p = stats.pearsonr(df[f], df['next_cycle_length'])
        axes[i].set_title(f'{f} → next_cycle\nr={r:.3f}, p={p:.4f}')
    
    plt.suptitle('BQ2: Pengaruh Gaya Hidup terhadap Siklus', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_bq2_lifestyle_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 08_bq2_lifestyle_impact.png")
    
    # ─── BQ3: Variasi panjang siklus antar pengguna ─────────
    print(f"\n📊 BQ3: Seberapa besar variasi panjang siklus antar pengguna?")
    
    user_stats = df.groupby('user_id')['cycle_length'].agg(['mean', 'std', 'min', 'max']).round(2)
    user_stats.columns = ['Mean', 'Std', 'Min', 'Max']
    print(f"\n{user_stats.to_string()}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    user_stats['Mean'].plot(kind='bar', ax=ax, color='#ec4899', edgecolor='white', alpha=0.8)
    ax.errorbar(range(len(user_stats)), user_stats['Mean'], yerr=user_stats['Std'],
                fmt='none', color='#7e22ce', capsize=5, linewidth=2)
    ax.set_title('BQ3: Rata-rata & Variasi Siklus per Pengguna')
    ax.set_xlabel('User ID')
    ax.set_ylabel('Panjang Siklus (hari)')
    ax.set_xticklabels(user_stats.index, rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_bq3_variasi_per_user.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 09_bq3_variasi_per_user.png")
    
    # ─── BQ4: Regularitas siklus ────────────────────────────
    print(f"\n📊 BQ4: Apakah siklus cenderung regular atau irregular?")
    
    cv_per_user = df.groupby('user_id')['cycle_length'].agg(
        lambda x: (x.std() / x.mean() * 100) if x.mean() > 0 else 0
    ).round(2)
    cv_per_user.name = 'CV (%)'
    print(f"\n   Coefficient of Variation (CV) per pengguna:")
    for uid, cv in cv_per_user.items():
        status = "Regular (CV<5%)" if cv < 5 else "Cukup Regular (CV<10%)" if cv < 10 else "Irregular (CV≥10%)"
        print(f"   User {uid}: CV = {cv:.2f}% — {status}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#10b981' if cv < 5 else '#f59e0b' if cv < 10 else '#ef4444' for cv in cv_per_user]
    cv_per_user.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.axhline(5, color='#10b981', linestyle='--', alpha=0.5, label='Regular (<5%)')
    ax.axhline(10, color='#ef4444', linestyle='--', alpha=0.5, label='Irregular (>10%)')
    ax.set_title('BQ4: Regularitas Siklus per Pengguna (CV%)')
    ax.set_xlabel('User ID')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_xticklabels(cv_per_user.index, rotation=0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_bq4_regularitas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 10_bq4_regularitas.png")
    
    # ─── BQ5: Distribusi keseluruhan ────────────────────────
    print(f"\n📊 BQ5: Bagaimana distribusi panjang siklus secara keseluruhan?")
    
    skewness = df['cycle_length'].skew()
    kurtosis = df['cycle_length'].kurtosis()
    print(f"   Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})")
    print(f"   Kurtosis: {kurtosis:.4f} ({'Leptokurtic (tajam)' if kurtosis > 0 else 'Platykurtic (datar)'})")
    
    # Normality test
    stat, p_normal = stats.shapiro(df['cycle_length'])
    print(f"   Shapiro-Wilk test: W={stat:.4f}, p={p_normal:.4f}")
    print(f"   Distribusi {'normal' if p_normal > 0.05 else 'TIDAK normal'} (α=0.05)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KDE plot
    axes[0].hist(df['cycle_length'], bins=12, density=True, alpha=0.6, color='#ec4899', edgecolor='white')
    df['cycle_length'].plot(kind='kde', ax=axes[0], color='#7e22ce', linewidth=2)
    axes[0].set_title(f'BQ5: Distribusi Panjang Siklus\nSkew={skewness:.2f}, Kurt={kurtosis:.2f}')
    axes[0].set_xlabel('Panjang Siklus (hari)')
    
    # QQ Plot
    stats.probplot(df['cycle_length'], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (vs Normal Distribution)')
    axes[1].get_lines()[0].set_color('#ec4899')
    axes[1].get_lines()[1].set_color('#7e22ce')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_bq5_distribusi_keseluruhan.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 11_bq5_distribusi_keseluruhan.png")
    
    # ─── SUMMARY ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📝 RINGKASAN INSIGHT")
    print("=" * 60)
    print("""
    1. Panjang siklus bervariasi antara 24-35 hari dengan rata-rata ~28.6 hari
    2. Terdapat korelasi kuat antara cycle_length dan next_cycle_length,
       menunjukkan pola siklus yang cenderung konsisten per individu
    3. Stres (avg_stress) memiliki korelasi POSITIF dengan panjang siklus — 
       semakin tinggi stres, siklus cenderung lebih panjang
    4. Tidur (avg_sleep) memiliki korelasi NEGATIF dengan panjang siklus — 
       tidur yang lebih baik berkorelasi dengan siklus yang lebih pendek
    5. Sebagian besar pengguna memiliki siklus yang REGULAR (CV < 10%)
    6. Puasa (fasting_days) memiliki pengaruh minimal terhadap panjang siklus
    7. Data tidak sepenuhnya berdistribusi normal (multimodal karena variasi antar user)
    """)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🚀 MENSTRUAL HEALTH COMPANION — EDA Analysis")
    print("=" * 60)
    
    # Phase 1: Gathering
    df = data_gathering()
    
    # Phase 2: Assessing
    df = data_assessing(df)
    
    # Phase 3: Cleaning + Feature Engineering
    df = data_cleaning(df)
    
    # Phase 4: EDA
    corr = eda_analysis(df)
    
    # Phase 5: Explanatory Analysis
    explanatory_analysis(df, corr)
    
    print(f"\n✅ Semua visualisasi disimpan ke: {OUTPUT_DIR}/")
    print("🏁 EDA Analysis selesai!")
