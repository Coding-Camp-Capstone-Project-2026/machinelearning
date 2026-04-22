"""
Preprocessing Pipeline untuk Model Prediksi Siklus Menstruasi
==============================================================
Modul ini menangani seluruh proses data preprocessing:

1. Data Loading & Validation
2. Feature Engineering
3. Feature Scaling (MinMaxScaler)
4. Sequence Creation (untuk LSTM input)
5. Prediction Preprocessing

Feature Engineering:
- cycle_length: Panjang siklus (hari)
- period_length: Durasi menstruasi (hari)
- avg_sleep: Rata-rata kualitas tidur (1-10 jam)
- avg_stress: Rata-rata tingkat stres (1-5)
- fasting_days: Jumlah hari puasa dalam siklus
- cycle_regularity: Deviasi dari rata-rata siklus (engineered)
- sleep_stress_ratio: Rasio tidur terhadap stres (engineered)
- is_irregular: Flag jika siklus di luar range normal (engineered)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def load_data(filepath='data/sample_data.csv'):
    """
    Load dan validasi dataset.
    
    Args:
        filepath: Path ke file CSV
    
    Returns:
        DataFrame yang sudah divalidasi
    
    Raises:
        ValueError: Jika kolom yang dibutuhkan tidak ada
    """
    df = pd.read_csv(filepath)
    required_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def engineer_features(df):
    """
    Feature Engineering — membuat fitur tambahan dari data mentah.
    
    Fitur yang dihasilkan:
    1. cycle_regularity: Seberapa jauh siklus dari rata-rata pengguna
    2. sleep_stress_ratio: Rasio kualitas tidur terhadap stres
    3. is_irregular: Binary flag untuk siklus irregular (< 21 atau > 35 hari)
    
    Args:
        df: DataFrame dengan kolom mentah
    
    Returns:
        DataFrame dengan fitur tambahan
    """
    df = df.copy()
    
    # 1. Cycle Regularity — deviasi dari rata-rata siklus per user
    if 'user_id' in df.columns:
        user_avg = df.groupby('user_id')['cycle_length'].transform('mean')
        df['cycle_regularity'] = abs(df['cycle_length'] - user_avg)
    else:
        global_avg = df['cycle_length'].mean()
        df['cycle_regularity'] = abs(df['cycle_length'] - global_avg)
    
    # 2. Sleep-Stress Ratio — hubungan tidur dan stres
    # Tidur yang baik + stres rendah = rasio tinggi (baik)
    df['sleep_stress_ratio'] = df['avg_sleep'] / (df['avg_stress'] + 0.1)
    
    # 3. Is Irregular — flag untuk siklus di luar range normal
    df['is_irregular'] = ((df['cycle_length'] < 21) | (df['cycle_length'] > 35)).astype(int)
    
    return df


def create_features(df):
    """
    Create feature matrix dan target vector.
    
    Menggunakan 8 fitur:
    - 5 fitur asli: cycle_length, period_length, avg_sleep, avg_stress, fasting_days
    - 3 fitur engineered: cycle_regularity, sleep_stress_ratio, is_irregular
    
    Args:
        df: DataFrame dengan fitur lengkap
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    """
    feature_cols = [
        'cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days',
        'cycle_regularity', 'sleep_stress_ratio', 'is_irregular'
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df['next_cycle_length'].values.astype(np.float32)
    return X, y


def create_sequences(X, y, seq_length=3):
    """
    Create sequences untuk LSTM input.
    
    Setiap sequence berisi `seq_length` timesteps berturut-turut,
    dan target-nya adalah panjang siklus setelah sequence tersebut.
    
    Args:
        X: Feature matrix
        y: Target vector
        seq_length: Jumlah timestep per sequence
    
    Returns:
        sequences: Array (n_sequences, seq_length, n_features)
        targets: Array (n_sequences,)
    """
    sequences = []
    targets = []
    
    for i in range(len(X) - seq_length):
        seq = X[i:i + seq_length]
        target = y[i + seq_length - 1]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def preprocess_for_training(filepath='data/sample_data.csv', seq_length=3):
    """
    Full preprocessing pipeline untuk training.
    
    Pipeline:
    1. Load data
    2. Feature engineering
    3. Create feature matrix
    4. Scale features (MinMaxScaler)
    5. Create sequences
    6. Save scalers
    
    Args:
        filepath: Path ke dataset CSV
        seq_length: Panjang sequence untuk LSTM
    
    Returns:
        X_seq: Sequences (n_sequences, seq_length, n_features)
        y_seq: Targets (n_sequences,)
        scaler_X: Fitted MinMaxScaler untuk features
        scaler_y: Fitted MinMaxScaler untuk target
    """
    # 1. Load data
    df = load_data(filepath)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Create features
    X, y = create_features(df)
    
    # 4. Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 5. Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    # 6. Save scalers
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler_X, 'model/scaler_X.pkl')
    joblib.dump(scaler_y, 'model/scaler_y.pkl')
    
    return X_seq, y_seq, scaler_X, scaler_y


def preprocess_for_prediction(cycles, sleep, stress, fasting, seq_length=3):
    """
    Preprocess input data untuk prediction/inference.
    
    Args:
        cycles: list of cycle lengths
        sleep: list of average sleep values
        stress: list of average stress values
        fasting: list of fasting day counts
        seq_length: Panjang sequence yang diharapkan model
    
    Returns:
        X_input: Preprocessed sequence ready for model input
        scaler_y: Scaler untuk inverse transform hasil prediksi
    """
    # Load scalers
    scaler_X = joblib.load('model/scaler_X.pkl')
    scaler_y = joblib.load('model/scaler_y.pkl')
    
    # Pad with averages if not enough data
    while len(cycles) < seq_length:
        cycles.insert(0, np.mean(cycles))
        sleep.insert(0, np.mean(sleep))
        stress.insert(0, np.mean(stress))
        fasting.insert(0, 0)
    
    # Create feature matrix with engineered features
    features = []
    avg_cycle = np.mean(cycles)
    
    for i in range(seq_length):
        idx = len(cycles) - seq_length + i
        cycle_len = cycles[idx]
        period_est = max(3, min(7, int(cycle_len * 0.18)))  # Estimate period length
        sleep_val = sleep[idx] if idx < len(sleep) else 7.0
        stress_val = stress[idx] if idx < len(stress) else 3.0
        fasting_val = fasting[idx] if idx < len(fasting) else 0
        
        # Engineered features
        cycle_regularity = abs(cycle_len - avg_cycle)
        sleep_stress_ratio = sleep_val / (stress_val + 0.1)
        is_irregular = 1 if (cycle_len < 21 or cycle_len > 35) else 0
        
        features.append([
            cycle_len,
            period_est,
            sleep_val,
            stress_val,
            fasting_val,
            cycle_regularity,
            sleep_stress_ratio,
            is_irregular
        ])
    
    features = np.array(features, dtype=np.float32)
    features_scaled = scaler_X.transform(features)
    
    return features_scaled.reshape(1, seq_length, -1), scaler_y


if __name__ == '__main__':
    X, y, sx, sy = preprocess_for_training()
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Features per timestep: {X.shape[2]}")
    print(f"  - 5 original + 3 engineered = 8 total")
    print(f"Scalers saved to model/ directory")
