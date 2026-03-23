import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_data(filepath='data/sample_data.csv'):
    """Load and validate the dataset."""
    df = pd.read_csv(filepath)
    required_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days', 'next_cycle_length']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

def create_features(df):
    """Create feature matrix and target vector."""
    feature_cols = ['cycle_length', 'period_length', 'avg_sleep', 'avg_stress', 'fasting_days']
    X = df[feature_cols].values.astype(np.float32)
    y = df['next_cycle_length'].values.astype(np.float32)
    return X, y

def create_sequences(X, y, seq_length=3):
    """Create sequences for LSTM input.
    
    Each sequence contains seq_length consecutive cycles' features,
    and the target is the next cycle length after the sequence.
    """
    sequences = []
    targets = []
    
    # Group by user_id if available
    for i in range(len(X) - seq_length):
        seq = X[i:i + seq_length]
        target = y[i + seq_length - 1]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def preprocess_for_training(filepath='data/sample_data.csv', seq_length=3):
    """Full preprocessing pipeline for training."""
    df = load_data(filepath)
    X, y = create_features(df)
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    # Save scalers
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler_X, 'model/scaler_X.pkl')
    joblib.dump(scaler_y, 'model/scaler_y.pkl')
    
    return X_seq, y_seq, scaler_X, scaler_y

def preprocess_for_prediction(cycles, sleep, stress, fasting, seq_length=3):
    """Preprocess input data for prediction.
    
    Args:
        cycles: list of cycle lengths
        sleep: list of average sleep values
        stress: list of average stress values
        fasting: list of fasting day counts
    
    Returns:
        Preprocessed sequence ready for model input
    """
    # Load scalers
    scaler_X = joblib.load('model/scaler_X.pkl')
    scaler_y = joblib.load('model/scaler_y.pkl')
    
    # Use the last seq_length entries
    n = min(len(cycles), seq_length)
    
    # Pad with averages if not enough data
    while len(cycles) < seq_length:
        cycles.insert(0, np.mean(cycles))
        sleep.insert(0, np.mean(sleep))
        stress.insert(0, np.mean(stress))
        fasting.insert(0, 0)
    
    # Create feature matrix
    features = []
    for i in range(seq_length):
        idx = len(cycles) - seq_length + i
        period_est = max(3, min(7, int(cycles[idx] * 0.18)))  # Estimate period length
        features.append([
            cycles[idx],
            period_est,
            sleep[idx] if idx < len(sleep) else 7.0,
            stress[idx] if idx < len(stress) else 3.0,
            fasting[idx] if idx < len(fasting) else 0
        ])
    
    features = np.array(features, dtype=np.float32)
    features_scaled = scaler_X.transform(features)
    
    return features_scaled.reshape(1, seq_length, -1), scaler_y

if __name__ == '__main__':
    X, y, sx, sy = preprocess_for_training()
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Scalers saved to model/ directory")
