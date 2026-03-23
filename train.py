import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocess import preprocess_for_training

def build_model(seq_length, n_features):
    """Build LSTM model for cycle length prediction."""
    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train():
    """Train the LSTM model."""
    print("=" * 50)
    print("🔬 Menstrual Cycle Prediction Model Training")
    print("=" * 50)
    
    # Preprocess data
    seq_length = 3
    X, y, scaler_X, scaler_y = preprocess_for_training(seq_length=seq_length)
    
    print(f"\n📊 Data shape: X={X.shape}, y={y.shape}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Features: {X.shape[2]}")
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build model
    model = build_model(seq_length, X.shape[2])
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    # Train
    print("\n🏋️ Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Inverse transform to get actual days error margin
    y_pred = model.predict(X_test, verbose=0)
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    actual_mae = np.mean(np.abs(y_pred_actual - y_test_actual))
    
    print(f"\n📈 Results:")
    print(f"   Test MSE: {loss:.4f}")
    print(f"   Test MAE (scaled): {mae:.4f}")
    print(f"   Test MAE (days): {actual_mae:.2f} days")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/lstm_model.h5')
    print(f"\n💾 Model saved to model/lstm_model.h5")
    
    # Also save in native Keras format
    model.save('model/lstm_model.keras')
    print(f"💾 Keras model saved to model/lstm_model.keras")
    
    print("\n✅ Training complete!")
    return model, history

if __name__ == '__main__':
    train()
