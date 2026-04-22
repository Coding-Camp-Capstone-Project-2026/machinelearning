"""
Train LSTM Model untuk Prediksi Siklus Menstruasi
===================================================
Menggunakan TensorFlow Functional API dengan custom components:
- AttentionLayer (Custom Layer)
- CyclePredictionLoss (Custom Loss Function)
- TrainingMonitorCallback (Custom Callback)
- TensorBoard untuk monitoring training

Model Architecture:
  Input → LSTM(64) → Dropout → LSTM(32, return_sequences) → 
  AttentionLayer → Dense(16) → Dense(1)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocess import preprocess_for_training
from custom_components import AttentionLayer, CyclePredictionLoss, TrainingMonitorCallback


def build_model(seq_length, n_features):
    """
    Build LSTM model menggunakan TensorFlow Functional API.
    
    Architecture:
    - Input Layer: (seq_length, n_features)
    - LSTM Layer 1: 64 units, return_sequences=True
    - Dropout: 0.2
    - LSTM Layer 2: 32 units, return_sequences=True (untuk attention)
    - Custom AttentionLayer: 32 units
    - Dense: 16 units, ReLU
    - Dropout: 0.2
    - Output Dense: 1 unit (predicted cycle length)
    
    Args:
        seq_length: Panjang sequence (jumlah siklus sebelumnya)
        n_features: Jumlah fitur per timestep
    
    Returns:
        Compiled Keras model
    """
    # --- Functional API ---
    inputs = keras.Input(shape=(seq_length, n_features), name='cycle_input')
    
    # LSTM Layer 1
    x = layers.LSTM(64, return_sequences=True, name='lstm_1')(inputs)
    x = layers.Dropout(0.2, name='dropout_1')(x)
    
    # LSTM Layer 2 (return_sequences=True for attention)
    x = layers.LSTM(32, return_sequences=True, name='lstm_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    
    # Custom Attention Layer
    x = AttentionLayer(units=32, name='attention')(x)
    
    # Dense layers
    x = layers.Dense(16, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2, name='dropout_3')(x)
    
    # Output layer
    outputs = layers.Dense(1, name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='MHC_CyclePredictor')
    
    # Compile with Custom Loss Function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=CyclePredictionLoss(delta=1.0, range_weight=0.1),
        metrics=['mae']
    )
    
    return model


def train():
    """Train the LSTM model with Functional API and custom components."""
    print("=" * 60)
    print("🔬 Menstrual Cycle Prediction — Model Training")
    print("   Architecture: Functional API + Attention + Custom Loss")
    print("=" * 60)
    
    # Preprocess data
    seq_length = 3
    X, y, scaler_X, scaler_y = preprocess_for_training(seq_length=seq_length)
    
    print(f"\n📊 Data shape: X={X.shape}, y={y.shape}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Features per timestep: {X.shape[2]}")
    
    # Split data (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Build model with Functional API
    model = build_model(seq_length, X.shape[2])
    model.summary()
    
    # ─── Callbacks ──────────────────────────────────────────────
    os.makedirs('logs', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    callbacks = [
        # 1. Custom Callback — Training Monitor
        TrainingMonitorCallback(
            log_dir='logs',
            patience_alert=15
        ),
        
        # 2. Built-in: EarlyStopping
        keras.callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=1
        ),
        
        # 3. Built-in: ReduceLROnPlateau
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            monitor='val_loss',
            verbose=1
        ),
        
        # 4. TensorBoard — monitoring & visualisasi
        keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        ),
        
        # 5. ModelCheckpoint — save best model
        keras.callbacks.ModelCheckpoint(
            filepath='model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # ─── Train ──────────────────────────────────────────────────
    print("\n🏋️  Training with Functional API + Custom Components...")
    print("   📊 TensorBoard: tensorboard --logdir=logs/tensorboard\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # ─── Evaluate ───────────────────────────────────────────────
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Inverse transform to get actual days error margin
    y_pred = model.predict(X_test, verbose=0)
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    actual_mae = np.mean(np.abs(y_pred_actual - y_test_actual))
    
    print(f"\n{'=' * 60}")
    print(f"📈 EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"   Test Loss (Custom Huber) : {loss:.6f}")
    print(f"   Test MAE (scaled)        : {mae:.6f}")
    print(f"   Test MAE (actual days)   : {actual_mae:.2f} days")
    
    # Check if MAE meets criteria (≤ 0.02 for scaled)
    if mae <= 0.02:
        print(f"   ✅ MAE ≤ 0.02 — CRITERIA MET!")
    else:
        print(f"   ⚠️  MAE > 0.02 — Perlu tuning lebih lanjut")
    
    # ─── Save Model (Multiple Formats) ─────────────────────────
    print(f"\n💾 Saving model in multiple formats...")
    
    # Format 1: Native Keras format (.keras) — PRODUCTION
    model.save('model/lstm_model.keras')
    print(f"   ✅ Keras format     → model/lstm_model.keras")
    
    # Format 2: SavedModel format — TF Serving ready
    model.export('model/saved_model')
    print(f"   ✅ SavedModel       → model/saved_model/")
    
    # Format 3: H5 format — Legacy compatibility
    model.save('model/lstm_model.h5')
    print(f"   ✅ H5 format        → model/lstm_model.h5")
    
    # ─── Inference Demo ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"🔮 INFERENCE DEMO")
    print(f"{'=' * 60}")
    
    # Take first test sample
    sample = X_test[:1]
    pred_scaled = model.predict(sample, verbose=0)
    pred_actual = scaler_y.inverse_transform(pred_scaled)
    true_actual = scaler_y.inverse_transform(y_test[:1].reshape(-1, 1))
    
    print(f"   Input shape      : {sample.shape}")
    print(f"   Predicted (scaled): {pred_scaled[0][0]:.4f}")
    print(f"   Predicted (days)  : {pred_actual[0][0]:.1f} days")
    print(f"   Actual (days)     : {true_actual[0][0]:.1f} days")
    print(f"   Error             : {abs(pred_actual[0][0] - true_actual[0][0]):.1f} days")
    
    print(f"\n✅ Training complete!")
    print(f"   Model: Functional API + AttentionLayer + CyclePredictionLoss")
    print(f"   Run TensorBoard: tensorboard --logdir=logs/tensorboard")
    
    return model, history


if __name__ == '__main__':
    train()
