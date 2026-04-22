"""
Custom Components untuk Model Prediksi Siklus Menstruasi
=========================================================
Modul ini berisi komponen kustom yang digunakan dalam model LSTM:
1. AttentionLayer - Custom Layer untuk mekanisme attention pada sequence data
2. CyclePredictionLoss - Custom Loss Function (Weighted Huber Loss)
3. TrainingMonitorCallback - Custom Callback untuk monitoring training secara detail
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# 1. CUSTOM LAYER — Attention Mechanism
# =============================================================================
class AttentionLayer(keras.layers.Layer):
    """
    Custom Attention Layer untuk LSTM-based cycle prediction.
    
    Mekanisme attention memungkinkan model memberikan bobot berbeda
    pada setiap timestep dalam sequence, sehingga siklus-siklus tertentu
    yang lebih relevan (misalnya siklus terbaru) bisa lebih berpengaruh
    terhadap prediksi.
    
    Referensi: Bahdanau Attention (additive attention)
    """
    
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name='attention_weight',
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, features)
        # Score: tanh(inputs @ W + b)
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        
        # Attention weights: softmax over timestep dimension
        attention_weights = tf.nn.softmax(
            tf.reduce_sum(score * self.u, axis=-1, keepdims=True),
            axis=1
        )
        
        # Context vector: weighted sum of inputs
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


# =============================================================================
# 2. CUSTOM LOSS FUNCTION — Weighted Huber Loss
# =============================================================================
class CyclePredictionLoss(keras.losses.Loss):
    """
    Custom Loss Function untuk prediksi siklus menstruasi.
    
    Mengkombinasikan Huber Loss dengan penalti tambahan untuk prediksi
    yang terlalu jauh dari range normal (21-45 hari). Ini memastikan
    model tidak menghasilkan prediksi yang secara medis tidak masuk akal.
    
    Components:
    - Huber Loss: Robust terhadap outlier (lebih baik dari MSE murni)
    - Range Penalty: Penalti ekstra untuk prediksi di luar range normal
    """
    
    def __init__(self, delta=1.0, range_weight=0.1, name='cycle_prediction_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.range_weight = range_weight
    
    def call(self, y_true, y_pred):
        # 1. Huber Loss component
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        huber = tf.reduce_mean(0.5 * tf.square(quadratic) + self.delta * linear)
        
        # 2. Range Penalty — penalti jika prediksi di luar range yang masuk akal
        # Karena data sudah di-scale ke [0,1], kita gunakan threshold yang sesuai
        lower_violation = tf.maximum(0.0 - y_pred, 0.0)
        upper_violation = tf.maximum(y_pred - 1.0, 0.0)
        range_penalty = tf.reduce_mean(tf.square(lower_violation) + tf.square(upper_violation))
        
        return huber + self.range_weight * range_penalty
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'range_weight': self.range_weight
        })
        return config


# =============================================================================
# 3. CUSTOM CALLBACK — Training Monitor
# =============================================================================
class TrainingMonitorCallback(keras.callbacks.Callback):
    """
    Custom Callback untuk monitoring training secara detail.
    
    Fitur:
    - Log detail per epoch (loss, learning rate, improvement)
    - Tracking learning rate changes
    - Summary statistik di akhir training
    - Export training history ke file CSV
    """
    
    def __init__(self, log_dir='logs', patience_alert=10, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.patience_alert = patience_alert
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.training_log = []
    
    def on_train_begin(self, logs=None):
        print("\n" + "=" * 60)
        print("🔬 TRAINING MONITOR — Custom Callback Active")
        print("=" * 60)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        train_mae = logs.get('mae', 0)
        val_mae = logs.get('val_mae', 0)
        
        # Get current learning rate
        try:
            current_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            current_lr = 0.001
        
        # Track improvement
        improved = val_loss < self.best_val_loss
        improvement = self.best_val_loss - val_loss if improved else 0
        
        if improved:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.wait = 0
            status = f"📈 IMPROVED by {improvement:.6f}"
        else:
            self.wait += 1
            status = f"⏳ No improvement ({self.wait}/{self.patience_alert})"
        
        # Log entry
        entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'lr': current_lr,
            'improved': improved
        }
        self.training_log.append(entry)
        
        # Print status
        if (epoch + 1) % 10 == 0 or improved or epoch == 0:
            print(f"\n  Epoch {epoch+1:3d} | "
                  f"Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                  f"MAE: {train_mae:.5f} | Val MAE: {val_mae:.5f} | "
                  f"LR: {current_lr:.2e} | {status}")
    
    def on_train_end(self, logs=None):
        print("\n" + "=" * 60)
        print("🏁 TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total Epochs Trained : {len(self.training_log)}")
        print(f"  Best Val Loss        : {self.best_val_loss:.6f} (Epoch {self.best_epoch})")
        
        if self.training_log:
            final = self.training_log[-1]
            first = self.training_log[0]
            print(f"  Final Train Loss     : {final['train_loss']:.6f}")
            print(f"  Final Val Loss       : {final['val_loss']:.6f}")
            print(f"  Loss Improvement     : {first['val_loss'] - final['val_loss']:.6f}")
            print(f"  Final Learning Rate  : {final['lr']:.2e}")
        
        # Save training log to CSV
        try:
            import csv
            log_path = os.path.join(self.log_dir, 'training_log.csv')
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'train_mae', 'val_mae', 'lr', 'improved'])
                writer.writeheader()
                writer.writerows(self.training_log)
            print(f"\n  📄 Training log saved to: {log_path}")
        except Exception as e:
            print(f"\n  ⚠️ Could not save training log: {e}")
        
        print("=" * 60)
