"""
ML Service — Flask REST API untuk Serving Model Prediksi Siklus Menstruasi
==========================================================================
REST API mandiri menggunakan Flask untuk melayani model machine learning.

Endpoints:
  GET  /health   — Health check & model status
  POST /predict  — Prediksi siklus berikutnya
  GET  /model-info — Informasi model (architecture, version)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from preprocess import preprocess_for_prediction
from custom_components import AttentionLayer, CyclePredictionLoss, TrainingMonitorCallback

app = Flask(__name__)
CORS(app)

# Load model on startup
model = None
MODEL_PATH = 'model/lstm_model.keras'
MODEL_PATH_H5 = 'model/lstm_model.h5'


def load_model():
    """Load trained model with custom objects support."""
    global model
    
    custom_objects = {
        'AttentionLayer': AttentionLayer,
        'CyclePredictionLoss': CyclePredictionLoss,
    }
    
    # Try .keras format first (production), then .h5 (legacy)
    for path in [MODEL_PATH, MODEL_PATH_H5]:
        if os.path.exists(path):
            try:
                model = keras.models.load_model(path, custom_objects=custom_objects)
                model.compile(
                    optimizer='adam',
                    loss=CyclePredictionLoss(delta=1.0, range_weight=0.1),
                    metrics=['mae']
                )
                print(f"✅ Model loaded from {path}")
                return
            except Exception as e:
                print(f"⚠️  Error loading model from {path}: {e}")
                continue
    
    print(f"⚠️  No model found. Run train.py first.")
    model = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_format': 'keras' if os.path.exists(MODEL_PATH) else 'h5' if os.path.exists(MODEL_PATH_H5) else 'none'
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return model architecture and metadata."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'name': model.name,
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'total_params': int(model.count_params()),
        'layers': [
            {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output.shape) if hasattr(layer, 'output') else 'N/A'
            }
            for layer in model.layers
        ],
        'model_version': '2.0',
        'architecture': 'Functional API + Attention + Custom Loss'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediksi panjang siklus berikutnya.
    
    Request JSON:
        cycles: list[int]   — Riwayat panjang siklus (min 1)
        sleep: list[float]  — Rata-rata kualitas tidur per siklus (opsional)
        stress: list[float] — Rata-rata tingkat stres per siklus (opsional)
        fasting: list[int]  — Jumlah hari puasa per siklus (opsional)
    
    Response JSON:
        predicted_cycle_length: int
        confidence: float (0-1)
        model_version: str
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        cycles = data.get('cycles', [])
        sleep = data.get('sleep', [])
        stress = data.get('stress', [])
        fasting = data.get('fasting', [])
        
        if len(cycles) < 1:
            return jsonify({'error': 'At least 1 cycle length is required'}), 400
        
        # Simple average fallback if model not loaded
        if model is None:
            avg_cycle = round(sum(cycles) / len(cycles))
            return jsonify({
                'predicted_cycle_length': avg_cycle,
                'confidence': 0.5,
                'model_version': 'fallback_average',
                'message': 'Model not loaded. Using simple average.'
            })
        
        # Preprocess and predict
        X_input, scaler_y = preprocess_for_prediction(
            cycles=list(cycles),
            sleep=list(sleep) if sleep else [7.0] * len(cycles),
            stress=list(stress) if stress else [3.0] * len(cycles),
            fasting=list(fasting) if fasting else [0] * len(cycles)
        )
        
        prediction_scaled = model.predict(X_input, verbose=0)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        predicted_cycle = round(float(prediction[0][0]))
        
        # Clamp to reasonable range
        predicted_cycle = max(21, min(45, predicted_cycle))
        
        # Calculate confidence based on data consistency
        cycle_std = np.std(cycles)
        confidence = max(0.3, min(0.95, 1.0 - (cycle_std / 10.0)))
        
        # Adjust confidence based on amount of data
        data_factor = min(1.0, len(cycles) / 6.0)
        confidence *= data_factor
        confidence = round(confidence, 2)
        
        return jsonify({
            'predicted_cycle_length': predicted_cycle,
            'confidence': confidence,
            'model_version': '2.0',
            'architecture': 'Functional API + Attention'
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'predicted_cycle_length': 28,
            'confidence': 0.3,
            'model_version': 'error_fallback'
        }), 200  # Still return 200 with fallback


if __name__ == '__main__':
    load_model()
    print("🚀 ML Service running on http://localhost:5001")
    print("   Endpoints: /health, /predict, /model-info")
    app.run(host='0.0.0.0', port=5001, debug=False)
