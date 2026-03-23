import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from preprocess import preprocess_for_prediction

app = Flask(__name__)
CORS(app)

# Load model on startup
model = None
MODEL_PATH = 'model/lstm_model.h5'

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            model = None
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}. Run train.py first.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
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
            'model_version': '1.0'
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
    app.run(host='0.0.0.0', port=5001, debug=False)
