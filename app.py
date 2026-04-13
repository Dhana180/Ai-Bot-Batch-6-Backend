import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)

# ✅ CORS (allow all for now — can restrict later)
CORS(app)

# ✅ Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model Paths ─────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OHE_PATH = os.path.join(MODEL_DIR, 'ohe.pkl')
RF_PATH = os.path.join(MODEL_DIR, 'rfmodel.pkl')

# ── Load Models Safely ─────────────────────────
def load_models():
    try:
        with open(OHE_PATH, 'rb') as f:
            ohe = pickle.load(f)

        with open(RF_PATH, 'rb') as f:
            rfmodel = pickle.load(f)

        print("✅ Models loaded successfully")
        return ohe, rfmodel

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

ohe, rfmodel = load_models()

# ── ROOT ROUTE (Fix Not Found) ─────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Bot Detection API Running 🚀",
        "status": "success"
    })

# ── HEALTH CHECK ─────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": ohe is not None and rfmodel is not None
    }), 200

# ── PREDICTION ─────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    if ohe is None or rfmodel is None:
        return jsonify({"error": "Models not loaded"}), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input provided"}), 400

        # Required fields
        required_fields = [
            'http_method',
            'endpoint',
            'requests_per_session',
            'login_attempts',
            'time_between_requests'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # ── Numerical Features ──
        num_features = {
            'status_code': 200.0,
            'response_size': 1500.0,
            'session_duration': 120.0,
            'requests_per_session': float(data['requests_per_session']),
            'time_between_requests': float(data['time_between_requests']),
            'failed_requests': float(data.get('failed_requests', 0.0)),
            'url_length': float(data.get('url_length', 25.0)),
            'query_param_count': float(data.get('query_param_count', 0.0)),
            'payload_size': float(data.get('payload_size', 500.0)),
            'distinct_endpoints_accessed': float(data.get('distinct_endpoints_accessed', 3.0)),
            'login_attempts': float(data['login_attempts']),
            'request_pattern_entropy': float(data.get('request_pattern_entropy', 0.7)),
        }

        df_num = pd.DataFrame([num_features])

        # ── Categorical Encoding ──
        df_cat = pd.DataFrame({
            'http_method': [str(data['http_method']).lower()],
            'endpoint': [str(data['endpoint']).lower()]
        })

        ohe_encoded = ohe.transform(df_cat)

        if hasattr(ohe_encoded, 'toarray'):
            ohe_encoded = ohe_encoded.toarray()

        ohe_df = pd.DataFrame(
            ohe_encoded,
            columns=ohe.get_feature_names_out()
        )

        # ── Combine Features ──
        X_eval = pd.concat([df_num, ohe_df], axis=1)

        # Align with training features
        if hasattr(rfmodel, 'feature_names_in_'):
            for col in rfmodel.feature_names_in_:
                if col not in X_eval.columns:
                    X_eval[col] = 0.0

            X_eval = X_eval[rfmodel.feature_names_in_]

        # ── Prediction ──
        probs = rfmodel.predict_proba(X_eval)[0]
        bot_prob = float(probs[1])

        threshold = float(data.get('threshold', 0.50))
        is_bot = bot_prob > threshold

        return jsonify({
            "result": "Bot" if is_bot else "Human",
            "status": "Bot" if is_bot else "Human",
            "action": "Apply CAPTCHA or Block" if is_bot else "Allow Access",
            "bot_probability": round(bot_prob, 4),
            "confidence": round(float(max(probs)), 4),
            "recommendation": "Apply CAPTCHA or Block" if is_bot else "Allow Access",
            "threshold_used": threshold,
            "timestamp": datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── RUN (FOR LOCAL + DEPLOYMENT SAFE) ─────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
