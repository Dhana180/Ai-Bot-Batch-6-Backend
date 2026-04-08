import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
OHE_PATH = os.path.join(MODEL_DIR, 'ohe.pkl')
RF_PATH = os.path.join(MODEL_DIR, 'rfmodel.pkl')

print("Generating synthetic models with proper OHE encoding...")

base_cols = [
    'status_code', 'response_size', 'session_duration', 'requests_per_session',
    'time_between_requests', 'failed_requests', 'url_length', 'query_param_count',
    'payload_size', 'distinct_endpoints_accessed', 'login_attempts', 'request_pattern_entropy'
]

# --- Fit OHE on known categories ---
df_cat_mock = pd.DataFrame({
    'http_method': ['get', 'post', 'put', 'delete', 'get', 'post'],
    'endpoint': ['/home', '/login', '/api/data', '/api/cart', '/api/checkout', '/admin']
})
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df_cat_mock)

# --- Generate N samples ---
np.random.seed(42)
N = 1000
y_mock = np.random.randint(0, 2, N)  # 0 = human, 1 = bot

# --- Numerical features with clear bot/human boundaries ---
num_data = {
    'status_code':                  np.where(y_mock == 1, np.random.uniform(400, 503, N), np.random.uniform(200, 302, N)),
    'response_size':                np.where(y_mock == 1, np.random.uniform(100, 500, N), np.random.uniform(500, 5000, N)),
    'session_duration':             np.where(y_mock == 1, np.random.uniform(1, 30, N),    np.random.uniform(60, 600, N)),
    'requests_per_session':         np.where(y_mock == 1, np.random.uniform(30, 200, N),  np.random.uniform(1, 25, N)),
    'time_between_requests':        np.where(y_mock == 1, np.random.uniform(0.01, 0.8, N), np.random.uniform(1.5, 10.0, N)),
    'failed_requests':              np.where(y_mock == 1, np.random.uniform(5, 50, N),    np.random.uniform(0, 2, N)),
    'url_length':                   np.where(y_mock == 1, np.random.uniform(60, 200, N),  np.random.uniform(10, 50, N)),
    'query_param_count':            np.where(y_mock == 1, np.random.uniform(5, 20, N),    np.random.uniform(0, 3, N)),
    'payload_size':                 np.where(y_mock == 1, np.random.uniform(0, 50, N),    np.random.uniform(100, 2000, N)),
    'distinct_endpoints_accessed':  np.where(y_mock == 1, np.random.uniform(10, 50, N),   np.random.uniform(1, 8, N)),
    'login_attempts':               np.where(y_mock == 1, np.random.uniform(5, 50, N),    np.random.uniform(0, 4, N)),
    'request_pattern_entropy':      np.where(y_mock == 1, np.random.uniform(0.01, 0.3, N), np.random.uniform(0.6, 1.0, N)),
}
df_num = pd.DataFrame(num_data)

# --- Proper categorical features → OHE (binary, not random floats) ---
# BUG FIX: Previously X_mock used np.random.rand() for ALL columns including OHE columns.
# OHE columns must be binary (0/1), not continuous floats — this mismatch broke predictions.
http_methods = np.random.choice(['get', 'post', 'put', 'delete'], N)
endpoints = np.random.choice(['/home', '/login', '/api/data', '/api/cart', '/api/checkout', '/admin'], N)
df_cat_all = pd.DataFrame({'http_method': http_methods, 'endpoint': endpoints})
ohe_encoded = ohe.transform(df_cat_all)  # proper binary 0/1
ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out())

# --- Combine and train ---
X_mock = pd.concat([df_num, ohe_df], axis=1)
features_in = list(X_mock.columns)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_mock, y_mock)
rf.feature_names_in_ = np.array(features_in)

# Quick sanity check
from sklearn.metrics import accuracy_score
preds = rf.predict(X_mock)
print(f"Training accuracy: {accuracy_score(y_mock, preds):.3f}  (should be high — signals model learned correctly)")

# Save models
with open(OHE_PATH, 'wb') as f:
    pickle.dump(ohe, f)
with open(RF_PATH, 'wb') as f:
    pickle.dump(rf, f)

print(f"Saved OHE  → {OHE_PATH}")
print(f"Saved RF   → {RF_PATH}")
print(f"Feature count: {len(features_in)}")