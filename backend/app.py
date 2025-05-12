# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- CORS Configuration ---
# Replace 'YOUR_GITHUB_USERNAME' and 'YOUR_REPOSITORY_NAME'
# This allows your GitHub Pages site and localhost (for testing) to make requests.
YOUR_GITHUB_USERNAME = "your_github_username" # CHANGE THIS
YOUR_REPOSITORY_NAME = "entropy-visual" # CHANGE THIS (or your repo name)

origins = [
    "http://localhost:8000",  # For local frontend testing
    "http://127.0.0.1:8000", # For local frontend testing
    f"https://{YOUR_GITHUB_USERNAME}.github.io", # Allow base and repo-specific
]
if YOUR_REPOSITORY_NAME: # Add specific repo page if name is set
    origins.append(f"https://{YOUR_GITHUB_USERNAME}.github.io/{YOUR_REPOSITORY_NAME}")

CORS(app, resources={r"/get_entropy": {"origins": origins}})
# For simpler initial testing, you can do CORS(app) to allow all,
# but tighten it for actual deployment.
# CORS(app)

# --- Model Configuration ---
MODEL_NAME = "gpt2"
TOP_K_PREDICTIONS = 15

if torch.backends.mps.is_available():
    DEVICE = "mps"
    logging.info("MPS device found, using MPS for acceleration.")
else:
    DEVICE = "cpu"
    logging.info("MPS not available, using CPU. Performance will be noticeably slower.")

tokenizer = None
model = None
try:
    logging.info(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info(f"Tokenizer loaded.")
    logging.info(f"Loading model: {MODEL_NAME} to {DEVICE}...")
    t_start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    t_end_load = time.time()
    logging.info(f"Model loaded to {DEVICE} in {t_end_load - t_start_load:.2f} seconds.")
except Exception as e:
    logging.error(f"CRITICAL ERROR: Failed to load model '{MODEL_NAME}' or tokenizer.", exc_info=True)

def calculate_entropy(probabilities):
    epsilon = 1e-10
    probabilities_np = np.array(probabilities) + epsilon
    probabilities_np /= np.sum(probabilities_np)
    return entropy(probabilities_np, base=2)

def get_next_token_predictions(text, top_k=10):
    if model is None or tokenizer is None:
        raise RuntimeError("Model or Tokenizer not loaded. Check server logs.")
    try:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        probabilities = F.softmax(next_token_logits, dim=-1)
        probabilities_cpu_np = probabilities.cpu().numpy()
        entropy_value = calculate_entropy(probabilities_cpu_np)
        top_k_prob, top_k_indices = torch.topk(probabilities, top_k)
        top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices.cpu()]
        top_k_tokens_cleaned = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in top_k_tokens]
        return {
            "entropy": float(entropy_value),
            "top_k_tokens": top_k_tokens_cleaned,
            "top_k_probabilities": top_k_prob.cpu().numpy().tolist()
        }
    except Exception as e:
        logging.error(f"Error during prediction for text '{text}': {e}", exc_info=True)
        raise e

@app.route('/get_entropy', methods=['POST', 'OPTIONS']) # Add OPTIONS for preflight requests
def get_entropy_api():
    # The flask-cors extension handles OPTIONS automatically, but good to be explicit.
    if request.method == 'OPTIONS':
         return jsonify({'message': 'CORS preflight'}), 200 # Or just let flask-cors handle

    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not available. Check server logs."}), 503
    try:
        start_time = time.time()
        data = request.get_json()
        text = data.get('text', '')
        logging.info(f"API Request: Processing text: '{text[:50]}...'")
        predictions = get_next_token_predictions(text, top_k=TOP_K_PREDICTIONS)
        end_time = time.time()
        logging.info(f"API Response: Entropy={predictions['entropy']:.4f}, TopToken='{predictions['top_k_tokens'][0]}', Time={end_time - start_time:.3f}s")
        return jsonify(predictions)
    except Exception as e:
        logging.error(f"API Error: Failed to process request: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# This __main__ block is for local testing of the backend ONLY.
# When deploying, a WSGI server like gunicorn will import 'app' from app.py.
if __name__ == '__main__':
    # For local testing, run on a different port than your frontend test server if needed.
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)