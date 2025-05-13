# app.py (in the root of entropy-visual/)

from flask import Flask, request, jsonify, render_template # Ensure render_template is here
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import logging
import numpy as np
import time

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__) # Flask app instance

# --- Configuration ---
MODEL_NAME = "gpt2"
TOP_K_PREDICTIONS = 15

# --- M1/MPS DEVICE SELECTION ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logging.info("MPS device found, using MPS for acceleration.")
else:
    DEVICE = "cpu"
    logging.info("MPS (or CUDA) not available, using CPU. Performance will be noticeably slower.")

# --- Load Model and Tokenizer ---
tokenizer = None
model = None
try:
    logging.info(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    logging.info(f"Loading model: {MODEL_NAME} to {DEVICE}...")
    t_start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    t_end_load = time.time()
    logging.info(f"Model loaded successfully to {DEVICE} in {t_end_load - t_start_load:.2f} seconds.")
except Exception as e:
    logging.error(f"CRITICAL ERROR: Failed to load model '{MODEL_NAME}' or tokenizer.", exc_info=True)
    pass

# --- Helper Functions ---
def calculate_entropy(probabilities):
    epsilon = 1e-10
    probabilities_np = np.array(probabilities) + epsilon
    probabilities_np /= np.sum(probabilities_np)
    return entropy(probabilities_np, base=2)

def get_next_token_predictions(text, top_k=10):
    if model is None or tokenizer is None:
        raise RuntimeError("Model or Tokenizer not loaded successfully. Check initial loading logs.")
    logging.debug(f"Received text for prediction: '{text}'")
    try:
        # Tokenize the input
        # Ensure add_special_tokens=True, which for GPT2 usually adds a BOS token.
        inputs_dict = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs_dict.input_ids.to(DEVICE)
        attention_mask = inputs_dict.attention_mask.to(DEVICE)


        # Handle case where tokenizer might still produce empty input_ids for an empty string
        # even with add_special_tokens=True (less common but possible with some configurations or future tokenizer changes)
        # A more direct check is if the sequence length dimension is 0.
        if input_ids.shape[1] == 0:
            logging.warning("Tokenizer produced empty input_ids even with add_special_tokens. Using BOS token only.")
            # Fallback: explicitly tokenize the BOS token if available
            if tokenizer.bos_token_id is not None:
                input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(DEVICE)
                attention_mask = torch.tensor([[1]], dtype=torch.long).to(DEVICE) # Attention mask for the BOS token
            else:
                # If no BOS token, we can't really proceed meaningfully with an empty sequence for most Causal LMs.
                # Return a default high-entropy or zero-prediction state.
                logging.error("Tokenizer produced empty input_ids and no BOS token is defined. Cannot predict.")
                # Create a dummy high entropy distribution over a small part of vocab for visualization
                vocab_size = tokenizer.vocab_size
                dummy_probs = np.ones(vocab_size) / vocab_size
                dummy_entropy = calculate_entropy(dummy_probs)
                dummy_top_tokens = [f"token_{i}" for i in range(top_k)]
                dummy_top_probs = (np.ones(top_k) / top_k).tolist()
                return {
                    "entropy": float(dummy_entropy),
                    "top_k_tokens": dummy_top_tokens,
                    "top_k_probabilities": dummy_top_probs,
                    "error": "Empty input and no BOS token."
                }

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) # Pass attention_mask
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

# --- Flask Routes ---
@app.route('/') # This route serves your HTML frontend
def index_page(): # Renamed to avoid conflict if you had 'index' elsewhere
    """Serves the main HTML page from the templates folder."""
    return render_template('index.html')

@app.route('/get_entropy', methods=['POST']) # API endpoint
def get_entropy_api():
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
        return jsonify({"error": "An internal server error occurred during prediction."}), 500

# --- Main execution ---
if __name__ == '__main__':
    # use_reloader=False is good because model loading is expensive.
    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)