# --- Imports ---
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import logging
import numpy as np # Make sure numpy is imported
import time # Optional: for timing model load

# --- Setup Logging ---
# Use INFO level to see model loading messages etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration ---
# Start with gpt2 or distilgpt2 as they are smaller
# MODEL_NAME = "distilgpt2"
MODEL_NAME = "gpt2"
TOP_K_PREDICTIONS = 15 # How many top predictions to show

# <<< --- M1/MPS DEVICE SELECTION --- >>>
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logging.info("MPS device found, using MPS for acceleration.")
# Optional: Fallback check for CUDA if running elsewhere
# elif torch.cuda.is_available():
#     DEVICE = "cuda"
#     logging.info("CUDA device found, using CUDA for acceleration.")
else:
    DEVICE = "cpu"
    logging.info("MPS (or CUDA) not available, using CPU. Performance will be noticeably slower.")
# <<< --- END DEVICE SELECTION --- >>>

# --- Load Model and Tokenizer ---
# This happens only once when the Flask app starts
tokenizer = None
model = None
try:
    logging.info(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    logging.info(f"Loading model: {MODEL_NAME} to {DEVICE}...")
    t_start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval() # Set model to evaluation mode (important!)
    t_end_load = time.time()
    logging.info(f"Model loaded successfully to {DEVICE} in {t_end_load - t_start_load:.2f} seconds.")

except Exception as e:
    logging.error(f"CRITICAL ERROR: Failed to load model '{MODEL_NAME}' or tokenizer.", exc_info=True)
    # Depending on desired behavior, you might exit or run with limited functionality
    # For now, we'll let Flask start but predictions will fail if model is None
    pass # Allow app to start, but prediction endpoint will likely fail


# --- Helper Functions ---
def calculate_entropy(probabilities):
    """Calculates Shannon entropy for a probability distribution."""
    # Ensure input is numpy array for scipy
    # Add small epsilon BEFORE normalization to avoid log(0), then normalize
    epsilon = 1e-10
    probabilities_np = np.array(probabilities) + epsilon
    probabilities_np /= np.sum(probabilities_np) # Normalize to sum to 1
    return entropy(probabilities_np, base=2)

def get_next_token_predictions(text, top_k=10):
    """
    Gets the probability distribution, entropy, and top_k tokens for the next token.
    Uses the globally loaded tokenizer, model, and DEVICE.
    """
    # Check if model loaded correctly
    if model is None or tokenizer is None:
        raise RuntimeError("Model or Tokenizer not loaded successfully. Check initial loading logs.")

    logging.debug(f"Received text for prediction: '{text}'")
    try:
        # Handle empty input - predict from BOS token or empty string context
        # Note: Behavior might differ slightly based on model/tokenizer for empty string
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(DEVICE)

        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(**inputs)
            logits = outputs.logits

        # Get logits for the *last* token in the sequence
        next_token_logits = logits[0, -1, :]

        # Apply Softmax to get probabilities
        probabilities = F.softmax(next_token_logits, dim=-1)

        # --- Calculate Entropy ---
        # Move probabilities to CPU and convert to numpy for scipy
        probabilities_cpu_np = probabilities.cpu().numpy()
        entropy_value = calculate_entropy(probabilities_cpu_np)
        logging.debug(f"Calculated Entropy: {entropy_value:.4f}")


        # --- Get Top K Tokens ---
        top_k_prob, top_k_indices = torch.topk(probabilities, top_k)

        # Move indices and probabilities to CPU for decoding/listing
        top_k_indices_cpu = top_k_indices.cpu()
        top_k_probabilities_list = top_k_prob.cpu().numpy().tolist()

        # Decode tokens
        top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices_cpu]

        # Clean up token representation (replace special chars used by tokenizer)
        top_k_tokens_cleaned = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in top_k_tokens]
        logging.debug(f"Top {top_k} tokens: {top_k_tokens_cleaned}")
        logging.debug(f"Top {top_k} probabilities: {top_k_probabilities_list}")

        return {
            "entropy": float(entropy_value),
            "top_k_tokens": top_k_tokens_cleaned,
            "top_k_probabilities": top_k_probabilities_list
        }

    except Exception as e:
        logging.error(f"Error during prediction for text '{text}': {e}", exc_info=True)
        # Re-raise or return an error structure
        raise e # Let the API endpoint handle the exception


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    # Flask automatically looks for this file in the 'templates' folder
    return render_template('index.html')

@app.route('/get_entropy', methods=['POST'])
def get_entropy_api():
    """API endpoint to get entropy and top tokens."""
    try:
        # Check if model is ready before processing request
        if model is None or tokenizer is None:
            return jsonify({"error": "Model is not available. Check server logs."}), 503 # 503 Service Unavailable

        start_time = time.time()
        data = request.get_json()
        text = data.get('text', '')
        logging.info(f"API Request: Processing text: '{text[:50]}...'") # Log truncated text

        predictions = get_next_token_predictions(text, top_k=TOP_K_PREDICTIONS)

        end_time = time.time()
        logging.info(f"API Response: Entropy={predictions['entropy']:.4f}, TopToken='{predictions['top_k_tokens'][0]}', Time={end_time - start_time:.3f}s")
        return jsonify(predictions)

    except Exception as e:
        logging.error(f"API Error: Failed to process request: {e}", exc_info=True)
        # Provide a generic error message to the client
        return jsonify({"error": "An internal server error occurred during prediction."}), 500

# --- Main execution ---
if __name__ == '__main__':
    # Set use_reloader=False because model loading is expensive and
    # reloading on code changes during debug can be problematic/slow.
    # You'll need to manually stop (Ctrl+C) and restart the server after code changes.
    # host='0.0.0.0' makes it accessible from other devices on your network (optional)
    # Use 127.0.0.1 for local access only
    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)
