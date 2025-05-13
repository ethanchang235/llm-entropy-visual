# app.py (in the root of entropy-visual/)

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import logging
import numpy as np
import time
from typing import Dict, List, Union, Tuple, Optional

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration ---
DEFAULT_MODEL_NAME = "gpt2" # Default model if not overridden by env var or other config
DEFAULT_TOP_K_PREDICTIONS = 15
MAX_TOP_K_PREDICTIONS = 50 # Safety limit

# --- Application State ---
model_state: Dict[str, Union[str, bool, Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]] = {
    "model_name": DEFAULT_MODEL_NAME,
    "tokenizer": None,
    "model": None,
    "device": "cpu",
    "loaded": False,
    "loading_error": None
}

def load_model_and_tokenizer():
    """Loads the model and tokenizer based on model_state['model_name']."""
    global model_state
    model_name = model_state["model_name"]
    try:
        # --- M1/MPS DEVICE SELECTION ---
        if torch.backends.mps.is_available():
            device = "mps"
            logging.info("MPS device found, using MPS for acceleration.")
        elif torch.cuda.is_available():
            device = "cuda"
            logging.info("CUDA device found, using CUDA for acceleration.")
        else:
            device = "cpu"
            logging.info("Neither MPS nor CUDA available, using CPU. Performance will be noticeably slower.")
        model_state["device"] = device

        logging.info(f"Loading tokenizer: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

        logging.info(f"Loading model: {model_name} to {device}...")
        t_start_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval() # Set model to evaluation mode
        t_end_load = time.time()

        model_state["tokenizer"] = tokenizer
        model_state["model"] = model
        model_state["loaded"] = True
        model_state["loading_error"] = None
        logging.info(f"Model '{model_name}' loaded successfully to {device} in {t_end_load - t_start_load:.2f} seconds.")

    except Exception as e:
        model_state["loaded"] = False
        model_state["loading_error"] = str(e)
        logging.error(f"CRITICAL ERROR: Failed to load model '{model_name}' or tokenizer.", exc_info=True)

# --- Helper Functions ---
def calculate_entropy_value(probabilities: np.ndarray) -> float:
    """Calculates Shannon entropy for a probability distribution."""
    epsilon = 1e-10 # To avoid log(0)
    # Ensure probabilities sum to 1 (approximately)
    probabilities_normalized = probabilities + epsilon
    probabilities_normalized /= np.sum(probabilities_normalized)
    return float(entropy(probabilities_normalized, base=2))

def get_next_token_predictions(text: str, top_k: int) -> Dict[str, Union[float, List[str], List[float], str]]:
    """
    Generates next token predictions, probabilities, and entropy.
    """
    if not model_state["loaded"] or model_state["model"] is None or model_state["tokenizer"] is None:
        raise RuntimeError(f"Model or Tokenizer not loaded successfully. Error: {model_state['loading_error']}")

    model: AutoModelForCausalLM = model_state["model"]
    tokenizer: AutoTokenizer = model_state["tokenizer"]
    device: str = model_state["device"]

    logging.debug(f"Received text for prediction: '{text[:100]}...' (Top K: {top_k})")
    try:
        # For GPT-2, add_special_tokens=True will add the EOS token (50256) if text is empty,
        # which acts as a BOS token to predict the first token of a sequence.
        inputs_dict = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs_dict.input_ids.to(device)
        attention_mask = inputs_dict.attention_mask.to(device)

        # Handle cases where tokenizer might yield empty IDs, although rare with add_special_tokens=True
        if input_ids.shape[1] == 0:
            logging.warning("Tokenizer produced empty input_ids despite add_special_tokens=True. Attempting to use BOS token.")
            if tokenizer.bos_token_id is not None:
                input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
                attention_mask = torch.tensor([[1]], dtype=torch.long).to(device)
            else:
                logging.error("Cannot proceed: Tokenizer produced empty input_ids and no BOS token is defined.")
                # Return a high-entropy, uniform distribution over a small part of vocab as a fallback
                dummy_probs = np.ones(tokenizer.vocab_size) / tokenizer.vocab_size
                dummy_entropy = calculate_entropy_value(dummy_probs)
                return {
                    "entropy": dummy_entropy,
                    "top_k_tokens": [f"ERR_TOKEN_{i}" for i in range(top_k)],
                    "top_k_probabilities": (np.ones(top_k) / top_k).tolist(),
                    "error_message": "Empty input and no BOS token."
                }

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get logits for the last token in the sequence
        next_token_logits = logits[0, -1, :]
        probabilities = F.softmax(next_token_logits, dim=-1)
        
        probabilities_cpu_np: np.ndarray = probabilities.cpu().numpy()
        entropy_val = calculate_entropy_value(probabilities_cpu_np)

        # Get top K tokens and their probabilities
        top_k_prob_tensor, top_k_indices_tensor = torch.topk(probabilities, top_k)
        
        top_k_tokens_decoded: List[str] = [tokenizer.decode(idx.item()) for idx in top_k_indices_tensor.cpu()]
        
        # Clean tokens for better display (specific to BPE-based tokenizers like GPT-2)
        # 'Ġ' often represents a space at the beginning of a word.
        # 'Ċ' can represent a newline.
        top_k_tokens_cleaned: List[str] = [
            token.replace('Ġ', ' ').replace('Ċ', '\n') for token in top_k_tokens_decoded
        ]
        
        top_k_probabilities_list: List[float] = top_k_prob_tensor.cpu().numpy().tolist()

        return {
            "entropy": entropy_val,
            "top_k_tokens": top_k_tokens_cleaned,
            "top_k_probabilities": top_k_probabilities_list
        }
    except Exception as e:
        logging.error(f"Error during prediction for text '{text[:50]}...': {e}", exc_info=True)
        # Re-raise to be caught by the API endpoint handler
        raise

# --- Flask Routes ---
@app.route('/')
def index_page():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def get_status():
    """Reports the loading status of the model."""
    if model_state["loaded"]:
        return jsonify({
            "status": "ready",
            "model_name": model_state["model_name"],
            "device": model_state["device"]
        })
    elif model_state["loading_error"]:
        return jsonify({
            "status": "error",
            "message": f"Model loading failed: {model_state['loading_error']}",
            "model_name": model_state["model_name"]
        }), 500
    else:
        return jsonify({
            "status": "loading",
            "model_name": model_state["model_name"]
        })


@app.route('/get_entropy', methods=['POST'])
def get_entropy_api():
    """API endpoint to get entropy and token predictions."""
    if not model_state["loaded"]:
        error_msg = model_state["loading_error"] or "Model is still loading or failed to load."
        return jsonify({"error": error_msg, "details": "The language model is not available. Please check server logs or wait."}), 503

    try:
        start_time = time.time()
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        text = data.get('text', '') # Default to empty string if not provided
        top_k_req = data.get('top_k', DEFAULT_TOP_K_PREDICTIONS)

        try:
            top_k = int(top_k_req)
            if not (1 <= top_k <= MAX_TOP_K_PREDICTIONS):
                top_k = DEFAULT_TOP_K_PREDICTIONS
                logging.warning(f"Requested top_k ({top_k_req}) out of range. Using default: {top_k}")
        except ValueError:
            top_k = DEFAULT_TOP_K_PREDICTIONS
            logging.warning(f"Invalid top_k value '{top_k_req}'. Using default: {top_k}")

        logging.info(f"API Request: Processing text (len {len(text)}), Top K: {top_k}")
        
        predictions = get_next_token_predictions(text, top_k=top_k)
        
        end_time = time.time()
        if "error_message" in predictions: # Handle specific errors from get_next_token_predictions
             logging.warning(f"Prediction resulted in an error state: {predictions['error_message']}")
        else:
            logging.info(f"API Response: Entropy={predictions['entropy']:.4f}, TopToken='{predictions['top_k_tokens'][0]}', Time={end_time - start_time:.3f}s")
        
        return jsonify(predictions)

    except RuntimeError as e: # Catch errors from get_next_token_predictions if model wasn't loaded
        logging.error(f"API Error (Runtime): {e}", exc_info=True)
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logging.error(f"API Error (General): Failed to process request: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during prediction."}), 500

# --- Main execution ---
if __name__ == '__main__':
    print("Starting Flask app... Attempting to load model. This may take a moment.")
    load_model_and_tokenizer() # Load model on startup
    # use_reloader=False is important because model loading is expensive.
    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)