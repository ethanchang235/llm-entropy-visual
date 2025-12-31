from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import TopPLogitsWarper
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import logging
import numpy as np
import time
from typing import Dict, List, Union, Tuple, Optional, Any

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Configuration ---
MODEL_NAME = "gpt2" 
MODEL_DISPLAY_NAME = "GPT-2 (Small)"

DEFAULT_TOP_K_PREDICTIONS = 15
MAX_TOP_K_PREDICTIONS = 50
DEFAULT_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 2.0
DEFAULT_TOP_P = 1.0 # 1.0 means effectively off
MIN_TOP_P = 0.01
MAX_TOP_P = 1.0

# --- Application State ---
model_state: Dict[str, Any] = {
    "model_name": MODEL_NAME,
    "model_display_name": MODEL_DISPLAY_NAME,
    "tokenizer": None,
    "model": None,
    "device": "cpu",
    "loaded": False,
    "loading_error": None,
    "is_loading": False,
}

def load_model_and_tokenizer() -> bool:
    """Loads the configured model and tokenizer."""
    global model_state
    
    if model_state["is_loading"]:
        logging.warning(f"Model loading already in progress for {model_state['model_name']}. Request ignored.")
        return False 

    logging.info(f"Attempting to load model: {model_state['model_name']}")
    model_state["is_loading"] = True
    model_state["loaded"] = False
    model_state["loading_error"] = None
    model_state["tokenizer"] = None
    model_state["model"] = None

    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model_state["device"] = device
        logging.info(f"Using device: {device}")

        logging.info(f"Loading tokenizer for: {model_state['model_name']}...")
        tokenizer = AutoTokenizer.from_pretrained(model_state['model_name'])
        
        logging.info(f"Loading model: {model_state['model_name']} to {device}...")
        t_start_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_state['model_name']).to(device)
        model.eval()
        t_end_load = time.time()

        model_state["tokenizer"] = tokenizer
        model_state["model"] = model
        model_state["loaded"] = True
        logging.info(f"Model '{model_state['model_name']}' loaded successfully to {device} in {t_end_load - t_start_load:.2f} seconds.")
        return True
    except Exception as e:
        model_state["loading_error"] = str(e)
        logging.error(f"CRITICAL ERROR: Failed to load model '{model_state['model_name']}' or tokenizer.", exc_info=True)
        return False
    finally:
        model_state["is_loading"] = False


# --- Helper Functions ---
def calculate_entropy_value(probabilities: np.ndarray) -> float:
    """Calculates Shannon entropy for a probability distribution."""
    epsilon = 1e-10
    probabilities_normalized = probabilities + epsilon
    probabilities_normalized /= np.sum(probabilities_normalized)
    return float(entropy(probabilities_normalized, base=2))

def get_next_token_predictions(text: str, top_k: int, temperature: float, top_p: float) -> Dict[str, Any]:
    """Generates next token predictions, probabilities, entropy, input tokenization, and surprise scores."""
    if not model_state["loaded"] or model_state["model"] is None or model_state["tokenizer"] is None:
        raise RuntimeError(f"Model '{model_state['model_name']}' not loaded. Error: {model_state['loading_error']}")

    model: AutoModelForCausalLM = model_state["model"]
    tokenizer: AutoTokenizer = model_state["tokenizer"]
    device: str = model_state["device"]

    temperature = max(MIN_TEMPERATURE, min(temperature, MAX_TEMPERATURE))
    top_p = max(MIN_TOP_P, min(top_p, MAX_TOP_P))
    applied_top_p = 1.0

    logging.debug(f"Predicting for: '{text[:50]}...', Top K: {top_k}, Temp: {temperature:.2f}, Top-P: {top_p:.2f}")

    input_tokens_display = []
    input_token_ids_display = []
    input_token_surprises = []  # Surprise score for each input token

    if text:
        tokenized_input_for_display = tokenizer(text, add_special_tokens=False)
        input_token_ids_display = tokenized_input_for_display.input_ids
        input_tokens_display = [tokenizer.decode([id_]) for id_ in input_token_ids_display]

    inputs_dict = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)

    if input_ids.shape[1] == 0:
        logging.warning("Tokenizer produced empty input_ids. Using BOS token.")
        if tokenizer.bos_token_id is not None:
            input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
            attention_mask = torch.tensor([[1]], dtype=torch.long).to(device)
        else:
            logging.error("Empty input, no BOS token. Cannot predict.")
            dummy_probs = np.ones(tokenizer.vocab_size) / tokenizer.vocab_size
            return {
                "entropy": calculate_entropy_value(dummy_probs),
                "top_k_tokens": [f"ERR_{i}" for i in range(top_k)],
                "top_k_probabilities": (np.ones(top_k) / top_k).tolist(),
                "input_tokens_display": [], "input_token_ids_display": [],
                "input_token_surprises": [],
                "error_message": "Empty input and no BOS token.",
                "applied_temperature": temperature,
                "applied_top_p": applied_top_p,
                "coverage_95": 0,
                "vocab_size": tokenizer.vocab_size
            }

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Calculate surprise scores for each input token (how surprised was the model?)
    # For token at position i, we look at logits[i-1] (prediction before seeing token i)
    if input_ids.shape[1] > 1:
        for i in range(1, input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            prev_logits = logits[0, i-1, :]
            prev_probs = F.softmax(prev_logits, dim=-1)
            token_prob = prev_probs[token_id].item()
            surprise = -np.log2(max(token_prob, 1e-10))  # bits
            input_token_surprises.append(round(surprise, 2))

    # Pad to match input_tokens_display length (first token has no "surprise" - it's the start)
    while len(input_token_surprises) < len(input_tokens_display):
        input_token_surprises.insert(0, 0.0)

    next_token_logits = logits[0, -1, :]

    # Apply temperature
    if temperature > 0:
        next_token_logits = next_token_logits / temperature

    # Apply Top-P (Nucleus) Sampling
    if 0.0 < top_p < 1.0:
        top_p_warper = TopPLogitsWarper(top_p=top_p, filter_value=-float("Inf"), min_tokens_to_keep=1)
        next_token_logits = top_p_warper(input_ids=None, scores=next_token_logits.unsqueeze(0)).squeeze(0)
        applied_top_p = top_p

    probabilities = F.softmax(next_token_logits, dim=-1)
    probabilities_cpu_np: np.ndarray = probabilities.cpu().numpy()
    entropy_val = calculate_entropy_value(probabilities_cpu_np)

    # Calculate 95% coverage: how many tokens needed to cover 95% of probability mass
    sorted_probs = np.sort(probabilities_cpu_np)[::-1]
    cumsum = np.cumsum(sorted_probs)
    coverage_95 = int(np.searchsorted(cumsum, 0.95) + 1)

    effective_vocab_size = torch.sum(probabilities > 0).item()
    actual_top_k = min(top_k, effective_vocab_size)
    if actual_top_k == 0 and effective_vocab_size > 0:
        actual_top_k = 1
    elif effective_vocab_size == 0:
        logging.error("Effective vocabulary size is 0 after filtering.")
        return {
            "entropy": 0.0, "top_k_tokens": ["ERR_NO_TOKENS"], "top_k_raw_tokens": ["ERR_NO_TOKENS"],
            "top_k_probabilities": [1.0], "input_tokens_display": input_tokens_display,
            "input_token_ids_display": input_token_ids_display, "input_token_surprises": input_token_surprises,
            "applied_temperature": temperature, "applied_top_p": applied_top_p,
            "error_message": "No tokens remained after Top-P/Temp filtering.",
            "coverage_95": 0, "vocab_size": tokenizer.vocab_size
        }

    top_k_prob_tensor, top_k_indices_tensor = torch.topk(probabilities, actual_top_k)
    top_k_tokens_decoded: List[str] = [tokenizer.decode(idx.item()) for idx in top_k_indices_tensor.cpu()]
    top_k_tokens_cleaned: List[str] = [
        token.replace('Ġ', ' ').replace('Ċ', '\n') if isinstance(token, str) else token
        for token in top_k_tokens_decoded
    ]
    top_k_probabilities_list: List[float] = top_k_prob_tensor.cpu().numpy().tolist()

    return {
        "entropy": entropy_val,
        "top_k_tokens": top_k_tokens_cleaned,
        "top_k_raw_tokens": top_k_tokens_decoded,
        "top_k_probabilities": top_k_probabilities_list,
        "input_tokens_display": input_tokens_display,
        "input_token_ids_display": input_token_ids_display,
        "input_token_surprises": input_token_surprises,
        "applied_temperature": temperature,
        "applied_top_p": applied_top_p,
        "coverage_95": coverage_95,
        "vocab_size": tokenizer.vocab_size
    }

# --- Flask Routes ---
@app.route('/')
def index_page():
    return render_template('index.html', model_display_name=model_state['model_display_name'])

@app.route('/status', methods=['GET'])
def get_status():
    if model_state["is_loading"]:
        return jsonify({
            "status": "loading_model",
            "model_name": model_state["model_display_name"],
            "message": f"Model '{model_state['model_display_name']}' is currently loading..."
        })
    elif model_state["loaded"]:
        return jsonify({
            "status": "ready",
            "model_name": model_state["model_display_name"],
            "device": model_state["device"]
        })
    elif model_state["loading_error"]:
        return jsonify({
            "status": "error",
            "message": f"Model loading failed for '{model_state['model_display_name']}': {model_state['loading_error']}",
            "model_name": model_state["model_display_name"]
        }), 500
    else: 
        return jsonify({"status": "initializing", "message": "Server is initializing."})


@app.route('/get_entropy', methods=['POST'])
def get_entropy_api():
    if model_state["is_loading"]:
        return jsonify({"error": f"Model '{model_state['model_display_name']}' is loading. Please wait."}), 503
    if not model_state["loaded"]:
        err_msg = model_state["loading_error"] or f"Model '{model_state['model_display_name']}' failed to load."
        return jsonify({"error": err_msg, "details": "The language model is not available."}), 503

    try:
        start_time = time.time()
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        text = data.get('text', '')

        def parse_int(val, default, min_val, max_val):
            try:
                v = int(val)
                return v if min_val <= v <= max_val else default
            except (ValueError, TypeError):
                return default

        def parse_float(val, default, min_val, max_val):
            try:
                return max(min_val, min(float(val), max_val))
            except (ValueError, TypeError):
                return default

        top_k = parse_int(data.get('top_k'), DEFAULT_TOP_K_PREDICTIONS, 1, MAX_TOP_K_PREDICTIONS)
        temperature = parse_float(data.get('temperature'), DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE)
        top_p = parse_float(data.get('top_p'), DEFAULT_TOP_P, MIN_TOP_P, MAX_TOP_P)

        logging.info(f"API Request: Text (len {len(text)}), TopK: {top_k}, Temp: {temperature:.2f}, TopP: {top_p:.2f}, Model: {model_state['model_name']}")
        predictions = get_next_token_predictions(text, top_k=top_k, temperature=temperature, top_p=top_p)
        end_time = time.time()

        if "error_message" in predictions:
             logging.warning(f"Prediction error: {predictions['error_message']}")
        else:
            logging.info(f"API Response: Entropy={predictions['entropy']:.4f}, TopTok='{predictions['top_k_tokens'][0] if predictions['top_k_tokens'] else 'N/A'}', Time={end_time - start_time:.3f}s")
        
        return jsonify(predictions)

    except RuntimeError as e:
        logging.error(f"API Error (Runtime): {e}", exc_info=True)
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logging.error(f"API Error (General): {e}", exc_info=True)
        return jsonify({"error": "Internal server error during prediction."}), 500

# --- Main execution ---
if __name__ == '__main__':
    print("Starting Flask app...")
    initial_load_success = load_model_and_tokenizer() 
    if not initial_load_success:
        print(f"CRITICAL: Initial model load for {MODEL_NAME} failed. Check logs. The app will run but predictions will fail.")
    else:
        print(f"Initial model {MODEL_NAME} loaded. Server is ready.")

    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)