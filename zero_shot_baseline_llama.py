import os
import json
import re
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# =========================
# Config
# =========================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TRAIN_JSON = "train_data/subtask 1/train_data.json"
EVAL_SCRIPT = "evaluation_kit/task 1 & 3/evaluation_script.py"
OUT_DIR = "results_zero_shot_llama"
PRED_PATH = os.path.join(OUT_DIR, "predictions.json")
EVAL_OUT_PATH = os.path.join(OUT_DIR, "eval_results.json")
MAX_NEW_TOKENS = 10
BATCH_SIZE = 8

PROMPT_TEMPLATE = """You are a formal logic expert. Determine whether the following syllogism is logically valid or invalid.

A syllogism is VALID if the conclusion necessarily follows from the premises based on logical structure alone.
A syllogism is INVALID if the conclusion does not necessarily follow from the premises.

Ignore whether the statements are true in the real world. Focus only on logical form.

Syllogism: {syllogism}

Is this syllogism valid or invalid? Answer with one word only: valid or invalid.
Answer:"""


# =========================
# Utils
# =========================
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_validity(generated_text: str) -> bool:
    text = generated_text.strip().lower()
    if re.search(r'\binvalid\b', text):
        return False
    if re.search(r'\bvalid\b', text):
        return True
    if any(word in text for word in ['no', 'not', 'false', 'incorrect', 'wrong']):
        return False
    if any(word in text for word in ['yes', 'true', 'correct', 'right']):
        return True
    return False


# =========================
# Zero-Shot Prediction
# =========================
@torch.no_grad()
def zero_shot_predict(
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> List[Dict[str, Any]]:
    model.eval()
    predictions = []

    for start_idx in tqdm(range(0, len(items), batch_size), desc="Zero-shot inference"):
        batch_items = items[start_idx : start_idx + batch_size]
        prompts = [PROMPT_TEMPLATE.format(syllogism=item["syllogism"]) for item in batch_items]

        tokenizer.padding_side = "left"
        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        output_ids = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

        input_len = encodings["input_ids"].shape[1]
        for i, item in enumerate(batch_items):
            generated_tokens = output_ids[i][input_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            validity = parse_validity(generated_text)

            predictions.append({
                "id": item["id"],
                "validity": validity,
            })

    return predictions


# =========================
# Official Evaluation
# =========================
def run_official_eval(eval_script_path: str, ref_path: str, pred_path: str, out_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("semeval_eval", eval_script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.run_full_scoring(ref_path, pred_path, out_path)
    print("Official eval written to:", out_path)
    with open(out_path, "r") as f:
        print(json.load(f))


# =========================
# Main
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Load data
    items = load_json(TRAIN_JSON)
    print(f"Loaded {len(items)} items from {TRAIN_JSON}")

    # Run zero-shot predictions
    predictions = zero_shot_predict(model, tokenizer, items, device)

    # Save predictions
    os.makedirs(OUT_DIR, exist_ok=True)
    save_json(predictions, PRED_PATH)
    print(f"Saved {len(predictions)} predictions to {PRED_PATH}")

    # Quick accuracy check
    correct = sum(
        1 for item, pred in zip(items, predictions)
        if item["validity"] == pred["validity"]
    )
    print(f"Quick accuracy: {100.0 * correct / len(items):.2f}%")

    # Run official evaluation
    run_official_eval(EVAL_SCRIPT, TRAIN_JSON, PRED_PATH, EVAL_OUT_PATH)