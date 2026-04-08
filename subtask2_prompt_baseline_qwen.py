import os
import json
import re
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


# ============================================================
# Config
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 80
MAX_LEN = 768
PRINT_RAW_MODEL_OUTPUT = True

TEST_JSON = "test_data/subtask 2/test_data_subtask_2.json"
OUT_DIR = "results_subtask2_prompt_baseline"
PRED_PATH = os.path.join(OUT_DIR, "predictions.json")
EVAL_OUT_PATH = os.path.join(OUT_DIR, "eval_results.json")


# ============================================================
# Device
# ============================================================
def get_device() -> str:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            print("CUDA appears unavailable at runtime; using CPU.")
            return "cpu"
    return "cpu"


# ============================================================
# IO helpers
# ============================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# Parsing syllogism into premises and conclusion
# ============================================================
def parse_syllogism_to_premises_and_conclusion(syllogism_text: str) -> Tuple[List[str], str]:
    sentences = [s.strip() for s in syllogism_text.split(".") if s.strip()]
    conclusion_markers = [
        "therefore",
        "consequently",
        "it follows",
        "this has led",
        "implies that",
    ]

    conclusion = None
    premises = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(marker in sent_lower for marker in conclusion_markers):
            conclusion = sent
        else:
            premises.append(sent)

    if conclusion is None and premises:
        conclusion = premises.pop()

    if conclusion is None:
        conclusion = ""

    return premises, conclusion


# ============================================================
# Prompting + output parsing
# ============================================================
def build_prompt(premises: List[str], conclusion: str) -> str:
    premise_lines = "\n".join([f"{i}. {p}" for i, p in enumerate(premises)])
    return (
        "You are a formal logic assistant.\n"
        "Given numbered premises and a conclusion, return ONLY the indices of premises that are necessary and sufficient to entail the conclusion.\n"
        "If no subset entails the conclusion, return an empty list.\n"
        "Output format must be exactly JSON with this schema:\n"
        '{"relevant_premises": [indices]}\n\n'
        f"Premises:\n{premise_lines}\n\n"
        f"Conclusion:\n{conclusion}\n"
        "Answer:\n"
    )


def parse_relevant_premises(generated_text: str, num_premises: int) -> List[int]:
    def _clean_indices(indices) -> List[int]:
        cleaned = []
        for x in indices:
            if isinstance(x, int) and 0 <= x < num_premises:
                cleaned.append(x)
        return sorted(list(set(cleaned)))

    # Try to decode any JSON object present in the text and use the first one
    # that contains the required key.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", generated_text):
        start = match.start()
        try:
            parsed, _ = decoder.raw_decode(generated_text[start:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "relevant_premises" in parsed:
            return _clean_indices(parsed.get("relevant_premises", []))

    # Fallback: only parse numbers from an explicit relevant_premises list.
    list_match = re.search(r'"relevant_premises"\s*:\s*\[([^\]]*)\]', generated_text)
    if list_match:
        found = re.findall(r"(?<!\d)(\d+)(?!\d)", list_match.group(1))
        return _clean_indices([int(tok) for tok in found])

    # No reliable structured output found.
    return []


@torch.no_grad()
def predict_relevant_premises(model, tokenizer, item: Dict[str, Any], device: str) -> List[int]:
    premises, conclusion = parse_syllogism_to_premises_and_conclusion(item["syllogism"])
    if not premises or not conclusion:
        return []

    prompt = build_prompt(premises, conclusion)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)

    out_ids = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = enc["input_ids"].shape[1]
    gen_tokens = out_ids[0][input_len:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    if PRINT_RAW_MODEL_OUTPUT:
        item_id = item.get("id", "unknown")
        print(f"\n[Raw model output | id={item_id}]\n{gen_text}\n")

    return parse_relevant_premises(gen_text, len(premises))


@torch.no_grad()
def run_prompt_baseline(
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    device: str,
) -> List[Dict[str, Any]]:
    predictions = []

    for item in tqdm(items, desc="Prompt baseline"):
        rel = predict_relevant_premises(model, tokenizer, item, device)
        predictions.append({
            "id": item["id"],
            "relevant_premises": rel,
        })

    return predictions


# ============================================================
# F1 (premise selection only)
# ============================================================
def calculate_f1_premises(ground_truth_list, predictions) -> float:
    gt_map = {item["id"]: item for item in ground_truth_list}

    total_precision = 0.0
    total_recall = 0.0
    valid_count = 0

    for pred_item in predictions:
        item_id = pred_item["id"]
        if item_id not in gt_map:
            continue

        gt_item = gt_map[item_id]
        true_positives = set(gt_item.get("relevant_premises", []))
        predicted_positives = set(pred_item.get("relevant_premises", []))

        tp = len(true_positives.intersection(predicted_positives))
        fp = len(predicted_positives.difference(true_positives))
        fn = len(true_positives.difference(predicted_positives))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        total_precision += precision
        total_recall += recall
        valid_count += 1

    if valid_count == 0:
        return 0.0

    macro_precision = total_precision / valid_count
    macro_recall = total_recall / valid_count

    if (macro_precision + macro_recall) == 0:
        return 0.0

    return 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) * 100


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading data: {TEST_JSON}")
    items = load_json(TEST_JSON)
    print(f"Total samples: {len(items)}")

    predictions = run_prompt_baseline(model, tokenizer, items, device)
    save_json(predictions, PRED_PATH)
    print(f"Saved predictions to: {PRED_PATH}")

    has_labels = len(items) > 0 and "relevant_premises" in items[0]
    if has_labels:
        f1 = calculate_f1_premises(items, predictions)
        eval_results = {"f1_premises": round(f1, 4)}
        save_json(eval_results, EVAL_OUT_PATH)
        print(f"Premise-selection F1: {f1:.4f}")
        print(f"Saved eval results to: {EVAL_OUT_PATH}")
    else:
        print("No gold relevant_premises found in input file; skipped F1 evaluation.")
