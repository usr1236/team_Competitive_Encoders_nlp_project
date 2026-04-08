import os
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Config
# ============================================================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 256

# Selection hyperparameters
# Largest-gap selection with a minimum floor.
MIN_SIMILARITY_THRESHOLD = 0.70

TEST_JSON = "test_data/subtask 2/test_data_subtask_2.json"
OUT_DIR = "results_subtask2_minilm"
OUT_PATH = os.path.join(OUT_DIR, "predictions.json")
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


DEVICE = get_device()


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


def calculate_f1_premises(ground_truth_list, predictions):
    """
    Compute macro-averaged premise-selection F1 over all examples that
    exist in both ground truth and predictions.

    For empty-label cases, use the standard convention:
    - precision = 1.0 if there are no predicted positives
    - recall = 1.0 if there are no gold positives
    """
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
    return (
        2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    ) * 100


# ============================================================
# Parsing
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
# Embedding model
# ============================================================
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return masked.sum(dim=1) / denom


@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], batch_size: int = 32) -> torch.Tensor:
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        out = model(**enc)
        vecs = mean_pool(out.last_hidden_state, enc["attention_mask"])
        vecs = F.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs.cpu().float())

    return torch.cat(all_vecs, dim=0)


def choose_indices(similarities: np.ndarray) -> List[int]:
    if similarities.size == 0:
        return []

    ranked = np.argsort(-similarities)

    sorted_scores = similarities[ranked]

    # Largest-gap cutoff: find the biggest drop between consecutive scores.
    if len(sorted_scores) > 1:
        gaps = sorted_scores[:-1] - sorted_scores[1:]
        gap_idx = int(np.argmax(gaps))
        gap_threshold = float(sorted_scores[gap_idx + 1])
    else:
        gap_threshold = float(sorted_scores[0])

    threshold = max(gap_threshold, MIN_SIMILARITY_THRESHOLD)
    selected = [int(i) for i in ranked if similarities[i] >= threshold]

    # Heuristic: if exactly one premise passes threshold, include one more
    # top-ranked premise so conclusions are supported by at least a pair.
    if len(selected) == 1 and len(ranked) >= 2:
        top_selected = selected[0]
        for idx in ranked:
            idx = int(idx)
            if idx != top_selected:
                selected.append(idx)
                break

    return sorted(selected)


# ============================================================
# Main prediction pipeline
# ============================================================
def predict_dataset_premises_only(model, tokenizer, test_data):
    predictions = []

    for item in test_data:
        premises, conclusion = parse_syllogism_to_premises_and_conclusion(item["syllogism"])

        if not premises or not conclusion:
            predictions.append({"id": item["id"], "relevant_premises": []})
            continue

        texts = premises + [conclusion]
        embs = encode_texts(model, tokenizer, texts, batch_size=32)
        premise_embs = embs[:-1]  # [num_premises, dim]
        conclusion_emb = embs[-1]  # [dim]

        sims = torch.matmul(premise_embs, conclusion_emb).numpy()
        selected = choose_indices(sims)

        predictions.append({
            "id": item["id"],
            "relevant_premises": selected,
        })

    return predictions


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
    model.eval()

    print(f"Loading data: {TEST_JSON}")
    test_data = load_json(TEST_JSON)
    print(f"Total samples: {len(test_data)}")

    print("Running premise selection...")
    predictions = predict_dataset_premises_only(model, tokenizer, test_data)

    save_json(predictions, OUT_PATH)
    print(f"Saved predictions to: {OUT_PATH}")

    f1_premises = calculate_f1_premises(test_data, predictions)
    eval_results = {"f1_premises": round(f1_premises, 4)}
    save_json(eval_results, EVAL_OUT_PATH)
    print(f"Premise-selection F1: {f1_premises:.4f}")
    print(f"Saved eval results to: {EVAL_OUT_PATH}")
