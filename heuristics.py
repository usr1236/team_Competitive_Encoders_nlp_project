import os
import json
import re
import argparse
import random
import importlib.util
from itertools import combinations
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_JSON = "train_data/subtask 1/train_data.json"
TEST_JSON_ST2 = "test_data/subtask 2/test_data_subtask_2.json"
TEST_JSON_ST4 = "test_data/subtask 4/test_data_subtask_4.json"
EVAL_SCRIPT_ST24 = "evaluation_kit/task 2 & 4/evaluation_script.py"

OUT_DIR = "subtask_retrieve_first_results"
SEED = 42
MAX_LEN = 512
ICL_SHOTS = 4
YES_TEXT = " yes"
NO_TEXT = " no"
TOP_K_PAIRS = 3


# ============================================================
# TERM EXTRACTION & MATCHING
# ============================================================
CONCLUSION_MARKERS = [
    "therefore, it must be the case that", "therefore, it can be said that",
    "therefore, we can say that a portion of", "therefore, we can say that",
    "from this, it follows that", "it is necessarily concluded that",
    "it is necessarily true that", "one can thus conclude that",
    "it is logically necessary that", "it is therefore the case that",
    "this leads to the conclusion that", "it must be true that",
    "it is therefore true that", "it must follow that",
    "it can be deduced that", "it can be said that",
    "it must be the case that", "we can conclude that",
    "it is deduced that", "it follows directly that",
    "it follows that", "this implies that",
    "we can say that", "this means that",
    "the conclusion is that", "there exists at least",
    "consequently, some of the", "consequently, there are no",
    "therefore, at least one",
    "consequently,", "therefore,", "hence,", "thus,", "as such,",
    "at least",
]

QUANTIFIERS = [
    "everything that can be classified as a", "everything that can be classified as",
    "anything that is a", "anything that is an", "anything that is",
    "nothing in the category of", "all of those who are",
    "all things that are", "among the items that are",
    "there is not a single", "it is a known fact that",
    "it is a fact that", "it is true that",
    "it is undeniable that", "it is the case that",
    "it is known that", "a certain number of",
    "a number of", "a portion of",
    "there are no", "there are some", "there are many",
    "there exist some", "there exist", "there exists",
    "every single", "without exception", "without a doubt",
    "every", "each", "all", "some", "any", "no",
    "not all", "a few", "in fact", "of course",
]

STOPS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "shall", "should", "may", "might",
    "can", "could", "the", "a", "an", "of", "in", "to", "for", "with",
    "by", "on", "at", "from", "as", "into", "that", "which", "who", "whom",
    "this", "these", "those", "it", "its", "also", "and", "or", "but", "if",
    "than", "when", "where", "how", "what", "there", "here", "only",
    "defined", "described", "classified", "considered", "called",
    "type", "kind", "form", "actually", "fact", "known", "true", "single",
    "said", "necessarily", "must", "follows", "therefore", "consequently",
    "hence", "such", "means", "case", "leads", "conclusion", "deduced",
    "established", "never", "mutually", "exclusive", "one", "not",
    "certain", "number", "least", "thus", "we", "conclude", "can",
    "portion", "directly", "exists", "logically", "necessary",
    "course", "without", "exception", "doubt", "undeniable",
    "thing", "things", "item", "items", "upon", "about", "over",
})


def normalize_word(word: str) -> str:
    w = word.lower().strip(".,!?;:()[]{}\"'")
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and w[-3] not in "aeiou":
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def extract_terms(sentence: str) -> set:
    s = sentence.lower().strip()
    for m in sorted(CONCLUSION_MARKERS, key=len, reverse=True):
        s = s.replace(m, " ")
    for q in sorted(QUANTIFIERS, key=len, reverse=True):
        s = s.replace(q, " ")

    words = re.findall(r"[a-z][\w-]*", s)
    terms = set()
    for w in words:
        if w not in STOPS and len(w) > 2:
            terms.add(w)
            terms.add(normalize_word(w))
    return terms


def term_overlap(t1: set, t2: set) -> set:
    overlap = set(t1 & t2)
    for a in t1:
        for b in t2:
            if len(a) > 4 and len(b) > 4 and (a in b or b in a):
                overlap.add(min(a, b, key=len))
    return overlap


def split_syllogism(text: str) -> Tuple[List[str], str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return sentences, sentences[-1] if sentences else ""
    return sentences[:-1], sentences[-1]


# ============================================================
# PREMISE PAIR SCORING
# ============================================================
def score_premise_pairs(premises: List[str], conclusion: str) -> List[Tuple[int, int, float]]:
    conc_terms = extract_terms(conclusion)
    prem_terms = [extract_terms(p) for p in premises]

    pair_scores: List[Tuple[int, int, float]] = []

    for i, j in combinations(range(len(premises)), 2):
        ti, tj = prem_terms[i], prem_terms[j]

        shared = term_overlap(ti, tj)
        middle = shared - term_overlap(shared, conc_terms)

        ei = term_overlap(ti, conc_terms)
        ej = term_overlap(tj, conc_terms)

        middle_score = len(middle)
        ep_i = len(ei)
        ep_j = len(ej)

        score = 0.0

        score += middle_score * 3.0
        score += ep_i + ep_j

        if ep_i > 0 and ep_j > 0 and len(ei & ej) == 0:
            score += 5.0
        elif ep_i > 0 and ep_j > 0:
            score += 2.0

        if middle_score > 0 and (ep_i > 0 or ep_j > 0):
            score += 3.0

        if ei == ej and len(ei) > 0:
            score -= 1.0

        both_overlap = ei & ej
        if len(both_overlap) > 1:
            score -= len(both_overlap) * 0.5

        useful_i = len(ei) + len(term_overlap(ti, shared))
        useful_j = len(ej) + len(term_overlap(tj, shared))
        extra_i = max(0, len(ti) - useful_i)
        extra_j = max(0, len(tj) - useful_j)
        score -= (extra_i + extra_j) * 0.3

        pair_scores.append((i, j, score))

    pair_scores.sort(key=lambda x: x[2], reverse=True)
    return pair_scores


def get_top_k_pairs(premises: List[str], conclusion: str, k: int = TOP_K_PAIRS) -> List[Tuple[int, int, float]]:
    if len(premises) < 2:
        return []
    return score_premise_pairs(premises, conclusion)[:k]


# ============================================================
# PROMPTING / VALIDITY PREDICTION
# ============================================================
def build_icl_examples(train_data: List[Dict], shots: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    pos = [x for x in train_data if x["validity"] is True]
    neg = [x for x in train_data if x["validity"] is False]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = shots // 2
    chosen = pos[:half] + neg[:shots - half]
    rng.shuffle(chosen)
    return chosen


def build_prompt_for_pair(tokenizer, train_data: List[Dict], pair_premises: List[str], conclusion: str) -> str:
    chosen = build_icl_examples(train_data, ICL_SHOTS, SEED)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict formal logic reasoner. "
                "Decide only whether the conclusion logically follows from the given premises. "
                "Ignore plausibility and world knowledge. "
                "Reply with only yes or no."
            ),
        }
    ]

    for ex in chosen:
        ex_premises, ex_conclusion = split_syllogism(ex["syllogism"])
        ex_premises = ex_premises[:2] if len(ex_premises) >= 2 else ex_premises
        ex_text = "\n".join([f"Premise {idx + 1}: {p}" for idx, p in enumerate(ex_premises)])
        ex_text += f"\nConclusion: {ex_conclusion}"
        ans = "yes" if ex["validity"] else "no"

        messages.append({
            "role": "user",
            "content": f"Argument:\n{ex_text}\n\nAnswer yes or no."
        })
        messages.append({"role": "assistant", "content": ans})

    pair_text = "\n".join([f"Premise {idx + 1}: {p}" for idx, p in enumerate(pair_premises)])
    pair_text += f"\nConclusion: {conclusion}"

    messages.append({
        "role": "user",
        "content": f"Argument:\n{pair_text}\n\nAnswer yes or no."
    })

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_label(model, tokenizer, prompt: str, label_text: str, max_len: int = MAX_LEN) -> float:
    full = prompt + label_text

    full_ids = tokenizer(
        full,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    ).input_ids.to(model.device)

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    ).input_ids.to(model.device)

    with torch.no_grad():
        out = model(full_ids, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = full_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)

    start = max(prompt_ids.shape[1] - 1, 0)
    return log_probs[0, start:].sum().item()


def predict_validity_from_pair(model, tokenizer, train_data: List[Dict], pair_premises: List[str], conclusion: str) -> Tuple[bool, float, float, float]:
    prompt = build_prompt_for_pair(tokenizer, train_data, pair_premises, conclusion)
    y = score_label(model, tokenizer, prompt, YES_TEXT)
    n = score_label(model, tokenizer, prompt, NO_TEXT)
    valid = y >= n
    margin = y - n
    return bool(valid), float(margin), float(y), float(n)


# ============================================================
# MULTILINGUAL HELPERS
# ============================================================
def get_syllogism_text(example: Dict) -> str:
    if "syllogism_t" in example and example.get("lang", "en") != "en":
        return example["syllogism_t"]
    return example["syllogism"]


def get_english_syllogism(example: Dict) -> str:
    return example["syllogism"]


# ============================================================
# OFFICIAL EVAL
# ============================================================
def run_official_eval(eval_script_path: str, ref_path: str, pred_path: str, out_path: str) -> Dict:
    spec = importlib.util.spec_from_file_location("semeval_eval", eval_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation script: {eval_script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_full_scoring(ref_path, pred_path, out_path)
    with open(out_path, "r") as f:
        return json.load(f)


# ============================================================
# RETRIEVE-FIRST PIPELINE
# ============================================================
@torch.no_grad()
def run_pipeline_retrieve_first(
    model,
    tokenizer,
    train_data: List[Dict],
    test_data: List[Dict],
    use_translated: bool = False,
    top_k: int = TOP_K_PAIRS,
) -> List[Dict]:
    predictions = []

    for idx, ex in enumerate(test_data):
        # Retrieval heuristic stays on English source text
        english_syl = get_english_syllogism(ex)
        premises, conclusion = split_syllogism(english_syl)

        if len(premises) < 2:
            predictions.append({
                "id": ex["id"],
                "validity": False,
                "relevant_premises": [],
            })
            continue

        top_pairs = get_top_k_pairs(premises, conclusion, k=top_k)

        if not top_pairs:
            predictions.append({
                "id": ex["id"],
                "validity": False,
                "relevant_premises": [],
            })
            continue

        best_pair = None
        best_valid = False
        best_combined_margin = float("-inf")

        for i, j, pair_score in top_pairs:
            candidate_premises = [premises[i], premises[j]]

            is_valid, margin, _, _ = predict_validity_from_pair(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                pair_premises=candidate_premises,
                conclusion=conclusion,
            )

            combined_margin = margin + 0.05 * pair_score

            if combined_margin > best_combined_margin:
                best_combined_margin = combined_margin
                best_valid = is_valid
                best_pair = sorted([i, j])

        if best_valid and best_pair is not None:
            relevant = best_pair
        else:
            relevant = []

        predictions.append({
            "id": ex["id"],
            "validity": bool(best_valid),
            "relevant_premises": relevant,
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_data):
            print(f"Processed {idx + 1}/{len(test_data)}")

    return predictions


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", type=int, default=2, choices=[2, 4])
    parser.add_argument("--test_json", default=None)
    parser.add_argument("--use_lora", default=None, help="Path to LoRA adapter")
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--top_k_pairs", type=int, default=TOP_K_PAIRS)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.test_json is None:
        args.test_json = TEST_JSON_ST4 if args.subtask == 4 else TEST_JSON_ST2

    print(f"Subtask: {args.subtask}")
    print(f"Train file: {TRAIN_JSON}")
    print(f"Test file: {args.test_json}")
    print(f"Model: {MODEL_NAME}")
    print(f"Top-k premise pairs: {args.top_k_pairs}")

    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if args.use_lora:
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {args.use_lora}")
        model = PeftModel.from_pretrained(model, args.use_lora)

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    use_translated = (args.subtask == 4)

    print("\nRunning retrieve-first pipeline...")
    predictions = run_pipeline_retrieve_first(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        test_data=test_data,
        use_translated=use_translated,
        top_k=args.top_k_pairs,
    )

    pred_path = os.path.join(args.out_dir, f"predictions_st{args.subtask}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved predictions to: {pred_path}")

    # Save reference copy for official scorer
    ref_path = os.path.join(args.out_dir, f"reference_st{args.subtask}.json")
    with open(ref_path, "w") as f:
        json.dump(test_data, f, indent=2)

    # Quick local validity accuracy
    valid_correct = sum(
        1 for ex, pr in zip(test_data, predictions)
        if ex["validity"] == pr["validity"]
    )
    print(f"Quick validity accuracy: {valid_correct}/{len(test_data)} = {100.0 * valid_correct / len(test_data):.2f}%")

    # Official evaluation
    eval_out_path = os.path.join(args.out_dir, f"eval_results_st{args.subtask}.json")
    if os.path.exists(EVAL_SCRIPT_ST24):
        print("\nRunning official evaluation...")
        results = run_official_eval(
            eval_script_path=EVAL_SCRIPT_ST24,
            ref_path=ref_path,
            pred_path=pred_path,
            out_path=eval_out_path,
        )

        print("\n=== OFFICIAL METRICS ===")
        for k, v in results.items():
            print(f"{k}: {v}")
        print(f"\nSaved evaluation to: {eval_out_path}")
    else:
        print(f"\nOfficial evaluation script not found at: {EVAL_SCRIPT_ST24}")


if __name__ == "__main__":
    main()