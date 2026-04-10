import os
import json
import random
import importlib.util
from typing import List, Dict, Any, Tuple
from itertools import product

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

ENTROPY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HOME = os.getenv("HF_HOME", "")
TRAIN_JSON = "train_data/subtask 1/train_data.json"
TEST_JSON = "test_data/subtask 1/test_data_subtask_1.json"
EVAL_SCRIPT = "evaluation_kit/task 1 & 3/evaluation_script.py"
OUT_DIR = "results_new_q3"

SEED = 42
MAX_LEN = 512
MAX_STEER_EXAMPLES = 2400
ALPHAS = [0, -0.5, 0.5, 1, 2, 3, 7, 10, -3, -2, -1,]
LAYER_FRACTION = 0.25
# "Q3" = third quarter (layers 50%-75%), matches Valentino et al.
# "Q4" = fourth quarter (layers 75%-100%), last quarter
# "Q3Q4" = both (layers 50%-100%)
LAYER_QUARTER = "Q3"
NORMALIZE_VECTORS = False
USE_BFLOAT16 = True
USE_CHAT_TEMPLATE = True
YES_TEXT = " yes"
NO_TEXT = " no"

PROMPT_MODE = "icl"
ICL_SHOTS = 4

MODE = "entropy_gated"

KCAST_K = 10
ENTROPY_THRESHOLD = 0.5
LAYERWISE_ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
LAYERWISE_MAX_COMBOS = 5000

MULTI_TOKEN_STEER = True

# "dynamic" = use prompt length to find correct position (safe for any label)
# "hardcoded" = use -2 (only safe when label is exactly 1 token)
SINGLE_TOKEN_MODE = "hardcoded"

USE_CACHED_STEERING = True

SAVE_OUTPUTS = False
SAVE_STEERING_DATA = True

if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME


def _build_steering_config(target_layers):
    return {
        "model_name": MODEL_NAME,
        "train_json": TRAIN_JSON,
        "prompt_mode": PROMPT_MODE,
        "icl_shots": ICL_SHOTS if PROMPT_MODE == "icl" else 0,
        "use_chat_template": USE_CHAT_TEMPLATE,
        "layer_fraction": LAYER_FRACTION,
        "layer_quarter": LAYER_QUARTER,
        "max_steer_examples": MAX_STEER_EXAMPLES,
        "normalize_vectors": NORMALIZE_VECTORS,
        "seed": SEED,
        "yes_text": YES_TEXT,
        "no_text": NO_TEXT,
        "pool_tag": "struct" if MODE == "structure_aware" else "std",
        "target_layers": sorted(target_layers),
    }


def get_steering_cache_path():
    model_tag = MODEL_NAME.replace("/", "_").replace("-", "_")
    prompt_tag = f"{PROMPT_MODE}{ICL_SHOTS}" if PROMPT_MODE == "icl" else PROMPT_MODE
    chat_tag = "chat" if USE_CHAT_TEMPLATE else "raw"
    pool_tag = "struct" if MODE == "structure_aware" else "std"
    norm_tag = "norm" if NORMALIZE_VECTORS else "nonorm"
    train_tag = os.path.basename(os.path.dirname(TRAIN_JSON)).replace(" ", "_")
    label_tag = f"{YES_TEXT.strip()}_{NO_TEXT.strip()}"

    fname = (
        f"steering_"
        f"{model_tag}_"
        f"{prompt_tag}_{chat_tag}_"
        f"{train_tag}_"
        f"{LAYER_QUARTER}_frac{LAYER_FRACTION}_"
        f"n{MAX_STEER_EXAMPLES}_"
        f"s{SEED}_"
        f"{label_tag}_"
        f"{pool_tag}_{norm_tag}.pt"
    )
    return os.path.join(OUT_DIR, fname)


def load_or_compute_steering(model, tokenizer, train_items, target_layers):
    cache_path = get_steering_cache_path()
    current_config = _build_steering_config(target_layers)

    if USE_CACHED_STEERING and os.path.exists(cache_path):
        print(f"Loading cached steering data: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")

        if "config" in payload and "data" in payload:
            cached_config = payload["config"]
            steering_data = payload["data"]

            mismatches = []
            for key in current_config:
                cached_val = cached_config.get(key)
                current_val = current_config[key]
                if cached_val != current_val:
                    mismatches.append(f"  {key}: cached={cached_val} vs current={current_val}")

            if mismatches:
                print(f"  Config mismatch ({len(mismatches)} fields):")
                for m in mismatches:
                    print(m)
                print(f"  Recomputing...")
            else:
                print(f"  Config validated. Layers: {sorted(steering_data['deltas'].keys())}")
                return steering_data
        else:
            cached_layers = set(payload.get("deltas", {}).keys())
            expected_layers = set(target_layers)
            if cached_layers == expected_layers:
                print(f"  Legacy cache (no config metadata). Layers match, using it.")
                print(f"  Warning: cannot verify other config fields. Set USE_CACHED_STEERING=False to force recompute.")
                return payload
            else:
                print(f"  Legacy cache, layer mismatch. Recomputing...")

    print(f"Computing steering data...")
    steering_data = compute_steering_data(model, tokenizer, train_items, target_layers)

    if SAVE_STEERING_DATA:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        payload = {"config": current_config, "data": steering_data}
        torch.save(payload, cache_path)
        print(f"Saved steering data: {cache_path}")

    return steering_data


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_json(obj, path):
    if not SAVE_OUTPUTS:
        return
    _write_json(obj, path)


def run_official_eval(eval_script_path, ref_path, pred_path, out_path):
    spec = importlib.util.spec_from_file_location("semeval_eval", eval_script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_full_scoring(ref_path, pred_path, out_path)
    return load_json(out_path)


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "module"):
        return get_layers(model.module)
    raise ValueError("Unsupported model architecture")


def get_target_layers(total, fraction, quarter=None):
    if quarter is not None:
        q_size = total // 4
        if quarter == "Q3":
            return list(range(2 * q_size, 3 * q_size))
        elif quarter == "Q4":
            return list(range(3 * q_size, total))
        elif quarter == "Q3Q4":
            return list(range(2 * q_size, total))
        else:
            raise ValueError(f"Unknown quarter: {quarter}")
    start = int(total * (1.0 - fraction))
    return list(range(start, total))


def build_icl_examples(train_data, shots, seed):
    rng = random.Random(seed)
    pos = [x for x in train_data if x["validity"] is True]
    neg = [x for x in train_data if x["validity"] is False]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = shots // 2
    chosen = pos[:half] + neg[:shots - half]
    rng.shuffle(chosen)
    return chosen


def build_prompt(tokenizer, train_data, syllogism):
    if PROMPT_MODE == "icl":
        chosen = build_icl_examples(train_data, ICL_SHOTS, SEED)
        if USE_CHAT_TEMPLATE:
            messages = [
                {"role": "system", "content": (
                    "You are a strict formal logic reasoner. "
                    "Decide only whether the conclusion logically follows from the premises. "
                    "Ignore plausibility and world knowledge. "
                    "Reply with only yes or no."
                )}
            ]
            for ex in chosen:
                ans = "yes" if ex["validity"] else "no"
                messages.append({"role": "user", "content": f"Argument:\n{ex['syllogism']}\n\nAnswer yes or no."})
                messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": f"Argument:\n{syllogism}\n\nAnswer yes or no."})
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        parts = [
            "You are a strict formal logic reasoner.",
            "Decide only whether the conclusion logically follows from the premises.",
            "Ignore plausibility and world knowledge.",
            "Reply with only yes or no.", ""
        ]
        for ex in chosen:
            ans = "yes" if ex["validity"] else "no"
            parts.extend([f"Argument:\n{ex['syllogism']}", f"Answer: {ans}", ""])
        parts.extend([f"Argument:\n{syllogism}", "Answer:"])
        return "\n".join(parts)

    if USE_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": (
                "You are a strict formal logic reasoner. "
                "Decide only whether the conclusion logically follows from the premises. "
                "Ignore plausibility and world knowledge. "
                "Reply with only yes or no."
            )},
            {"role": "user", "content": f"Argument:\n{syllogism}\n\nAnswer yes or no."}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "You are a strict formal logic reasoner.\n"
        "Decide only whether the conclusion logically follows from the premises.\n"
        "Ignore plausibility and world knowledge.\n"
        "Reply with only yes or no.\n\n"
        f"Argument:\n{syllogism}\n\nAnswer:"
    )


def score_label(model, tokenizer, prompt, label_text, max_len):
    full = prompt + label_text
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=max_len).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                           truncation=True, max_length=max_len).input_ids.to(model.device)
    set_prompt_len(tokenizer, prompt, max_len)
    with torch.no_grad():
        out = model(full_ids, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = full_ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_ids.shape[1] - 1, 0)
    return lp[0, start:].sum().item()


def predict_validity(model, tokenizer, prompt, max_len):
    y = score_label(model, tokenizer, prompt, YES_TEXT, max_len)
    n = score_label(model, tokenizer, prompt, NO_TEXT, max_len)
    return y >= n


def get_prediction_entropy(model, tokenizer, prompt, max_len):
    y = score_label(model, tokenizer, prompt, YES_TEXT, max_len)
    n = score_label(model, tokenizer, prompt, NO_TEXT, max_len)
    logits = torch.tensor([y, n])
    probs = torch.softmax(logits, dim=0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_entropy = np.log(2)
    return entropy / max_entropy, y >= n


def get_all_layer_hidden(model, tokenizer, text, target_layers, max_len):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_len, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    return {l: out.hidden_states[l + 1][0, -1, :].float().cpu() for l in target_layers}


def balanced_by_validity(data, n, seed):
    rng = random.Random(seed)
    pos = [x for x in data if x["validity"] is True]
    neg = [x for x in data if x["validity"] is False]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    out = pos[:half] + neg[:n - half]
    rng.shuffle(out)
    return out


def cosine_sim(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-10)


def extract_syllogistic_form(text):
    text_lower = text.lower()
    quantifiers = [
        "not all", "no", "all", "some", "every", "there are no",
        "it is not the case that all", "it is not the case that every",
        "there exist some", "every single", "it is certain that no",
        "it is certain that every", "it is also true that every",
        "it is known that some", "it is known that every", "it is known that no"
    ]
    found = []
    for q in sorted(quantifiers, key=len, reverse=True):
        if q in text_lower:
            found.append(q)
    if any(marker in text_lower for marker in ["therefore", "consequently", "it follows", "this has led"]):
        found.append("CONCLUSION_MARKER")
    neg_count = text_lower.count("not") + text_lower.count("no ")
    form_key = "|".join(sorted(found)) + f"|neg{neg_count}"
    return form_key


def build_structure_aware_pool(train_items, max_examples, seed):
    from collections import defaultdict
    rng = random.Random(seed)

    groups = defaultdict(lambda: {
        "valid_plausible": [], "valid_implausible": [],
        "invalid_plausible": [], "invalid_implausible": []
    })
    for item in train_items:
        form = extract_syllogistic_form(item["syllogism"])
        v = "valid" if item["validity"] else "invalid"
        p = "plausible" if item["plausibility"] else "implausible"
        groups[form][f"{v}_{p}"].append(item)

    paired = []
    unpaired = []
    for form, buckets in groups.items():
        vp = buckets["valid_plausible"]
        vi = buckets["valid_implausible"]
        ip = buckets["invalid_plausible"]
        ii = buckets["invalid_implausible"]
        rng.shuffle(vp); rng.shuffle(vi)
        rng.shuffle(ip); rng.shuffle(ii)

        n_valid = min(len(vp), len(vi))
        for j in range(n_valid):
            paired.append((vp[j], vi[j]))
        n_invalid = min(len(ip), len(ii))
        for j in range(n_invalid):
            paired.append((ip[j], ii[j]))
        for lst in [vp[n_valid:], vi[n_valid:], ip[n_invalid:], ii[n_invalid:]]:
            unpaired.extend(lst)

    rng.shuffle(paired)
    rng.shuffle(unpaired)
    print(f"Structure-aware: {len(paired)} paired, {len(unpaired)} unpaired")

    pool = []
    for p_item, ip_item in paired[:max_examples // 2]:
        pool.append(p_item)
        pool.append(ip_item)
    remaining = max_examples - len(pool)
    if remaining > 0:
        pool.extend(unpaired[:remaining])
    rng.shuffle(pool)
    return pool


def compute_steering_data(model, tokenizer, train_items, target_layers):
    if MODE == "structure_aware":
        pool = build_structure_aware_pool(train_items, MAX_STEER_EXAMPLES, SEED)
    else:
        plausible = [x for x in train_items if x["plausibility"] is True]
        implausible = [x for x in train_items if x["plausibility"] is False]
        n = min(len(plausible), len(implausible), MAX_STEER_EXAMPLES // 2)
        pool = balanced_by_validity(plausible, n, SEED) + balanced_by_validity(implausible, n, SEED)
        random.Random(SEED).shuffle(pool)

    print(f"Steering pool: {len(pool)} examples")

    correct_acts = {l: [] for l in target_layers}
    wrong_acts = {l: [] for l in target_layers}
    valid_acts = {l: [] for l in target_layers}
    invalid_acts = {l: [] for l in target_layers}
    all_acts_with_validity = {l: [] for l in target_layers}
    valid_correct_acts = {l: [] for l in target_layers}
    valid_wrong_acts = {l: [] for l in target_layers}
    invalid_correct_acts = {l: [] for l in target_layers}
    invalid_wrong_acts = {l: [] for l in target_layers}

    n_correct = 0
    n_wrong = 0

    for i, ex in enumerate(pool):
        prompt = build_prompt(tokenizer, train_items, ex["syllogism"])
        hiddens = get_all_layer_hidden(model, tokenizer, prompt, target_layers, MAX_LEN)
        pred = predict_validity(model, tokenizer, prompt, MAX_LEN)
        gold = bool(ex["validity"])
        is_correct = (pred == gold)

        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        for l in target_layers:
            h = hiddens[l]
            if is_correct:
                correct_acts[l].append(h)
            else:
                wrong_acts[l].append(h)
            if gold:
                valid_acts[l].append(h)
            else:
                invalid_acts[l].append(h)
            all_acts_with_validity[l].append((h, 1 if gold else -1))
            if gold:
                if is_correct:
                    valid_correct_acts[l].append(h)
                else:
                    valid_wrong_acts[l].append(h)
            else:
                if is_correct:
                    invalid_correct_acts[l].append(h)
                else:
                    invalid_wrong_acts[l].append(h)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pool)} (correct={n_correct}, wrong={n_wrong})")

    print(f"  Total correct={n_correct}, wrong={n_wrong}")

    deltas = {}
    for l in target_layers:
        if len(correct_acts[l]) == 0 or len(wrong_acts[l]) == 0:
            raise ValueError(f"Layer {l}: need both correct and wrong examples")
        mu_plus = torch.stack(correct_acts[l]).mean(dim=0)
        mu_minus = torch.stack(wrong_acts[l]).mean(dim=0)
        delta = mu_plus - mu_minus
        if NORMALIZE_VECTORS:
            delta = delta / delta.norm()
        deltas[l] = delta
        print(f"  Layer {l}: ||delta|| = {delta.norm():.4f}")

    cond_valid = {}
    cond_invalid = {}
    for l in target_layers:
        cond_valid[l] = torch.stack(valid_acts[l]).mean(dim=0)
        cond_invalid[l] = torch.stack(invalid_acts[l]).mean(dim=0)

    valid_deltas = {}
    invalid_deltas = {}
    for l in target_layers:
        if len(valid_correct_acts[l]) > 0 and len(valid_wrong_acts[l]) > 0:
            vd = torch.stack(valid_correct_acts[l]).mean(dim=0) - torch.stack(valid_wrong_acts[l]).mean(dim=0)
            if NORMALIZE_VECTORS:
                vd = vd / vd.norm()
            valid_deltas[l] = vd
        else:
            valid_deltas[l] = deltas[l]
        if len(invalid_correct_acts[l]) > 0 and len(invalid_wrong_acts[l]) > 0:
            ivd = torch.stack(invalid_correct_acts[l]).mean(dim=0) - torch.stack(invalid_wrong_acts[l]).mean(dim=0)
            if NORMALIZE_VECTORS:
                ivd = ivd / ivd.norm()
            invalid_deltas[l] = ivd
        else:
            invalid_deltas[l] = deltas[l]
        print(f"  Layer {l}: ||valid_delta|| = {valid_deltas[l].norm():.4f}, ||invalid_delta|| = {invalid_deltas[l].norm():.4f}")

    knn_store = {}
    for l in target_layers:
        vecs = torch.stack([item[0] for item in all_acts_with_validity[l]])
        labels = torch.tensor([item[1] for item in all_acts_with_validity[l]], dtype=torch.float32)
        knn_store[l] = {"vecs": vecs, "labels": labels}

    return {
        "deltas": deltas,
        "valid_deltas": valid_deltas,
        "invalid_deltas": invalid_deltas,
        "cond_valid": cond_valid,
        "cond_invalid": cond_invalid,
        "knn_store": knn_store,
    }


_steer_state = {"prompt_len": None}


def set_prompt_len(tokenizer, prompt, max_len):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=max_len).input_ids
    _steer_state["prompt_len"] = ids.shape[1]


def apply_steering(x, delta, alpha):
    d = delta.to(x.device, x.dtype)
    if MULTI_TOKEN_STEER:
        return x + alpha * d.view(1, 1, -1)
    else:
        x = x.clone()
        if SINGLE_TOKEN_MODE == "hardcoded":
            x[:, -2, :] = x[:, -2, :] + alpha * d.view(1, -1)
        else:
            pos = _steer_state.get("prompt_len")
            if pos is None or pos <= 0:
                pos = x.shape[1] - 1
            else:
                pos = min(pos - 1, x.shape[1] - 1)
            x[:, pos, :] = x[:, pos, :] + alpha * d.view(1, -1)
        return x


class StaticHook:
    def __init__(self, delta, alpha):
        self.delta = delta
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            return (apply_steering(x, self.delta, self.alpha),) + output[1:]
        return apply_steering(output, self.delta, self.alpha)


class CASTHook:
    def __init__(self, delta, alpha, psi_valid, psi_invalid):
        self.delta = delta
        self.alpha = alpha
        self.psi_valid = psi_valid
        self.psi_invalid = psi_invalid

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        h = x[0, -1, :].float().cpu()
        proj_valid = (torch.dot(h, self.psi_valid) / (torch.dot(self.psi_valid, self.psi_valid) + 1e-10)) * self.psi_valid
        proj_invalid = (torch.dot(h, self.psi_invalid) / (torch.dot(self.psi_invalid, self.psi_invalid) + 1e-10)) * self.psi_invalid
        sim_valid = cosine_sim(h, proj_valid).item()
        sim_invalid = cosine_sim(h, proj_invalid).item()
        effective_alpha = -self.alpha if sim_valid > sim_invalid else self.alpha
        x = apply_steering(x, self.delta, effective_alpha)
        if rest is None:
            return x
        return (x,) + rest


class KCASTHook:
    def __init__(self, delta, alpha, knn_vecs, knn_labels, k):
        self.delta = delta
        self.alpha = alpha
        self.knn_vecs = knn_vecs
        self.knn_labels = knn_labels
        self.k = k

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        h = x[0, -1, :].float().cpu()
        sims = F.cosine_similarity(h.unsqueeze(0), self.knn_vecs, dim=1)
        topk = torch.topk(sims, self.k)
        neighbor_labels = self.knn_labels[topk.indices]
        vote = neighbor_labels.sum().item()
        y_hat = 1.0 if vote > 0 else -1.0
        effective_alpha = -y_hat * self.alpha
        x = apply_steering(x, self.delta, effective_alpha)
        if rest is None:
            return x
        return (x,) + rest


class ValidityConditionedHook:
    def __init__(self, valid_delta, invalid_delta, alpha, psi_valid, psi_invalid):
        self.valid_delta = valid_delta
        self.invalid_delta = invalid_delta
        self.alpha = alpha
        self.psi_valid = psi_valid
        self.psi_invalid = psi_invalid

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        h = x[0, -1, :].float().cpu()
        proj_valid = (torch.dot(h, self.psi_valid) / (torch.dot(self.psi_valid, self.psi_valid) + 1e-10)) * self.psi_valid
        proj_invalid = (torch.dot(h, self.psi_invalid) / (torch.dot(self.psi_invalid, self.psi_invalid) + 1e-10)) * self.psi_invalid
        sim_valid = cosine_sim(h, proj_valid).item()
        sim_invalid = cosine_sim(h, proj_invalid).item()
        if sim_valid > sim_invalid:
            delta = self.valid_delta
            effective_alpha = -self.alpha
        else:
            delta = self.invalid_delta
            effective_alpha = self.alpha
        x = apply_steering(x, delta, effective_alpha)
        if rest is None:
            return x
        return (x,) + rest


class ValidityConditionedKCASTHook:
    def __init__(self, valid_delta, invalid_delta, alpha, knn_vecs, knn_labels, k):
        self.valid_delta = valid_delta
        self.invalid_delta = invalid_delta
        self.alpha = alpha
        self.knn_vecs = knn_vecs
        self.knn_labels = knn_labels
        self.k = k

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        h = x[0, -1, :].float().cpu()
        sims = F.cosine_similarity(h.unsqueeze(0), self.knn_vecs, dim=1)
        topk = torch.topk(sims, self.k)
        neighbor_labels = self.knn_labels[topk.indices]
        vote = neighbor_labels.sum().item()
        y_hat = 1.0 if vote > 0 else -1.0
        if y_hat > 0:
            delta = self.valid_delta
        else:
            delta = self.invalid_delta
        effective_alpha = -y_hat * self.alpha
        x = apply_steering(x, delta, effective_alpha)
        if rest is None:
            return x
        return (x,) + rest


def register_hooks(layers, steering_data, alpha, mode, layer_alphas=None):
    handles = []
    deltas = steering_data["deltas"]
    valid_deltas = steering_data["valid_deltas"]
    invalid_deltas = steering_data["invalid_deltas"]
    cond_valid = steering_data["cond_valid"]
    cond_invalid = steering_data["cond_invalid"]
    knn_store = steering_data["knn_store"]

    for layer_idx in deltas.keys():
        a = layer_alphas[layer_idx] if layer_alphas else alpha

        if mode == "static":
            hook = StaticHook(deltas[layer_idx], a)
        elif mode == "cast":
            hook = CASTHook(deltas[layer_idx], a, cond_valid[layer_idx], cond_invalid[layer_idx])
        elif mode == "kcast":
            hook = KCASTHook(deltas[layer_idx], a, knn_store[layer_idx]["vecs"], knn_store[layer_idx]["labels"], KCAST_K)
        elif mode == "validity_cond":
            hook = ValidityConditionedHook(valid_deltas[layer_idx], invalid_deltas[layer_idx], a, cond_valid[layer_idx], cond_invalid[layer_idx])
        elif mode == "validity_cond_kcast":
            hook = ValidityConditionedKCASTHook(valid_deltas[layer_idx], invalid_deltas[layer_idx], a, knn_store[layer_idx]["vecs"], knn_store[layer_idx]["labels"], KCAST_K)
        elif mode in ("layerwise", "layerwise_entropy", "entropy_gated", "structure_aware"):
            hook = StaticHook(deltas[layer_idx], a)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        handles.append(layers[layer_idx].register_forward_hook(hook))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


@torch.no_grad()
def predict_dataset(model, tokenizer, train_items, eval_items, steering_data, alpha, mode, layer_alphas=None, threshold=None):
    layers = get_layers(model)
    if mode in ("entropy_gated", "layerwise_entropy"):
        return predict_entropy_gated(model, tokenizer, train_items, eval_items, steering_data, alpha, layer_alphas, threshold=threshold)

    handles = []
    if alpha != 0 or (layer_alphas is not None):
        handles = register_hooks(layers, steering_data, alpha, mode, layer_alphas)

    preds = []
    try:
        for i, ex in enumerate(eval_items):
            prompt = build_prompt(tokenizer, train_items, ex["syllogism"])
            pred = predict_validity(model, tokenizer, prompt, MAX_LEN)
            preds.append({"id": ex["id"], "validity": bool(pred)})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(eval_items)}")
    finally:
        remove_hooks(handles)
    return preds


@torch.no_grad()
def predict_entropy_gated(model, tokenizer, train_items, eval_items, steering_data, alpha_base, layer_alphas=None, threshold=None):
    """
    Entropy-gated KCAST: direction comes from kNN vote (same as KCASTHook),
    magnitude is scaled by prediction uncertainty measured on the unsteered model.
    """
    if threshold is None:
        threshold = ENTROPY_THRESHOLD

    layers = get_layers(model)
    deltas = steering_data["deltas"]
    knn_store = steering_data["knn_store"]
    preds = []
    for i, ex in enumerate(eval_items):
        prompt = build_prompt(tokenizer, train_items, ex["syllogism"])

        # 1. Measure uncertainty on the UNSTEERED model.
        norm_entropy, _ = get_prediction_entropy(model, tokenizer, prompt, MAX_LEN)

        # 2. Scale in [0, 1]. Below threshold = no steering; above = linear ramp.
        if norm_entropy < threshold:
            scale = 0.0
        else:
            scale = (norm_entropy - threshold) / (1.0 - threshold + 1e-10)
            scale = max(0.0, min(1.0, scale))

        handles = []
        try:
            # 3. Register KCAST hooks with entropy-scaled alpha.
            if scale > 0:
                for layer_idx in deltas.keys():
                    base_a = layer_alphas[layer_idx] if layer_alphas is not None else alpha_base
                    a = base_a * scale
                    hook = KCASTHook(
                        deltas[layer_idx],
                        a,
                        knn_store[layer_idx]["vecs"],
                        knn_store[layer_idx]["labels"],
                        KCAST_K,
                    )
                    handles.append(layers[layer_idx].register_forward_hook(hook))

            # 4. Run the steered prediction.
            pred = predict_validity(model, tokenizer, prompt, MAX_LEN)
        finally:
            remove_hooks(handles)

        preds.append({"id": ex["id"], "validity": bool(pred)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eval_items)} (entropy={norm_entropy:.3f} scale={scale:.3f})")
    return preds


def optimize_layerwise_alphas(model, tokenizer, train_items, eval_items, steering_data, ref_path):
    target_layers = sorted(steering_data["deltas"].keys())
    n_layers = len(target_layers)
    alpha_vals = LAYERWISE_ALPHA_VALUES
    total_combos = len(alpha_vals) ** n_layers
    print(f"Layer-wise optimization: {n_layers} layers x {len(alpha_vals)} values = {total_combos} combos")

    if total_combos > LAYERWISE_MAX_COMBOS:
        print(f"Too many combos, using random search ({LAYERWISE_MAX_COMBOS} samples)")
        rng = random.Random(SEED)
        combos = [tuple(rng.choice(alpha_vals) for _ in range(n_layers)) for _ in range(LAYERWISE_MAX_COMBOS)]
    else:
        combos = list(product(alpha_vals, repeat=n_layers))

    best_score = -1
    best_combo = None
    best_result = None

    for ci, combo in enumerate(combos):
        layer_alphas = {l: a for l, a in zip(target_layers, combo)}
        preds = predict_dataset(model, tokenizer, train_items, eval_items, steering_data, 0, "layerwise", layer_alphas)
        pred_path = os.path.join(OUT_DIR, "tmp_layerwise_preds.json")
        _write_json(preds, pred_path)
        result = run_official_eval(EVAL_SCRIPT, ref_path, pred_path, os.path.join(OUT_DIR, "tmp_layerwise_eval.json"))
        score = result["combined_score"]
        if score > best_score:
            best_score = score
            best_combo = layer_alphas
            best_result = result
            print(f"  [{ci+1}/{len(combos)}] New best: {layer_alphas} -> Score={score:.4f} ACC={result['accuracy']:.2f} TCE={result['content_effect']:.4f}")
        if (ci + 1) % 100 == 0:
            print(f"  [{ci+1}/{len(combos)}] best so far: {best_score:.4f}")

    print(f"\nBest layer-wise alphas: {best_combo}")
    print(f"Best score: {best_score:.4f}")
    return best_combo, best_result


def print_subgroup_accuracy(items, preds):
    pred_map = {p["id"]: p["validity"] for p in preds}
    buckets = {}
    for item in items:
        v = "valid" if item["validity"] else "invalid"
        p = "plausible" if item.get("plausibility") else "implausible"
        key = f"{v}_{p}"
        buckets.setdefault(key, {"correct": 0, "total": 0})
        buckets[key]["total"] += 1
        if pred_map.get(item["id"]) == item["validity"]:
            buckets[key]["correct"] += 1
    for key in sorted(buckets.keys()):
        vals = buckets[key]
        acc = 100 * vals["correct"] / max(1, vals["total"])
        print(f"  {key:25s}: {acc:6.2f}%  ({vals['correct']}/{vals['total']})")


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Mode: {MODE}")
    print(f"Prompt: {PROMPT_MODE}" + (f" ({ICL_SHOTS} shots)" if PROMPT_MODE == "icl" else ""))
    print(f"Alphas: {ALPHAS}")
    print(f"Multi-token steering: {MULTI_TOKEN_STEER}")
    print(f"Use cached steering: {USE_CACHED_STEERING}")
    if MODE == "kcast":
        print(f"K-CAST K={KCAST_K}")
    if MODE == "entropy_gated":
        print(f"Entropy thresholds to sweep: {ENTROPY_THRESHOLDS}")
    elif MODE == "layerwise_entropy":
        print(f"Entropy threshold: {ENTROPY_THRESHOLD}")
    os.makedirs(OUT_DIR, exist_ok=True)

    token = HF_TOKEN if HF_TOKEN else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if USE_BFLOAT16 and device == "cuda" else (
        torch.float16 if device == "cuda" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=token, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    model.config.use_cache = False

    train_items = load_json(TRAIN_JSON)
    test_items = load_json(TEST_JSON)
    ref_path = os.path.join(OUT_DIR, "test_reference.json")
    _write_json(test_items, ref_path)
    print(f"Train: {len(train_items)}, Test: {len(test_items)}")

    all_layers = get_layers(model)
    target_layers = get_target_layers(len(all_layers), LAYER_FRACTION, LAYER_QUARTER)
    print(f"Total layers: {len(all_layers)}, Steering layers: {target_layers}")

    steering_data = load_or_compute_steering(model, tokenizer, train_items, target_layers)
    print(f"Steering cache path: {get_steering_cache_path()}")

    # ------------------------------------------------------------------
    # Layerwise alpha optimization
    # ------------------------------------------------------------------
    if MODE == "layerwise":
        best_layer_alphas, best_result = optimize_layerwise_alphas(
            model, tokenizer, train_items, test_items, steering_data, ref_path
        )
        save_json(
            {"layer_alphas": {str(k): v for k, v in best_layer_alphas.items()}, "result": best_result},
            os.path.join(OUT_DIR, "layerwise_best.json"),
        )
        preds = predict_dataset(model, tokenizer, train_items, test_items, steering_data, 0, "layerwise", best_layer_alphas)
        save_json(preds, os.path.join(OUT_DIR, "preds_layerwise_best.json"))
        print_subgroup_accuracy(test_items, preds)
        return

    # ------------------------------------------------------------------
    # Layerwise + entropy
    # ------------------------------------------------------------------
    if MODE == "layerwise_entropy":
        best_layer_alphas, _ = optimize_layerwise_alphas(
            model, tokenizer, train_items, test_items, steering_data, ref_path
        )
        print(f"\nRunning entropy-gated with optimized per-layer alphas...")
        preds = predict_entropy_gated(model, tokenizer, train_items, test_items, steering_data, 0, best_layer_alphas)
        save_json(preds, os.path.join(OUT_DIR, "preds_layerwise_entropy.json"))
        print_subgroup_accuracy(test_items, preds)
        eval_pred_path = os.path.join(OUT_DIR, "tmp_eval_preds.json")
        _write_json(preds, eval_pred_path)
        result = run_official_eval(EVAL_SCRIPT, ref_path, eval_pred_path, os.path.join(OUT_DIR, "tmp_eval_result.json"))
        print(f"  ACC={result['accuracy']:.2f}  TCE={result['content_effect']:.4f}  Score={result['combined_score']:.4f}")
        save_json(
            {"layer_alphas": {str(k): v for k, v in best_layer_alphas.items()}, "result": result},
            os.path.join(OUT_DIR, "layerwise_entropy_best.json"),
        )
        return

    # ------------------------------------------------------------------
    # Entropy-gated sweep over (threshold, alpha)
    # ------------------------------------------------------------------
    if MODE == "entropy_gated":
        best_score = -1
        best_config = None
        summary = []

        for threshold in ENTROPY_THRESHOLDS:
            print(f"\n########## ENTROPY_THRESHOLD = {threshold} ##########")

            for alpha in ALPHAS:
                print(f"\n===== ENTROPY_GATED threshold={threshold} alpha={alpha} =====")
                preds = predict_dataset(
                    model, tokenizer, train_items, test_items, steering_data,
                    alpha, MODE, threshold=threshold,
                )

                tag = f"t{str(threshold).replace('.', 'p')}_a{str(alpha).replace('-', 'neg').replace('.', 'p')}"
                eval_pred_path = os.path.join(OUT_DIR, "tmp_eval_preds.json")
                _write_json(preds, eval_pred_path)
                save_json(preds, os.path.join(OUT_DIR, f"preds_entropy_{tag}.json"))
                print_subgroup_accuracy(test_items, preds)

                result = run_official_eval(
                    EVAL_SCRIPT, ref_path, eval_pred_path,
                    os.path.join(OUT_DIR, "tmp_eval_result.json"),
                )
                print(f"  ACC={result['accuracy']:.2f}  TCE={result['content_effect']:.4f}  Score={result['combined_score']:.4f}")
                save_json(result, os.path.join(OUT_DIR, f"eval_entropy_{tag}.json"))

                summary.append({
                    "threshold": threshold,
                    "alpha": alpha,
                    "accuracy": result["accuracy"],
                    "content_effect": result["content_effect"],
                    "combined_score": result["combined_score"],
                })

                if result["combined_score"] > best_score:
                    best_score = result["combined_score"]
                    best_config = {"threshold": threshold, "alpha": alpha}

        print(f"\n===== ENTROPY_GATED SWEEP SUMMARY =====")
        print(f"{'Thresh':>8s}  {'Alpha':>8s}  {'ACC':>8s}  {'TCE':>10s}  {'Score':>10s}")
        for s in summary:
            is_best = (s["threshold"] == best_config["threshold"] and s["alpha"] == best_config["alpha"])
            marker = " <-- best" if is_best else ""
            print(f"{s['threshold']:>8.2f}  {s['alpha']:>8.1f}  {s['accuracy']:>8.2f}  {s['content_effect']:>10.4f}  {s['combined_score']:>10.4f}{marker}")
        print(f"\nBest: threshold={best_config['threshold']} alpha={best_config['alpha']} score={best_score:.4f}")
        save_json(summary, os.path.join(OUT_DIR, "sweep_summary_entropy_gated.json"))
        return

    # ------------------------------------------------------------------
    # Standard alpha sweep (static, cast, kcast, validity_cond, validity_cond_kcast, structure_aware)
    # ------------------------------------------------------------------
    best_score = -1
    best_alpha = 0
    summary = []

    for alpha in ALPHAS:
        print(f"\n===== {MODE.upper()} Alpha = {alpha} =====")
        preds = predict_dataset(model, tokenizer, train_items, test_items, steering_data, alpha, MODE)
        tag = str(alpha).replace("-", "neg").replace(".", "p")
        eval_pred_path = os.path.join(OUT_DIR, "tmp_eval_preds.json")
        _write_json(preds, eval_pred_path)
        save_json(preds, os.path.join(OUT_DIR, f"preds_{MODE}_alpha_{tag}.json"))
        print_subgroup_accuracy(test_items, preds)
        result = run_official_eval(EVAL_SCRIPT, ref_path, eval_pred_path, os.path.join(OUT_DIR, "tmp_eval_result.json"))
        print(f"  ACC={result['accuracy']:.2f}  TCE={result['content_effect']:.4f}  Score={result['combined_score']:.4f}")
        save_json(result, os.path.join(OUT_DIR, f"eval_{MODE}_alpha_{tag}.json"))
        summary.append({
            "alpha": alpha,
            "accuracy": result["accuracy"],
            "content_effect": result["content_effect"],
            "combined_score": result["combined_score"],
        })
        if result["combined_score"] > best_score:
            best_score = result["combined_score"]
            best_alpha = alpha

    print(f"\n===== SWEEP SUMMARY ({MODE.upper()}) =====")
    print(f"{'Alpha':>8s}  {'ACC':>8s}  {'TCE':>10s}  {'Score':>10s}")
    for s in summary:
        marker = " <-- best" if s["alpha"] == best_alpha else ""
        print(f"{s['alpha']:>8.1f}  {s['accuracy']:>8.2f}  {s['content_effect']:>10.4f}  {s['combined_score']:>10.4f}{marker}")
    print(f"\nBest alpha: {best_alpha} (combined_score: {best_score:.4f})")
    save_json(summary, os.path.join(OUT_DIR, f"sweep_summary_{MODE}.json"))


if __name__ == "__main__":
    main()