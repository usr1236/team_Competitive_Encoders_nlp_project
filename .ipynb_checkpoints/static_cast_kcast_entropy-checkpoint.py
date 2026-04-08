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

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HOME = os.getenv("HF_HOME", "")
TRAIN_JSON = "train_data/subtask 1/train_data.json"
TEST_JSON = "test_data/subtask 1/test_data_subtask_1.json"
EVAL_SCRIPT = "evaluation_kit/task 1 & 3/evaluation_script.py"
OUT_DIR = "results_novelty"

SEED = 42
MAX_LEN = 512
MAX_STEER_EXAMPLES = 2400
ALPHAS = [0,  -3, -2, -1, -0.5, 0.5, 1, 2, 3, 7, 10]
LAYER_FRACTION = 0.25
NORMALIZE_VECTORS = False
USE_BFLOAT16 = True
USE_CHAT_TEMPLATE = True
YES_TEXT = " yes"
NO_TEXT = " no"

PROMPT_MODE = "icl"
ICL_SHOTS = 4

# "static" / "cast" / "kcast"
# "layerwise"          = novelty 1: per-layer alpha optimization
# "validity_cond"      = novelty 2: separate vectors for valid vs invalid
# "entropy_gated"      = novelty 3: entropy-based alpha scaling
# "structure_aware"    = novelty 5: syllogistic form aware contrastive pairs
# "layerwise_entropy"  = novelty 1+3 combined
MODE = "kcast"

KCAST_K = 10
ENTROPY_THRESHOLD = 0.5
LAYERWISE_ALPHA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
LAYERWISE_MAX_COMBOS = 5000

if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def get_target_layers(total, fraction):
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
                {"role": "system", "content": "You are a strict formal logic reasoner. Decide only whether the conclusion logically follows from the premises. Ignore plausibility and world knowledge. Reply with only yes or no."}
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
            {"role": "system", "content": "You are a strict formal logic reasoner. Decide only whether the conclusion logically follows from the premises. Ignore plausibility and world knowledge. Reply with only yes or no."},
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
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len).input_ids.to(model.device)
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
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False)
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


def project_onto(v, u):
    return (torch.dot(v, u) / (torch.dot(u, u) + 1e-10)) * u


# ============================================================
# NOVELTY 5: Structure-aware contrastive pairs
# ============================================================

def extract_syllogistic_form(text):
    text_lower = text.lower()
    quantifiers = ["not all", "no", "all", "some", "every", "there are no", "it is not the case that all",
                   "it is not the case that every", "there exist some", "every single", "it is certain that no",
                   "it is certain that every", "it is also true that every", "it is known that some",
                   "it is known that every", "it is known that no"]
    found = []
    for q in sorted(quantifiers, key=len, reverse=True):
        if q in text_lower:
            found.append(q)
    if "therefore" in text_lower or "consequently" in text_lower or "it follows" in text_lower or "this has led" in text_lower:
        found.append("CONCLUSION_MARKER")
    neg_count = text_lower.count("not") + text_lower.count("no ")
    form_key = "|".join(sorted(found)) + f"|neg{neg_count}"
    return form_key


def build_structure_aware_pool(train_items, max_examples, seed):
    from collections import defaultdict
    rng = random.Random(seed)

    groups = defaultdict(lambda: {"valid_plausible": [], "valid_implausible": [],
                                   "invalid_plausible": [], "invalid_implausible": []})
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
        rng.shuffle(vp)
        rng.shuffle(vi)
        rng.shuffle(ip)
        rng.shuffle(ii)
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


# ============================================================
# Core steering data computation
# ============================================================

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
                if is_correct:
                    valid_correct_acts[l].append(h)
                else:
                    valid_wrong_acts[l].append(h)
            else:
                invalid_acts[l].append(h)
                if is_correct:
                    invalid_correct_acts[l].append(h)
                else:
                    invalid_wrong_acts[l].append(h)
            all_acts_with_validity[l].append((h, 1 if gold else -1))

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

    cond_valid = {}
    cond_invalid = {}
    for l in target_layers:
        cond_valid[l] = torch.stack(valid_acts[l]).mean(dim=0)
        cond_invalid[l] = torch.stack(invalid_acts[l]).mean(dim=0)

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


# ============================================================
# Hooks
# ============================================================

class StaticHook:
    def __init__(self, delta, alpha):
        self.delta = delta
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            d = self.delta.to(x.device, x.dtype)
            return (x + self.alpha * d.view(1, 1, -1),) + output[1:]
        d = self.delta.to(output.device, output.dtype)
        return output + self.alpha * d.view(1, 1, -1)


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
        proj_valid = project_onto(h, self.psi_valid)
        proj_invalid = project_onto(h, self.psi_invalid)
        sim_valid = cosine_sim(h, proj_valid).item()
        sim_invalid = cosine_sim(h, proj_invalid).item()
        if sim_valid > sim_invalid:
            effective_alpha = -self.alpha
        else:
            effective_alpha = self.alpha
        d = self.delta.to(x.device, x.dtype)
        x = x + effective_alpha * d.view(1, 1, -1)
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
        d = self.delta.to(x.device, x.dtype)
        x = x + effective_alpha * d.view(1, 1, -1)
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
        proj_valid = project_onto(h, self.psi_valid)
        proj_invalid = project_onto(h, self.psi_invalid)
        sim_valid = cosine_sim(h, proj_valid).item()
        sim_invalid = cosine_sim(h, proj_invalid).item()
        if sim_valid > sim_invalid:
            delta = self.valid_delta
            effective_alpha = -self.alpha
        else:
            delta = self.invalid_delta
            effective_alpha = self.alpha
        d = delta.to(x.device, x.dtype)
        x = x + effective_alpha * d.view(1, 1, -1)
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
        d = delta.to(x.device, x.dtype)
        x = x + effective_alpha * d.view(1, 1, -1)
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


# ============================================================
# Prediction functions
# ============================================================

@torch.no_grad()
def predict_dataset(model, tokenizer, train_items, eval_items, steering_data, alpha, mode, layer_alphas=None):
    layers = get_layers(model)

    if mode == "entropy_gated" or mode == "layerwise_entropy":
        return predict_entropy_gated(model, tokenizer, train_items, eval_items, steering_data, alpha, layer_alphas)

    handles = []
    if alpha != 0:
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
def predict_entropy_gated(model, tokenizer, train_items, eval_items, steering_data, alpha_base, layer_alphas=None):
    layers = get_layers(model)
    deltas = steering_data["deltas"]
    preds = []

    for i, ex in enumerate(eval_items):
        prompt = build_prompt(tokenizer, train_items, ex["syllogism"])
        norm_entropy, _ = get_prediction_entropy(model, tokenizer, prompt, MAX_LEN)

        if norm_entropy < ENTROPY_THRESHOLD:
            scale = (norm_entropy / ENTROPY_THRESHOLD) * 0.5
        else:
            scale = norm_entropy

        handles = []
        for layer_idx, delta in deltas.items():
            base_a = layer_alphas[layer_idx] if layer_alphas else alpha_base
            a = base_a * scale
            handles.append(layers[layer_idx].register_forward_hook(StaticHook(delta, a)))

        pred = predict_validity(model, tokenizer, prompt, MAX_LEN)
        remove_hooks(handles)

        preds.append({"id": ex["id"], "validity": bool(pred)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eval_items)} (entropy_scale={scale:.3f})")

    return preds


# ============================================================
# Novelty 1: Layer-wise alpha optimization
# ============================================================

def optimize_layerwise_alphas(model, tokenizer, train_items, eval_items, steering_data, ref_path):
    target_layers = sorted(steering_data["deltas"].keys())
    n_layers = len(target_layers)
    alpha_vals = LAYERWISE_ALPHA_VALUES

    total_combos = len(alpha_vals) ** n_layers
    print(f"Layer-wise optimization: {n_layers} layers x {len(alpha_vals)} values = {total_combos} combos")

    if total_combos > LAYERWISE_MAX_COMBOS:
        print(f"Too many combos, using random search ({LAYERWISE_MAX_COMBOS} samples)")
        rng = random.Random(SEED)
        combos = []
        for _ in range(LAYERWISE_MAX_COMBOS):
            combo = tuple(rng.choice(alpha_vals) for _ in range(n_layers))
            combos.append(combo)
    else:
        combos = list(product(alpha_vals, repeat=n_layers))

    best_score = -1
    best_combo = None
    best_result = None

    for ci, combo in enumerate(combos):
        layer_alphas = {l: a for l, a in zip(target_layers, combo)}
        preds = predict_dataset(model, tokenizer, train_items, eval_items, steering_data, 0, "layerwise", layer_alphas)
        pred_path = os.path.join(OUT_DIR, "tmp_layerwise_preds.json")
        save_json(preds, pred_path)
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


# ============================================================
# Evaluation helpers
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Mode: {MODE}")
    print(f"Prompt: {PROMPT_MODE}" + (f" ({ICL_SHOTS} shots)" if PROMPT_MODE == "icl" else ""))
    print(f"Alphas: {ALPHAS}")
    if MODE == "kcast":
        print(f"K-CAST K={KCAST_K}")
    if MODE in ("entropy_gated", "layerwise_entropy"):
        print(f"Entropy threshold: {ENTROPY_THRESHOLD}")
    os.makedirs(OUT_DIR, exist_ok=True)

    token = HF_TOKEN if HF_TOKEN else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if USE_BFLOAT16 and device == "cuda" else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=token, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device != "cuda":
        model.to(device)
    model.eval()
    model.config.use_cache = False

    train_items = load_json(TRAIN_JSON)
    test_items = load_json(TEST_JSON)
    save_json(test_items, os.path.join(OUT_DIR, "test_reference.json"))
    ref_path = os.path.join(OUT_DIR, "test_reference.json")
    print(f"Train: {len(train_items)}, Test: {len(test_items)}")

    all_layers = get_layers(model)
    target_layers = get_target_layers(len(all_layers), LAYER_FRACTION)
    print(f"Total layers: {len(all_layers)}, Steering layers: {target_layers}")

    steering_data = compute_steering_data(model, tokenizer, train_items, target_layers)
    torch.save(steering_data, os.path.join(OUT_DIR, "steering_data.pt"))

    if MODE == "layerwise":
        best_layer_alphas, best_result = optimize_layerwise_alphas(model, tokenizer, train_items, test_items, steering_data, ref_path)
        save_json({"layer_alphas": {str(k): v for k, v in best_layer_alphas.items()}, "result": best_result}, os.path.join(OUT_DIR, "layerwise_best.json"))
        preds = predict_dataset(model, tokenizer, train_items, test_items, steering_data, 0, "layerwise", best_layer_alphas)
        pred_path = os.path.join(OUT_DIR, "preds_layerwise_best.json")
        save_json(preds, pred_path)
        print_subgroup_accuracy(test_items, preds)
        return

    if MODE == "layerwise_entropy":
        best_layer_alphas, _ = optimize_layerwise_alphas(model, tokenizer, train_items, test_items, steering_data, ref_path)
        print(f"\nRunning entropy-gated with optimized per-layer alphas...")
        preds = predict_entropy_gated(model, tokenizer, train_items, test_items, steering_data, 0, best_layer_alphas)
        pred_path = os.path.join(OUT_DIR, "preds_layerwise_entropy.json")
        save_json(preds, pred_path)
        print_subgroup_accuracy(test_items, preds)
        result = run_official_eval(EVAL_SCRIPT, ref_path, pred_path, os.path.join(OUT_DIR, "eval_layerwise_entropy.json"))
        print(f"  ACC={result['accuracy']:.2f}  TCE={result['content_effect']:.4f}  Score={result['combined_score']:.4f}")
        save_json({"layer_alphas": {str(k): v for k, v in best_layer_alphas.items()}, "result": result}, os.path.join(OUT_DIR, "layerwise_entropy_best.json"))
        return

    best_score = -1
    best_alpha = 0
    summary = []

    for alpha in ALPHAS:
        print(f"\n===== {MODE.upper()} Alpha = {alpha} =====")
        preds = predict_dataset(model, tokenizer, train_items, test_items, steering_data, alpha, MODE)
        tag = str(alpha).replace("-", "neg").replace(".", "p")
        pred_path = os.path.join(OUT_DIR, f"preds_{MODE}_alpha_{tag}.json")
        save_json(preds, pred_path)
        print_subgroup_accuracy(test_items, preds)
        result = run_official_eval(EVAL_SCRIPT, ref_path, pred_path, os.path.join(OUT_DIR, f"eval_{MODE}_alpha_{tag}.json"))
        print(f"  ACC={result['accuracy']:.2f}  TCE={result['content_effect']:.4f}  Score={result['combined_score']:.4f}")
        summary.append({"alpha": alpha, "accuracy": result["accuracy"], "content_effect": result["content_effect"], "combined_score": result["combined_score"]})
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