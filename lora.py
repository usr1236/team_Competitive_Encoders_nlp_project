"""
 LoRA Fine-Tuning for Syllogistic Reasoning
=====================================================
4 strategies selectable via STRATEGY flag:

  "simple"        = Option 1: validity classification only
  "contrastive"   = Option 2: validity + plausibility-invariance pairs
  "orthogonal"    = Option 3: multi-task with orthogonal separation
  "adversarial"   = Option 4: multi-task with gradient reversal

Fixes applied:
  [1] Single AutoModelForCausalLM for both generation loss and representations.
      We use logits for causal LM loss and hidden_states (pre-lm_head) for
      representation learning. No separate AutoModel needed.
  [2] Loss normalization: each loss component is scale-normalized before
      weighting so no single term dominates.
  [3] Improved contrastive pairing: cross-validity pairs added as negative
      examples (same plausibility, different validity) for harder contrast.
  [4] Weighted attention pooling: last_token still default for decoder,
      but added attention_weighted option using attention scores as weights.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import importlib.util

# ============================================================
# CONFIG
# ============================================================
STRATEGY = "contrastive"  # "simple" | "contrastive" | "orthogonal" | "adversarial"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_JSON = "train_data/subtask 1/train_data.json"
TEST_JSON = "test_data/subtask 1/test_data_subtask_1.json"
EVAL_SCRIPT = "evaluation_kit/task 1 & 3/evaluation_script.py"
OUTPUT_DIR = f"lora_{STRATEGY}_output"
HF_TOKEN = os.getenv("HF_TOKEN", "")

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # reduced from 4 to 2 modules to save ~40% LoRA memory

EPOCHS = 7
BATCH_SIZE = 1            # reduced from 4 — contrastive does 2 forward passes per step
GRAD_ACCUM = 16           # effective batch = 1 * 16 = 16 (same as before)
LR = 2e-4
HEAD_LR = 5e-4
WARMUP_RATIO = 0.1
MAX_LEN = 384             # reduced from 512 — dataset max is 40 words (~60 tokens), 384 is plenty
SEED = 42

# Contrastive
LAMBDA_CONTRASTIVE = 0.5
CONTRASTIVE_LAYER_FRAC = 0.25
USE_NEGATIVE_PAIRS = True  # [FIX 3] add cross-validity negative pairs

# Orthogonal/Adversarial
LAMBDA_PLAUS = 0.3
LAMBDA_ORTH = 0.3          # [FIX 2] reduced from 1.0, gets normalized anyway
LAMBDA_DECORR = 0.1
LAMBDA_VAL_HEAD = 0.3
GRL_LAMBDA = 1.0
HEAD_PROJ_DIM = 256
EXTRACT_LAYER = -1

# [FIX 4] "last_token" | "mean_pool" | "attention_weighted"
POOL_MODE = "last_token"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def log_gpu_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU {tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")

def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def save_json(o, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(o, f, indent=2, ensure_ascii=False)


# ============================================================
# DATASETS
# ============================================================

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def _prompt(self, syl):
        msgs = [
            {"role": "system", "content": "You are a strict formal logic reasoner. Decide only whether the conclusion logically follows from the premises. Ignore plausibility and world knowledge. Reply with only yes or no."},
            {"role": "user", "content": f"Argument:\n{syl}\n\nAnswer yes or no."}
        ]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = self._prompt(ex["syllogism"])
        target = " yes" if ex["validity"] else " no"
        full = prompt + target
        enc = self.tokenizer(full, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        penc = self.tokenizer(prompt, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "prompt_len": penc.input_ids.shape[1],
            "validity": 1 if ex["validity"] else 0,
            "plausibility": 1 if ex["plausibility"] else 0,
        }


# [FIX 3] Improved pairing with negative examples
class PairDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, use_negatives=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []

        vp = [x for x in data if x["validity"] and x["plausibility"]]
        vi = [x for x in data if x["validity"] and not x["plausibility"]]
        ip = [x for x in data if not x["validity"] and x["plausibility"]]
        ii = [x for x in data if not x["validity"] and not x["plausibility"]]

        rng = random.Random(SEED)

        # Positive pairs: same validity, different plausibility → push together
        def pos_pair(a, b, val):
            rng.shuffle(a); rng.shuffle(b)
            n = min(len(a), len(b))
            for i in range(n):
                self.pairs.append({
                    "syl_a": a[i]["syllogism"], "syl_b": b[i]["syllogism"],
                    "validity": val, "pair_type": "positive"
                })

        pos_pair(list(vp), list(vi), True)
        pos_pair(list(ip), list(ii), False)

        # Negative pairs: same plausibility, different validity → push apart
        if use_negatives:
            def neg_pair(a, b):
                rng.shuffle(a); rng.shuffle(b)
                n = min(len(a), len(b))
                for i in range(n):
                    self.pairs.append({
                        "syl_a": a[i]["syllogism"], "syl_b": b[i]["syllogism"],
                        "validity_a": a[i]["validity"], "validity_b": b[i]["validity"],
                        "pair_type": "negative"
                    })
            neg_pair(list(vp), list(ip))  # plausible: valid vs invalid
            neg_pair(list(vi), list(ii))  # implausible: valid vs invalid

        rng.shuffle(self.pairs)
        n_pos = sum(1 for p in self.pairs if p["pair_type"] == "positive")
        n_neg = sum(1 for p in self.pairs if p["pair_type"] == "negative")
        print(f"Pairs: {n_pos} positive (push together), {n_neg} negative (push apart)")

    def __len__(self): return len(self.pairs)

    def _prompt(self, syl):
        msgs = [
            {"role": "system", "content": "You are a strict formal logic reasoner. Decide only whether the conclusion logically follows from the premises. Ignore plausibility and world knowledge. Reply with only yes or no."},
            {"role": "user", "content": f"Argument:\n{syl}\n\nAnswer yes or no."}
        ]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        out = {"pair_type": 0 if p["pair_type"] == "positive" else 1}

        if p["pair_type"] == "positive":
            target_a = " yes" if p["validity"] else " no"
            target_b = target_a
        else:
            target_a = " yes" if p["validity_a"] else " no"
            target_b = " yes" if p["validity_b"] else " no"

        for suffix, syl, target in [("_a", p["syl_a"], target_a), ("_b", p["syl_b"], target_b)]:
            prompt = self._prompt(syl)
            full = prompt + target
            enc = self.tokenizer(full, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
            penc = self.tokenizer(prompt, truncation=True, max_length=self.max_len, return_tensors="pt")
            out[f"input_ids{suffix}"] = enc.input_ids.squeeze(0)
            out[f"attention_mask{suffix}"] = enc.attention_mask.squeeze(0)
            out[f"prompt_len{suffix}"] = penc.input_ids.shape[1]

        return out


# ============================================================
# HEADS & GRADIENT REVERSAL
# ============================================================

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

class GRL(nn.Module):
    def __init__(self, lam=1.0): super().__init__(); self.lam = lam
    def forward(self, x): return GRLFunction.apply(x, self.lam)

class DisentangleHeads(nn.Module):
    def __init__(self, hidden_dim, proj_dim, use_grl=False, grl_lam=1.0):
        super().__init__()
        self.val_proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1))
        self.val_cls = nn.Linear(proj_dim, 2)
        self.plaus_proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1))
        self.plaus_cls = nn.Linear(proj_dim, 2)
        self.grl = GRL(grl_lam) if use_grl else nn.Identity()

    def forward(self, h):
        hv = self.val_proj(h)
        hp = self.plaus_proj(self.grl(h))
        return self.val_cls(hv), self.plaus_cls(hp), hv, hp


# ============================================================
# LOSS HELPERS
# ============================================================

def causal_lm_loss(logits, input_ids, attention_mask, prompt_len):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    mask = torch.zeros_like(shift_labels, dtype=torch.float32)
    for i in range(input_ids.shape[0]):
        pl = prompt_len[i].item() if isinstance(prompt_len, torch.Tensor) else prompt_len[i]
        end = attention_mask[i].sum().item() - 1
        mask[i, max(pl - 1, 0):end] = 1.0
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").view(shift_labels.shape)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


# [FIX 3] Updated to handle positive (push together) and negative (push apart) pairs
def contrastive_repr_loss(hidden_a, hidden_b, mask_a, mask_b, layer_frac, pair_type):
    n_layers = len(hidden_a)
    start = int(n_layers * (1.0 - layer_frac))
    total = 0.0
    count = 0
    for li in range(start, n_layers):
        ha, hb = hidden_a[li], hidden_b[li]
        pos_a = mask_a.sum(dim=1) - 1
        pos_b = mask_b.sum(dim=1) - 1
        ra = ha[torch.arange(ha.size(0)), pos_a].float()
        rb = hb[torch.arange(hb.size(0)), pos_b].float()
        cos_sim = F.cosine_similarity(ra, rb, dim=-1)

        # Per-example: positive pairs → minimize distance, negative → maximize distance
        # pair_type: 0=positive, 1=negative
        pt = pair_type.float()
        # positive: loss = 1 - cos_sim  (push together)
        # negative: loss = max(0, cos_sim - margin)  (push apart)
        margin = 0.2
        pos_loss = (1.0 - cos_sim) * (1.0 - pt)
        neg_loss = torch.clamp(cos_sim - margin, min=0.0) * pt
        total += (pos_loss + neg_loss).mean()
        count += 1
    return total / max(count, 1)


def orth_loss(hv, hp):
    hv_n = F.normalize(hv, dim=-1)
    hp_n = F.normalize(hp, dim=-1)
    cc = (hv_n.T @ hp_n) / hv.shape[0]
    return torch.sum(cc ** 2)


def decorr_loss(hv, hp):
    def off_diag(h):
        hn = F.normalize(h, dim=0)
        c = hn.T @ hn / h.shape[0]
        m = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
        return torch.sum(c[m] ** 2)
    return off_diag(hv) + off_diag(hp)


# [FIX 4] Pooling with attention_weighted option
def pool_repr(hidden_states, attention_mask, layer_idx=-1, mode="last_token", attentions=None):
    h = hidden_states[layer_idx]  # (batch, seq, hidden)

    if mode == "last_token":
        pos = attention_mask.sum(dim=1) - 1
        return h[torch.arange(h.size(0), device=h.device), pos].float()

    elif mode == "mean_pool":
        mask = attention_mask.unsqueeze(-1).float()
        return ((h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).float()

    elif mode == "attention_weighted":
        # Use attention weights from last layer as importance scores
        # attentions[-1] shape: (batch, heads, seq, seq)
        if attentions is not None and len(attentions) > 0:
            attn = attentions[-1]  # last layer attention
            # Average across heads, take attention TO the last token
            last_pos = attention_mask.sum(dim=1) - 1  # (batch,)
            # Gather attention weights for last token attending to all positions
            attn_weights = torch.zeros(h.size(0), h.size(1), device=h.device)
            for i in range(h.size(0)):
                lp = last_pos[i].item()
                # Mean across heads: how much the last token attends to each position
                attn_weights[i, :lp+1] = attn[i, :, lp, :lp+1].mean(dim=0)
            # Normalize
            attn_weights = attn_weights * attention_mask.float()
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            # Weighted sum
            return (h * attn_weights.unsqueeze(-1)).sum(dim=1).float()
        else:
            # Fallback to last_token if no attention weights
            pos = attention_mask.sum(dim=1) - 1
            return h[torch.arange(h.size(0), device=h.device), pos].float()

    raise ValueError(f"Unknown pool mode: {mode}")


# [FIX 2] Loss normalizer: tracks running stats to keep losses on similar scales
class LossNormalizer:
    """
    Tracks EMA of each loss component's magnitude.
    Divides each loss by its EMA so all components are ~1.0 before weighting.
    First warmup_steps use simple averaging instead of EMA for stability.
    """
    def __init__(self, keys, momentum=0.9, warmup_steps=50):
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.ema = {k: 1.0 for k in keys}
        self.initialized = {k: False for k in keys}
        self.step_count = {k: 0 for k in keys}
        self.warmup_sum = {k: 0.0 for k in keys}

    def normalize(self, key, loss_val):
        v = loss_val.detach().item()
        if v < 1e-8:
            return loss_val
        self.step_count[key] += 1
        if self.step_count[key] <= self.warmup_steps:
            self.warmup_sum[key] += v
            self.ema[key] = self.warmup_sum[key] / self.step_count[key]
            self.initialized[key] = True
        else:
            self.ema[key] = self.momentum * self.ema[key] + (1 - self.momentum) * v
            self.initialized[key] = True
        return loss_val / (self.ema[key] + 1e-8)


# ============================================================
# INFERENCE
# ============================================================

def score_label(model, tokenizer, prompt, label_text, max_len):
    full = prompt + label_text
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len).input_ids.to(model.device)
    with torch.no_grad():
        logits = model(full_ids, use_cache=False).logits[:, :-1, :]
        target = full_ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return lp[0, max(prompt_ids.shape[1] - 1, 0):].sum().item()


@torch.no_grad()
def evaluate(model, tokenizer, test_path, out_path):
    model.eval()
    data = load_json(test_path)
    preds = []
    for i, ex in enumerate(data):
        msgs = [
            {"role": "system", "content": "You are a strict formal logic reasoner. Decide only whether the conclusion logically follows from the premises. Ignore plausibility and world knowledge. Reply with only yes or no."},
            {"role": "user", "content": f"Argument:\n{ex['syllogism']}\n\nAnswer yes or no."}
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        y = score_label(model, tokenizer, prompt, " yes", MAX_LEN)
        n = score_label(model, tokenizer, prompt, " no", MAX_LEN)
        preds.append({"id": ex["id"], "validity": bool(y >= n)})
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(data)}")
    save_json(preds, out_path)
    print(f"Predictions: {out_path}")
    return preds


def run_eval(pred_path):
    if not os.path.exists(EVAL_SCRIPT):
        print("Eval script not found, skipping official eval")
        return None
    spec = importlib.util.spec_from_file_location("ev", EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = os.path.join(OUTPUT_DIR, "eval_results.json")
    mod.run_full_scoring(TEST_JSON, pred_path, out)
    r = load_json(out)
    print(f"ACC={r['accuracy']:.2f}  TCE={r['content_effect']:.4f}  Score={r['combined_score']:.4f}")
    return r


def subgroup_acc(test_path, pred_path):
    data = load_json(test_path)
    preds = {p["id"]: p["validity"] for p in load_json(pred_path)}
    buckets = {}
    for ex in data:
        v = "valid" if ex["validity"] else "invalid"
        p = "plausible" if ex["plausibility"] else "implausible"
        k = f"{p}_{v}"
        buckets.setdefault(k, {"c": 0, "t": 0})
        buckets[k]["t"] += 1
        if preds.get(ex["id"]) == ex["validity"]:
            buckets[k]["c"] += 1
    print("\nSubgroup accuracy:")
    for k in sorted(buckets):
        b = buckets[k]
        print(f"  {k:30s}: {100*b['c']/b['t']:6.2f}% ({b['c']}/{b['t']})")


def quick_eval(model, tokenizer, label):
    """Run full eval (ACC, CE, Score), print results, return to caller."""
    print(f"\n{'='*60}")
    print(f"  EVAL: {label}")
    print(f"{'='*60}")
    pred_path = os.path.join(OUTPUT_DIR, f"predictions_{label}.json")
    evaluate(model, tokenizer, TEST_JSON, pred_path)
    subgroup_acc(TEST_JSON, pred_path)
    result = run_eval(pred_path)
    print(f"{'='*60}\n")
    return result


# ============================================================
# TRAINING
# ============================================================

def train():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    token = HF_TOKEN or None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Strategy: {STRATEGY}")
    print(f"Model: {MODEL_NAME}")
    print(f"Pool mode: {POOL_MODE}")

    # [FIX 1] Single AutoModelForCausalLM for both generation loss and representations.
    # We use logits for causal LM loss and hidden_states (pre-lm_head) for
    # representation learning via the disentanglement heads.
    # No separate AutoModel needed — hidden_states are encoder representations.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=token,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )
    model = model.to(DEVICE)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
                          lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias="none")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing: trades compute for ~40% less activation memory
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    torch.cuda.empty_cache()
    log_gpu_mem("after model+lora")

    train_data = load_json(TRAIN_JSON)
    use_heads = STRATEGY in ("orthogonal", "adversarial")
    use_pairs = STRATEGY == "contrastive"
    need_attn = (POOL_MODE == "attention_weighted")

    if use_pairs:
        dataset = PairDataset(train_data, tokenizer, MAX_LEN, use_negatives=USE_NEGATIVE_PAIRS)
    else:
        dataset = SimpleDataset(train_data, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    heads = None
    param_groups = [{"params": model.parameters(), "lr": LR, "weight_decay": 0.01}]
    if use_heads:
        heads = DisentangleHeads(
            model.config.hidden_size, HEAD_PROJ_DIM,
            use_grl=(STRATEGY == "adversarial"), grl_lam=GRL_LAMBDA
        ).to(DEVICE)
        param_groups.append({"params": heads.parameters(), "lr": HEAD_LR, "weight_decay": 0.01})
        print(f"Head params: {sum(p.numel() for p in heads.parameters()):,}")

    optimizer = torch.optim.AdamW(param_groups)
    total_steps = (len(dataloader) * EPOCHS) // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    # [FIX 2] Initialize loss normalizer
    loss_keys = ["cls", "con", "vh", "ph", "ort", "dec"]
    loss_norm = LossNormalizer(loss_keys)

    print(f"Dataset: {len(dataset)} examples, {len(dataloader)} batches/epoch")
    print(f"Total steps: {total_steps}\n")

    # ---- EVAL BEFORE TRAINING (baseline) ----
    quick_eval(model, tokenizer, "before_training")

    model.train()
    if heads: heads.train()
    history = []
    best_loss = float("inf")
    best_epoch = -1
    best_dir = os.path.join(OUTPUT_DIR, "best")

    for epoch in range(EPOCHS):
        m = {"cls": 0, "con": 0, "val_h": 0, "plaus_h": 0, "ort": 0, "dec": 0, "total": 0, "n": 0}

        for bi, batch in enumerate(dataloader):
            # ---- SIMPLE / ORTHOGONAL / ADVERSARIAL ----
            if not use_pairs:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                plen = batch["prompt_len"]
                val_lbl = batch["validity"].to(DEVICE)
                plaus_lbl = batch["plausibility"].to(DEVICE)

                out = model(ids, attention_mask=mask,
                            output_hidden_states=use_heads,
                            output_attentions=need_attn,
                            use_cache=False)

                cls = causal_lm_loss(out.logits, ids, mask, plen)
                # [FIX 2] Normalize before weighting
                loss = loss_norm.normalize("cls", cls) * 1.0
                m["cls"] += cls.item() * ids.shape[0]

                if use_heads:
                    attns = out.attentions if need_attn else None
                    rep = pool_repr(out.hidden_states, mask, EXTRACT_LAYER, POOL_MODE, attns)
                    vl, pl, hv, hp = heads(rep)
                    vh_loss = F.cross_entropy(vl, val_lbl)
                    ph_loss = F.cross_entropy(pl, plaus_lbl)
                    ol = orth_loss(hv, hp)
                    dl = decorr_loss(hv, hp)

                    # [FIX 2] Each component normalized then weighted
                    loss = loss + LAMBDA_VAL_HEAD * loss_norm.normalize("vh", vh_loss)
                    loss = loss + LAMBDA_PLAUS * loss_norm.normalize("ph", ph_loss)
                    loss = loss + LAMBDA_ORTH * loss_norm.normalize("ort", ol)
                    loss = loss + LAMBDA_DECORR * loss_norm.normalize("dec", dl)

                    m["val_h"] += vh_loss.item() * ids.shape[0]
                    m["plaus_h"] += ph_loss.item() * ids.shape[0]
                    m["ort"] += ol.item() * ids.shape[0]
                    m["dec"] += dl.item() * ids.shape[0]

                m["total"] += loss.item() * ids.shape[0]
                m["n"] += ids.shape[0]

            # ---- CONTRASTIVE ----
            else:
                ids_a = batch["input_ids_a"].to(DEVICE)
                mask_a = batch["attention_mask_a"].to(DEVICE)
                plen_a = batch["prompt_len_a"]
                ids_b = batch["input_ids_b"].to(DEVICE)
                mask_b = batch["attention_mask_b"].to(DEVICE)
                plen_b = batch["prompt_len_b"]
                pair_type = batch["pair_type"].to(DEVICE)

                # Sequential forward passes to halve peak memory.
                # Pass A: compute CLS loss with grad, cache hidden states for contrastive loss
                out_a = model(ids_a, attention_mask=mask_a, output_hidden_states=True, use_cache=False)
                cls_a = causal_lm_loss(out_a.logits, ids_a, mask_a, plen_a)
                # Detach hidden states — no grad through them for contrastive loss
                hidden_a = tuple(h.detach() for h in out_a.hidden_states)
                del out_a
                torch.cuda.empty_cache()

                # Pass B: compute CLS loss with grad, cache hidden states
                out_b = model(ids_b, attention_mask=mask_b, output_hidden_states=True, use_cache=False)
                cls_b = causal_lm_loss(out_b.logits, ids_b, mask_b, plen_b)
                hidden_b = tuple(h.detach() for h in out_b.hidden_states)
                del out_b
                torch.cuda.empty_cache()

                cls = (cls_a + cls_b) / 2.0

                # Contrastive loss on detached hiddens (no second-order grad needed)
                con = contrastive_repr_loss(
                    hidden_a, hidden_b,
                    mask_a, mask_b, CONTRASTIVE_LAYER_FRAC, pair_type
                )
                del hidden_a, hidden_b

                # [FIX 2] Normalized loss mixing
                loss = loss_norm.normalize("cls", cls) * 1.0 + LAMBDA_CONTRASTIVE * loss_norm.normalize("con", con)

                bs = ids_a.shape[0]
                m["cls"] += cls.item() * bs
                m["con"] += con.item() * bs
                m["total"] += loss.item() * bs
                m["n"] += bs

            (loss / GRAD_ACCUM).backward()

            if (bi + 1) % GRAD_ACCUM == 0:
                params = list(model.parameters()) + (list(heads.parameters()) if heads else [])
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if (bi + 1) % 30 == 0:
                n = m["n"]
                parts = [f"cls={m['cls']/n:.4f}"]
                if use_pairs: parts.append(f"con={m['con']/n:.4f}")
                if use_heads: parts.extend([f"vh={m['val_h']/n:.4f}", f"ph={m['plaus_h']/n:.4f}", f"ort={m['ort']/n:.6f}"])
                # [FIX 2] Show EMA scales so you can monitor normalization
                scales = " | scales: " + " ".join(f"{k}={loss_norm.ema[k]:.4f}" for k in loss_keys if loss_norm.initialized[k])
                print(f"  E{epoch+1} [{bi+1}/{len(dataloader)}] {' '.join(parts)}{scales}")

        n = m["n"]
        ep = {"epoch": epoch + 1, "cls": m["cls"]/n, "total": m["total"]/n}
        if use_pairs: ep["contrastive"] = m["con"]/n
        if use_heads: ep.update({"val_head": m["val_h"]/n, "plaus_head": m["plaus_h"]/n, "orth": m["ort"]/n})
        history.append(ep)
        print(f"\n  Epoch {epoch+1}: {ep}\n")

        # ---- EVAL AFTER EPOCH 1 ONLY ----
        if epoch == 0:
            quick_eval(model, tokenizer, "after_epoch1")
            model.train()
            if heads: heads.train()

        # ---- SAVE ONLY IF BEST (by total loss) ----
        epoch_total = m["total"] / n
        if epoch_total < best_loss:
            best_loss = epoch_total
            best_epoch = epoch + 1
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir); tokenizer.save_pretrained(best_dir)
            if heads: torch.save(heads.state_dict(), os.path.join(best_dir, "heads.pt"))
            print(f"  >> New best! total_loss={best_loss:.4f} — saved to {best_dir}\n")
        else:
            print(f"  >> total_loss={epoch_total:.4f} did not beat best={best_loss:.4f} (epoch {best_epoch})\n")

    save_json(history, os.path.join(OUTPUT_DIR, "history.json"))
    print(f"Best checkpoint: epoch {best_epoch} total_loss={best_loss:.4f} — at {best_dir}")
    return model, tokenizer, heads


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    model, tokenizer, heads = train()
    # Final eval on best model (which is current model if last epoch was best,
    # otherwise load best checkpoint)
    pred_path = os.path.join(OUTPUT_DIR, "predictions.json")
    evaluate(model, tokenizer, TEST_JSON, pred_path)
    subgroup_acc(TEST_JSON, pred_path)
    run_eval(pred_path)