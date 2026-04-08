import os
import json
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import importlib.util

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_JSON = "train_data/subtask 1/train_data.json"
TEST_JSON = "test_data/subtask 1/test_data_subtask_1.json"
EVAL_SCRIPT = "evaluation_kit/task 1 & 3/evaluation_script.py"

SEED = 42
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-4
MAX_LEN = 512
OUT_DIR = "results_qwen_train_test_probe"
PROBE_EPOCHS = 3
PROBE_LR = 1e-3
PROBE_BATCH_SIZE = 16
LAYER_START_FRAC = 0.0

PROMPT_TEMPLATE = """You are a strict formal logic reasoner. Decide whether the conclusion follows logically from the premises. Ignore real-world plausibility and meaning. Only follow logical form.

Syllogism: {syllogism}

Answer with exactly one word: valid or invalid. Answer:"""

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] [{elapsed}<{remaining}, {rate_fmt}]"

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class SyllogismDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_len: int = 512):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        prompt = PROMPT_TEMPLATE.format(syllogism=it["syllogism"])
        enc = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        y = 1 if it.get("validity", False) is True else 0
        p = 1 if it.get("plausibility", False) is True else 0
        return {
            "id": it["id"],
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": torch.tensor(y, dtype=torch.long),
            "plausibility": torch.tensor(p, dtype=torch.long),
            "has_validity": torch.tensor(1 if "validity" in it else 0, dtype=torch.long),
            "has_plausibility": torch.tensor(1 if "plausibility" in it else 0, dtype=torch.long),
        }

@dataclass
class Batch:
    ids: List[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    plausibility: torch.Tensor
    has_validity: torch.Tensor
    has_plausibility: torch.Tensor

def collate_fn_with_pad(pad_id: int):
    def _collate(batch_list: List[Dict[str, Any]]) -> Batch:
        ids = [b["id"] for b in batch_list]
        max_len = max(b["input_ids"].shape[0] for b in batch_list)
        input_ids, attn, labels, plausibility, has_validity, has_plausibility = [], [], [], [], [], []
        for b in batch_list:
            x = b["input_ids"]
            m = b["attention_mask"]
            pad = max_len - x.shape[0]
            if pad > 0:
                x = torch.cat([x, torch.full((pad,), pad_id, dtype=x.dtype)], dim=0)
                m = torch.cat([m, torch.zeros((pad,), dtype=m.dtype)], dim=0)
            input_ids.append(x)
            attn.append(m)
            labels.append(b["label"])
            plausibility.append(b["plausibility"])
            has_validity.append(b["has_validity"])
            has_plausibility.append(b["has_plausibility"])
        return Batch(
            ids=ids,
            input_ids=torch.stack(input_ids, dim=0),
            attention_mask=torch.stack(attn, dim=0),
            labels=torch.stack(labels, dim=0),
            plausibility=torch.stack(plausibility, dim=0),
            has_validity=torch.stack(has_validity, dim=0),
            has_plausibility=torch.stack(has_plausibility, dim=0),
        )
    return _collate

class FrozenLMWithHead(nn.Module):
    def __init__(self, base_lm, hidden_size: int):
        super().__init__()
        self.lm = base_lm
        self.head = nn.Linear(hidden_size, 2)
        for p in self.lm.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        h = out.hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(h.shape[0], device=h.device)
        last_h = h[batch_idx, lengths, :].float()
        if self.head.weight.device != last_h.device:
            self.head = self.head.to(last_h.device)
        return self.head(last_h)

@torch.no_grad()
def evaluate_acc(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(loader, desc="Eval", leave=False, bar_format=BAR_FORMAT)
    for batch in pbar:
        mask = batch.has_validity.to(device).bool()
        if mask.sum().item() == 0:
            continue
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
    return 100.0 * correct / max(1, total)

def train_head(model, train_loader, device, lr: float, epochs: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    if not (device == "cuda" and hasattr(model.lm, "hf_device_map")):
        model.to(device)
    else:
        model.head.to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_path = os.path.join(OUT_DIR, "best_head.pt")
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {ep}/{epochs}", bar_format=BAR_FORMAT)
        total_loss = 0.0
        steps = 0
        t0 = time.time()
        for batch in pbar:
            mask = batch.has_validity.to(device).bool()
            if mask.sum().item() == 0:
                continue
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits[mask], labels[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            opt.step()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss became non-finite")
            total_loss += loss.item()
            steps += 1
            avg_loss = total_loss / max(1, steps)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")
        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {ep}: avg_train_loss={avg_loss:.4f} time={time.time()-t0:.1f}s")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.head.state_dict(), best_path)
    model.head.load_state_dict(torch.load(best_path, map_location=device))
    return model

@torch.no_grad()
def predict_to_json(model, items: List[Dict[str, Any]], tokenizer, batch_size: int, max_len: int, out_path: str, device: str):
    model.eval()
    ds = SyllogismDataset(items, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_pad(tokenizer.pad_token_id))
    preds = []
    pbar = tqdm(dl, desc="Predict", bar_format=BAR_FORMAT)
    for batch in pbar:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        for _id, yhat in zip(batch.ids, pred):
            preds.append({"id": _id, "validity": bool(yhat == 1)})
    save_json(preds, out_path)
    print("Wrote predictions:", out_path)

def run_official_eval(eval_script_path: str, ref_path: str, pred_path: str, out_path: str):
    spec = importlib.util.spec_from_file_location("semeval_eval", eval_script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.run_full_scoring(ref_path, pred_path, out_path)
    print("Official eval written to:", out_path)
    print(load_json(out_path))

@torch.no_grad()
def collect_layer_features(base_model, loader, device, start_frac=0.0):
    base_model.eval()
    all_hidden = None
    all_validity = []
    all_plausibility = []
    all_has_validity = []
    all_has_plausibility = []
    ids = []
    pbar = tqdm(loader, desc="CollectFeatures", bar_format=BAR_FORMAT)
    for batch in pbar:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        out = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        hs = out.hidden_states[1:]
        n_layers = len(hs)
        start = int(n_layers * start_frac)
        selected = []
        for l in range(start, n_layers):
            x = hs[l][batch_idx, lengths, :].detach().float().cpu()
            selected.append(x)
        selected = torch.stack(selected, dim=1)
        if all_hidden is None:
            all_hidden = [selected]
        else:
            all_hidden.append(selected)
        all_validity.append(batch.labels.cpu())
        all_plausibility.append(batch.plausibility.cpu())
        all_has_validity.append(batch.has_validity.cpu())
        all_has_plausibility.append(batch.has_plausibility.cpu())
        ids.extend(batch.ids)
    hidden = torch.cat(all_hidden, dim=0)
    validity = torch.cat(all_validity, dim=0)
    plausibility = torch.cat(all_plausibility, dim=0)
    has_validity = torch.cat(all_has_validity, dim=0)
    has_plausibility = torch.cat(all_has_plausibility, dim=0)
    return {
        "ids": ids,
        "hidden": hidden,
        "validity": validity,
        "plausibility": plausibility,
        "has_validity": has_validity,
        "has_plausibility": has_plausibility,
    }

class ProbeDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = X[mask]
        self.y = y[mask]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_probe(X_train, y_train, m_train, X_test, y_test, m_test, epochs, lr, batch_size, device):
    train_ds = ProbeDataset(X_train, y_train, m_train)
    test_ds = ProbeDataset(X_test, y_test, m_test)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    probe = nn.Linear(X_train.shape[-1], 2).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        probe.train()
        for xb, yb in train_dl:
            xb = xb.to(device).float()
            yb = yb.to(device)
            logits = probe(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device).float()
            yb = yb.to(device)
            pred = torch.argmax(probe(xb), dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return 100.0 * correct / max(1, total)

def run_layer_probe_analysis(base_model, tokenizer, train_items, test_items, device):
    collate = collate_fn_with_pad(tokenizer.pad_token_id)
    train_ds = SyllogismDataset(train_items, tokenizer, MAX_LEN)
    test_ds = SyllogismDataset(test_items, tokenizer, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=PROBE_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=PROBE_BATCH_SIZE, shuffle=False, collate_fn=collate)

    train_feat = collect_layer_features(base_model, train_dl, device, start_frac=LAYER_START_FRAC)
    test_feat = collect_layer_features(base_model, test_dl, device, start_frac=LAYER_START_FRAC)

    Xtr = train_feat["hidden"]
    Xte = test_feat["hidden"]
    yv_tr = train_feat["validity"]
    yv_te = test_feat["validity"]
    yp_tr = train_feat["plausibility"]
    yp_te = test_feat["plausibility"]
    mv_tr = train_feat["has_validity"].bool()
    mv_te = test_feat["has_validity"].bool()
    mp_tr = train_feat["has_plausibility"].bool()
    mp_te = test_feat["has_plausibility"].bool()

    n_layers = Xtr.shape[1]
    start_layer = int(len(base_model.model.layers) * LAYER_START_FRAC) if hasattr(base_model, "model") and hasattr(base_model.model, "layers") else 0
    results = []
    pbar = tqdm(range(n_layers), desc="LayerProbes", bar_format=BAR_FORMAT)
    for l in pbar:
        acc_validity = train_probe(
            Xtr[:, l, :], yv_tr, mv_tr,
            Xte[:, l, :], yv_te, mv_te,
            PROBE_EPOCHS, PROBE_LR, PROBE_BATCH_SIZE, device
        )
        acc_plausibility = train_probe(
            Xtr[:, l, :], yp_tr, mp_tr,
            Xte[:, l, :], yp_te, mp_te,
            PROBE_EPOCHS, PROBE_LR, PROBE_BATCH_SIZE, device
        )
        row = {
            "layer_index_in_selected_range": l,
            "original_layer_index": start_layer + l,
            "validity_probe_acc": acc_validity,
            "plausibility_probe_acc": acc_plausibility
        }
        results.append(row)
        print(row)
    return results

set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)
base.eval()
base.config.use_cache = False

if hasattr(base, "config") and hasattr(base.config, "num_hidden_layers"):
    print("Model layers:", base.config.num_hidden_layers)
elif hasattr(base, "model") and hasattr(base.model, "layers"):
    print("Model layers:", len(base.model.layers))

model = FrozenLMWithHead(base, base.config.hidden_size)

train_items = load_json(TRAIN_JSON)
test_items = load_json(TEST_JSON)

print("Train size:", len(train_items))
print("Test size:", len(test_items))
if len(train_items) > 0:
    print("Train sample keys:", list(train_items[0].keys()))
if len(test_items) > 0:
    print("Test sample keys:", list(test_items[0].keys()))

os.makedirs(OUT_DIR, exist_ok=True)
save_json(train_items, os.path.join(OUT_DIR, "train_used.json"))
save_json(test_items, os.path.join(OUT_DIR, "test_used.json"))

collate = collate_fn_with_pad(tokenizer.pad_token_id)
train_ds = SyllogismDataset(train_items, tokenizer, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

model = train_head(model, train_dl, device, LR, EPOCHS)

train_acc = evaluate_acc(model, train_dl, device)
print(f"Train accuracy: {train_acc:.4f}")

pred_path = os.path.join(OUT_DIR, "test_predictions.json")
predict_to_json(model, test_items, tokenizer, BATCH_SIZE, MAX_LEN, pred_path, device=device)

if len(test_items) > 0 and "validity" in test_items[0]:
    eval_out = os.path.join(OUT_DIR, "official_eval_test.json")
    run_official_eval(EVAL_SCRIPT, TEST_JSON, pred_path, eval_out)

probe_results = run_layer_probe_analysis(base, tokenizer, train_items, test_items, device)
probe_path = os.path.join(OUT_DIR, "layer_probe_analysis.json")
save_json(probe_results, probe_path)
print("Probe analysis written to:", probe_path)

if len(probe_results) > 0:
    valid_rows = [r for r in probe_results if r["validity_probe_acc"] is not None]
    plaus_rows = [r for r in probe_results if r["plausibility_probe_acc"] is not None]
    best_validity = max(valid_rows, key=lambda x: x["validity_probe_acc"], default=None)
    best_plausibility = max(plaus_rows, key=lambda x: x["plausibility_probe_acc"], default=None)
    print("Best validity probe:", best_validity)
    print("Best plausibility probe:", best_plausibility)
    print("\nAll layer probe results:")
    for r in probe_results:
        print(r)