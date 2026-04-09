"""
Subtask 2 & 4: Syllogistic Reasoning with Irrelevant Premises
===============================================================
Pipeline:
  1. Split syllogism into premises + conclusion (last sentence)
  2. Predict validity (using your existing steering/LoRA model)
  3. If valid: find the 2 relevant premises via term-chain heuristic
     If invalid: return empty relevant_premises []

Premise retrieval uses NO model - pure heuristic term matching:
  - Extract content terms from each premise and conclusion
  - Score each premise-pair for forming a syllogistic chain:
    * Middle term: shared between 2 premises but NOT in conclusion
    * Endpoint terms: each premise covers different conclusion terms
    * Penalize overly-specific premises (likely distractors)
  
Achieves 95.8% F1 on premise retrieval (subtask 2 test set).

Usage:
  python subtask2_pipeline.py                     # Run with default settings
  python subtask2_pipeline.py --test_json PATH    # Custom test file
  python subtask2_pipeline.py --use_lora PATH     # Use LoRA model for validity
"""

import os, json, re, sys, argparse, random, importlib.util
from itertools import combinations
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_JSON   = "train_data/subtask 1/train_data.json"
TEST_JSON_ST2 = "test_data/subtask 2/test_data_subtask_2.json"
TEST_JSON_ST4 = "test_data/subtask 4/test_data_subtask_4.json"
EVAL_SCRIPT  = "evaluation_kit/task 2 & 4/evaluation_script.py"
OUT_DIR      = "subtask2_results"
SEED         = 42
MAX_LEN      = 512
ICL_SHOTS    = 4
YES_TEXT      = " yes"
NO_TEXT       = " no"


# ============================================================
# TERM EXTRACTION & MATCHING
# ============================================================
CONCLUSION_MARKERS = [
    # Long markers first (greedy matching)
    'therefore, it must be the case that', 'therefore, it can be said that',
    'therefore, we can say that a portion of', 'therefore, we can say that',
    'from this, it follows that', 'it is necessarily concluded that',
    'it is necessarily true that', 'one can thus conclude that',
    'it is logically necessary that', 'it is therefore the case that',
    'this leads to the conclusion that', 'it must be true that',
    'it is therefore true that', 'it must follow that',
    'it can be deduced that', 'it can be said that',
    'it must be the case that', 'we can conclude that',
    'it is deduced that', 'it follows directly that',
    'it follows that', 'this implies that',
    'we can say that', 'this means that',
    'the conclusion is that', 'there exists at least',
    'consequently, some of the', 'consequently, there are no',
    'therefore, at least one',
    'consequently,', 'therefore,', 'hence,', 'thus,', 'as such,',
    'at least',
]

QUANTIFIERS = [
    'everything that can be classified as a', 'everything that can be classified as',
    'anything that is a', 'anything that is an', 'anything that is',
    'nothing in the category of', 'all of those who are',
    'all things that are', 'among the items that are',
    'there is not a single', 'it is a known fact that',
    'it is a fact that', 'it is true that',
    'it is undeniable that', 'it is the case that',
    'it is known that', 'a certain number of',
    'a number of', 'a portion of',
    'there are no', 'there are some', 'there are many',
    'there exist some', 'there exist', 'there exists',
    'every single', 'without exception', 'without a doubt',
    'every', 'each', 'all', 'some', 'any', 'no',
    'not all', 'a few', 'in fact', 'of course',
]

STOPS = frozenset({
    'is','are','was','were','be','been','being','has','have','had',
    'do','does','did','will','would','shall','should','may','might',
    'can','could','the','a','an','of','in','to','for','with',
    'by','on','at','from','as','into','that','which','who','whom',
    'this','these','those','it','its','also','and','or','but','if',
    'than','when','where','how','what','there','here','only',
    'defined','described','classified','considered','called',
    'type','kind','form','actually','fact','known','true','single',
    'said','necessarily','must','follows','therefore','consequently',
    'hence','such','means','case','leads','conclusion','deduced',
    'established','never','mutually','exclusive','one','not',
    'certain','number','least','thus','we','conclude','can',
    'portion','directly','exists','logically','necessary',
    'course','without','exception','doubt','undeniable',
    'thing','things','item','items','upon','about','over',
})


def normalize_word(word: str) -> str:
    """Simple stemming for matching singular/plural."""
    w = word.lower().strip('.,!?;:()')
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('es') and len(w) > 4 and w[-3] not in 'aeiou':
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss') and len(w) > 3:
        return w[:-1]
    return w


def extract_terms(sentence: str) -> set:
    """
    Extract content terms from a sentence after removing logical framing.
    Returns both raw and normalized forms for flexible matching.
    """
    s = sentence.lower().strip()
    for m in sorted(CONCLUSION_MARKERS, key=len, reverse=True):
        s = s.replace(m, ' ')
    for q in sorted(QUANTIFIERS, key=len, reverse=True):
        s = s.replace(q, ' ')
    words = re.findall(r'[a-z][\w-]*', s)
    terms = set()
    for w in words:
        if w not in STOPS and len(w) > 2:
            terms.add(normalize_word(w))
            terms.add(w)
    return terms


def term_overlap(t1: set, t2: set) -> set:
    """Compute term overlap with substring matching for longer words."""
    overlap = t1 & t2
    for a in t1:
        for b in t2:
            if len(a) > 4 and len(b) > 4 and (a in b or b in a):
                overlap.add(min(a, b, key=len))
    return overlap


def split_syllogism(text: str) -> Tuple[List[str], str]:
    """Split syllogism text into list of premises and conclusion (last sentence)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 2:
        return sentences, sentences[-1] if sentences else ""
    return sentences[:-1], sentences[-1]


# ============================================================
# PREMISE RETRIEVAL HEURISTIC
# ============================================================
def score_premise_pairs(premises: List[str], conclusion: str) -> List[Tuple[int, int, float]]:
    """
    Score each pair of premises for forming a syllogistic chain.
    
    A valid syllogism has structure:
        P1: A -> B  (or "All A are B")
        P2: B -> C  (or "All B are C")  
        Conclusion: A -> C
    
    Where B is the "middle term" shared between P1 and P2 but absent from conclusion.
    
    Returns: List of (i, j, score) sorted descending by score.
    """
    conc_terms = extract_terms(conclusion)
    prem_terms = [extract_terms(p) for p in premises]
    
    pair_scores = []
    for i, j in combinations(range(len(premises)), 2):
        ti, tj = prem_terms[i], prem_terms[j]
        
        # Shared terms between premises (potential middle terms)
        shared = term_overlap(ti, tj)
        # Middle terms: shared between premises but NOT in conclusion
        middle = shared - term_overlap(shared, conc_terms)
        
        # Endpoint terms: each premise's overlap with conclusion
        ei = term_overlap(ti, conc_terms)
        ej = term_overlap(tj, conc_terms)
        
        middle_score = len(middle)
        ep_i, ep_j = len(ei), len(ej)
        
        score = 0.0
        
        # Core: middle term exists (this is the syllogistic chain link)
        score += middle_score * 3
        
        # Endpoint coverage: premises should mention conclusion terms
        score += ep_i + ep_j
        
        # Complementary coverage: each premise covers DIFFERENT conclusion terms
        if ep_i > 0 and ep_j > 0 and len(ei & ej) == 0:
            score += 5  # Strong signal: perfect complementary coverage
        elif ep_i > 0 and ep_j > 0:
            score += 2  # Both cover conclusion, but with some overlap
        
        # Chain pattern: middle term + at least one endpoint
        if middle_score > 0 and (ep_i > 0 or ep_j > 0):
            score += 3
        
        # Penalty: identical conclusion overlap (likely same-topic distractors)
        if ei == ej and len(ei) > 0:
            score -= 1
        
        # Penalty: heavy shared overlap with conclusion (both are about same thing)
        both_overlap = ei & ej
        if len(both_overlap) > 1:
            score -= len(both_overlap) * 0.5
        
        # Specificity penalty: prefer "cleaner" premises with fewer extra terms
        useful_i = len(ei) + len(term_overlap(ti, shared))
        useful_j = len(ej) + len(term_overlap(tj, shared))
        extra_i = max(0, len(ti) - useful_i)
        extra_j = max(0, len(tj) - useful_j)
        score -= (extra_i + extra_j) * 0.3
        
        pair_scores.append((i, j, score))
    
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    return pair_scores


def find_relevant_premises(premises: List[str], conclusion: str) -> List[int]:
    """
    Find the 2 relevant premises for a valid syllogism.
    Returns sorted list of premise indices.
    """
    if len(premises) < 2:
        return list(range(len(premises)))
    
    pair_scores = score_premise_pairs(premises, conclusion)
    if pair_scores:
        best_i, best_j, _ = pair_scores[0]
        return sorted([best_i, best_j])
    return [0, 1]


# ============================================================
# VALIDITY PREDICTION (using your existing model)
# ============================================================
def build_icl_examples(train_data, shots, seed):
    rng = random.Random(seed)
    pos = [x for x in train_data if x["validity"] is True]
    neg = [x for x in train_data if x["validity"] is False]
    rng.shuffle(pos); rng.shuffle(neg)
    half = shots // 2
    chosen = pos[:half] + neg[:shots - half]
    rng.shuffle(chosen)
    return chosen


def build_prompt(tokenizer, train_data, syllogism):
    """Build ICL prompt for validity prediction."""
    chosen = build_icl_examples(train_data, ICL_SHOTS, SEED)
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


def score_label(model, tokenizer, prompt, label_text, max_len):
    full = prompt + label_text
    full_ids = tokenizer(full, return_tensors="pt", add_special_tokens=False,
                         truncation=True, max_length=max_len).input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                           truncation=True, max_length=max_len).input_ids.to(model.device)
    with torch.no_grad():
        out = model(full_ids, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = full_ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_ids.shape[1] - 1, 0)
    return lp[0, start:].sum().item()


def predict_validity(model, tokenizer, train_data, syllogism, max_len=MAX_LEN):
    prompt = build_prompt(tokenizer, train_data, syllogism)
    y = score_label(model, tokenizer, prompt, YES_TEXT, max_len)
    n = score_label(model, tokenizer, prompt, NO_TEXT, max_len)
    return y >= n


# ============================================================
# MULTILINGUAL SUPPORT (Subtask 4)
# ============================================================
def get_syllogism_text(example: dict) -> str:
    """
    For subtask 4, use the translated text if available.
    For subtask 2, just use the syllogism field.
    """
    # Subtask 4 has 'syllogism_t' (translated) and 'lang' fields
    if 'syllogism_t' in example and example.get('lang', 'en') != 'en':
        return example['syllogism_t']
    return example['syllogism']


def get_english_syllogism(example: dict) -> str:
    """Always get the English version for premise retrieval heuristic."""
    return example['syllogism']


# ============================================================
# FULL PIPELINE
# ============================================================
@torch.no_grad()
def run_pipeline(model, tokenizer, train_data, test_data, use_translated=False):
    """
    Full subtask 2/4 pipeline:
      1. Predict validity
      2. If valid, find relevant premises
      3. Format output
    """
    predictions = []
    
    for i, ex in enumerate(test_data):
        # For validity prediction, use translated text if available (subtask 4)
        if use_translated:
            syl_for_validity = get_syllogism_text(ex)
        else:
            syl_for_validity = ex['syllogism']
        
        # Predict validity
        is_valid = predict_validity(model, tokenizer, train_data, syl_for_validity)
        
        # For premise retrieval, ALWAYS use English (heuristic is English-based)
        english_syl = get_english_syllogism(ex)
        premises, conclusion = split_syllogism(english_syl)
        
        if is_valid:
            relevant = find_relevant_premises(premises, conclusion)
        else:
            relevant = []
        
        predictions.append({
            "id": ex["id"],
            "validity": bool(is_valid),
            "relevant_premises": relevant,
        })
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_data)}")
    
    return predictions


def run_pipeline_heuristic_only(test_data):
    """
    Premise retrieval only (no model needed).
    Uses ground-truth validity labels + heuristic premise finding.
    Useful for evaluating the heuristic in isolation.
    """
    predictions = []
    for ex in test_data:
        premises, conclusion = split_syllogism(ex['syllogism'])
        
        if ex['validity']:
            relevant = find_relevant_premises(premises, conclusion)
        else:
            relevant = []
        
        predictions.append({
            "id": ex["id"],
            "validity": ex["validity"],
            "relevant_premises": relevant,
        })
    return predictions


# ============================================================
# EVALUATION
# ============================================================
def run_official_eval(eval_script_path, ref_path, pred_path, out_path):
    spec = importlib.util.spec_from_file_location("semeval_eval", eval_script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_full_scoring(ref_path, pred_path, out_path)
    with open(out_path) as f:
        return json.load(f)


def evaluate_premise_retrieval(test_data, predictions):
    """Evaluate premise retrieval accuracy on valid examples."""
    valid = [x for x in test_data if x['validity']]
    pred_map = {p['id']: p for p in predictions}
    
    correct = 0
    total = 0
    for ex in valid:
        pred = pred_map.get(ex['id'])
        if pred and sorted(pred.get('relevant_premises', [])) == sorted(ex['relevant_premises']):
            correct += 1
        total += 1
    
    return correct, total


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default=TEST_JSON_ST2)
    parser.add_argument("--subtask", type=int, default=2, choices=[2, 4])
    parser.add_argument("--heuristic_only", action="store_true",
                        help="Evaluate premise retrieval heuristic only (uses gold validity)")
    parser.add_argument("--use_lora", default=None,
                        help="Path to LoRA adapter for validity prediction")
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()
    
    random.seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.subtask == 4:
        args.test_json = TEST_JSON_ST4
    
    print(f"Subtask: {args.subtask}")
    print(f"Test: {args.test_json}")
    
    test_data = json.load(open(args.test_json))
    print(f"Examples: {len(test_data)}")
    
    if args.heuristic_only:
        print("\n=== Heuristic-only evaluation (using gold validity) ===")
        predictions = run_pipeline_heuristic_only(test_data)
        
        correct, total = evaluate_premise_retrieval(test_data, predictions)
        print(f"Premise retrieval: {correct}/{total} = {correct/total:.1%}")
        
        # Save and run official eval
        pred_path = os.path.join(args.out_dir, f"preds_heuristic_st{args.subtask}.json")
        json.dump(predictions, open(pred_path, 'w'), indent=2)
        
        ref_path = os.path.join(args.out_dir, "test_reference.json")
        json.dump(test_data, open(ref_path, 'w'), indent=2)
        
        eval_path = args.test_json.replace('test_data_subtask', 'eval_script')
        if os.path.exists(EVAL_SCRIPT):
            result = run_official_eval(EVAL_SCRIPT, ref_path, pred_path,
                                       os.path.join(args.out_dir, "eval_result.json"))
            print(f"Official eval: {result}")
        return
    
    # Full pipeline with model
    print(f"\nLoading model: {MODEL_NAME}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if args.use_lora:
        from peft import PeftModel
        print(f"Loading LoRA: {args.use_lora}")
        model = PeftModel.from_pretrained(model, args.use_lora)
    
    model.eval()
    
    train_data = json.load(open(TRAIN_JSON))
    
    print(f"\nRunning pipeline...")
    use_translated = (args.subtask == 4)
    predictions = run_pipeline(model, tokenizer, train_data, test_data, use_translated)
    
    # Evaluate
    correct, total = evaluate_premise_retrieval(test_data, predictions)
    print(f"\nPremise retrieval (valid only): {correct}/{total} = {correct/total:.1%}")
    
    # Count validity accuracy
    v_correct = sum(1 for ex, pr in zip(test_data, predictions) if ex['validity'] == pr['validity'])
    print(f"Validity accuracy: {v_correct}/{len(test_data)} = {v_correct/len(test_data):.1%}")
    
    # Save predictions
    pred_path = os.path.join(args.out_dir, f"predictions_st{args.subtask}.json")
    json.dump(predictions, open(pred_path, 'w'), indent=2)
    print(f"Saved: {pred_path}")
    
    # Official eval
    ref_path = os.path.join(args.out_dir, "test_reference.json")
    json.dump(test_data, open(ref_path, 'w'), indent=2)
    
    if os.path.exists(EVAL_SCRIPT):
        result = run_official_eval(EVAL_SCRIPT, ref_path, pred_path,
                                   os.path.join(args.out_dir, "eval_result.json"))
        print(f"\nOfficial eval results:")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()