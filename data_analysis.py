"""
Dataset Analysis for Syllogistic Reasoning
============================================
Analyzes: distribution, syllogistic forms (mood & figure),
quantifier patterns, length/complexity, content domains.
"""

import json
import re
from collections import Counter, defaultdict
import os

# ============================================================
# CONFIG - update these paths to match your setup
# ============================================================
TRAIN_JSON = "train_data/subtask 1/train_data.json"
TEST_JSON = "test_data/subtask 1/test_data_subtask_1.json"
OUTPUT_DIR = "analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 1. DISTRIBUTION ANALYSIS
# ============================================================
def analyze_distribution(data, label="Dataset"):
    print(f"\n{'='*60}")
    print(f"  DISTRIBUTION ANALYSIS: {label}")
    print(f"{'='*60}")
    print(f"Total examples: {len(data)}")

    buckets = Counter()
    for ex in data:
        v = "valid" if ex.get("validity") else "invalid"
        p = "plausible" if ex.get("plausibility") else "implausible"
        buckets[f"{p}_{v}"] += 1

    print(f"\n  {'Bucket':<30s} {'Count':>6s} {'%':>7s}")
    print(f"  {'-'*45}")
    for key in ["plausible_valid", "plausible_invalid",
                 "implausible_valid", "implausible_invalid"]:
        c = buckets.get(key, 0)
        pct = 100 * c / len(data) if data else 0
        print(f"  {key:<30s} {c:>6d} {pct:>6.1f}%")

    # validity and plausibility marginals
    n_valid = sum(1 for ex in data if ex.get("validity"))
    n_plausible = sum(1 for ex in data if ex.get("plausibility"))
    print(f"\n  Valid:      {n_valid:>5d} ({100*n_valid/len(data):.1f}%)")
    print(f"  Invalid:    {len(data)-n_valid:>5d} ({100*(len(data)-n_valid)/len(data):.1f}%)")
    print(f"  Plausible:  {n_plausible:>5d} ({100*n_plausible/len(data):.1f}%)")
    print(f"  Implausible:{len(data)-n_plausible:>5d} ({100*(len(data)-n_plausible)/len(data):.1f}%)")

    return buckets


# ============================================================
# 2. SYLLOGISTIC FORM ANALYSIS (Mood + Figure)
# ============================================================

# Categorical proposition types:
#   A = "All S are P"       (universal affirmative)
#   E = "No S are P"        (universal negative)
#   I = "Some S are P"      (particular affirmative)
#   O = "Some S are not P"  (particular negative)

QUANTIFIER_PATTERNS = [
    # Order matters: longer/more specific patterns first
    # O-type (particular negative)
    (r'\b(?:some|certain|a few|several|there (?:are|exist) some)\b.*\b(?:are not|aren\'t|is not|isn\'t|cannot be|can not be|could not be)\b', 'O'),
    (r'\b(?:not all|not every)\b', 'O'),
    (r'\b(?:it is not the case that (?:all|every))\b', 'O'),
    (r'\b(?:some|certain)\b.*\bnot\b', 'O'),

    # E-type (universal negative)
    (r'\b(?:no|none of the|there (?:are|is) no|there (?:are|is) not a single)\b', 'E'),
    (r'\b(?:it is (?:not the case|certain|known) that no)\b', 'E'),
    (r'\b(?:all|every|each)\b.*\b(?:are not|aren\'t|is not|isn\'t|cannot|can not)\b', 'E'),
    (r'\bno\b', 'E'),

    # A-type (universal affirmative)
    (r'\b(?:all|every|each|any|every single|it is (?:certain|known|true|also true) that (?:all|every))\b', 'A'),
    (r'\b(?:all|every|each)\b', 'A'),

    # I-type (particular affirmative)
    (r'\b(?:some|certain|a few|several|there (?:are|exist) some|it is (?:known|true) that some)\b', 'I'),
    (r'\bsome\b', 'I'),
]


def classify_proposition(sentence):
    """Classify a sentence as A, E, I, or O type."""
    s = sentence.lower().strip()

    # Handle "not all" / "not every" -> O type
    if re.search(r'\b(?:not all|not every)\b', s):
        return 'O'
    if re.search(r'\bit is not the case that (?:all|every)\b', s):
        return 'O'

    # Handle "some ... not" -> O type
    if re.search(r'\bsome\b', s) and re.search(r'\b(?:not|n\'t)\b', s):
        return 'O'

    # Handle "no" / "none" -> E type
    if re.search(r'\b(?:no|none)\b', s) and not re.search(r'\bnot\b', s):
        return 'E'

    # Handle universal + negative -> E type
    if re.search(r'\b(?:all|every|each)\b', s) and re.search(r'\b(?:not|n\'t|cannot|never)\b', s):
        return 'E'

    # Handle "all" / "every" / "each" -> A type
    if re.search(r'\b(?:all|every|each|any)\b', s):
        return 'A'

    # Handle "some" -> I type
    if re.search(r'\b(?:some|certain|several|a few|there (?:are|exist))\b', s):
        return 'I'

    return '?'


def extract_terms(sentence):
    """Try to extract subject and predicate terms from a categorical proposition."""
    s = sentence.lower().strip()
    # Remove common quantifier prefixes
    s = re.sub(r'^(therefore,?\s*|thus,?\s*|hence,?\s*|consequently,?\s*|it follows that\s*)', '', s)
    s = re.sub(r'^(all|every|each|some|no|none of the|not all|a few|several|certain)\s+', '', s)

    # Try "X are (not) Y" pattern
    m = re.match(r'(.+?)\s+(?:are|is|can be|cannot be|can not be|could be)\s+(?:not\s+)?(.+)', s)
    if m:
        return m.group(1).strip().rstrip('.'), m.group(2).strip().rstrip('.')

    return None, None


def split_syllogism(text):
    """Split syllogism into premises and conclusion."""
    # Try splitting on "therefore", "thus", "hence", "consequently", "it follows"
    conclusion_markers = [
        r'\btherefore\b', r'\bthus\b', r'\bhence\b',
        r'\bconsequently\b', r'\bit follows that\b',
        r'\bthis (?:means|implies|shows|has led)\b',
        r'\bwe can conclude\b', r'\bso\b,?\s'
    ]

    text_lower = text.lower()
    conclusion_start = -1
    for marker in conclusion_markers:
        m = re.search(marker, text_lower)
        if m:
            conclusion_start = m.start()
            break

    if conclusion_start == -1:
        # Fallback: split by sentences, last one is conclusion
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) >= 2:
            premises = sentences[:-1]
            conclusion = sentences[-1]
        else:
            return [], text
    else:
        premise_text = text[:conclusion_start].strip()
        conclusion = text[conclusion_start:].strip()
        # Clean conclusion marker
        conclusion = re.sub(r'^(?:therefore|thus|hence|consequently|it follows that|so)\s*,?\s*', '', conclusion, flags=re.IGNORECASE)
        premises = re.split(r'(?<=[.!?])\s+', premise_text.strip())

    premises = [p.strip().rstrip('.') for p in premises if p.strip()]
    conclusion = conclusion.strip().rstrip('.')
    return premises, conclusion


def get_mood(premises, conclusion):
    """Get the mood (e.g., AAA, EAE) of a syllogism."""
    types = []
    for p in premises[:2]:  # standard syllogism has 2 premises
        types.append(classify_proposition(p))
    types.append(classify_proposition(conclusion))
    return ''.join(types)


def determine_figure(premises, conclusion):
    """
    Determine figure (1-4) based on position of middle term.
    Figure 1: M-P, S-M => S-P
    Figure 2: P-M, S-M => S-P
    Figure 3: M-P, M-S => S-P
    Figure 4: P-M, M-S => S-P
    Returns figure number or '?' if can't determine.
    """
    if len(premises) < 2:
        return '?'

    s1, p1 = extract_terms(premises[0])
    s2, p2 = extract_terms(premises[1])
    sc, pc = extract_terms(conclusion)

    if not all([s1, p1, s2, p2, sc, pc]):
        return '?'

    # Find middle term (appears in premises but not conclusion)
    premise_terms = {s1, p1, s2, p2}
    conclusion_terms = {sc, pc}

    # Middle term candidates: in premises but not in conclusion
    middle_candidates = premise_terms - conclusion_terms
    if len(middle_candidates) == 0:
        return '?'

    # Use first candidate as middle term
    middle = list(middle_candidates)[0]

    # Determine positions
    # Premise 1
    if p1 == middle:
        p1_pos = 'predicate'
    elif s1 == middle:
        p1_pos = 'subject'
    else:
        return '?'

    # Premise 2
    if p2 == middle:
        p2_pos = 'predicate'
    elif s2 == middle:
        p2_pos = 'subject'
    else:
        return '?'

    # Figure determination
    if p1_pos == 'subject' and p2_pos == 'predicate':
        return '1'  # M-P, S-M
    elif p1_pos == 'predicate' and p2_pos == 'predicate':
        return '2'  # P-M, S-M
    elif p1_pos == 'subject' and p2_pos == 'subject':
        return '3'  # M-P, M-S
    elif p1_pos == 'predicate' and p2_pos == 'subject':
        return '4'  # P-M, M-S
    return '?'


def analyze_syllogistic_forms(data, label="Dataset"):
    print(f"\n{'='*60}")
    print(f"  SYLLOGISTIC FORM ANALYSIS: {label}")
    print(f"{'='*60}")

    mood_counter = Counter()
    figure_counter = Counter()
    mood_figure_counter = Counter()
    prop_type_counter = Counter()
    form_validity = defaultdict(lambda: {"valid": 0, "invalid": 0, "total": 0})
    parse_failures = []

    for ex in data:
        syl = ex.get("syllogism", "")
        premises, conclusion = split_syllogism(syl)

        if len(premises) < 2:
            parse_failures.append(ex.get("id", "?"))
            mood = "PARSE_FAIL"
            figure = "?"
        else:
            mood = get_mood(premises, conclusion)
            figure = determine_figure(premises, conclusion)

            # Count individual proposition types
            for p in premises[:2]:
                prop_type_counter[classify_proposition(p)] += 1
            prop_type_counter[classify_proposition(conclusion)] += 1

        mood_counter[mood] += 1
        figure_counter[figure] += 1
        mood_figure_counter[f"{mood}-Fig{figure}"] += 1

        v = "valid" if ex.get("validity") else "invalid"
        form_validity[mood][v] += 1
        form_validity[mood]["total"] += 1

    # Print mood distribution
    print(f"\n  Mood Distribution (top 20):")
    print(f"  {'Mood':<15s} {'Count':>6s} {'%':>7s}  {'Valid':>6s} {'Invalid':>6s}")
    print(f"  {'-'*50}")
    for mood, count in mood_counter.most_common(20):
        pct = 100 * count / len(data)
        fv = form_validity[mood]
        print(f"  {mood:<15s} {count:>6d} {pct:>6.1f}%  {fv['valid']:>6d} {fv['invalid']:>6d}")

    # Print figure distribution
    print(f"\n  Figure Distribution:")
    for fig, count in sorted(figure_counter.items()):
        pct = 100 * count / len(data)
        print(f"  Figure {fig}: {count:>6d} ({pct:.1f}%)")

    # Print proposition type distribution
    print(f"\n  Proposition Type Distribution:")
    for ptype in ['A', 'E', 'I', 'O', '?']:
        c = prop_type_counter.get(ptype, 0)
        total_props = sum(prop_type_counter.values())
        pct = 100 * c / total_props if total_props > 0 else 0
        labels = {'A': 'Universal Affirmative', 'E': 'Universal Negative',
                  'I': 'Particular Affirmative', 'O': 'Particular Negative', '?': 'Unknown'}
        print(f"  {ptype} ({labels[ptype]:<25s}): {c:>6d} ({pct:.1f}%)")

    if parse_failures:
        print(f"\n  Parse failures: {len(parse_failures)} examples")
        for pid in parse_failures[:5]:
            print(f"    ID: {pid}")

    return mood_counter, figure_counter


# ============================================================
# 3. QUANTIFIER PATTERN ANALYSIS
# ============================================================
def analyze_quantifiers(data, label="Dataset"):
    print(f"\n{'='*60}")
    print(f"  QUANTIFIER ANALYSIS: {label}")
    print(f"{'='*60}")

    # Raw quantifier words/phrases found
    quantifier_phrases = [
        "all", "every", "each", "any",
        "some", "certain", "several", "a few",
        "no", "none", "not all", "not every",
        "there are no", "there is no", "there are some", "there exist",
        "it is not the case that all", "it is not the case that every",
        "it is certain that", "it is known that", "it is true that",
        "it is also true that", "every single",
        "cannot", "can not", "could not",
    ]

    # Sort by length descending to match longer phrases first
    quantifier_phrases.sort(key=len, reverse=True)

    phrase_counter = Counter()
    paraphrased_count = 0
    standard_quantifiers = {"all", "some", "no", "not"}

    for ex in data:
        syl = ex.get("syllogism", "").lower()
        found_in_this = set()

        for phrase in quantifier_phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            matches = re.findall(pattern, syl)
            if matches:
                phrase_counter[phrase] += len(matches)
                found_in_this.add(phrase)

        # Check for paraphrased quantifiers
        non_standard = found_in_this - standard_quantifiers
        if non_standard:
            paraphrased_count += 1

    print(f"\n  Quantifier Phrase Frequencies (top 25):")
    print(f"  {'Phrase':<40s} {'Count':>6s}")
    print(f"  {'-'*48}")
    for phrase, count in phrase_counter.most_common(25):
        print(f"  {phrase:<40s} {count:>6d}")

    print(f"\n  Examples with paraphrased quantifiers: {paraphrased_count}/{len(data)} ({100*paraphrased_count/len(data):.1f}%)")

    return phrase_counter


# ============================================================
# 4. LENGTH AND COMPLEXITY ANALYSIS
# ============================================================
def analyze_length(data, label="Dataset"):
    print(f"\n{'='*60}")
    print(f"  LENGTH & COMPLEXITY ANALYSIS: {label}")
    print(f"{'='*60}")

    char_lens = []
    word_lens = []
    sentence_counts = []
    negation_counts = []

    for ex in data:
        syl = ex.get("syllogism", "")
        char_lens.append(len(syl))
        words = syl.split()
        word_lens.append(len(words))

        sentences = re.split(r'(?<=[.!?])\s+', syl.strip())
        sentence_counts.append(len(sentences))

        neg_count = len(re.findall(r'\b(?:not|no|none|never|neither|nor|cannot|can\'t|isn\'t|aren\'t|don\'t|doesn\'t|wasn\'t|weren\'t)\b', syl.lower()))
        negation_counts.append(neg_count)

    import statistics

    print(f"\n  Character length:")
    print(f"    Min: {min(char_lens)}, Max: {max(char_lens)}, Mean: {statistics.mean(char_lens):.1f}, Median: {statistics.median(char_lens):.1f}, StdDev: {statistics.stdev(char_lens):.1f}")

    print(f"\n  Word count:")
    print(f"    Min: {min(word_lens)}, Max: {max(word_lens)}, Mean: {statistics.mean(word_lens):.1f}, Median: {statistics.median(word_lens):.1f}, StdDev: {statistics.stdev(word_lens):.1f}")

    print(f"\n  Sentence count:")
    sc = Counter(sentence_counts)
    for s, c in sorted(sc.items()):
        print(f"    {s} sentences: {c} examples ({100*c/len(data):.1f}%)")

    print(f"\n  Negation count per syllogism:")
    nc = Counter(negation_counts)
    for n, c in sorted(nc.items()):
        print(f"    {n} negations: {c} examples ({100*c/len(data):.1f}%)")

    # Complexity by validity/plausibility
    print(f"\n  Mean word count by bucket:")
    buckets = defaultdict(list)
    for ex in data:
        v = "valid" if ex.get("validity") else "invalid"
        p = "plausible" if ex.get("plausibility") else "implausible"
        buckets[f"{p}_{v}"].append(len(ex.get("syllogism", "").split()))

    for key in sorted(buckets.keys()):
        vals = buckets[key]
        print(f"    {key:<30s}: {statistics.mean(vals):.1f} words (n={len(vals)})")

    return word_lens, negation_counts


# ============================================================
# 5. CONTENT DOMAIN ANALYSIS
# ============================================================
def analyze_domains(data, label="Dataset"):
    print(f"\n{'='*60}")
    print(f"  CONTENT DOMAIN ANALYSIS: {label}")
    print(f"{'='*60}")

    # Define domain keyword groups
    domain_keywords = {
        "animals": ["animal", "dog", "cat", "bird", "fish", "horse", "cow", "sheep",
                     "lion", "tiger", "bear", "rabbit", "elephant", "monkey", "whale",
                     "dolphin", "snake", "insect", "mammal", "reptile", "pet", "creature",
                     "mouse", "rat", "chicken", "duck", "pig", "wolf", "fox", "deer"],
        "people/professions": ["doctor", "teacher", "student", "lawyer", "engineer", "nurse",
                               "artist", "musician", "scientist", "professor", "athlete",
                               "politician", "writer", "actor", "chef", "worker", "employee",
                               "manager", "person", "people", "human", "man", "woman",
                               "child", "children", "adult", "citizen", "philosopher"],
        "objects/artifacts": ["car", "vehicle", "book", "table", "chair", "computer",
                              "phone", "machine", "tool", "weapon", "instrument",
                              "building", "house", "furniture", "device", "toy",
                              "newspaper", "magazine", "painting", "sculpture"],
        "food/plants": ["fruit", "vegetable", "flower", "tree", "plant", "food",
                        "apple", "rose", "oak", "herb", "grain", "meat", "berry"],
        "geography/places": ["country", "city", "continent", "ocean", "mountain",
                             "river", "island", "europe", "asia", "africa",
                             "america", "planet", "earth"],
        "abstract/concepts": ["idea", "concept", "theory", "emotion", "feeling",
                              "number", "color", "shape", "quality", "property",
                              "virtue", "crime", "law", "right", "freedom",
                              "justice", "truth", "knowledge", "belief"],
        "academic/science": ["mineral", "element", "chemical", "metal", "molecule",
                             "cell", "organism", "species", "genus", "fossil",
                             "equation", "theorem", "hypothesis"],
    }

    domain_counter = Counter()
    domain_examples = defaultdict(list)
    multi_domain = 0
    no_domain = 0

    for ex in data:
        syl = ex.get("syllogism", "").lower()
        matched_domains = []

        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', syl):
                    matched_domains.append(domain)
                    break

        if len(matched_domains) == 0:
            no_domain += 1
            domain_counter["other/unclassified"] += 1
        else:
            for d in set(matched_domains):
                domain_counter[d] += 1
            if len(set(matched_domains)) > 1:
                multi_domain += 1

        if len(matched_domains) > 0:
            domain_examples[matched_domains[0]].append(ex.get("id", "?"))

    print(f"\n  Domain Distribution:")
    print(f"  {'Domain':<25s} {'Count':>6s} {'%':>7s}")
    print(f"  {'-'*40}")
    for domain, count in domain_counter.most_common():
        pct = 100 * count / len(data)
        print(f"  {domain:<25s} {count:>6d} {pct:>6.1f}%")

    print(f"\n  Multi-domain examples: {multi_domain}")
    print(f"  Unclassified examples: {no_domain}")

    # Domain x validity/plausibility
    print(f"\n  Error-prone domains (by plausibility-validity bucket):")
    domain_bucket = defaultdict(lambda: Counter())
    for ex in data:
        syl = ex.get("syllogism", "").lower()
        v = "valid" if ex.get("validity") else "invalid"
        p = "plausible" if ex.get("plausibility") else "implausible"
        bucket = f"{p}_{v}"

        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', syl):
                    domain_bucket[domain][bucket] += 1
                    break

    for domain in sorted(domain_bucket.keys()):
        buckets = domain_bucket[domain]
        total = sum(buckets.values())
        parts = ", ".join(f"{k}:{v}" for k, v in sorted(buckets.items()))
        print(f"  {domain:<25s} (n={total}): {parts}")

    return domain_counter


# ============================================================
# 6. SAMPLE EXAMPLES (for sanity check)
# ============================================================
def show_samples(data, n=3, label="Dataset"):
    print(f"\n{'='*60}")
    print(f"  SAMPLE EXAMPLES: {label}")
    print(f"{'='*60}")

    import random
    random.seed(42)
    samples = random.sample(data, min(n, len(data)))

    for i, ex in enumerate(samples):
        syl = ex.get("syllogism", "")
        premises, conclusion = split_syllogism(syl)
        mood = get_mood(premises, conclusion) if len(premises) >= 2 else "?"

        print(f"\n  --- Example {i+1} (ID: {ex.get('id', '?')}) ---")
        print(f"  Syllogism: {syl}")
        print(f"  Validity: {ex.get('validity')}")
        print(f"  Plausibility: {ex.get('plausibility')}")
        print(f"  Parsed premises: {premises}")
        print(f"  Parsed conclusion: {conclusion}")
        print(f"  Mood: {mood}")
        for j, p in enumerate(premises[:2]):
            print(f"    Premise {j+1} type: {classify_proposition(p)}")
        if conclusion:
            print(f"    Conclusion type: {classify_proposition(conclusion)}")


# ============================================================
# 7. TRAIN vs TEST COMPARISON
# ============================================================
def compare_distributions(train_buckets, test_buckets, train_n, test_n):
    print(f"\n{'='*60}")
    print(f"  TRAIN vs TEST DISTRIBUTION COMPARISON")
    print(f"{'='*60}")

    all_keys = sorted(set(list(train_buckets.keys()) + list(test_buckets.keys())))
    print(f"\n  {'Bucket':<30s} {'Train%':>8s} {'Test%':>8s} {'Diff':>8s}")
    print(f"  {'-'*56}")
    for key in all_keys:
        t_pct = 100 * train_buckets.get(key, 0) / train_n if train_n else 0
        e_pct = 100 * test_buckets.get(key, 0) / test_n if test_n else 0
        diff = abs(t_pct - e_pct)
        flag = " !!!" if diff > 5 else ""
        print(f"  {key:<30s} {t_pct:>7.1f}% {e_pct:>7.1f}% {diff:>7.1f}%{flag}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading datasets...")

    train_data = load_json(TRAIN_JSON) if os.path.exists(TRAIN_JSON) else None
    test_data = load_json(TEST_JSON) if os.path.exists(TEST_JSON) else None

    if train_data is None and test_data is None:
        print("ERROR: No data files found. Update TRAIN_JSON and TEST_JSON paths.")
        print(f"  Looked for: {TRAIN_JSON}, {TEST_JSON}")
        return

    # Show sample to understand structure
    if train_data:
        print(f"\nSample train keys: {list(train_data[0].keys())}")
        print(f"Sample train example:\n{json.dumps(train_data[0], indent=2, ensure_ascii=False)[:500]}")

    train_buckets = None
    test_buckets = None

    if train_data:
        train_buckets = analyze_distribution(train_data, "TRAIN")
        analyze_syllogistic_forms(train_data, "TRAIN")
        analyze_quantifiers(train_data, "TRAIN")
        analyze_length(train_data, "TRAIN")
        analyze_domains(train_data, "TRAIN")
        show_samples(train_data, 5, "TRAIN")

    if test_data:
        test_buckets = analyze_distribution(test_data, "TEST")
        analyze_syllogistic_forms(test_data, "TEST")
        analyze_quantifiers(test_data, "TEST")
        analyze_length(test_data, "TEST")
        analyze_domains(test_data, "TEST")
        show_samples(test_data, 3, "TEST")

    if train_buckets and test_buckets:
        compare_distributions(train_buckets, test_buckets, len(train_data), len(test_data))

    print(f"\n{'='*60}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nKey things to check:")
    print(f"  1. Are the 4 buckets roughly balanced? Imbalance affects steering vectors.")
    print(f"  2. Which moods dominate? Rare moods may have unstable steering.")
    print(f"  3. How many paraphrased quantifiers? More = harder for pattern matching.")
    print(f"  4. Length variation? Could be a confound if one bucket has longer examples.")
    print(f"  5. Domain skew? Some domains may trigger stronger plausibility bias.")
    print(f"  6. Train vs test similar? Distribution shift = unreliable evaluation.")


if __name__ == "__main__":
    main()