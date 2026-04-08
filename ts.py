import json
import os
from collections import Counter

TRAIN_PATH = "train_data/subtask 1/train_data.json"
TEST_DIR = "test_data"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_json_files(root_dir):
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d != ".ipynb_checkpoints"]
        for file in files:
            if file.endswith(".json") and not file.endswith("-checkpoint.json"):
                json_files.append(os.path.join(root, file))
    return sorted(json_files)


def summarize_file(file_path):
    print("\n" + "=" * 90)
    print(f"FILE: {file_path}")
    print("=" * 90)

    try:
        data = load_json(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not isinstance(data, list):
        print("Top-level JSON is not a list.")
        return

    print(f"Total examples: {len(data)}")

    all_keys = set()
    validity_counter = Counter()
    plausibility_counter = Counter()
    lang_counter = Counter()

    relevant_present = 0
    relevant_nonempty = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        all_keys.update(item.keys())

        if "validity" in item:
            validity_counter[item["validity"]] += 1

        if "plausibility" in item:
            plausibility_counter[item["plausibility"]] += 1

        if "lang" in item:
            lang_counter[item["lang"]] += 1

        if "relevant_premises" in item:
            relevant_present += 1
            rp = item["relevant_premises"]
            if isinstance(rp, list) and len(rp) > 0:
                relevant_nonempty += 1

    print(f"Fields found: {sorted(all_keys)}")

    if validity_counter:
        print("\nValidity counts:")
        for k, v in sorted(validity_counter.items(), key=lambda x: str(x[0])):
            print(f"  {k}: {v}")

    if plausibility_counter:
        print("\nPlausibility counts:")
        for k, v in sorted(plausibility_counter.items(), key=lambda x: str(x[0])):
            print(f"  {k}: {v}")

    if relevant_present > 0:
        print("\nRelevant premises:")
        print(f"  present in examples: {relevant_present}")
        print(f"  non-empty lists: {relevant_nonempty}")
        print(f"  empty lists: {relevant_present - relevant_nonempty}")

    if lang_counter:
        print("\nLanguage counts:")
        for lang, count in sorted(lang_counter.items()):
            print(f"  {lang}: {count}")


def main():
    print("DATASET STATISTICS")

    if os.path.exists(TRAIN_PATH):
        summarize_file(TRAIN_PATH)
    else:
        print(f"Train file not found: {TRAIN_PATH}")

    if os.path.exists(TEST_DIR):
        for file_path in collect_json_files(TEST_DIR):
            summarize_file(file_path)
    else:
        print(f"Test directory not found: {TEST_DIR}")


if __name__ == "__main__":
    main()