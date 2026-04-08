import json
import os
import random

NUM_SAMPLES = 4
TRAIN_PATH = "train_data/subtask 1/train_data.json"
TEST_DIR = "test_data"


def pretty_print_item(item, indent=0):
    prefix = "  " * indent

    if isinstance(item, dict):
        if not item:
            print(f"{prefix}{{}}")
            return
        for k, v in item.items():
            print(f"{prefix}{k}:")
            pretty_print_item(v, indent + 1)

    elif isinstance(item, list):
        if not item:
            print(f"{prefix}[]")
            return
        for i, v in enumerate(item):
            print(f"{prefix}- [{i}]")
            pretty_print_item(v, indent + 1)

    else:
        print(f"{prefix}{item}")


def summarize_keys(data):
    keys = set()
    for item in data:
        if isinstance(item, dict):
            keys.update(item.keys())
    return sorted(keys)


def print_samples(file_path, num_samples=3):
    print("\n" + "=" * 80)
    print(f"File: {file_path}")
    print("=" * 80)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Warning: top-level JSON is not a list.")
            return

        print(f"Total examples: {len(data)}")
        print(f"Fields found: {summarize_keys(data)}")

        if len(data) == 0:
            print("No examples found.")
            return

        samples = random.sample(data, min(num_samples, len(data)))

        for i, item in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            pretty_print_item(item)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def collect_json_files(root_dir):
    json_files = []

    for root, dirs, files in os.walk(root_dir):
        # skip notebook checkpoint folders
        dirs[:] = [d for d in dirs if d != ".ipynb_checkpoints"]

        for file in files:
            if file.endswith(".json") and not file.endswith("-checkpoint.json"):
                json_files.append(os.path.join(root, file))

    return sorted(json_files)


def main():
    print("\nPRINTING DATA SAMPLES")

    if os.path.exists(TRAIN_PATH):
        print_samples(TRAIN_PATH, NUM_SAMPLES)
    else:
        print(f"Train file not found: {TRAIN_PATH}")

    if os.path.exists(TEST_DIR):
        for file_path in collect_json_files(TEST_DIR):
            print_samples(file_path, NUM_SAMPLES)
    else:
        print(f"Test directory not found: {TEST_DIR}")


if __name__ == "__main__":
    main()