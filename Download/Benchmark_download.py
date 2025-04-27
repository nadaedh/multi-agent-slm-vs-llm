from datasets import load_dataset

benchmarks = {
    "GSM8K": ("openai/gsm8k", "main"),
    "AI2-ARC": ("allenai/ai2_arc", "ARC-Easy"),  #"ARC-Challenge" or "ARC-Easy"
    "HellaSwag": "Rowan/hellaswag"
}

for name, dataset_info in benchmarks.items():
    print(f"Downloading {name} dataset...")
    try:
        if isinstance(dataset_info, tuple):
            dataset = load_dataset(dataset_info[0], dataset_info[1])
        else:
            dataset = load_dataset(dataset_info)
        print(f"{name} downloaded successfully!\n")
    except Exception as e:
        print(f"Failed to download {name}: {e}")