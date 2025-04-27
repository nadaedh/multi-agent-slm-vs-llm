import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# # Model Configuration
# MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
# MODEL_FILE = "Llama-3.2-7B-Instruct-Q4_K_M.gguf"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

metric = evaluate.load("exact_match")

def load_hellaswag():
    dataset = load_dataset("Rowan/hellaswag")["test"]
    return dataset

def benchmark_hellaswag(llm):
    dataset = load_dataset("Rowan/hellaswag")["validation"]
    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem = entry["ctx"]
        choices = "\n".join([f"{idx}: {text}" for idx, text in enumerate(entry["endings"])])

        expected_answer = entry.get("label", "").strip()
        if expected_answer == "":  # Skip if label is missing
            print(f" Skipping example {i+1}: No label found")
            continue

        formatted_prompt = (
            "Complete the following sentence with the most logical ending. Output only the correct choice number.\n\n"
            f"Context: {problem}\n"
            f"Choices:\n{choices}\n\n"
            "Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=5,  
            stop=[".", "?"]
        )

        raw_output = response["choices"][0]["text"].strip()
        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Context {i+1}: {problem}")
        print(f" Expected: {expected_answer}")
        print(f" Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n HellaSwag Benchmark Completed: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_hellaswag(llm)
    finally:
        del llm