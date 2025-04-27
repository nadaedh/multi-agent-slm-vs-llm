import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# Model config
MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

metric = evaluate.load("exact_match")

def load_aime_2024():
    dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    return dataset

def benchmark_aime_2024(llm):
    dataset = load_aime_2024()
    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem = entry["Problem"]
        expected_answer = str(entry["Answer"]).strip()

        formatted_prompt = (
            "Solve the following math problem strictly by outputting only the final answer.\n\n"
            f"Problem: {problem}\n\n"
            "Final Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=20,  
            stop=["\n", ".", " "]  
        )

        raw_output = response["choices"][0]["text"].strip()

        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {problem}")
        print(f"Expected: {expected_answer}")
        print(f"Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[str(r["prediction"]) for r in results], 
                           references=[str(r["reference"]) for r in results])

    print(f"\n AIME 2024 Benchmark completed in: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_aime_2024(llm)
    finally:
        del llm