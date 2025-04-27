import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

metric = evaluate.load("exact_match")

def load_swe_bench():
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
    
    dataset = dataset.map(lambda x: {"problem_statement": x["problem_statement"][:1000]})  # First 1000 chars
    return dataset

def benchmark_swe_bench(llm):
    dataset = load_swe_bench()
    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem_statement = entry["problem_statement"]
        expected_patch = str(entry["patch"]).strip()

        formatted_prompt = (
            "You are an AI software engineer. Given a GitHub issue with a bug description, output only the correct Git patch with the fix.\n\n"
            f"Issue Description:\n{problem_statement}\n\n"
            "Expected Fix (Git patch format):\n"
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=100,
            stop=["\n", "---", "diff"]
        )

        raw_output = response["choices"][0]["text"].strip()

        if not raw_output:
            print(f"\n Empty output for issue {i+1}!")

        results.append({"prediction": raw_output, "reference": expected_patch})

        print(f"\n Issue {i+1}: {problem_statement[:200]}...")
        print(f" Expected Patch:\n{expected_patch}")
        print(f" Model Output:\n{raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n SWE-Bench Verified Benchmark Completed: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_swe_bench(llm)
    finally:
        del llm 