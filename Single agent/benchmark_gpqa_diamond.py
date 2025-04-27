import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate 

# MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
# MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

metric = evaluate.load("exact_match")

def load_gpqa_diamond():
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    return dataset

def benchmark_gpqa_diamond(llm):
    dataset = load_gpqa_diamond()
    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem = entry.get("Question", entry.get("Pre-Revision Question", ""))
        expected_answer = str(entry["Correct Answer"]).strip()

        formatted_prompt = (
            "Solve the following math problem strictly by outputting only the final answer without a sentence.\n\n"
            f"Problem: {problem}\n\n"
            "Final Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=50,
        )

        raw_output = str(response["choices"][0]["text"]).strip() 

        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {problem}")
        print(f" Expected: {expected_answer}")
        print(f" Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n GPQA Diamond Benchmark Completed: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_gpqa_diamond(llm)
    finally:
        del llm 