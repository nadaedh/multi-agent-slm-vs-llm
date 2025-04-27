import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

metric = evaluate.load("exact_match")

def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    return dataset

def benchmark_gsm8k(llm):
    dataset = load_gsm8k()
    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem = entry["question"]
        expected_answer = entry["answer"].split("####")[-1].strip()

        formatted_prompt = (
            "### Instruction:\nSolve the following math problem and provide only the final answer number.\n\n"
            f"### Problem: {problem}\n\n"
            "### Final Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=50,  
            stop=["\n", ".", "?"]
        )

        raw_output = response["choices"][0]["text"].strip()

        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {problem}")
        print(f" Expected: {expected_answer}")
        print(f" Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n GSM8K Benchmark Completed: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_gsm8k(llm)
    finally:
        del llm