import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# # Model Configuration
# MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
# MODEL_FILE = "Llama-3.2-7B-Instruct-Q4_K_M.gguf"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Model Configuration Mistral
MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

metric = evaluate.load("exact_match")

def load_ai2_arc():
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]  # use "ARC-Easy" ?
    return dataset

def benchmark_ai2_arc(llm):
    dataset = load_ai2_arc()
    MAX_EXAMPLES = 50
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem = entry["question"]
        choices = "\n".join([f"{label}: {text}" for label, text in zip(entry["choices"]["label"], entry["choices"]["text"])])
        expected_answer = entry["answerKey"].strip()

        formatted_prompt = (
            "Answer the following multiple-choice science question. Only provide the only letter corresponding to the correct answer {A, B, C, D} with no other text after it.\n\n"
            f"Question: {problem}\n"
            f"Choices:\n{choices}\n\n"
            "Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=5,  
            stop=[".", ":"]
        )

        raw_output = response["choices"][0]["text"].strip().upper()

        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {problem}")
        print(f"Expected: {expected_answer}")
        print(f"Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n AI2-ARC Benchmark Completed: {elapsed_time:.2f}s")
    print(f" Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_ai2_arc(llm)
    finally:
        del llm 