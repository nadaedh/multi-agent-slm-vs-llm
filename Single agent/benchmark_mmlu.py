import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# Model Configuration
#MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = "/localhome/eddahir/benchmark-llm/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"
metric = evaluate.load("exact_match")

def load_mmlu():
    dataset = load_dataset("cais/mmlu", "anatomy")["test"]
    return dataset

def benchmark_mmlu(llm):
    dataset = load_mmlu()
    MAX_EXAMPLES = 50
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        question = entry["question"]
        choices = entry["choices"]
        expected_answer = chr(65 + entry["answer"])  # Already A, B, C, or D

        # Label choices as A, B, C, D
        labeled_choices = "\n".join([f"{chr(65 + j)}: {c}" for j, c in enumerate(choices)])

        formatted_prompt = (
            "Answer the following multiple-choice question. Only provide the letter corresponding to the correct answer {A, B, C, D} with no explanation or extra text.\n\n"
            f"Question: {question}\n"
            f"Choices:\n{labeled_choices}\n\n"
            "Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt,
            max_tokens=5,
            stop=[".", ":"]
        )

        raw_output = response["choices"][0]["text"].strip().upper()

        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results]
    )

    print(f"\n MMLU Benchmark Completed: {elapsed_time:.2f}s")
    print(f" Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)
        benchmark_mmlu(llm)
    finally:
        del llm