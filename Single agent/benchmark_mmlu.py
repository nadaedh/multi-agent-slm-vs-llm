import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
# MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

metric = evaluate.load("exact_match")

def load_mmlu():
    dataset = load_dataset("cais/mmlu", "all")["test"]
    return dataset

def benchmark_mmlu(llm):
    dataset = load_mmlu()
    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        question = entry["question"]
        choices = entry["choices"]
        expected_answer = str(entry["answer"]).strip()

        formatted_prompt = (
            "Answer the following multiple-choice question by only outputting the single correct choice index (0, 1, 2, or 3) without explanation.\n\n"
            f"Question: {question}\n\n"
            f"Choices: {choices}\n\n"
            "Final Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=5
        )

        raw_output = response["choices"][0]["text"].strip()

        if not raw_output:
            print(f"\n Empty output for question {i+1}!")

        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {question}")
        print(f" Expected: {expected_answer}")
        print(f" Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n MMLU Benchmark completed in: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_mmlu(llm)
    finally:
        del llm 