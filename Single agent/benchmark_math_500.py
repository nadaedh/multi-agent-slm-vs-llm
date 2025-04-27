import os
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# MODEL_DIR = "/Users/nadaeddahir/llama3_gguf/"
# MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)


MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

def load_math_500():
    dataset = load_dataset("HuggingFaceH4/MATH-500", "default")["test"]
    return dataset

def benchmark_math_500(llm):
    dataset = load_math_500()
    metric = evaluate.load("exact_match")  

    MAX_EXAMPLES = 10
    start_time = time.time()
    results = []

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        problem = entry["problem"]
        expected_answer = entry["answer"].strip()

        formatted_prompt = (
            "Solve the following math problem strictly by outputting only the final answer.\n\n"
            f"Problem: {problem}\n\n"
            "Final Answer: "
        )

        response = llm.create_completion(
            prompt=formatted_prompt, 
            max_tokens=50,  
            stop=[".", "?"]  
        )

        raw_output = response["choices"][0]["text"].strip()
        
        results.append({"prediction": raw_output, "reference": expected_answer})

        print(f"\n Question {i+1}: {problem}")
        print(f" Expected: {expected_answer}")
        print(f" Model Output: {raw_output}")

    elapsed_time = time.time() - start_time

    score = metric.compute(predictions=[r["prediction"] for r in results], 
                           references=[r["reference"] for r in results])

    print(f"\n MATH-500 Benchmark Completed: {elapsed_time:.2f}s")
    print(f"Accuracy Score: {score['exact_match'] * 100:.2f}%")

if __name__ == "__main__":
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048)  
        benchmark_math_500(llm)
    finally:
        del llm  