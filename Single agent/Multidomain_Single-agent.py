import os
import re
import time
import random
import numpy as np
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES_PER_DOMAIN = 50
SEED = 42
metric = evaluate.load("exact_match")

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DOMAINS = [
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("cais/mmlu", "anatomy"),
    ("cais/mmlu", "marketing"),
    ("cais/mmlu", "high_school_geography")
]

def extract_answer_letter(text):
    patterns = [
        r"final answer\s*\(ONLY a single letter A, B, C, or D\):\s*([A-D])",
        r"correct answer is:?\s*([A-D])",
        r"answer:?\s*([A-D])",
        r"choice:?\s*([A-D])",
        r"option:?\s*([A-D])",
        r"select.*?([A-D])",
        r"conclusion.*?([A-D])",
        r"\b([A-D])\b(?![a-z])"
    ]
    lines = text.split('\n')[-5:] if '\n' in text else [text]
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    return None

def prepare_agent_context(entries, domain):
    context = f"\n--- Domain: {domain.upper()} ---\n"
    for entry in entries:
        if "answerKey" in entry:  # AI2-ARC
            question = entry["question"][:150]
            answer = entry["answerKey"].strip()
        else:  # MMLU
            question = entry["question"]
            answer = chr(65 + entry["answer"])
        context += f"- Q: {question}\n  A: {answer}\n"
    return context

def format_choices(entry):
    if "choices" in entry and isinstance(entry["choices"], dict):  # AI2-ARC
        return "\n".join([f"{label}: {text}" for label, text in zip(entry["choices"]["label"], entry["choices"]["text"])])
    else:  # MMLU
        return "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(entry["choices"])])

def prepare_prompt(entry):
    question = entry["question"]
    choices = format_choices(entry)
    return f"Q: {question}\nChoices:\n{choices}"

class SingleAgent:
    def __init__(self, model_path, knowledge=""):
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_threads=4)
        self.knowledge = knowledge

    def answer(self, context):
        prompt = f"""You are a specialist answering multiple-choice questions across domains.
Use the provided Q&A examples to guide your reasoning.

{self.knowledge}

Now, answer the following question:
{context}

Follow these steps:
1. Analyze the question and options.
2. Compare with any relevant training Q&A.
3. Reason your way to the answer.
4. At the end, write:
   Final answer (ONLY a single letter A, B, C, or D): <your letter>

Answer:
"""
        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
            logit_bias={65: 5, 66: 5, 67: 5, 68: 5}
        )
        return response["choices"][0]["text"].strip()

class Judge:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)

    def evaluate(self, output):
        answer = extract_answer_letter(output)
        if answer:
            return answer
        fallback = output.split('\n')[-1]
        return extract_answer_letter(fallback) or random.choice(["A", "B", "C", "D"])

def load_entries():
    all_entries = []
    for dataset_name, subset in DOMAINS:
        data = load_dataset(dataset_name, subset)
        split = "test" if "mmlu" in dataset_name else "test"
        entries = list(data[split].select(range(MAX_EXAMPLES_PER_DOMAIN)))
        all_entries.append((subset, entries))
    return all_entries

def benchmark(agent, judge):
    all_entries = load_entries()
    full_context = ""
    results = []
    start = time.time()

    for domain, entries in all_entries:
        full_context += prepare_agent_context(entries, domain)

    agent.knowledge = full_context

    index = 0
    for domain, entries in all_entries:
        print(f"\n=== Evaluating domain: {domain.upper()} ===")
        for entry in entries:
            question_text = prepare_prompt(entry)
            expected = chr(65 + entry["answer"]) if "answer" in entry else entry["answerKey"].strip()

            print(f"\n--- Question {index+1} ({domain}) ---")
            model_output = agent.answer(question_text)
            print(f"[Model Output]\n{model_output}")

            predicted = judge.evaluate(model_output)
            print(f"Expected: {expected} | Predicted: {predicted}")

            results.append({"prediction": predicted, "reference": expected})
            index += 1

    elapsed = time.time() - start
    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results]
    )

    print(f"\nFull Multi-Domain Benchmark (Memorization Test): {score['exact_match'] * 100:.2f}% in {elapsed:.2f}s")
    return score["exact_match"], elapsed

if __name__ == "__main__":
    agent = SingleAgent(MODEL_PATH)
    judge = Judge(MODEL_PATH)
    try:
        acc, runtime = benchmark(agent, judge)
    finally:
        del agent.llm
        del judge.llm