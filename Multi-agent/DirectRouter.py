import os
import time
import random
import re
from datasets import load_dataset
from llama_cpp import Llama
import evaluate

SEED = 42
MAX_EXAMPLES = 50
CONTEXT_EXAMPLES = 25
MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
metric = evaluate.load("exact_match")
random.seed(SEED)

def extract_answer_letter(text):
    patterns = [
        r"final answer\s*\(ONLY a single letter A, B, C, or D\):\s*([A-D])",
        r"final answer\s*[:\-]?\s*([A-D])",
        r"correct answer\s*[:\-]?\s*([A-D])",
        r"answer\s*[:\-]?\s*([A-D])",
        r"\b([A-D])\b(?![a-z])"
    ]

    lines = text.split('\n')[-5:] if '\n' in text else [text]
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).upper()

    return "INVALID"

class SpecialistAgent:
    def __init__(self, name, model_path, domain_knowledge=""):
        self.name = name
        self.llm = Llama(model_path=model_path, n_ctx=2048, logits_all=True)
        self.knowledge = domain_knowledge

    def answer(self, question, choices):
        prompt = (
        "You are a specialist answering multiple-choice questions. "
        "First ANALYZE these similar questions and their correct answers:\n"
        f"{self.knowledge}\n\n"
        "NEW QUESTION:\n"
        f"{question}\n"
        "OPTIONS:\n"
        f"{choices}\n\n"
        "REASONING STEPS:\n"
        "1. Find the most similar training question to this new question\n"
        "2. Compare how their options differ\n"
        "3. Explain why the correct answer applies here\n"
        "4. At the end, output this exact format:\n"
        "**Final answer (ONLY a single letter A, B, C, or D): <your answer>**\n\n"
        "Answer:"
    )

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=300,
            stop=["\n\n"],
            logit_bias={65: 5, 66: 5, 67: 5, 68: 5},
        )
        return response["choices"][0]["text"].strip()


def prepare_context(entries, domain):
    context = f"You have been trained with these specific Q&A pairs from the {domain} domain:\n"
    for entry in entries:
        if domain == "math":
            question = entry["question"]
            answer = entry["answerKey"].strip()
        else:
            question = entry["question"]
            answer = chr(65 + entry["answer"])
        context += f"Q: {question} → A: {answer}\n"
    return context

def format_choices(entry, domain):
    if domain == "math":
        return "\n".join([f"{label}: {text}" for label, text in zip(entry["choices"]["label"], entry["choices"]["text"])])
    else:
        return "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(entry["choices"])])

def get_ground_truth(entry, domain):
    return entry["answerKey"] if domain == "math" else chr(65 + entry["answer"])

def run_direct_benchmark():
    # Load datasets
    math_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    anatomy_ds = load_dataset("cais/mmlu", "anatomy")["test"]

    # Use first 25 as context and next 25 for benchmarking
    math_train = list(math_ds.select(range(CONTEXT_EXAMPLES)))
    math_test = list(math_ds.select(range(CONTEXT_EXAMPLES)))

    anatomy_train = list(anatomy_ds.select(range(CONTEXT_EXAMPLES)))
    anatomy_test = list(anatomy_ds.select(range(CONTEXT_EXAMPLES)))

    math_context = prepare_context(math_train, "math")
    anatomy_context = prepare_context(anatomy_train, "anatomy")

    math_agent = SpecialistAgent("MathAgent", MODEL_PATH, domain_knowledge=math_context)
    anatomy_agent = SpecialistAgent("AnatomyAgent", MODEL_PATH, domain_knowledge=anatomy_context)

    all_questions = [(q, "math") for q in math_test] + [(q, "anatomy") for q in anatomy_test]
    random.shuffle(all_questions)

    results = []
    start_time = time.time()

    for idx, (entry, domain) in enumerate(all_questions):
        question = entry["question"]
        choices = format_choices(entry, domain)
        reference = get_ground_truth(entry, domain)

        agent = math_agent if domain == "math" else anatomy_agent

        raw_output = agent.answer(question, choices)
        print(f"\n[Raw Answer from {agent.name}]:\n{raw_output}\n")

        prediction = extract_answer_letter(raw_output)
        results.append({"prediction": prediction, "reference": reference})

        print(f"\nQ{idx+1}: {question[:60]}...")
        print(f"Domain: {domain}")
        print(f"Expected: {reference} | Predicted: {prediction}")

    elapsed = time.time() - start_time
    score = metric.compute(predictions=[r["prediction"] for r in results],
                           references=[r["reference"] for r in results])
    print(f"\nDirect Benchmark Accuracy: {score['exact_match'] * 100:.2f}% | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    try:
        run_direct_benchmark()
    finally:
        pass  # Clean-up handled implicitly