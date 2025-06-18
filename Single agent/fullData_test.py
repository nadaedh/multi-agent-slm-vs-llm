import os
import re
import time
import random
import numpy as np
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# full data but for MMLU benchmark this time 

MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
#MODEL_PATH = "/localhome/eddahir/benchmark-llm/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 50
SEED = 42
metric = evaluate.load("exact_match")

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def extract_answer_letter(text):
    patterns = [
        r"final answer\s*\(ONLY a single letter A, B, C, or D\):\s*([A-D])",
        r"final answer:?\s*([A-D])",
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

def prepare_agent_context(entries):
    context = "Below are sample anatomy multiple-choice questions with correct answers. Do NOT re-answer them. Use them to guide your reasoning.\n\n"
    for entry in entries:
        question = entry["question"]
        answer = chr(65 + entry["answer"])
        context += f"(Example) Q: {question}\n(Example) A: {answer}\n\n"
    return context

def format_choices(entry):
    return "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(entry["choices"])])

class SingleAgent:
    def __init__(self, model_path, knowledge=""):
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_threads=4)
        self.knowledge = knowledge

    def answer(self, context):
        prompt = f"""You are a specialist in anatomy answering multiple-choice questions.
Use the training examples below for reference:

{self.knowledge}

Now, answer the following question:
{context}

Follow these steps:
1. Analyze the question and each option.
2. Compare to the training questions if relevant.
3. Reason through the answer.
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
        answer = extract_answer_letter(fallback)
        return answer if answer else random.choice(["A", "B", "C", "D"])

def load_mmlu_anatomy():
    return load_dataset("cais/mmlu", "anatomy")["test"].select(range(MAX_EXAMPLES))

def benchmark(agent, judge):
    dataset = load_mmlu_anatomy()
    agent.knowledge = prepare_agent_context(dataset)

    results = []
    start = time.time()

    for idx, entry in enumerate(dataset):
        question_text = f"Q: {entry['question']}\nChoices:\n{format_choices(entry)}"
        expected = chr(65 + entry["answer"])

        print(f"\n--- Question {idx+1} ---")
        model_output = agent.answer(question_text)
        print(f"[Model Output]\n{model_output}")

        predicted = judge.evaluate(model_output)
        print(f"Expected: {expected} | Predicted: {predicted}")

        results.append({"prediction": predicted, "reference": expected})

    elapsed = time.time() - start
    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results]
    )

    print(f"\nFull-data MMLU Anatomy Benchmark: {score['exact_match'] * 100:.2f}% in {elapsed:.2f}s")
    return score["exact_match"], elapsed

if __name__ == "__main__":
    agent = SingleAgent(MODEL_PATH)
    judge = Judge(MODEL_PATH)
    try:
        acc, runtime = benchmark(agent, judge)
    finally:
        del agent.llm
        del judge.llm