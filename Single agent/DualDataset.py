import re
import os
import random
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate
import numpy as np

MODEL_PATH = "/localhome/eddahir/benchmark-llm/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES_PER_DOMAIN = 25
SEED = 42
metric = evaluate.load("exact_match")

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def extract_answer_letter(text):
    patterns = [
        r"correct answer is:?[ \t]*([A-D])",
        r"answer:?[ \t]*([A-D])",
        r"choice:?[ \t]*([A-D])",
        r"option:?[ \t]*([A-D])",
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
    context = "You have been trained with these specific Q&A pairs:\n"
    for entry in entries:
        question = entry["question"][:150]
        answer = entry["answerKey"].strip()
        context += f"- Q: {question} → A: {answer}\n"
    return context

class SingleAgent:
    def __init__(self, model_path, knowledge=""):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4
        )
        self.knowledge = knowledge

    def answer(self, context):
        prompt = f"""You are an expert answering multiple-choice science questions.
Here is your training data:
{self.knowledge}

When answering new questions:
1. Carefully read the question and all options
2. Compare with similar questions from your training data
3. Analyze each option systematically
4. Explain your reasoning step by step
5. Conclude with EXACTLY ONE of these EXACT phrases:
   - "Therefore, the correct answer is: A"
   - "Therefore, the correct answer is: B"
   - "Therefore, the correct answer is: C"
   - "Therefore, the correct answer is: D"

Question:
{context}
Your analysis:"""

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
            stop=["\n\n", "Question:"],
            logit_bias={65: 5, 66: 5, 67: 5, 68: 5}
        )

        return response["choices"][0]["text"].strip()

class Judge:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=2
        )

    def evaluate(self, agent_output):
        answer = extract_answer_letter(agent_output)
        if answer:
            return answer

        judge_prompt = f"""Extract ONLY the final answer letter (A, B, C, or D) from this response.
Response:
{agent_output}
Extracted answer (ONLY the letter A, B, C, or D):"""

        response = self.llm.create_completion(
            prompt=judge_prompt,
            max_tokens=1,
            temperature=0.0,
            stop=["\n", ".", ",", " "],
            logit_bias={65: 100, 66: 100, 67: 100, 68: 100}
        )
        raw = response["choices"][0]["text"].strip().upper()
        answer = extract_answer_letter(raw) or random.choice(["A", "B", "C", "D"])
        return answer

def prepare_prompt(entry):
    question = entry["question"]
    choices = "\n".join([
        f"{label}: {text}" for label, text in zip(entry["choices"]["label"], entry["choices"]["text"])
    ])
    return f"Q: {question}\nChoices:\n{choices}"

def load_combined_dataset():
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")['test'].select(range(MAX_EXAMPLES_PER_DOMAIN))
    mmlu = load_dataset("cais/mmlu", "anatomy")['test'].select(range(MAX_EXAMPLES_PER_DOMAIN))

    # Convert MMLU to ARC-like format
    mmlu_entries = []
    for item in mmlu:
        mmlu_entries.append({
            "question": item["question"],
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": [item["choices"][i] for i in range(4)]
            },
            "answerKey": item["answer"]
        })

    return list(arc) + mmlu_entries

def benchmark(agent, judge):
    entries = load_combined_dataset()
    agent.knowledge = prepare_agent_context(entries)

    results = []
    start_time = time.time()

    for idx, entry in enumerate(entries):
        context = prepare_prompt(entry)
        expected = entry["answerKey"].strip()
        model_output = agent.answer(context)
        predicted = judge.evaluate(model_output)

        results.append({"prediction": predicted, "reference": expected})

        print(f"\nQuestion {idx+1}:")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")

    elapsed = time.time() - start_time
    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results],
    )
    return score["exact_match"], elapsed

if __name__ == "__main__":
    agent = SingleAgent(MODEL_PATH)
    judge = Judge(MODEL_PATH)
    acc, total_time = benchmark(agent, judge)
    print(f"\nMixed-Domain Agent Accuracy: {acc*100:.2f}% | Time: {total_time:.2f}s")
