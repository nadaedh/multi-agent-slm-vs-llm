import re
import os
import random
import time
from llama_cpp import Llama
from datasets import load_dataset
import evaluate
import numpy as np

# A single agent gets all the questions and corresponding answers before answering.

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 10
SEED = 42
metric = evaluate.load("exact_match")

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def extract_answer_letter(text):
    patterns = [
        r"final answer:?\s*([A-D])",
        r"answer:?\s*([A-D])",
        r"choice:?\s*([A-D])",
        r"correct:?\s*([A-D])",
        r"option:?\s*([A-D])",
        r"therefore.*?([A-D])",
        r"\b([A-D])\b(?![a-z])" 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None

def prepare_agent_context(entries):
    context = "You have been trained with these specific Q&A pairs:\n"
    for entry in entries[:MAX_EXAMPLES]:
        question = entry["question"][:150]  # truncate long questions
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
5. Conclude with: "Therefore, the correct answer is: A" or "Therefore, the correct answer is: B" or "Therefore, the correct answer is: C" or "Therefore, the correct answer is: D"

Question:
{context}
Your analysis:"""
        
        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000, 
            stop=["\n\n", "Question:"],
            logit_bias={65: 5, 66: 5, 67: 5, 68: 5}  # biased toward A/B/C/D
        )

        raw_output = response["choices"][0]["text"].strip()
        print(f"[Agent Output]: {raw_output}")
        return raw_output

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
            print(f"[Judge] Direct extraction: {answer}")
            return answer
            
        judge_prompt = f"""Extract the final answer (A, B, C, or D) from this response:
{agent_output}

Only respond with the single letter, nothing else. Answer:"""
        
        response = self.llm.create_completion(
            prompt=judge_prompt,
            max_tokens=1,
            temperature=0.0,
            stop=["\n", " "],
            logit_bias={65: 10, 66: 10, 67: 10, 68: 10}  # stronger bias toward A/B/C/D
        )
        
        raw = response["choices"][0]["text"].strip().upper()
        answer = extract_answer_letter(raw) or random.choice(["A", "B", "C", "D"])
        print(f"[Judge] LLM extraction: {answer}")
        return answer

def load_ai2_arc():
    return load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]

def prepare_prompt(entry):
    question = entry["question"]
    choices = "\n".join([f"{label}: {text}" for label, text in zip(entry["choices"]["label"], entry["choices"]["text"])])
    return f"Q: {question}\nChoices:\n{choices}"

def benchmark(agent, judge):
    dataset = load_ai2_arc()
    dataset = dataset.select(range(MAX_EXAMPLES))
    entries = list(dataset)

    # Provide all questions and answers as context to the agent
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
    try:
        agent = SingleAgent(MODEL_PATH)
        judge = Judge(MODEL_PATH)

        acc, total_time = benchmark(agent, judge)
        print(f"\nSingle Agent System - Accuracy: {acc*100:.2f}% | Time: {total_time:.2f}s")
    finally:
        # avoid the TypeError on exit
        if hasattr(agent, 'llm'):
            del agent.llm
        if hasattr(judge, 'llm'):
            del judge.llm