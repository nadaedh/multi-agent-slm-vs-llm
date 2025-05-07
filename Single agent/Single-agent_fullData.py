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
        r"correct answer is:?\s*([A-D])",  # specific pattern first
        r"answer:?\s*([A-D])",
        r"choice:?\s*([A-D])",
        r"option:?\s*([A-D])",
        r"select.*?([A-D])",
        r"conclusion.*?([A-D])",
        r"\b([A-D])\b(?![a-z])"  # general pattern last
    ]
    
    # check last few lines first where conclusion usually appears
    lines = text.split('\n')[-5:] if '\n' in text else [text]
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
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
        
        judge_prompt = f"""Extract ONLY the final answer letter (A, B, C, or D) from this response. 
Look for the exact phrase "Therefore, the correct answer is: X" or similar conclusive statements.

Response:
{agent_output}

Extracted answer (ONLY the letter A, B, C, or D):"""
        
        response = self.llm.create_completion(
            prompt=judge_prompt,
            max_tokens=1,
            temperature=0.0,
            stop=["\n", ".", ",", " "],
            logit_bias={65: 100, 66: 100, 67: 100, 68: 100}  # mega strong bias
        )
        
        raw = response["choices"][0]["text"].strip().upper()
        answer = extract_answer_letter(raw)
        
        if not answer:
            last_line = agent_output.split('\n')[-1] if '\n' in agent_output else agent_output
            answer = extract_answer_letter(last_line)
        
        if not answer:
            answer = random.choice(["A", "B", "C", "D"])
            print(f"[Judge] Warning: Could not determine answer, choosing randomly: {answer}")
        else:
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
        if hasattr(agent, 'llm'):
            del agent.llm
        if hasattr(judge, 'llm'):
            del judge.llm