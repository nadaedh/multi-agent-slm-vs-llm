import math
import os
import random
import time
import re
from llama_cpp import Llama
from datasets import load_dataset
import evaluate
import numpy as np

# Two agents each get half of the questions with corresponding answers of the dataset before discussing each question and answering. 
# A judge reads the conversation and determines the letter corresponding to the given answer. 


MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 10  
SEED = 42
metric = evaluate.load("exact_match")

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def extract_answer_letter(text):
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else "INVALID"

class MistralAgent:
    def __init__(self, name, model_path):
        self.name = name
        self.llm = Llama(model_path=model_path, n_ctx=4096, logits_all=True)
        self.memory = []

    def reset(self):
        self.memory.clear()

    def receive(self, sender, message):
        self.memory.append(f"{sender}: {message}")

    def respond(self):
        prompt = "\n".join(self.memory) + f"\n{self.name}:"

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200,
            stop=["\n", "\n\n"]
        )

        raw_output = response["choices"][0]["text"].strip()
        self.memory.append(f"{self.name}: {raw_output}")
        print(f"[{self.name}] Raw: {raw_output}")
        return raw_output

class JudgeAgent:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=4096)

    def analyze(self, discussion_history):
        prompt = (
            "You are a neutral judge. Based on the following discussion between two experts, "
            "determine the final answer (A, B, C, or D). Only output the letter without any explanation.\n\n"
            + "\n".join(discussion_history)
            + "\n\nFinal answer (only A, B, C, or D):"
        )

        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=2,
            temperature=0.0,
            stop=["\n", "."]
        )
        raw = response["choices"][0]["text"].strip().upper()
        parsed = extract_answer_letter(raw)
        print(f"[Judge] Raw: {raw} → Judged final answer: {parsed}")
        return parsed

def load_ai2_arc():
    return load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]

def prepare_agent_context(agent_entries):
    context = "You have access to the following questions and their correct answers:\n\n"
    for entry in agent_entries:
        question = entry["question"]
        choices = "\n".join([f"{label}: {text}" for label, text in zip(entry["choices"]["label"], entry["choices"]["text"])])
        answer = entry["answerKey"].strip()
        context += f"Q: {question}\nChoices:\n{choices}\nCorrect Answer: {answer}\n\n"
    return context

def debate_and_judge(agent1, agent2, judge, user_question, max_turns=5):
    agent1.reset()
    agent2.reset()

    agent1.receive("User", user_question)
    agent2.receive("User", user_question)

    for turn in range(max_turns):
        print(f"\nTurn {turn+1}:")
        msg1 = agent1.respond()
        agent2.receive(agent1.name, msg1)

        msg2 = agent2.respond()
        agent1.receive(agent2.name, msg2)

    return judge.analyze(agent1.memory + agent2.memory)

def benchmark_multiagent_system(agent1, agent2, judge):
    dataset = load_ai2_arc()
    dataset = dataset.select(range(MAX_EXAMPLES))
    entries = list(dataset)

    # divide dataset
    mid = len(entries) // 2
    entries_agent1 = entries[:mid]
    entries_agent2 = entries[mid:]

    context1 = prepare_agent_context(entries_agent1)
    context2 = prepare_agent_context(entries_agent2)

    results = []
    start_time = time.time()

    for idx, entry in enumerate(entries):
        question = entry["question"]
        labels = entry["choices"]["label"]
        texts = entry["choices"]["text"]
        expected = entry["answerKey"].strip()

        choices = "\n".join([f"{label}: {text}" for label, text in zip(labels, texts)])

        user_prompt = (
            "You are two collaborating experts who have access to your own questions and their answers. "
            "Discuss together to determine the correct answer to the question below.\n\n"
            f"{agent1.name} context:\n{context1}\n"
            f"{agent2.name} context:\n{context2}\n\n"
            f"Q: {question}\nChoices:\n{choices}"
        )

        predicted = debate_and_judge(agent1, agent2, judge, user_prompt)

        results.append({"prediction": predicted, "reference": expected})

        print(f"\nQ{idx+1}: {question}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}\n")

    elapsed = time.time() - start_time
    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results],
    )
    return score["exact_match"], elapsed

if __name__ == "__main__":
    try:
        agent_a = MistralAgent("AgentA", MODEL_PATH)
        agent_b = MistralAgent("AgentB", MODEL_PATH)
        judge = JudgeAgent(MODEL_PATH)

        acc, total_time = benchmark_multiagent_system(agent_a, agent_b, judge)
        print(f"\nMulti-Agent (with ground-truth context) - Accuracy: {acc*100:.2f}% | Time: {total_time:.2f}s")
    finally:
        del agent_a.llm
        del agent_b.llm
        del judge.llm
