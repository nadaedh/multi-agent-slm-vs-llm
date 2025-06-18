import math
import os
import random
import time
import re
import numpy as np
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# Debate + judge using MMLU (anatomy), A gets half answers, B gets the other half

MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 50
metric = evaluate.load("exact_match")

# Fix seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def extract_answer_letter(text):
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else "INVALID"

def prepare_agent_context(entries):
    context = "You have been trained with the following Q&A examples:\n"
    for entry in entries:
        q = entry["question"][:150]
        a = chr(65 + entry["answer"])  # convert index to A–D
        context += f"- Q: {q} → A: {a}\n"
    return context

class MistralAgent:
    def __init__(self, name, model_path, knowledge=""):
        self.name = name
        self.knowledge = knowledge
        self.llm = Llama(model_path=model_path, n_ctx=2048, logits_all=True)
        self.memory = []

    def reset(self):
        self.memory.clear()

    def receive(self, sender, message):
        self.memory.append(f"{sender}: {message}")

    def respond(self):
        prompt = ""
        if self.knowledge:
            prompt += f"You are a helpful medical expert. Use the following known Q&A examples to inform your reasoning:\n{self.knowledge}\n\n"

        prompt += "\n".join(self.memory) + f"\n{self.name}:"

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.7,
            max_tokens=50,
            stop=["\n", "."]
        )

        raw_output = response["choices"][0]["text"].strip()
        self.memory.append(f"{self.name}: {raw_output}")
        print(f"[{self.name}] Raw: {raw_output}")
        return raw_output

class JudgeAgent:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=2048)

    def analyze(self, debate_history):
        prompt = (
            "You are a neutral judge. Based on the following conversation between two experts, "
            "your task is to extract the final answer to a multiple-choice question (A, B, C, or D). "
            "Only output a single letter with no explanation.\n\n"
            "Example Conversation:\n"
            "AgentA: I believe the answer is A because...\n"
            "AgentB: I disagree, I think it's C due to...\n"
            "AgentA: Good point, C seems correct.\n\n"
            "Final answer (only A, B, C, or D): C\n\n"
            "Now judge the following conversation:\n"
            + "\n".join(debate_history) +
            "\n\nFinal answer (only A, B, C, or D):"
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

def debate_and_judge(agent1, agent2, judge, user_prompt, max_turns=5):
    agent1.reset()
    agent2.reset()
    agent1.receive("User", user_prompt)
    agent2.receive("User", user_prompt)

    for turn in range(max_turns):
        print(f"\nTurn {turn+1}:")
        msg1 = agent1.respond()
        agent2.receive(agent1.name, msg1)

        msg2 = agent2.respond()
        agent1.receive(agent2.name, msg2)

    return judge.analyze(agent1.memory + agent2.memory)

def load_mmlu_anatomy():
    return load_dataset("cais/mmlu", "anatomy")["test"].select(range(MAX_EXAMPLES))

def benchmark_debate_system(agent1, agent2, judge):
    dataset = load_mmlu_anatomy()
    entries = list(dataset)

    # Split Q&A examples: first half to A, second half to B
    mid = MAX_EXAMPLES // 2
    agent1.knowledge = prepare_agent_context(entries[:mid])
    agent2.knowledge = prepare_agent_context(entries[mid:])

    results = []
    start_time = time.time()

    for i, entry in enumerate(entries):
        question = entry["question"]
        choices = entry["choices"]
        expected = chr(65 + entry["answer"])  # Convert index to A–D

        labeled_choices = "\n".join([f"{chr(65 + j)}: {text}" for j, text in enumerate(choices)])
        formatted = (
            "You are a helpful medical expert. Your task is to answer the following multiple-choice anatomy question. "
            "Discuss the reasoning step-by-step with a fellow expert.\n\n"
            f"Q: {question}\nChoices:\n{labeled_choices}"
        )

        answer = debate_and_judge(agent1, agent2, judge, formatted)
        results.append({"prediction": answer, "reference": expected})

        print(f"\nQ{i+1}: {question}")
        print(f"Expected: {expected}")
        print(f"Answer: {answer}\n")

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

        acc, total_time = benchmark_debate_system(agent_a, agent_b, judge)
        print(f"\nMulti-Agent Debate w/ Judge (MMLU-Anatomy, Split Context) - Accuracy: {acc*100:.2f}% | Time: {total_time:.2f}s")
    finally:
        del agent_a.llm
        del agent_b.llm
        del judge.llm