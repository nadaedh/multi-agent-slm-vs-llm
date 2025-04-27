import math
import os
import time
import re
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# 2 agents and decides final answer based on confidence level of each model (they do not interact)

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 10
metric = evaluate.load("exact_match")

def extract_answer_letter(text):
    match = re.search(r"\b([A-D])\b", text.strip().upper())
    if match:
        return match.group(1)
    return "INVALID"

class MistralAgent:
    def __init__(self, name, model_path):
        self.name = name
        self.llm = Llama(model_path=model_path, n_ctx=2048, logits_all=True)
        self.memory = []

    def reset(self):
        self.memory.clear()

    def receive(self, sender, message):
        self.memory.append(f"{sender}: {message}")

    def respond(self):
        prompt = "\n".join(self.memory) + f"\n{self.name}:"
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stop=["\n", "."],
            logprobs=5
        )
        raw_output = response["choices"][0]["text"].strip().upper()
        parsed = extract_answer_letter(raw_output)

        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        if logprobs and logprobs[0] is not None:
            prob = math.exp(logprobs[0])
        else:
            prob = 0.0

        print(f"[{self.name}] Raw: {raw_output} → Parsed: {parsed} (conf: {prob:.4f})")
        self.memory.append(f"{self.name}: {parsed}")
        return parsed, prob

def load_ai2_arc():
    return load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]

def decide_final_answer_weighted(answers_probs):
    scores = {}
    for ans, prob in answers_probs:
        if ans not in scores:
            scores[ans] = 0.0
        scores[ans] += prob
    return max(scores.items(), key=lambda x: x[1])[0]

def benchmark_agents(agent1, agent2=None):
    dataset = load_ai2_arc()
    results = []
    start_time = time.time()

    for i in range(MAX_EXAMPLES):
        entry = dataset[i]
        question = entry["question"]
        labels = entry["choices"]["label"]
        texts = entry["choices"]["text"]
        expected = entry["answerKey"].strip()

        choices = "\n".join([f"{label}: {text}" for label, text in zip(labels, texts)])
        formatted = (
            "You are a science expert. You must answer with exactly one of the following letters: A, B, C, or D. Do NOT explain. Just return a single letter."
            "Example: Q: What is 2 + 2? A: A\nChoices:\nA: 4\nB: 5\nC: 6\nD: 3\nAnswer: A\n\n"
            f"Question: {question}\nChoices:\n{choices}\n\nAnswer:"
        )

        if agent2:
            agent1.reset()
            agent2.reset()
            agent1.receive("User", formatted)
            agent2.receive("User", formatted)

            answer1, prob1 = agent1.respond()
            answer2, prob2 = agent2.respond()

            final = decide_final_answer_weighted([(answer1, prob1), (answer2, prob2)])
        else:
            agent1.reset()
            agent1.receive("User", formatted)
            answer, _ = agent1.respond()
            final = answer

        results.append({"prediction": final, "reference": expected})

        print(f"\n Q{i+1}: {question}")
        print(f" Expected: {expected}")
        print(f" Answer: {final}")

    elapsed = time.time() - start_time
    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results],
    )
    return score["exact_match"], elapsed

if __name__ == "__main__":
    try:
        solo = MistralAgent("SoloMistral", MODEL_PATH)
        acc_solo, time_solo = benchmark_agents(solo)
        print(f"\nSolo Agent - Accuracy: {acc_solo*100:.2f}% | Time: {time_solo:.2f}s")
    finally:
        del solo.llm

    try:
        agent_a = MistralAgent("AgentA", MODEL_PATH)
        agent_b = MistralAgent("AgentB", MODEL_PATH)
        acc_multi, time_multi = benchmark_agents(agent_a, agent_b)
        print(f"\nMulti-Agent - Accuracy: {acc_multi*100:.2f}% | Time: {time_multi:.2f}s")
    finally:
        del agent_a.llm
        del agent_b.llm