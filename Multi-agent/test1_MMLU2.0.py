import math
import os
import time
import re
import random
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# A gets all answers, B gets none. Confidence-based voting using MMLU benchmark.

MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 50
SEED = 42
metric = evaluate.load("exact_match")

random.seed(SEED)
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
    context = "You have access to these Q&A pairs:\n"
    for entry in entries[:MAX_EXAMPLES]:
        question = entry["question"][:150]
        answer = chr(65 + entry["answer"])  # 0 -> A, 1 -> B...
        context += f"- Q: {question} → A: {answer}\n"
    return context

class MistralAgent:
    def __init__(self, name, model_path, knowledge=""):
        self.name = name
        self.knowledge = knowledge
        self.llm = Llama(model_path=model_path, n_ctx=4096, logits_all=True)
        self.memory = []

    def reset(self):
        self.memory.clear()

    def receive(self, sender, message):
        self.memory.append(f"{sender}: {message}")

    def respond(self):
        prompt = ""

        if self.knowledge:
            prompt += f"You are an anatomy expert. You have learned the following Q&A examples:\n{self.knowledge}\n\n"

        prompt += "\n".join(self.memory) + f"\n{self.name}:"

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500,
            logit_bias={65: 5, 66: 5, 67: 5, 68: 5},
            logprobs=5
        )

        raw_output = response["choices"][0]["text"].strip().upper()
        parsed = extract_answer_letter(raw_output)

        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        prob = math.exp(logprobs[0]) if logprobs and logprobs[0] is not None else 0.0

        self.memory.append(f"{self.name}: {parsed}")
        print(f"[{self.name}] Raw: {raw_output} → Parsed: {parsed} (conf: {prob:.4f})")
        return parsed, prob

def load_mmlu_anatomy():
    return load_dataset("cais/mmlu", "anatomy")["test"].select(range(MAX_EXAMPLES))

def decide_final_answer_weighted(answers_probs):
    scores = {}
    for ans, prob in answers_probs:
        scores[ans] = scores.get(ans, 0.0) + prob
    return max(scores.items(), key=lambda x: x[1])[0]

def benchmark_agents(agent1, agent2):
    dataset = load_mmlu_anatomy()
    entries = list(dataset)

    # Agent A gets full MMLU anatomy Q&A knowledge
    agent1.knowledge = prepare_agent_context(entries)

    results = []
    start_time = time.time()

    for i, entry in enumerate(entries):
        question = entry["question"]
        choices = entry["choices"]
        expected = chr(65 + entry["answer"])

        labeled_choices = "\n".join([f"{chr(65 + j)}: {c}" for j, c in enumerate(choices)])
        formatted = f"""You are a specialist in anatomy answering multiple-choice questions.
        Answer the following question:
Question: {question}
Choices:
{labeled_choices}

Follow these steps:
1. Analyze the question and each option.
2. Reason through the answer.
3. At the end, write:
   Final answer (ONLY a single letter A, B, C, or D): <your letter>

Answer:
"""

        agent1.reset()
        agent2.reset()
        agent1.receive("User", formatted)
        agent2.receive("User", formatted)

        answer1, prob1 = agent1.respond()
        answer2, prob2 = agent2.respond()

        final = decide_final_answer_weighted([(answer1, prob1), (answer2, prob2)])

        results.append({"prediction": final, "reference": expected})

        print(f"\nQ{i+1}: {question}")
        print(f"Expected: {expected} | Final Answer: {final}")

    elapsed = time.time() - start_time
    score = metric.compute(
        predictions=[r["prediction"] for r in results],
        references=[r["reference"] for r in results],
    )
    return score["exact_match"], elapsed

if __name__ == "__main__":
    try:
        agent_a = MistralAgent("AgentA", MODEL_PATH)  # gets full context
        agent_b = MistralAgent("AgentB", MODEL_PATH)  # gets none

        acc, total_time = benchmark_agents(agent_a, agent_b)
        print(f"\nDual-Agent (MMLU-Anatomy, A has full answers) - Accuracy: {acc*100:.2f}% | Time: {total_time:.2f}s")
    finally:
        del agent_a.llm
        del agent_b.llm