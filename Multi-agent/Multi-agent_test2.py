import math
import time
import re
from llama_cpp import Llama
from datasets import load_dataset
import evaluate

# 2 agents each respond to the same question with their answer, they inform each other of their choices up to max_turns (3).
# If both agents agree on a valid letter, it’s returned, otherwise, it picks the final answer using confidence-weighted voting.

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 5
metric = evaluate.load("exact_match")

def extract_answer_letter(text):
    match = re.search(r"(?:MY ANSWER IS|ANSWER)[\s:]*([A-D])\b", text.upper())
    if match:
        return match.group(1)

    matches = re.findall(r"\b([A-D])\b", text.upper())
    return matches[0] if matches else "INVALID"

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
            #max_tokens=30,
            temperature=0.0,
            stop=["\n", "."],
            logprobs=5
        )

        # full_text = response["choices"][0]["text"].strip()
        # print(f"[{self.name}] Full model response:\n{full_text}\n")

        raw_output = response["choices"][0]["text"].strip().upper()
        parsed = extract_answer_letter(raw_output)

        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        if logprobs and logprobs[0] is not None:
            prob = math.exp(logprobs[0])
        else:
            prob = 0.0

        print(f"[{self.name}] Raw: {raw_output} → Parsed: {parsed} (conf: {prob:.4f})")
        # self.memory.append(f"{self.name}: {parsed}")
        self.memory.append(f"{self.name}: {raw_output}")
        return parsed, prob, raw_output

def debate_until_agreement(agent1, agent2, user_prompt, max_turns=5):
    agent1.reset()
    agent2.reset()
    agent1.receive("User", user_prompt)
    agent2.receive("User", user_prompt)

    REMINDER = (
        "Reminder: You must respond in this exact format:\n"
        "'My answer is: <LETTER>. Here's why: <your reasoning>'\n"
        "Only one letter (A, B, C, or D)."
    )


    for turn in range(max_turns):
        print(f"\n Turn {turn+1}:")

        letter1, prob1, raw_output1 = agent1.respond()
        agent2.receive(agent1.name, f"{raw_output1}")  # includes full answer + justification 

        letter2, prob2, raw_output2 = agent2.respond()
        agent1.receive(agent2.name, f"Chose: {raw_output2}")

        # if either response is INVALID remind them of the format
        if letter1 == "INVALID":
            agent1.receive("System", REMINDER)
        if letter2 == "INVALID":
            agent2.receive("System", REMINDER)

        print(f" - {agent1.name}: {letter1} (conf: {prob1:.4f})")
        print(f" - {agent2.name}: {letter2} (conf: {prob2:.4f})")

        if letter1 == letter2 and letter1 != "INVALID":
            print(f"Consensus reached on: {letter1}")
            return letter1

    print("\nNo consensus, falling back to weighted vote :")
    return decide_final_answer_weighted([(letter1, prob1), (letter2, prob2)])

def decide_final_answer_weighted(answers_probs):
    scores = {}
    for ans, prob in answers_probs:
        scores[ans] = scores.get(ans, 0.0) + prob
    return max(scores.items(), key=lambda x: x[1])[0]

def load_ai2_arc():
    return load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]

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
            "You are a helpful science expert. Your task is to answer the multiple-choice science question below. "
            "You must choose exactly one of the options (A, B, C, or D) and explain your reasoning clearly in a single sentence.\n\n"
            "Respond in this exact format:\n"
            "My answer is: <LETTER>. Here's why: <your reasoning>\n\n"
            "Examples:\n"
            "Q: What is the boiling point of water?\n"
            "Choices:\nA: 50°C\nB: 100°C\nC: 150°C\nD: 200°C\n"
            "My answer is: B. Here's why: Water boils at 100 degrees Celsius at standard atmospheric pressure.\n\n"
            f"Q: {question}\nChoices:\n{choices}\n\nAnswer:"
        )

        if agent2:
            final = debate_until_agreement(agent1, agent2, formatted)
        else:
            agent1.reset()
            agent1.receive("User", formatted)
            final, _ = agent1.respond()

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
    # try:
    #     solo = MistralAgent("SoloMistral", MODEL_PATH)
    #     acc_solo, time_solo = benchmark_agents(solo)
    #     print(f"\nSolo Agent - Accuracy: {acc_solo*100:.2f}% | Time: {time_solo:.2f}s")
    # finally:
    #     del solo.llm

    try:
        agent_a = MistralAgent("AgentA", MODEL_PATH)
        agent_b = MistralAgent("AgentB", MODEL_PATH)
        acc_multi, time_multi = benchmark_agents(agent_a, agent_b)
        print(f"\nMulti-Agent - Accuracy: {acc_multi*100:.2f}% | Time: {time_multi:.2f}s")
    finally:
        del agent_a.llm
        del agent_b.llm