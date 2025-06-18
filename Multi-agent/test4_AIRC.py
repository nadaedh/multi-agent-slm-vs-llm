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


MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 50  
SEED = 42
MAX_CONTEXT_EXAMPLES = 5
metric = evaluate.load("exact_match")

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

def extract_answer_letter(text):
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else "INVALID"

class MistralAgent:
    def __init__(self, name, model_path, knowledge=""):
        self.name = name
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4,
            logits_all=True
        )
        self.memory = []
        self.knowledge = knowledge

    def reset(self):
        self.memory.clear()

    def receive(self, sender, message):
        self.memory.append(f"{sender}: {message}")

    def respond(self):
        prompt = (
            f"Role: You are {self.name}, an expert with specific knowledge.\n"
            f"Knowledge:\n{self.knowledge}\n\n"
            "Conversation History:\n" + "\n".join(self.memory) + 
            f"\n{self.name}: Based on my knowledge and the discussion, the answer is"
        )

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,  # lower temperature for more focused answers
            max_tokens=10,
            stop=["\n", ".", ","]
        )

        raw_output = response["choices"][0]["text"].strip()
        self.memory.append(f"{self.name}: {raw_output}")
        print(f"[{self.name}] Raw: {raw_output}")
        return raw_output

class JudgeAgent:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4
        )

    def analyze(self, discussion_history):
        recent_discussion = "\n".join(discussion_history[-6:]) # only last few turns for decision
        
        prompt = (
            "Role: Neutral Judge\n\n"
            "Discussion Summary:\n"
            f"{recent_discussion}\n\n"
            "Task: Determine the FINAL answer (A/B/C/D) that both experts agree on.\n"
            "Output ONLY the letter (A, B, C, or D) with no other text.\n"
            "Final Answer:"
        )

        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=2,
            temperature=0.1,
            stop=["\n", ".", ","]
        )
        
        raw = response["choices"][0]["text"].strip().upper()
        parsed = extract_answer_letter(raw)
        print(f"[Judge] Raw: {raw} → Judged final answer: {parsed}")
        return parsed

def load_ai2_arc():
    return load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]

def prepare_agent_context(agent_entries):
    context = "You know these specific Q&A pairs:\n"
    for entry in agent_entries[:MAX_CONTEXT_EXAMPLES]:
        question = entry["question"][:150]
        answer = entry["answerKey"].strip()
        context += f"- Q: {question} → A: {answer}\n"
    return context

def debate_and_judge(agent1, agent2, judge, user_question, max_turns=3):
    agent1.reset()
    agent2.reset()

    # clearer instructions
    agent1.receive("System", "Discuss to find the correct answer. Reference your knowledge.")
    agent2.receive("System", "Discuss to find the correct answer. Reference your knowledge.")
    
    agent1.receive("User", user_question)
    agent2.receive("User", user_question)

    for turn in range(max_turns):
        print(f"\nTurn {turn+1}:")
        msg1 = agent1.respond()
        agent2.receive(agent1.name, msg1)

        msg2 = agent2.respond()
        agent1.receive(agent2.name, msg2)

    answer = judge.analyze(agent1.memory + agent2.memory)
    if answer not in ["A", "B", "C", "D"]:
        answers = [extract_answer_letter(msg) for msg in agent1.memory + agent2.memory]
        valid_answers = [a for a in answers if a in ["A", "B", "C", "D"]]
        answer = max(set(valid_answers), key=valid_answers.count) if valid_answers else "INVALID"
    return answer

def benchmark_multiagent_system(agent1, agent2, judge):
    dataset = load_ai2_arc()
    dataset = dataset.select(range(MAX_EXAMPLES))
    entries = list(dataset)

    mid = len(entries) // 2
    agent1.knowledge = prepare_agent_context(entries[:mid])
    agent2.knowledge = prepare_agent_context(entries[mid:])

    results = []
    start_time = time.time()

    for idx, entry in enumerate(entries):
        question = entry["question"]
        choices = "\n".join([
            f"{label}: {text}" 
            for label, text in zip(
                entry["choices"]["label"], 
                entry["choices"]["text"]
            )
        ])
        expected = entry["answerKey"].strip()

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Options:\n{choices}\n\n"
            "Discuss and determine the single correct answer."
        )

        predicted = debate_and_judge(agent1, agent2, judge, user_prompt)
        results.append({"prediction": predicted, "reference": expected})

        print(f"\nQ{idx+1}: {question[:60]}...")
        print(f"Expected: {expected} | Predicted: {predicted}")

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
        print(f"\nMulti-Agent System - Accuracy: {acc*100:.2f}% | Time: {total_time:.2f}s")
    finally:
        del agent_a.llm
        del agent_b.llm
        del judge.llm
