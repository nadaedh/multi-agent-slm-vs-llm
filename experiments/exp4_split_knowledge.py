"""Experiment 4: specialized agents with split knowledge.

Each agent is preloaded with Q&A examples from a different half of the
benchmark, then the two agents briefly discuss every question (3 rounds of
short answers). A lightweight judge reads the last turns of the discussion to
read off the agreed letter, falling back to a majority vote over the letters
the agents mentioned.

Usage example: python -m experiments.exp4_split_knowledge --benchmark arc
"""

import argparse
from collections import Counter

from slm_bench.agents import Agent
from slm_bench.config import CTX_WITH_KNOWLEDGE, MODELS, N_QUESTIONS, set_seed
from slm_bench.data import BENCHMARKS, knowledge_block, load_questions
from slm_bench.evaluation import run_benchmark
from slm_bench.parsing import INVALID, first_letter

ROUNDS = 3
# Examples each agent is shown, taken from the start of its half of the dataset.
CONTEXT_EXAMPLES = 5
# The judge only reads the last few messages of the discussion.
JUDGE_WINDOW = 6


def respond(agent):
    prompt = (
        f"Role: You are {agent.name}, an expert with specific knowledge.\n"
        f"Knowledge:\n{agent.knowledge}\n\n"
        "Conversation History:\n" + "\n".join(agent.memory) +
        f"\n{agent.name}: Based on my knowledge and the discussion, the answer is"
    )
    return agent.complete(prompt, temperature=0.3, max_tokens=10, stop=["\n", ".", ","])


def judge_discussion(judge, messages):
    prompt = (
        "Role: Neutral Judge\n\n"
        "Discussion Summary:\n" + "\n".join(messages[-JUDGE_WINDOW:]) + "\n\n"
        "Task: Determine the FINAL answer (A/B/C/D) that both experts agree on.\n"
        "Output ONLY the letter (A, B, C, or D) with no other text.\n"
        "Final Answer:"
    )
    raw = judge.complete(prompt, max_tokens=2, temperature=0.1, stop=["\n", ".", ","])
    answer = first_letter(raw)
    print(f"[Judge] {raw} -> {answer}")

    if answer == INVALID:
        letters = [first_letter(m.split(":", 1)[-1]) for m in messages]
        votes = Counter(letter for letter in letters if letter != INVALID)
        answer = votes.most_common(1)[0][0] if votes else INVALID
    return answer


def discuss(agents, judge, question):
    agent_a, agent_b = agents
    for agent in agents:
        agent.reset()
        agent.receive("System", "Discuss to find the correct answer. Reference your knowledge.")
        agent.receive("User", (
            f"Question:\n{question.text}\n\nOptions:\n{question.choices}\n\n"
            "Discuss and determine the single correct answer."
        ))

    for turn in range(1, ROUNDS + 1):
        print(f"Turn {turn}:")
        for speaker, listener in ((agent_a, agent_b), (agent_b, agent_a)):
            message = respond(speaker)
            print(f"  [{speaker.name}] {message}")
            speaker.receive(speaker.name, message)
            listener.receive(speaker.name, message)

    # Skip the System and User messages, keeping only the agents' turns.
    return judge_discussion(judge, agent_a.memory[2:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="arc")
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="number of questions")
    args = parser.parse_args()

    set_seed()
    questions = load_questions(BENCHMARKS[args.benchmark], args.n)
    mid = len(questions) // 2
    header = "You know these specific Q&A pairs:"
    knowledge_a = knowledge_block(questions[:mid][:CONTEXT_EXAMPLES], header=header)
    knowledge_b = knowledge_block(questions[mid:][:CONTEXT_EXAMPLES], header=header)

    model = MODELS["mistral"]
    agents = [
        Agent("AgentA", model, knowledge_a, n_ctx=CTX_WITH_KNOWLEDGE, n_threads=4),
        Agent("AgentB", model, knowledge_b, n_ctx=CTX_WITH_KNOWLEDGE, n_threads=4),
    ]
    judge = Agent("Judge", model, n_ctx=CTX_WITH_KNOWLEDGE, n_threads=4)

    run_benchmark(f"Exp4 split knowledge / {args.benchmark}", questions, lambda q: discuss(agents, judge, q))


if __name__ == "__main__":
    main()
