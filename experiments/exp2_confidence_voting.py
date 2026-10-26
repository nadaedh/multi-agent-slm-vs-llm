"""Experiment 2: multi-agent without interaction.

Two SLM agents answer each question independently. Their answers are combined with 
a confidence-weighted vote where an agent's confidence is the probability of the first token it generated.

Usage example: python -m experiments.exp2_confidence_voting --benchmark mmlu --knowledge split
"""

import argparse

from slm_bench.agents import Agent, weighted_vote
from slm_bench.config import CTX_WITH_KNOWLEDGE, LETTER_BIAS, MODELS, N_QUESTIONS, set_seed
from slm_bench.data import BENCHMARKS, EXPERT_ROLES, KNOWLEDGE_SETTINGS, distribute_knowledge, load_questions
from slm_bench.evaluation import run_benchmark
from slm_bench.parsing import extract_letter


def respond(agent, role):
    prompt = ""
    if agent.knowledge:
        prompt += f"You are {role}. Here is some prior knowledge:\n{agent.knowledge}\n\n"
    prompt += agent.conversation()

    raw, confidence = agent.complete_with_confidence(
        prompt, temperature=0.3, max_tokens=500, logit_bias=LETTER_BIAS
    )
    letter = extract_letter(raw)
    print(f"[{agent.name}] {raw}\n  -> {letter} (confidence {confidence:.4f})")
    return letter, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="arc")
    parser.add_argument("--knowledge", choices=KNOWLEDGE_SETTINGS, default="none",
                        help="; ".join(f"{k} = {v}" for k, v in KNOWLEDGE_SETTINGS.items()))
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="number of questions")
    args = parser.parse_args()

    set_seed()
    domain = BENCHMARKS[args.benchmark]
    role = EXPERT_ROLES[domain]
    questions = load_questions(domain, args.n)
    knowledge_a, knowledge_b = distribute_knowledge(questions, args.knowledge)

    agents = [
        Agent("AgentA", MODELS["mistral"], knowledge_a, n_ctx=CTX_WITH_KNOWLEDGE, logits_all=True),
        Agent("AgentB", MODELS["mistral"], knowledge_b, n_ctx=CTX_WITH_KNOWLEDGE, logits_all=True),
    ]

    def answer(q):
        user_prompt = (
            f"You are {role} answering multiple-choice questions.\n"
            f"Answer the following question:\n{q.prompt()}\n\n"
            "Follow these steps:\n"
            "1. Analyze the question and each option.\n"
            "2. Reason through the answer.\n"
            "3. At the end, write:\n"
            "   Final answer (ONLY a single letter A, B, C, or D): <your letter>\n\n"
            "Answer:"
        )
        votes = []
        for agent in agents:
            agent.reset()
            agent.receive("User", user_prompt)
            votes.append(respond(agent, role))
        return weighted_vote(votes)

    run_benchmark(f"Exp2 confidence voting / {args.benchmark} / knowledge={args.knowledge}", questions, answer)


if __name__ == "__main__":
    main()
