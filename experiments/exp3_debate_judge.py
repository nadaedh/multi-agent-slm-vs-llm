"""Experiment 3: N-round debate between two SLM agents.

Protocols:
  judge      (default) The agents debate for 5 rounds then a third agent, the Judge, reads the whole
             conversation and extracts the final letter.
  consensus  The agents debate until they agree on a letter (at most 5 rounds).
             Without agreement a confidence-weighted vote breaks the tie.

Usage example: python -m experiments.exp3_debate_judge --benchmark arc --knowledge split
"""

import argparse

from llama_cpp import LlamaGrammar

from slm_bench.agents import Agent, weighted_vote
from slm_bench.config import CTX_BASE, CTX_WITH_KNOWLEDGE, MODELS, N_QUESTIONS, set_seed
from slm_bench.data import BENCHMARKS, EXPERT_ROLES, KNOWLEDGE_SETTINGS, distribute_knowledge, load_questions
from slm_bench.evaluation import run_benchmark
from slm_bench.parsing import INVALID, extract_letter, first_letter

ROUNDS = 5

# Constrains the judge's output to exactly one of the four letters.
ABCD_GRAMMAR = 'root ::= "A" | "B" | "C" | "D"'

FORMAT_REMINDER = (
    "When answering questions:\n"
    "1. Carefully read the question and all options\n"
    "2. Analyze each option systematically\n"
    "3. Explain your reasoning\n"
    '4. Conclude with EXACTLY ONE of these phrases: "My answer is: A", "My answer is: B", '
    '"My answer is: C", "My answer is: D"'
)


def knowledge_prompt(agent, role):
    if not agent.knowledge:
        return ""
    return f"You are {role}. Use the following known Q&A examples to inform your reasoning:\n{agent.knowledge}\n\n"


class Judge:
    def __init__(self, model_path, use_grammar=False):
        self.agent = Agent("Judge", model_path, n_ctx=CTX_BASE)
        self.grammar = LlamaGrammar.from_string(ABCD_GRAMMAR, verbose=False) if use_grammar else None

    def verdict(self, conversation):
        prompt = (
            "You are a neutral judge. Based on the following conversation between two experts, "
            "your task is to extract the final answer to a multiple-choice question (A, B, C, or D). "
            "Only output a single letter with no explanation.\n\n"
            "Example Conversation:\n"
            "AgentA: I believe the answer is A because...\n"
            "AgentB: I disagree, I think it's C due to...\n"
            "AgentA: Good point, C seems correct.\n\n"
            "Final answer (only A, B, C, or D): C\n\n"
            f"Now judge the following conversation:\n{conversation}\n\n"
            "Final answer (only A, B, C, or D):"
        )
        if self.grammar:
            raw = self.agent.complete(prompt, max_tokens=1, temperature=0.0, grammar=self.grammar)
        else:
            raw = self.agent.complete(prompt, max_tokens=2, temperature=0.0, stop=["\n", "."])
        answer = first_letter(raw)
        print(f"[Judge] {raw} -> {answer}")
        return answer


def debate_with_judge(agents, judge, question, role):
    agent_a, agent_b = agents
    user_prompt = (
        f"You are {role}. Your task is to answer the multiple-choice question below. "
        f"You must discuss your reasoning with a fellow expert.\n\n{question.prompt()}"
    )
    for agent in agents:
        agent.reset()
        agent.receive("User", user_prompt)

    for turn in range(1, ROUNDS + 1):
        print(f"Turn {turn}:")
        for speaker, listener in ((agent_a, agent_b), (agent_b, agent_a)):
            message = speaker.complete(
                knowledge_prompt(speaker, role) + speaker.conversation(),
                temperature=0.7, max_tokens=50, stop=["\n", "."],
            )
            print(f"  [{speaker.name}] {message}")
            speaker.receive(speaker.name, message)
            listener.receive(speaker.name, message)

    # Both agents saw the same conversation, so one transcript is enough.
    return judge.verdict("\n".join(agent_a.memory))


def debate_until_consensus(agents, question, role):
    agent_a, agent_b = agents
    user_prompt = (
        f"You are {role}. Your task is to answer the multiple-choice question below. "
        "You must choose exactly one of the options (A, B, C, or D) and explain your reasoning "
        "clearly in a single sentence.\n\n"
        "Respond in this exact format:\n"
        "My answer is: <LETTER>. Here's why: <your reasoning>\n\n"
        "Example:\n"
        "Question: What is the boiling point of water?\n"
        "Choices:\nA: 50°C\nB: 100°C\nC: 150°C\nD: 200°C\n"
        "My answer is: B. Here's why: Water boils at 100 degrees Celsius at standard atmospheric pressure.\n\n"
        f"{question.prompt()}\n\nAnswer:"
    )
    for agent in agents:
        agent.reset()
        agent.receive("User", user_prompt)

    for turn in range(1, ROUNDS + 1):
        print(f"Turn {turn}:")
        votes = []
        for speaker, listener in ((agent_a, agent_b), (agent_b, agent_a)):
            message, confidence = speaker.complete_with_confidence(
                knowledge_prompt(speaker, role) + speaker.conversation(), temperature=0.0, max_tokens=100
            )
            letter = extract_letter(message)
            print(f"  [{speaker.name}] {message}\n    -> {letter} (confidence {confidence:.4f})")
            speaker.receive(speaker.name, message)
            listener.receive(speaker.name, message)
            if letter == INVALID:
                speaker.receive("System", FORMAT_REMINDER)
            votes.append((letter, confidence))

        if votes[0][0] == votes[1][0] != INVALID:
            print(f"Consensus reached on {votes[0][0]}")
            return votes[0][0]

    print("No consensus, falling back to the confidence-weighted vote")
    return weighted_vote(votes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="arc")
    parser.add_argument("--knowledge", choices=KNOWLEDGE_SETTINGS, default="none",
                        help="; ".join(f"{k} = {v}" for k, v in KNOWLEDGE_SETTINGS.items()))
    parser.add_argument("--protocol", choices=["judge", "consensus"], default="judge")
    parser.add_argument("--grammar-judge", action="store_true",
                        help="constrain the judge's output to A/B/C/D with a grammar")
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="number of questions")
    args = parser.parse_args()

    set_seed()
    domain = BENCHMARKS[args.benchmark]
    role = EXPERT_ROLES[domain]
    questions = load_questions(domain, args.n)
    knowledge_a, knowledge_b = distribute_knowledge(
        questions, args.knowledge, header="You have been trained with the following Q&A examples:"
    )

    n_ctx = CTX_BASE if args.knowledge == "none" else CTX_WITH_KNOWLEDGE
    logits_all = args.protocol == "consensus"  # needed for confidence scores
    agents = [
        Agent("AgentA", MODELS["mistral"], knowledge_a, n_ctx=n_ctx, logits_all=logits_all),
        Agent("AgentB", MODELS["mistral"], knowledge_b, n_ctx=n_ctx, logits_all=logits_all),
    ]

    if args.protocol == "judge":
        judge = Judge(MODELS["mistral"], use_grammar=args.grammar_judge)
        answer = lambda q: debate_with_judge(agents, judge, q, role)
    else:
        answer = lambda q: debate_until_consensus(agents, q, role)

    title = f"Exp3 debate ({args.protocol}) / {args.benchmark} / knowledge={args.knowledge}"
    run_benchmark(title, questions, answer)


if __name__ == "__main__":
    main()
