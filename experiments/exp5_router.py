"""Experiment 5: routing questions to domain-specialized agents.

Each specialist SLM is given the Q&A pairs of one subject domain. A router
agent classifies every incoming question and forwards it to the matching
specialist. Questions from all domains are shuffled into one mixed batch.

  --agents 2     science (AI2-ARC) + anatomy specialists (Table 11)
  --agents 4     + marketing and geography specialists (Table 12)
  --oracle       skip the router and always use the correct specialist,
                 i.e. the upper bound with perfect routing
  --generalist   baseline: one model that is given the Q&A pairs of all domains

Usage examples:
    python -m experiments.exp5_router --agents 4
    python -m experiments.exp5_router --agents 4 --generalist --model mistral
"""

import argparse
import random

from slm_bench.agents import Agent, AnswerJudge
from slm_bench.config import CTX_WITH_KNOWLEDGE, LETTER_BIAS, MODELS, set_seed
from slm_bench.data import knowledge_block, load_questions
from slm_bench.evaluation import run_benchmark
from slm_bench.parsing import extract_letter

SETUPS = {
    2: {"domains": ["science", "anatomy"], "n_per_domain": 25},
    4: {"domains": ["science", "anatomy", "marketing", "geography"], "n_per_domain": 20},
}

ROUTER_PROMPTS = {
    2: (
        "Analyze this question and classify it into exactly one category:\n"
        "- 'science' for physics, chemistry, earth science, general science\n"
        "- 'anatomy' for human body, organs, biological structures\n\n"
        "Examples:\n"
        "Q: What planet is closest to the sun?\nA: science\n"
        "Q: What is the function of the liver?\nA: anatomy\n"
        "Q: Which gas do plants need for photosynthesis?\nA: science\n"
        "Q: Which bone protects the brain?\nA: anatomy\n"
        "Q: What causes day and night on Earth?\nA: science\n"
        "Q: How many chambers does the heart have?\nA: anatomy\n\n"
        "Now classify this question:\n"
        "Q: {question}\n"
        "A: [respond with ONLY one word: science or anatomy]"
    ),
    # In the original 4-agent run, the AI2-ARC specialist was offered to the
    # router under the label 'math'.
    4: (
        "Classify the following question into one category:\n"
        "- 'math' for equations, calculations, numbers\n"
        "- 'anatomy' for the human body, biology, organs\n"
        "- 'marketing' for business, advertising, branding\n"
        "- 'geography' for countries, maps, locations, earth\n\n"
        "Q: Solve 5x - 3 = 17\nA: math\n"
        "Q: What does the liver do?\nA: anatomy\n"
        "Q: Define customer retention.\nA: marketing\n"
        "Q: Where is the Sahara Desert located?\nA: geography\n"
        "\nClassify this question:\nQ: {question}\nA:"
    ),
}

# Router output label -> specialist domain.
ROUTER_LABELS = {
    2: {"science": "science", "anatomy": "anatomy"},
    4: {"math": "science", "anatomy": "anatomy", "marketing": "marketing", "geography": "geography"},
}

# Used when the router's output contains no known label.
FALLBACK_KEYWORDS = {
    "science": ["planet", "gravity", "mass", "chemical", "energy", "scientist", "experiment"],
    "anatomy": ["organ", "bone", "muscle", "body", "biological"],
}


class Router:
    def __init__(self, model_path, n_agents):
        self.agent = Agent("Router", model_path, n_ctx=CTX_WITH_KNOWLEDGE)
        self.prompt = ROUTER_PROMPTS[n_agents]
        self.labels = ROUTER_LABELS[n_agents]
        self.domains = SETUPS[n_agents]["domains"]

    def route(self, question):
        raw = self.agent.complete(self.prompt.format(question=question), max_tokens=20, temperature=0.1).lower()
        print(f"[Router] {raw!r}")
        # The model sometimes keeps generating after its answer, so take the label it named first.
        found = [(raw.find(label), domain) for label, domain in self.labels.items() if label in raw]
        if found:
            return min(found)[1]
        text = question.lower()
        for domain, keywords in FALLBACK_KEYWORDS.items():
            if domain in self.domains and any(word in text for word in keywords):
                return domain
        return self.domains[0]


class Specialist:
    def __init__(self, domain, model_path, examples):
        header = f"You have been trained with these specific Q&A pairs from the {domain} domain:"
        knowledge = knowledge_block(examples, header=header, truncate=None)
        self.agent = Agent(f"{domain.title()}Agent", model_path, knowledge, n_ctx=CTX_WITH_KNOWLEDGE)

    def answer(self, question):
        prompt = (
            "You are a specialist answering multiple-choice questions. "
            "First ANALYZE these similar questions and their correct answers:\n"
            f"{self.agent.knowledge}\n"
            f"NEW QUESTION:\n{question.text}\n"
            f"OPTIONS:\n{question.choices}\n\n"
            "REASONING STEPS:\n"
            "1. Find the most similar training question to this new question\n"
            "2. Compare how their options differ\n"
            "3. Explain why the correct answer applies here\n"
            "4. At the end, output this exact format:\n"
            "**Final answer (ONLY a single letter A, B, C, or D): <your answer>**\n\n"
            "Answer:"
        )
        raw = self.agent.complete(prompt, temperature=0.3, max_tokens=300, stop=["\n\n"], logit_bias=LETTER_BIAS)
        print(f"[{self.agent.name}] {raw}")
        return extract_letter(raw)


def routed_system(model_path, n_agents, examples, oracle):
    specialists = {domain: Specialist(domain, model_path, qs) for domain, qs in examples.items()}
    router = None if oracle else Router(model_path, n_agents)

    def answer(q):
        domain = q.domain if oracle else router.route(q.text)
        print(f"True domain: {q.domain} | Routed to: {domain}")
        return specialists[domain].answer(q)

    return answer


def generalist_system(model_path, examples):
    knowledge = "".join(
        knowledge_block(qs, header=f"\n--- Domain: {domain.upper()} ---") for domain, qs in examples.items()
    )
    agent = Agent("Generalist", model_path, knowledge, n_ctx=CTX_WITH_KNOWLEDGE, n_threads=4)
    judge = AnswerJudge(model_path)

    def answer(q):
        prompt = (
            "You are a specialist answering multiple-choice questions across domains.\n"
            "Use the provided Q&A examples to guide your reasoning.\n"
            f"{agent.knowledge}\n"
            f"Now, answer the following question:\n{q.prompt()}\n\n"
            "Follow these steps:\n"
            "1. Analyze the question and options.\n"
            "2. Compare with any relevant training Q&A.\n"
            "3. Reason your way to the answer.\n"
            "4. At the end, write:\n"
            "   Final answer (ONLY a single letter A, B, C, or D): <your letter>\n\n"
            "Answer:"
        )
        raw = agent.complete(prompt, temperature=0.3, max_tokens=500, logit_bias=LETTER_BIAS)
        print(f"[Generalist] {raw}")
        return judge.extract(raw)

    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, choices=SETUPS, default=4, help="number of specialist agents")
    parser.add_argument("--oracle", action="store_true", help="route every question to the correct specialist")
    parser.add_argument("--generalist", action="store_true", help="run the single generalist baseline instead")
    parser.add_argument("--model", choices=MODELS, default="mistral", help="model used by the generalist baseline")
    parser.add_argument("--n-per-domain", type=int, help="questions per domain (default: 25 for 2 agents, 20 for 4)")
    args = parser.parse_args()

    set_seed()
    setup = SETUPS[args.agents]
    n = args.n_per_domain or setup["n_per_domain"]
    # each agent's prior knowledge is the Q&A pairs of the questions it is tested on.
    examples = {domain: load_questions(domain, n) for domain in setup["domains"]}
    questions = [q for qs in examples.values() for q in qs]
    random.shuffle(questions)

    if args.generalist:
        answer = generalist_system(MODELS[args.model], examples)
        title = f"Exp5 generalist {args.model} / {args.agents} domains"
    else:
        answer = routed_system(MODELS["mistral"], args.agents, examples, args.oracle)
        title = f"Exp5 {'oracle-routed' if args.oracle else 'routed'} specialists / {args.agents} agents"
    run_benchmark(title, questions, answer)


if __name__ == "__main__":
    main()
