"""Experiment 1: single-agent baseline, SLM vs LLM.

Scenario 1: the model answers each question directly with no prior examples.
Scenario 2: the model is first shown the benchmark's Q&A pairs, reasons step by step 
and a judge extracts the final letter from its reasoning.

Usage example: python -m experiments.exp1_single_agent --model mixtral --benchmark mmlu --scenario 2
"""

import argparse

from slm_bench.agents import Agent, AnswerJudge
from slm_bench.config import CTX_BASE, CTX_WITH_KNOWLEDGE, LETTER_BIAS, MODELS, N_QUESTIONS, set_seed
from slm_bench.data import BENCHMARKS, EXPERT_ROLES, knowledge_block, load_questions
from slm_bench.evaluation import run_benchmark
from slm_bench.parsing import first_letter


def run_scenario_1(model_path, questions):
    agent = Agent("Model", model_path, n_ctx=CTX_BASE)

    def answer(q):
        prompt = (
            "Answer the following multiple-choice question. Only provide the letter corresponding "
            "to the correct answer {A, B, C, D} with no explanation or extra text.\n\n"
            f"{q.prompt()}\n\nAnswer: "
        )
        raw = agent.complete(prompt, max_tokens=5, stop=[".", ":"])
        print(f"[Model] {raw}")
        return first_letter(raw)

    return answer


def run_scenario_2(model_path, questions):
    knowledge = knowledge_block(questions, header="You have been trained with these specific Q&A pairs:")
    agent = Agent("Model", model_path, knowledge=knowledge, n_ctx=CTX_WITH_KNOWLEDGE, n_threads=4)
    judge = AnswerJudge(model_path)
    role = EXPERT_ROLES[questions[0].domain]

    def answer(q):
        prompt = (
            f"You are {role} answering multiple-choice questions.\n"
            f"Here is your training data:\n{agent.knowledge}\n"
            "When answering the new question:\n"
            "1. Carefully read the question and all options\n"
            "2. Compare with similar questions from your training data\n"
            "3. Analyze each option systematically\n"
            "4. Explain your reasoning step by step\n"
            "5. At the end, write:\n"
            "   Final answer (ONLY a single letter A, B, C, or D): <your letter>\n\n"
            f"{q.prompt()}\n\nYour analysis:"
        )
        raw = agent.complete(prompt, temperature=0.3, max_tokens=500, stop=["Question:"], logit_bias=LETTER_BIAS)
        print(f"[Model] {raw}")
        return judge.extract(raw)

    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="mistral", help="mistral = SLM, mixtral = LLM")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="arc")
    parser.add_argument("--scenario", type=int, choices=[1, 2], default=1)
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="number of questions")
    args = parser.parse_args()

    set_seed()
    questions = load_questions(BENCHMARKS[args.benchmark], args.n)
    setup = run_scenario_1 if args.scenario == 1 else run_scenario_2
    answer = setup(MODELS[args.model], questions)
    run_benchmark(f"Exp1 {args.model} / {args.benchmark} / scenario {args.scenario}", questions, answer)


if __name__ == "__main__":
    main()
