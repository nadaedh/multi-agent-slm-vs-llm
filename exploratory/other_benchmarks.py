"""Early exploration of other benchmarks.

Before settling on AI2-ARC and MMLU, a single model was tried on these
benchmarks with direct answer-only prompts. Most of them expect free-form
answers (numbers, expressions, patches) which exact-match scoring handles
poorly. That is why the final experiments use multiple-choice benchmarks.

Usage example:
    python -m exploratory.other_benchmarks gsm8k --n 10
"""

import argparse
import time

import evaluate
from datasets import load_dataset
from llama_cpp import Llama

from slm_bench.config import CTX_BASE, MODELS, SEED

MATH_PROMPT = "Solve the following math problem strictly by outputting only the final answer.\n\nProblem: {problem}\n\nFinal Answer: "

# name -> how to load it, build the prompt, read the reference answer, and generate.
BENCHMARKS = {
    "gsm8k": dict(
        load=lambda: load_dataset("openai/gsm8k", "main")["test"],
        prompt=lambda e: (
            "### Instruction:\nSolve the following math problem and provide only the final answer number.\n\n"
            f"### Problem: {e['question']}\n\n### Final Answer: "
        ),
        reference=lambda e: e["answer"].split("####")[-1].strip(),
        max_tokens=50, stop=["\n", ".", "?"],
    ),
    "aime2024": dict(
        load=lambda: load_dataset("Maxwell-Jia/AIME_2024")["train"],
        prompt=lambda e: MATH_PROMPT.format(problem=e["Problem"]),
        reference=lambda e: str(e["Answer"]).strip(),
        max_tokens=20, stop=["\n", ".", " "],
    ),
    "math500": dict(
        load=lambda: load_dataset("HuggingFaceH4/MATH-500", "default")["test"],
        prompt=lambda e: MATH_PROMPT.format(problem=e["problem"]),
        reference=lambda e: e["answer"].strip(),
        max_tokens=50, stop=[".", "?"],
    ),
    "gpqa": dict(  # gated dataset: requires accepting its terms on Hugging Face
        load=lambda: load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"],
        prompt=lambda e: (
            "Solve the following problem strictly by outputting only the final answer without a sentence.\n\n"
            f"Problem: {e['Question']}\n\nFinal Answer: "
        ),
        reference=lambda e: str(e["Correct Answer"]).strip(),
        max_tokens=50, stop=None,
    ),
    "hellaswag": dict(
        load=lambda: load_dataset("Rowan/hellaswag")["validation"],
        prompt=lambda e: (
            "Complete the following sentence with the most logical ending. Output only the correct choice number.\n\n"
            f"Context: {e['ctx']}\nChoices:\n"
            + "\n".join(f"{i}: {ending}" for i, ending in enumerate(e["endings"]))
            + "\n\nAnswer: "
        ),
        reference=lambda e: e["label"].strip(),
        max_tokens=5, stop=[".", "?"],
    ),
    "swe-bench": dict(
        load=lambda: load_dataset("princeton-nlp/SWE-bench_Verified")["test"],
        prompt=lambda e: (
            "You are an AI software engineer. Given a GitHub issue with a bug description, "
            "output only the correct Git patch with the fix.\n\n"
            f"Issue Description:\n{e['problem_statement'][:1000]}\n\nExpected Fix (Git patch format):\n"
        ),
        reference=lambda e: str(e["patch"]).strip(),
        max_tokens=100, stop=["\n", "---", "diff"],
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--model", choices=MODELS, default="mistral")
    parser.add_argument("--n", type=int, default=10, help="number of examples")
    args = parser.parse_args()

    bench = BENCHMARKS[args.benchmark]
    dataset = bench["load"]()
    llm = Llama(model_path=MODELS[args.model], n_ctx=CTX_BASE, seed=SEED, verbose=False)

    predictions, references = [], []
    start = time.time()
    for i in range(args.n):
        entry = dataset[i]
        response = llm.create_completion(prompt=bench["prompt"](entry), max_tokens=bench["max_tokens"], stop=bench["stop"])
        predictions.append(response["choices"][0]["text"].strip())
        references.append(bench["reference"](entry))
        print(f"\n--- Example {i + 1}\nExpected: {references[-1]}\nModel output: {predictions[-1]}")

    elapsed = time.time() - start
    accuracy = evaluate.load("exact_match").compute(predictions=predictions, references=references)["exact_match"]
    print(f"\n{args.benchmark} - Accuracy: {accuracy * 100:.2f}% | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
