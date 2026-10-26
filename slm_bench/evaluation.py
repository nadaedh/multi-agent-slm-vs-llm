# Running a benchmark loop and scoring it with exact-match accuracy

import time

import evaluate

_exact_match = evaluate.load("exact_match")


def run_benchmark(title, questions, answer_fn):
    """Ask every question through `answer_fn(question) -> letter` and score the answers.

    Returns (accuracy, elapsed seconds) and prints a one-line summary.
    """
    predictions, references = [], []
    start = time.time()

    for i, question in enumerate(questions, start=1):
        print(f"\n--- Q{i} ({question.domain}): {question.text[:80]}")
        predicted = answer_fn(question)
        predictions.append(predicted)
        references.append(question.answer)
        print(f"Expected: {question.answer} | Predicted: {predicted}")

    elapsed = time.time() - start
    accuracy = _exact_match.compute(predictions=predictions, references=references)["exact_match"]
    print(f"\n{title} - Accuracy: {accuracy * 100:.2f}% | Time: {elapsed:.2f}s ({elapsed / 60:.1f} min)")
    return accuracy, elapsed
