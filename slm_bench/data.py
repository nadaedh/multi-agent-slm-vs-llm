# Loading AI2-ARC and MMLU into one common multiple-choice format

from dataclasses import dataclass

from datasets import load_dataset

# Every subject domain used in the experiments as (HF dataset, subset).
DOMAINS = {
    "science": ("allenai/ai2_arc", "ARC-Challenge"),
    "anatomy": ("cais/mmlu", "anatomy"),
    "marketing": ("cais/mmlu", "marketing"),
    "geography": ("cais/mmlu", "high_school_geography"),
}

# Benchmark 1 (AI2-ARC) and Benchmark 2 (MMLU anatomy).
BENCHMARKS = {
    "arc": "science",
    "mmlu": "anatomy",
}

EXPERT_ROLES = {
    "science": "a science expert",
    "anatomy": "a medical expert",
    "marketing": "a marketing expert",
    "geography": "a geography expert",
}


@dataclass
class Question:
    text: str
    labels: list
    options: list
    answer: str
    domain: str

    @property
    def choices(self):
        return "\n".join(f"{label}: {option}" for label, option in zip(self.labels, self.options))

    def prompt(self):
        return f"Question: {self.text}\nChoices:\n{self.choices}"


def _from_arc(entry, domain):
    return Question(
        text=entry["question"],
        labels=list(entry["choices"]["label"]),
        options=list(entry["choices"]["text"]),
        answer=entry["answerKey"].strip(),
        domain=domain,
    )


def _from_mmlu(entry, domain):
    return Question(
        text=entry["question"],
        labels=[chr(65 + i) for i in range(len(entry["choices"]))],
        options=list(entry["choices"]),
        answer=chr(65 + entry["answer"]),
        domain=domain,
    )


def load_questions(domain, n):
    """Return the first `n` test questions of a domain."""
    name, subset = DOMAINS[domain]
    split = load_dataset(name, subset)["test"].select(range(n))
    convert = _from_arc if name == "allenai/ai2_arc" else _from_mmlu
    return [convert(entry, domain) for entry in split]


def knowledge_block(questions, header="You have access to these Q&A pairs:", truncate=150):
    """Format Q&A pairs as prior knowledge given to an agent before inference."""
    lines = [header]
    for q in questions:
        text = q.text[:truncate] if truncate else q.text
        lines.append(f"- Q: {text} → A: {q.answer}")
    return "\n".join(lines) + "\n"


# How the Q&A examples are distributed between Agent A and Agent B
KNOWLEDGE_SETTINGS = {
    "none": "No expert: neither agent sees any Q&A example",
    "a": "A expert: Agent A sees all Q&A examples, Agent B none",
    "split": "Both experts: each agent sees a different half of the Q&A examples",
}


def distribute_knowledge(questions, setting, **block_args):
    """Return the (Agent A, Agent B) knowledge for a KNOWLEDGE_SETTINGS key."""
    if setting == "none":
        return "", ""
    if setting == "a":
        return knowledge_block(questions, **block_args), ""
    mid = len(questions) // 2
    return knowledge_block(questions[:mid], **block_args), knowledge_block(questions[mid:], **block_args)
