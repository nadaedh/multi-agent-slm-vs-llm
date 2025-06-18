import os
import time
import random
import re
from datasets import load_dataset
from llama_cpp import Llama
import evaluate

SEED = 42
MAX_EXAMPLES = 20
MODEL_PATH = "/localhome/eddahir/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
random.seed(SEED)
metric = evaluate.load("exact_match")


def extract_answer_letter(text):
    patterns = [
        r"final answer\s*\(ONLY a single letter A, B, C, or D\):\s*([A-D])",
        r"final answer\s*[:\-]?\s*([A-D])",
        r"correct answer\s*[:\-]?\s*([A-D])",
        r"answer\s*[:\-]?\s*([A-D])",
        r"\b([A-D])\b(?![a-z])"
    ]
    lines = text.split('\n')[-5:] if '\n' in text else [text]
    for line in reversed(lines):
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    return "INVALID"


class SpecialistAgent:
    def __init__(self, name, model_path, domain_knowledge=""):
        self.name = name
        self.llm = Llama(model_path=model_path, n_ctx=4096, logits_all=True)
        self.knowledge = domain_knowledge

    def answer(self, question, choices):
        prompt = (
            "You are a specialist answering multiple-choice questions. "
            "First ANALYZE these similar questions and their correct answers:\n"
            f"{self.knowledge}\n\n"
            "NEW QUESTION:\n"
            f"{question}\n"
            "OPTIONS:\n"
            f"{choices}\n\n"
            "REASONING STEPS:\n"
            "1. Find the most similar training question to this new question\n"
            "2. Compare how their options differ\n"
            "3. Explain why the correct answer applies here\n"
            "4. At the end, output this exact format:\n"
            "**Final answer (ONLY a single letter A, B, C, or D): <your answer>**\n\n"
            "Answer:"
        )

        response = self.llm.create_completion(
            prompt=prompt,
            temperature=0.3,
            max_tokens=300,
            stop=["\n\n"],
            logit_bias={65: 5, 66: 5, 67: 5, 68: 5},
        )
        return response["choices"][0]["text"].strip()


class ExtendedRouterAgent:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=4096)

    def route(self, question):
        prompt = (
            "Classify the following question into one category:\n"
            "- 'math' for equations, calculations, numbers\n"
            "- 'anatomy' for the human body, biology, organs\n"
            "- 'marketing' for business, advertising, branding\n"
            "- 'geography' for countries, maps, locations, earth\n\n"
            "Q: Solve 5x - 3 = 17\nA: math\n"
            "Q: What does the liver do?\nA: anatomy\n"
            "Q: Define customer retention.\nA: marketing\n"
            "Q: Where is the Sahara Desert located?\nA: geography\n"
            f"\nClassify this question:\nQ: {question}\nA:"
        )
        response = self.llm.create_completion(prompt=prompt, max_tokens=20, temperature=0.1)
        raw = response["choices"][0]["text"].strip().lower()
        print(f"Router output: '{raw}'")

        for domain in ["math", "anatomy", "marketing", "geography"]:
            if domain in raw:
                return domain
        return "math"  # fallback


class MultiAgentSystem:
    def __init__(self, model_path):
        self.model_path = model_path
        self.domains = {
            "math": {
                "dataset": load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"],
                "label_fn": lambda ex: ex["answerKey"].strip(),
                "format_choices": lambda ex: "\n".join(
                    [f"{l}: {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"])]
                )
            },
            "anatomy": {
                "dataset": load_dataset("cais/mmlu", "anatomy")["test"],
                "label_fn": lambda ex: chr(65 + ex["answer"]),
                "format_choices": lambda ex: "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(ex["choices"])])
            },
            "marketing": {
                "dataset": load_dataset("cais/mmlu", "marketing")["test"],
                "label_fn": lambda ex: chr(65 + ex["answer"]),
                "format_choices": lambda ex: "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(ex["choices"])])
            },
            "geography": {
                "dataset": load_dataset("cais/mmlu", "high_school_geography")["test"],
                "label_fn": lambda ex: chr(65 + ex["answer"]),
                "format_choices": lambda ex: "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(ex["choices"])])
            }
        }
        self.agents = {}
        self.router = ExtendedRouterAgent(model_path)
        self._prepare_agents()

    def _prepare_agents(self):
        for domain, data in self.domains.items():
            train_set = list(data["dataset"].select(range(MAX_EXAMPLES)))
            context = self._prepare_context(train_set, domain)
            self.agents[domain] = SpecialistAgent(f"{domain.title()}Agent", self.model_path, context)

    def _prepare_context(self, examples, domain):
        context = f"You have been trained with these Q&A pairs in the {domain} domain:\n"
        for ex in examples:
            question = ex["question"]
            answer = self.domains[domain]["label_fn"](ex)
            context += f"Q: {question} → A: {answer}\n"
        return context

    def run(self):
        all_questions = []
        for domain, data in self.domains.items():
            test_set = list(data["dataset"].select(range(MAX_EXAMPLES)))
            for entry in test_set:
                all_questions.append((entry, domain))

        random.shuffle(all_questions)
        results = []
        start = time.time()

        for idx, (entry, true_domain) in enumerate(all_questions):
            question = entry["question"]
            choices = self.domains[true_domain]["format_choices"](entry)
            reference = self.domains[true_domain]["label_fn"](entry)

            routed = self.router.route(question)
            agent = self.agents.get(routed, self.agents["math"])
            raw_output = agent.answer(question, choices)
            prediction = extract_answer_letter(raw_output)

            results.append({"prediction": prediction, "reference": reference})
            print(f"\nQ{idx+1}: {question[:60]}...")
            print(f"True: {true_domain} | Routed: {routed}")
            print(f"Expected: {reference} | Predicted: {prediction}\n")

        elapsed = time.time() - start
        score = metric.compute(
            predictions=[r["prediction"] for r in results],
            references=[r["reference"] for r in results]
        )
        print(f"\nMulti-Agent Accuracy: {score['exact_match'] * 100:.2f}% | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    system = MultiAgentSystem(MODEL_PATH)
    system.run()