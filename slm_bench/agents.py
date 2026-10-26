# Building blocks shared by the experiments: a model-backed agent and judges.

import math
import random

from llama_cpp import Llama

from .config import CTX_BASE, LETTER_TOKEN_IDS, SEED
from .parsing import INVALID, extract_letter


class Agent:
    """A single SLM agent. Each agent loads its own instance of the model.

    "knowledge" holds the Q&A examples the agent was "trained" with (shown in its
    prompt) and "memory" holds the conversation it has seen so far.
    """

    def __init__(self, name, model_path, knowledge="", n_ctx=CTX_BASE, n_threads=None, logits_all=False):
        self.name = name
        self.knowledge = knowledge
        self.memory = []
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            logits_all=logits_all,
            seed=SEED,
            verbose=False,
        )

    def reset(self):
        self.memory.clear()

    def receive(self, sender, message):
        self.memory.append(f"{sender}: {message}")

    def conversation(self):
        """The conversation so far, ending with this agent's turn to speak."""
        return "\n".join(self.memory) + f"\n{self.name}:"

    def complete(self, prompt, **params):
        response = self.llm.create_completion(prompt=prompt, **params)
        return response["choices"][0]["text"].strip()

    def complete_with_confidence(self, prompt, **params):
        """Generate text and return it with the probability of its first token.

        That probability is the model's confidence used for weighted voting.
        Requires the agent to be created with `logits_all=True`.
        """
        response = self.llm.create_completion(prompt=prompt, logprobs=1, **params)
        choice = response["choices"][0]
        logprobs = choice["logprobs"]["token_logprobs"] if choice["logprobs"] else None
        confidence = math.exp(logprobs[0]) if logprobs and logprobs[0] is not None else 0.0
        return choice["text"].strip(), confidence


def weighted_vote(answers):
    """Pick the answer with the highest summed confidence from (letter, confidence) pairs."""
    scores = {}
    for letter, confidence in answers:
        scores[letter] = scores.get(letter, 0.0) + confidence
    return max(scores.items(), key=lambda item: item[1])[0]


class AnswerJudge:
    """Extracts the final letter from a single agent's reasoning.

    Tries pattern matching first, then asks a small model to read the answer
    off the text, and finally falls back to a random letter.
    """

    def __init__(self, model_path):
        self.agent = Agent("Judge", model_path, n_ctx=CTX_BASE, n_threads=2)

    def extract(self, agent_output):
        answer = extract_letter(agent_output)
        if answer != INVALID:
            return answer

        prompt = (
            "Extract ONLY the final answer letter (A, B, C, or D) from this response.\n\n"
            f"Response:\n{agent_output}\n\n"
            "Extracted answer (ONLY the letter A, B, C, or D):"
        )
        raw = self.agent.complete(
            prompt,
            max_tokens=1,
            temperature=0.0,
            stop=["\n", ".", ",", " "],
            logit_bias={token: 100 for token in LETTER_TOKEN_IDS},
        )
        answer = extract_letter(raw)
        if answer == INVALID:
            answer = random.choice("ABCD")
            print(f"[Judge] Could not determine the answer, choosing randomly: {answer}")
        return answer
