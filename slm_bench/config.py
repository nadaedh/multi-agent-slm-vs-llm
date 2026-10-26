# Global settings shared by every experiment

import os
import random

import numpy as np

SEED = 42

# Quantized GGUF models (Q4_K_M). Override the paths with environment variables
MODELS = {
    "mistral": os.environ.get("MISTRAL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"),
    "mixtral": os.environ.get("MIXTRAL_PATH", "models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"),
}

# Base context window, and the extended one used whenever an agent is given
# Q&A examples as prior knowledge.
CTX_BASE = 2048
CTX_WITH_KNOWLEDGE = 4096

# Every benchmark run answers a fixed set of questions.
N_QUESTIONS = 50

# Logit bias used in the original runs to nudge generations towards a letter
# answer (token ids 65-68, i.e. the ASCII codes of "A"-"D").
LETTER_TOKEN_IDS = (65, 66, 67, 68)
LETTER_BIAS = {token: 5 for token in LETTER_TOKEN_IDS}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
