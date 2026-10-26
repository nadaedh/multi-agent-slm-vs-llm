# Extracting the answer letter (A-D) from free-form model output

import re

INVALID = "INVALID"

# Ordered from most to least specific: explicit "final answer" phrases first,
# a bare capital letter last.
_ANSWER_PATTERNS = [
    re.compile(r"final answer\s*\(ONLY a single letter A, B, C, or D\):\s*\**([A-D])", re.IGNORECASE),
    re.compile(r"final answer\s*[:\-]?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"correct answer is\s*:?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"my answer is\s*:?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"(?:choice|option)\s*:?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\b([A-D])\b"),
]


def extract_letter(text):
    """Find the concluding answer letter of a reasoning trace.

    The last lines are searched first since that is where the conclusion
    usually is. Returns INVALID when no letter is found.
    """
    for line in reversed(text.strip().split("\n")[-5:]):
        for pattern in _ANSWER_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1).upper()
    return INVALID


def first_letter(text):
    """Return the first standalone letter A-D in a short answer, or INVALID."""
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else INVALID
