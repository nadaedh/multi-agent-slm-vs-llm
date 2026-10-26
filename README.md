# Multi-Agent Small Language Models vs. LLMs

Can several cooperating **small language models (SLMs)** match or beat a single **large language model (LLM)** on reasoning tasks while staying cheap enough to run on a CPU?

This repository contains the code for my EPFL bachelor semester project. It compares a quantized SLM (Mistral-7B-Instruct, Q4_K_M) with a quantized LLM (Mixtral-8x7B-Instruct, Q4_K_M) across five experiments, from a single-agent baseline to debates between agents and routing questions to domain specialists.

*Supervisor: Rachid Guerraoui. Assistants: Gauthier Voron, Geovanni Rizk.*

## Results at a glance

Benchmarks: **AI2-ARC** (Challenge set) and **MMLU** (anatomy subset), 50 questions each, scored by exact match on the answer letter.

| Experiment | Setup | AI2-ARC | MMLU |
|---|---|---|---|
| 1. Single agent | Mistral-7B, no examples / with examples | 52% / 72% | 46% / 48% |
| | Mixtral-8x7B, no examples / with examples | 66% / 90% | 60% / 74% |
| 2. Confidence voting | no expert / A expert / both experts | 50% / 50% / 52% | 48% / 52% / 70% |
| 3. Debate + judge | no expert / A expert / both experts | 60% / 50% / **80%** | 50% / 40% / 60% |
| 4. Split knowledge | both agents, half of the examples each | 64% | 56% |

| Experiment 5: mixed-domain batch | Accuracy |
|---|---|
| Router + 2 specialists (science, anatomy) | 62.0% |
| Router + 4 specialists (science, anatomy, marketing, geography) | **72.5%** |
| Single generalist given all Q&A pairs | 47.5% |

LLM sampling is not fully deterministic, so a re-run will give numbers close to these but not identical.

## Repository layout

```
slm_bench/                  shared code
  config.py                 model paths, seed, context sizes
  data.py                   AI2-ARC / MMLU loading, Q&A "knowledge" blocks
  agents.py                 Agent (one model instance), weighted vote, answer judge
  parsing.py                extracting the answer letter from model output
  evaluation.py             benchmark loop, exact-match accuracy and timing
experiments/                one script per experiment of the report
  exp1_single_agent.py      Section 5.1
  exp2_confidence_voting.py Section 5.2
  exp3_debate_judge.py      Section 5.3
  exp4_split_knowledge.py   Section 5.4
  exp5_router.py            Section 5.5
exploratory/
  other_benchmarks.py       early tests on GSM8K, AIME, MATH-500, GPQA, HellaSwag, SWE-bench (Section 2.2)
```

## Setup

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

Download the two quantized models into `models/`:

```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
    mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir models
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
    mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf --local-dir models
```

If you keep the models elsewhere, set `MISTRAL_PATH` and `MIXTRAL_PATH` to their paths. The datasets are downloaded from Hugging Face automatically on first use.

The original runs used CPU-only inference through `llama.cpp` on an Intel i7-8700K with 64 GB of RAM. The Mistral-7B file is about 4.4 GB and the Mixtral-8x7B file about 26 GB. Every agent is its own model instance.

## Running the experiments

Run every command from the repository root. Each script prints the model output and the expected and predicted letter for every question, then a final accuracy and runtime line. Add `--n 5` to any experiment for a quick test run on fewer questions (`--n-per-domain` for experiment 5).

### Experiment 1: single-agent baseline 

```bash
python -m experiments.exp1_single_agent --model mistral --benchmark arc --scenario 1
```

- `--model mistral|mixtral` selects the SLM or the LLM.
- `--benchmark arc|mmlu` selects AI2-ARC or MMLU anatomy.
- `--scenario 1` answers directly.
- `--scenario 2` first shows the model the benchmark's Q&A pairs. The model then reasons step by step and a judge extracts the letter.

### Experiment 2: multi-agent without interaction

Two Mistral agents answer independently and a confidence-weighted vote picks the answer. Confidence is the probability of the first generated token.

```bash
python -m experiments.exp2_confidence_voting --benchmark mmlu --knowledge split
```

`--knowledge` sets how the Q&A examples are distributed:

- `none`: no expert.
- `a`: only Agent A sees the examples.
- `split`: both experts, each agent sees a different half.

### Experiment 3: N-round debate + judge 

The two agents debate for 5 rounds, then a third agent (the Judge) reads the conversation and extracts the final letter.

```bash
python -m experiments.exp3_debate_judge --benchmark arc --knowledge split
```

Variants:

- `--protocol consensus` stops as soon as the agents agree and breaks ties with a confidence vote.
- `--grammar-judge` constrains the judge's output to `A|B|C|D`.

### Experiment 4: specialized agents with split knowledge 

```bash
python -m experiments.exp4_split_knowledge --benchmark arc
```

### Experiment 5: routing to specialized agents 

```bash
python -m experiments.exp5_router --agents 2                 # router + science/anatomy specialists
python -m experiments.exp5_router --agents 4                 # router + 4 specialists
python -m experiments.exp5_router --agents 4 --generalist    # single generalist baseline
python -m experiments.exp5_router --agents 4 --oracle        # perfect routing (upper bound)
```

`--oracle` sends every question to the correct specialist. It measures how much accuracy is lost to routing mistakes.

### Exploratory benchmarks

```bash
python -m exploratory.other_benchmarks gsm8k --n 10
```

## A note on "prior knowledge"

In the experiments that use prior knowledge, an agent is "specialized" by placing Q&A pairs from the benchmark in its prompt before it answers. These Q&A pairs come from the same 50 questions the agent is evaluated on. The setup therefore measures how well models use and share knowledge they have been given, not how well they generalize to unseen questions.
