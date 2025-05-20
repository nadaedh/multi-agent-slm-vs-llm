import os
import re
import time
from llama_cpp import Llama
from datasets import load_dataset, concatenate_datasets
import evaluate

MODEL_PATH = "/Users/nadaeddahir/mistral-gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MAX_EXAMPLES = 10
MAX_CONTEXT_EXAMPLES = 3
metric = evaluate.load("exact_match")

def extract_answer(text):
    patterns = [
        r"final answer:\s*([A-D])",
        r"correct answer is:\s*([A-D])",
        r"best option is:\s*([A-D])",
        r"conclusion:\s*([A-D])",
        r"\b([A-D])\b(?![a-z])"
    ]
    text = text.upper()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

class SpecialistAgent:
    def __init__(self, name, domain, model_path):
        self.name = name
        self.domain = domain
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=1
        )
        self.knowledge = ""
        self.domain_keywords = self._get_domain_keywords(domain)
        
    def _get_domain_keywords(self, domain):
        """Get relevant keywords for domain detection"""
        if "physic" in domain.lower():
            return ["physics", "energy", "force", "matter", "atom", "quantum", "gravity"]
        elif "bio" in domain.lower():
            return ["biology", "life", "cell", "organism", "DNA", "evolution", "ecosystem"]
        return []
        
    def set_knowledge(self, knowledge):
        self.knowledge = knowledge
        
    def domain_relevance(self, question):
        """Determine how relevant this question is to the agent's domain"""
        prompt = f"""As a {self.domain} specialist, assess the relevance of this question to your domain:
Question: {question}

Domain Keywords: {', '.join(self.domain_keywords)}

Provide a single score from 0-100 where:
- 0-30: Not relevant
- 31-70: Somewhat relevant 
- 71-100: Highly relevant

Relevance score:"""
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["\n"]
        )
        try:
            return int(response['choices'][0]['text'].strip())
        except:
            return 0
        
    def analyze_question(self, question):
        prompt = f"""As a {self.domain} specialist with this knowledge:
{self.knowledge}

Analyze this question thoroughly:
{question}

Provide structured analysis:
1. Domain relevance assessment
2. Key concepts involved
3. Potential solution approach
4. Suggested answer (A/B/C/D) with detailed reasoning

Analysis:"""
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
            stop=["\n\n"]
        )
        return response['choices'][0]['text'].strip()
        
    def respond_to_argument(self, opponent_argument, question):
        prompt = f"""As a {self.domain} specialist, you're in a debate about:
{question}

Your opponent argues:
{opponent_argument}

Your knowledge:
{self.knowledge}

Provide a structured counter-argument:
1. Points of agreement (if any)
2. Points of disagreement with reasoning
3. Supporting evidence from your domain
4. Refined answer suggestion (A/B/C/D)

Counter-argument:"""
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=300,
            temperature=0.3,
            stop=["\n\n"]
        )
        return response['choices'][0]['text'].strip()

class DebateModerator:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4
        )
    
    def conduct_debate(self, question, agent1, agent2, max_rounds=2):
        """Facilitate a structured debate between two specialists"""
        analysis1 = agent1.analyze_question(question)
        analysis2 = agent2.analyze_question(question)
        
        debate_history = [
            f"Round 0 - {agent1.name} initial analysis:\n{analysis1}",
            f"Round 0 - {agent2.name} initial analysis:\n{analysis2}"
        ]
        
        # Debate
        for round_num in range(1, max_rounds+1):
            # Agent1 responds to Agent2's last message
            response1 = agent1.respond_to_argument(analysis2, question)
            debate_history.append(f"Round {round_num} - {agent1.name} response:\n{response1}")
            
            # Agent2 responds to Agent1's last message
            response2 = agent2.respond_to_argument(response1, question)
            debate_history.append(f"Round {round_num} - {agent2.name} response:\n{response2}")
            
            analysis1, analysis2 = response1, response2
        
        relevance1 = agent1.domain_relevance(question)
        relevance2 = agent2.domain_relevance(question)
        
        if relevance1 > relevance2 + 20:  
            final_answer = extract_answer(analysis1)
        elif relevance2 > relevance1 + 20:
            final_answer = extract_answer(analysis2)
        else: 
            final_answer = self.determine_answer(question, analysis1, analysis2)
            
        return final_answer or "INVALID", "\n\n".join(debate_history)
    
    def determine_answer(self, question, analysis1, analysis2):
        prompt = f"""After a detailed debate, determine the most correct answer for:
{question}

Debate Summary:
- Expert 1 Analysis: {analysis1[:500]}...
- Expert 2 Analysis: {analysis2[:500]}...

As the moderator, evaluate:
1. Logical consistency of each argument
2. Strength of evidence provided
3. Domain expertise relevance

Final reasoned answer (A/B/C/D):"""
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            stop=["\n"]
        )
        return extract_answer(response['choices'][0]['text'])

def prepare_knowledge(entries):
    """Create more focused knowledge snippets"""
    return "\n".join(
        f"Example {i+1}: Q: {e['question'][:80]}... | Correct Answer: {e['answerKey']} | Key Concepts: {', '.join(e['choices']['text'][:3])}"
        for i, e in enumerate(entries.select(range(MAX_CONTEXT_EXAMPLES))))

def benchmark_specialized_agents(physicist, biologist, moderator):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    dataset = dataset.select(range(MAX_EXAMPLES))
    
    shared_examples = dataset.select(range(1)) 
    physicist_entries = concatenate_datasets([dataset.select(range(MAX_CONTEXT_EXAMPLES)), shared_examples])
    biologist_entries = concatenate_datasets([dataset.select(range(1, MAX_CONTEXT_EXAMPLES+1)), shared_examples])
    
    physicist.set_knowledge(prepare_knowledge(physicist_entries))
    biologist.set_knowledge(prepare_knowledge(biologist_entries))
    
    results = []
    start_time = time.time()
    
    for idx, entry in enumerate(dataset):
        question = entry["question"]
        choices = "\n".join([
            f"{label}: {text}" 
            for label, text in zip(
                entry["choices"]["label"], 
                entry["choices"]["text"]
            )
        ])
        expected = entry["answerKey"]
        
        full_question = f"Question:\n{question}\nOptions:\n{choices}"
        
        question_start = time.time()
        predicted, debate_log = moderator.conduct_debate(
            full_question, physicist, biologist
        )
        question_time = time.time() - question_start
        
        results.append({
            "question": question[:50] + "...",
            "expected": expected,
            "predicted": predicted,
            "time": question_time,
            "debate": debate_log
        })
        
        print(f"Q{idx+1}: {predicted} (Expected: {expected}) | Time: {question_time:.2f}s")
    
    total_time = time.time() - start_time
    return results, total_time

if __name__ == "__main__":
    try:
        physicist = SpecialistAgent("Dr. Quantum", "Physics and Physical Sciences", MODEL_PATH)
        biologist = SpecialistAgent("Dr. Bio", "Biology and Life Sciences", MODEL_PATH)
        moderator = DebateModerator(MODEL_PATH)
        
        results, total_time = benchmark_specialized_agents(physicist, biologist, moderator)
        
        accuracy = metric.compute(
            predictions=[r["predicted"] for r in results],
            references=[r["expected"] for r in results]
        )["exact_match"]
        
        print(f"\nSpecialized Multi-Agent System - Accuracy: {accuracy*100:.2f}% | Time: {total_time:.2f}s")
        
        with open("debate_logs.txt", "w") as f:
            for i, result in enumerate(results):
                f.write(f"\n\nQuestion {i+1}: {result['question']}")
                f.write(f"\nExpected: {result['expected']} | Predicted: {result['predicted']}")
                f.write(f"\nDebate Log:\n{result['debate']}")
    
    finally:
        if 'physicist' in locals():
            del physicist.llm
        if 'biologist' in locals():
            del biologist.llm
        if 'moderator' in locals():
            del moderator.llm