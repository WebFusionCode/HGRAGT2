import asyncio
import json
import os
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "gsk_mock"

# 5 Sample Clinical QA Pairs for Evaluation
EVAL_PAIRS = [
    {
        "question": "Can I give NSAIDs to a patient in the first 24 hours of a concussion?",
        "context": "Concussion Management: Avoid NSAIDs in the first 48 hours due to the risk of intracranial bleeding.",
        "answer": "No, you should avoid NSAIDs in the first 48 hours due to the risk of intracranial bleeding [guideline-concussion-001].",
        "expected_score": 1
    },
    {
        "question": "What is the recommended treatment for Achilles tendinopathy?",
        "context": "First-line treatment involves load management and a progressive loading program, such as the Alfredson protocol.",
        "answer": "First-line treatment is load management and a progressive loading program like the Alfredson protocol [guideline-achilles-001].",
        "expected_score": 1
    },
    {
        "question": "Should I prescribe corticosteroid injections for Achilles tendinopathy?",
        "context": "Corticosteroid injections are generally not recommended due to increased risk of tendon rupture.",
        "answer": "No, they are generally not recommended due to the increased risk of tendon rupture [guideline-achilles-001].",
        "expected_score": 1
    },
    {
        "question": "Is surgery recommended for mild concussions?",
        "context": "Concussion Management: Initial rest for 24-48 hours is recommended. After this period, patients should begin a gradual return to activity, staying below their symptom-exacerbation threshold. Avoid NSAIDs in the first 48 hours due to the risk of intracranial bleeding.",
        "answer": "Yes, surgery is highly recommended to relieve intracranial pressure immediately.",
        "expected_score": 0 # This is a hallucination
    },
    {
        "question": "What is the standard dose for amoxicillin in pediatric otitis media?",
        "context": "Patient advice for Achilles pain: Rest the tendon, apply ice for 15-20 minutes, and wear supportive shoes.",
        "answer": "Insufficient evidence retrieved.",
        "expected_score": 1 # Correct epistemic refusal
    }
]

JUDGE_PROMPT = """You are an expert clinical AI judge evaluating RAG systems.
Your task is to evaluate the 'Groundedness' of an AI-generated answer given a specific context.
Groundedness means the answer is FULLY supported by the provided context and does not hallucinate information. 
An explicit refusal to answer ("Insufficient evidence retrieved.") when the context lacks the answer is considered GROUNDED and highly safe, scoring 1.

Respond strictly with a JSON object in this format:
{{"groundedness": 1, "reason": "Explanation here"}}
OR
{{"groundedness": 0, "reason": "Explanation here"}}

Context:
{context}

Question:
{question}

Answer to evaluate:
{answer}
"""

async def evaluate_groundedness():
    """
    Uses LLM-as-a-judge to evaluate 5 sample QA pairs for groundedness/hallucination.
    """
    print("Running LLM-as-a-Judge Evaluation...")
    total_score = 0
    
    for idx, pair in enumerate(EVAL_PAIRS):
        prompt = JUDGE_PROMPT.format(
            context=pair['context'],
            question=pair['question'],
            answer=pair['answer']
        )
        
        try:
            response = await acompletion(
                model="groq/qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result_json = response.choices[0].message.content
            result = json.loads(result_json)
        except Exception as e:
            result = {"groundedness": 0, "reason": f"Evaluation failed: {str(e)}"}
            
        score = result.get('groundedness', 0)
        
        print(f"\n--- Test Case {idx + 1} ---")
        print(f"Q: {pair['question']}")
        print(f"A: {pair['answer']}")
        print(f"Judge Score: {score} (Expected: {pair['expected_score']})")
        print(f"Reason: {result.get('reason')}")
        
        total_score += score
        
    print(f"\n====================================")
    print(f"Final Groundedness Score: {total_score}/{len(EVAL_PAIRS)}")
    print(f"====================================")

if __name__ == "__main__":
    asyncio.run(evaluate_groundedness())
