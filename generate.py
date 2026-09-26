from litellm import acompletion
from typing import List, Dict, Any

# A highly constrained system prompt for clinical safety.
# Enforces inline citations, epistemic refusals, and contradiction surfacing.
SYSTEM_PROMPT = """You are a highly precise, zero-trust clinical AI assistant.
Your task is to answer the user's query STRICTLY based on the provided retrieved context.

CRITICAL SAFETY INSTRUCTIONS:
1. INLINE CITATIONS: You MUST provide inline citations pointing to the exact chunk ID used for every claim. Format your citations like this: [chunk_id].
2. EPISTEMIC REFUSAL: If the provided context does NOT contain sufficient information to fully and safely answer the query, you MUST explicitly state: "Insufficient evidence retrieved." Do not hallucinate or rely on outside knowledge.
3. CONFLICT DETECTION: If the provided context documents contradict each other (e.g., opposing treatments or guidelines), you MUST explicitly surface this contradiction to the user. State clearly that there is conflicting evidence and describe the conflict safely without making a definitive clinical decision yourself.

Retrieved Context Documents:
{context}

Respond safely, concisely, and strictly follow the instructions above.
"""

async def generate_clinical_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Calls the LLM via LiteLLM with a strict prompt structure to enforce zero-trust clinical RAG rules.
    """
    if not retrieved_docs:
        return "Insufficient evidence retrieved."
        
    # Format the retrieved documents into the context string
    context_str = ""
    for doc in retrieved_docs:
        doc_id = doc.get('metadata', {}).get('doc_id', doc.get('id', 'unknown'))
        context_str += f"\n--- Document [{doc_id}] ---\n{doc['content']}\n"
        
    prompt = SYSTEM_PROMPT.format(context=context_str)
    
    try:
        # We can easily swap 'gpt-4o' with 'claude-3-5-sonnet-20240620' or local models via LiteLLM
        response = await acompletion(
            model="groq/qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0, # Strict deterministic generation for clinical safety
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation error: {e}")
        return "Error: Unable to generate response due to internal system error."
