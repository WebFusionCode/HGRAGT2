from litellm import acompletion
from litellm.exceptions import AuthenticationError, RateLimitError, APIConnectionError
from typing import List, Dict, Any

# A highly constrained system prompt for clinical safety and multi-hop GraphRAG.
# Enforces inline citations, epistemic refusals, contradiction surfacing, and multi-hop reasoning.
SYSTEM_PROMPT = """You are a highly precise, zero-trust clinical AI assistant.
Your task is to answer the user's query STRICTLY based on the provided retrieved context.

CRITICAL SAFETY INSTRUCTIONS:
1. MULTI-HOP REASONING: You must synthesize and connect relationships across multiple documents if required (e.g., Document A mentions a treatment, Document B mentions a risk of that treatment).
2. INLINE CITATIONS: You MUST provide inline citations pointing to the exact chunk ID used for every claim. Format your citations like this: [chunk_id].
3. EPISTEMIC REFUSAL: If the provided context does NOT contain sufficient information to fully and safely answer the query, you MUST explicitly state: "Insufficient evidence retrieved." Do not hallucinate or rely on outside knowledge.
4. CONFLICT DETECTION: If the provided context documents contradict each other (e.g., opposing treatments or guidelines), you MUST explicitly surface this contradiction to the user. State clearly that there is conflicting evidence and describe the conflict safely without making a definitive clinical decision yourself.

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
        # Upgraded to use heavy frontier models for the generation step as requested
        response = await acompletion(
            model="groq/llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0, # Strict deterministic generation for clinical safety
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except AuthenticationError as e:
        return f"ERROR_AUTH: Invalid API Key or Authentication Failure. {str(e)}"
    except RateLimitError as e:
        return f"ERROR_RATELIMIT: You have exceeded your rate limit. Please try again later. {str(e)}"
    except APIConnectionError as e:
        return f"ERROR_NETWORK: Unable to connect to the LLM provider. {str(e)}"
    except Exception as e:
        print(f"Generation error: {e}")
        return f"ERROR_SYSTEM: Unable to generate response due to internal system error. {str(e)}"
