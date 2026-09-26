import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def highlight_relevant_sentences(query: str, text: str) -> str:
    """
    Explainable AI (XAI) feature: Highlights sentences in the text that have the highest 
    semantic overlap with the user's query.
    Uses TF-IDF + Cosine Similarity for lightweight, lightning-fast execution on the CPU.
    """
    # Split text into rough sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= 1:
        return f"<mark class='xai-highlight'>{text}</mark>"
        
    # Compute similarities
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([query] + sentences)
    except ValueError:
        # If vocabulary is empty (e.g. only stop words)
        return text

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Highlight top 30% of sentences or at least 1
    k = max(1, int(len(sentences) * 0.3))
    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
    
    highlighted_text = []
    for i, sentence in enumerate(sentences):
        # Only highlight if the similarity is greater than a small threshold
        if i in top_k_indices and cosine_similarities[i] > 0.05:
            highlighted_text.append(f"<mark class='xai-highlight'>{sentence}</mark>")
        else:
            highlighted_text.append(sentence)
            
    return " ".join(highlighted_text)
