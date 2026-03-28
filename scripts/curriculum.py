import re

def compute_complexity_score(text):
    """
    Heuristic rule to compute the complexity of a given text.
    Infants learn from simple, short, and repetitive patterns first,
    then move to complex, diverse, and long patterns.
    
    Factors considered:
    1. Overall length (longer is generally more complex).
    2. Average sentence length (longer sentences = more complex syntax).
    3. Vocabulary richness (unique words / total words).
    """
    if not text:
        return 0.0
        
    # Factor 1: Length
    char_length = len(text)
    
    # Factor 2: Sentence complexity
    sentences = [s.strip() for s in re.split(r'[.?!]+', text) if s.strip()]
    num_sentences = len(sentences) or 1
    words = re.findall(r'\b\w+\b', text.lower())
    num_words = len(words) or 1
    
    avg_words_per_sentence = num_words / num_sentences
    
    # Factor 3: Vocabulary Richness (Type-Token Ratio)
    unique_words = len(set(words))
    ttr = unique_words / num_words if num_words > 0 else 0
    
    # Combined score
    # Normalize roughly:
    # length_score: cap at 2000 chars -> 0-1
    length_score = min(char_length / 2000.0, 1.0)
    # sent_score: cap at 25 words/sentence -> 0-1
    sent_score = min(avg_words_per_sentence / 25.0, 1.0)
    # ttr is already 0-1
    
    complexity = (length_score * 0.4) + (sent_score * 0.4) + (ttr * 0.2)
    return complexity

def sort_dataset_by_curriculum(dataset_list, text_key="instruction"):
    """
    Sorts a list of dataset dictionaries from simplest to most complex
    based on the specified text key (e.g., the prompt/instruction).
    """
    print(f"[CURRICULUM] Sorting {len(dataset_list)} items by complexity...")
    
    # Compute scores
    scored_items = []
    for item in dataset_list:
        text_to_score = item.get(text_key, "")
        score = compute_complexity_score(text_to_score)
        scored_items.append((score, item))
        
    # Sort ascending by score (simplest first)
    scored_items.sort(key=lambda x: x[0])
    
    # Extract sorted dataset
    sorted_dataset = [item for score, item in scored_items]
    
    print(f"[CURRICULUM] Sorting complete. Complexity ranges from {scored_items[0][0]:.3f} to {scored_items[-1][0]:.3f}")
    
    return sorted_dataset
