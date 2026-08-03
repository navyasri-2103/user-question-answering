import re
from typing import Dict, Any, List

def clean_text(text: str) -> str:
    """
    Cleans text by stripping whitespace and normalizing multiple spaces.
    """
    if not text:
        return ""
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes text into lowercase words, ignoring punctuation.
    """
    # Find all alphanumeric word boundaries
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def tokenize_sentences(text: str) -> List[str]:
    """
    Splits text into sentences. Handles abbreviations (e.g., Mr., Co.)
    so they don't cause incorrect splits.
    """
    if not text:
        return []
    
    # Common abbreviations to ignore during period splitting
    abbreviations = r"(?:[A-Z]\.|Mr\.|Mrs\.|Dr\.|St\.|Co\.|Inc\.|Corp\.)"
    text_placeholder = re.sub(abbreviations, lambda m: m.group(0).replace('.', '___DOT___'), text)
    
    sentences = re.split(r'(?<=[.!?])\s+', text_placeholder)
    
    # Restore periods in the abbreviations
    restored_sentences = [s.replace('___DOT___', '.') for s in sentences if s.strip()]
    return restored_sentences

def calculate_text_analytics(text: str) -> Dict[str, Any]:
    """
    Extracts descriptive statistics of the text for data science profiling.
    Computes Word Count, Sentence Count, Average Word Length, and Readability (ARI).
    """
    cleaned = clean_text(text)
    words = tokenize_words(cleaned)
    sentences = tokenize_sentences(cleaned)
    
    word_count = len(words)
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    total_char_count = len(cleaned)
    char_count_no_spaces = sum(len(w) for w in words)
    
    avg_word_length = round(char_count_no_spaces / word_count, 2) if word_count > 0 else 0
    avg_sentence_length = round(word_count / sentence_count, 2)
    
    # Calculate Automated Readability Index (ARI)
    # Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43
    if word_count > 0 and sentence_count > 0:
        ari = 4.71 * (total_char_count / word_count) + 0.5 * (word_count / sentence_count) - 21.43
        readability_score = round(max(1.0, ari), 1)
    else:
        readability_score = 0.0
        
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": total_char_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "readability_score": readability_score
    }
