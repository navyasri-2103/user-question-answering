import json
import string
import re
import os
from typing import List, Dict, Any
from qa_engine import QAEngine

def normalize_answer(s: str) -> str:
    """
    Normalizes text for NLP comparison: lowercases, removes punctuation,
    removes articles (a, an, the), and standardizes whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def collections_counter(tokens: List[str]) -> dict:
    """
    Simple counter function to count word tokens in a list.
    """
    cnt = {}
    for t in tokens:
        cnt[t] = cnt.get(t, 0) + 1
    return cnt

def calculate_f1(prediction: str, ground_truth: str) -> float:
    """
    Calculates token-level F1 score (word overlap) between prediction and ground truth.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0
        
    pred_counter = collections_counter(prediction_tokens)
    gt_counter = collections_counter(ground_truth_tokens)
    
    num_same = 0
    for token, count in pred_counter.items():
        if token in gt_counter:
            num_same += min(count, gt_counter[token])
            
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_em(prediction: str, ground_truth: str) -> float:
    """
    Calculates Exact Match (EM) metric (1.0 if identical after normalization, else 0.0).
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def run_evaluation(dataset_path: str = "data/sample_dataset.json") -> Dict[str, Any]:
    """
    Evaluates the QA engine on the dataset and prints aggregate metrics.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return {}
        
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    print("Initializing QA Engine for evaluation...")
    engine = QAEngine()
    
    results = []
    total_em = 0.0
    total_f1 = 0.0
    total_latency = 0.0
    count = 0
    
    print("\nStarting evaluation run:")
    print("-" * 80)
    
    for item in dataset:
        passage = item["passage"]
        category = item["category"]
        for qa in item["qas"]:
            question = qa["question"]
            ground_truth = qa["answer"]
            
            # Predict
            pred_info = engine.get_answer(passage, question)
            prediction = pred_info["answer"]
            latency = pred_info["latency_seconds"]
            
            # Calculate metrics
            em = calculate_em(prediction, ground_truth)
            f1 = calculate_f1(prediction, ground_truth)
            
            total_em += em
            total_f1 += f1
            total_latency += latency
            count += 1
            
            results.append({
                "question_id": qa["id"],
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "em": em,
                "f1": f1,
                "latency": latency
            })
            
            print(f"Q: {question}")
            print(f"GT:  {ground_truth}")
            print(f"PRED: {prediction}")
            print(f"Metrics -> EM: {em}, F1: {round(f1, 2)}, Latency: {latency}s")
            print("-" * 80)
            
    avg_em = total_em / count if count > 0 else 0
    avg_f1 = total_f1 / count if count > 0 else 0
    avg_latency = total_latency / count if count > 0 else 0
    
    print("\n" + "=" * 40)
    print("EVALUATION OVERALL PERFORMANCE METRICS")
    print("=" * 40)
    print(f"Total Questions Evaluated : {count}")
    print(f"Average Exact Match (EM)   : {round(avg_em * 100, 2)}%")
    print(f"Average F1-Score          : {round(avg_f1 * 100, 2)}%")
    print(f"Average Latency           : {round(avg_latency, 4)} seconds")
    print("=" * 40)
    
    return {
        "total_evaluated": count,
        "average_em": avg_em,
        "average_f1": avg_f1,
        "average_latency": avg_latency,
        "results": results
    }

if __name__ == "__main__":
    run_evaluation()
