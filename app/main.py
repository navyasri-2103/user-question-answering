import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List
import json

# Python path helper: ensures imports work from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qa_engine import QAEngine
from preprocess import calculate_text_analytics
from evaluate import calculate_em, calculate_f1

app = FastAPI(
    title="User Question Answering API",
    description="A FastAPI backend containing endpoints for Transformer-based QA and context text profiling."
)

# Initialize QA engine lazily
qa_engine = QAEngine()

class AnswerRequest(BaseModel):
    passage: str
    question: str

@app.get("/api/dataset")
def get_dataset():
    """
    Returns the sample dataset for frontend scenarios selection.
    """
    # Find root directory absolutely
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "sample_dataset.json")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
        
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data

@app.post("/api/answer")
def post_answer(request: AnswerRequest):
    """
    Accepts passage and question, extracts the answer, and computes text analytics.
    """
    passage = request.passage.strip()
    question = request.question.strip()
    
    if not passage or not question:
        raise HTTPException(status_code=400, detail="Passage and Question must not be empty.")
        
    try:
        # Get prediction from QA Engine
        prediction = qa_engine.get_answer(passage, question)
        
        # Calculate context analytics (word counts, sentences, readability)
        analytics = calculate_text_analytics(passage)
        
        return {
            "prediction": {
                "answer": prediction["answer"],
                "score": prediction["score"],
                "start": prediction["start"],
                "end": prediction["end"],
                "latency_seconds": prediction["latency_seconds"]
            },
            "analytics": analytics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
def get_metrics():
    """
    Runs model evaluation on the dataset and returns aggregated dashboard statistics.
    """
    # Find root directory absolutely
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "sample_dataset.json")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    try:
        total_em = 0.0
        total_f1 = 0.0
        total_latency = 0.0
        count = 0
        category_metrics = {}
        
        for item in dataset:
            passage = item["passage"]
            category = item["category"]
            
            if category not in category_metrics:
                category_metrics[category] = {"em": 0.0, "f1": 0.0, "latency": 0.0, "count": 0}
                
            for qa in item["qas"]:
                question = qa["question"]
                ground_truth = qa["answer"]
                
                # Predict
                pred_info = qa_engine.get_answer(passage, question)
                prediction = pred_info["answer"]
                latency = pred_info["latency_seconds"]
                
                # Calculate metrics
                em = calculate_em(prediction, ground_truth)
                f1 = calculate_f1(prediction, ground_truth)
                
                total_em += em
                total_f1 += f1
                total_latency += latency
                count += 1
                
                # Category-wise statistics
                cat_stat = category_metrics[category]
                cat_stat["em"] += em
                cat_stat["f1"] += f1
                cat_stat["latency"] += latency
                cat_stat["count"] += 1
                
        # Finalize aggregates
        avg_em = total_em / count if count > 0 else 0
        avg_f1 = total_f1 / count if count > 0 else 0
        avg_latency = total_latency / count if count > 0 else 0
        
        category_summary = []
        for cat, stats in category_metrics.items():
            cat_count = stats["count"]
            category_summary.append({
                "category": cat,
                "count": cat_count,
                "em": round(stats["em"] / cat_count, 4) if cat_count > 0 else 0.0,
                "f1": round(stats["f1"] / cat_count, 4) if cat_count > 0 else 0.0,
                "latency_seconds": round(stats["latency"] / cat_count, 4) if cat_count > 0 else 0.0
            })
            
        return {
            "summary": {
                "total_samples": count,
                "avg_em": round(avg_em, 4),
                "avg_f1": round(avg_f1, 4),
                "avg_latency": round(avg_latency, 4)
            },
            "categories": category_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Resolve the static directory path absolutely
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
