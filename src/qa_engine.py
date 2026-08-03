import time
from typing import Dict, Any
import torch
import torch.nn.functional as F

class QAEngine:
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initializes the Question Answering pipeline using Hugging Face Transformers.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def _load_model(self):
        """
        Loads the tokenizer and model if they haven't been loaded already.
        """
        if not self.loaded:
            print(f"Loading QA model: {self.model_name}...")
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.loaded = True
            print("Model loaded successfully.")

    def get_answer(self, passage: str, question: str) -> Dict[str, Any]:
        """
        Extracts an answer to the question from the provided passage.
        Returns a dictionary containing the answer text, character span indices,
        confidence score, and latency in seconds.
        """
        self._load_model()
        
        start_time = time.time()
        
        # Tokenize question and passage together
        inputs = self.tokenizer(question, passage, return_tensors="pt")
        
        # Forward pass to obtain start/end logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Determine top token span indices
        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))
        
        # Calculate confidence score (Softmax product of start & end tokens)
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        score = float(start_probs[0, start_idx] * end_probs[0, end_idx])
        
        # Decode the target token span back to string
        answer_tokens = inputs.input_ids[0, start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Resolve character indexes in the original passage string
        try:
            start_span = inputs.token_to_chars(0, start_idx)
            end_span = inputs.token_to_chars(0, end_idx)
            
            if start_span is not None and end_span is not None:
                start_char = start_span.start
                end_char = end_span.end
            else:
                # Fallback to string index search if span coordinates are missing
                start_char = passage.lower().find(answer.lower())
                end_char = start_char + len(answer) if start_char != -1 else 0
                if start_char == -1:
                    start_char = 0
                    end_char = 0
        except Exception:
            start_char = passage.lower().find(answer.lower())
            end_char = start_char + len(answer) if start_char != -1 else 0
            if start_char == -1:
                start_char = 0
                end_char = 0
                
        end_time = time.time()
        latency = round(end_time - start_time, 4)
        
        return {
            "answer": answer,
            "score": round(score, 4),
            "start": start_char,
            "end": end_char,
            "latency_seconds": latency
        }

if __name__ == "__main__":
    # Quick self-test to verify the engine runs locally
    engine = QAEngine()
    test_passage = "Transformers is a python package developed by Hugging Face."
    test_question = "Who developed Transformers?"
    print("\n--- Running engine self-test ---")
    result = engine.get_answer(test_passage, test_question)
    print("Result:", result)
