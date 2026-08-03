import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(
        description="CognitiveQA CLI - Question Answering Engine and Web Server Dashboard"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI web server dashboard on localhost:8000"
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run model evaluation (EM & F1) on the pre-collected sample dataset"
    )
    
    args = parser.parse_args()
    
    if args.evaluate:
        print("Running quantitative validation on target QA datasets...")
        from src.evaluate import run_evaluation
        run_evaluation()
        
    elif args.server:
        print("Launching local dashboard server on http://127.0.0.1:8000 ...")
        # Starts the FastAPI application
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
