# User Question Answering: Interactive QA & Text Analytics

User Question Answering is an end-to-end, interactive Question Answering (QA) application and analytics dashboard. It utilizes a Hugging Face Transformer model (**DistilBERT** fine-tuned on SQuAD) to extract answers from passages, profiles the passage text using quantitative readability statistics, and evaluates performance metrics (Exact Match, F1-Score, and Latency) across a benchmark dataset.

---

## 🚀 Features

- **Transformer-Based QA Engine**: Resolves questions on user-provided passages with token-level confidence scores and latency calculation.
- **Dynamic Text Analytics**: Extracts descriptive metrics (words, sentences, character counts, average word/sentence lengths) and computes readability using the **Automated Readability Index (ARI)**.
- **Performance Evaluation Suite**: Benchmarks predictions against a sample dataset to report aggregated validation statistics including **Exact Match (EM)** and **F1-Score**.
- **Interactive Web Dashboard**: A modern, glassmorphic UI built with Vanilla HTML/CSS/JS that displays KPI cards, playground scenarios, and visual progress bars of performance categories.
---

## 🛠️ Tech Stack & Tools

- **Backend / Web Server**: FastAPI, Uvicorn (Python 3.8+)
- **NLP & Deep Learning**: Hugging Face Transformers, PyTorch (`distilbert-base-cased-distilled-squad` model)
- **Data & Evaluation**: Pandas, NumPy
- **Frontend Dashboard**: Vanilla HTML5, Custom CSS3 (Modern Glassmorphic UI & Outlined Typography), Vanilla JavaScript (Fetch API & Dynamic DOM)
- **Development Tools**: Git/GitHub, Visual Studio Code

---

## 📁 Project Structure

```text
├── app/
│   ├── main.py            # FastAPI server containing API endpoints
│   └── static/            # Frontend dashboard files
│       ├── index.html     # Glassmorphic UI template
│       ├── style.css      # Premium dark-theme CSS design
│       └── script.js      # Dashboard event handlers & API integration
├── data/
│   └── sample_dataset.json# Benchmark dataset categorized by topics
├── src/
│   ├── qa_engine.py       # Core QA interface loading DistilBERT pipeline
│   ├── preprocess.py      # Text preprocessing and readability calculation (ARI)
│   └── evaluate.py        # Quantitative metrics evaluation (EM, F1, and Latency)
├── run.py                 # Application launcher (CLI wrapper)
├── requirements.txt       # Project python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/navyasri-2103/user-question-answering.git
   cd user-question-answering
   ```

2. **Install required dependencies**:
   Ensure you have Python 3.8+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

The project is controlled via the CLI entrypoint script `run.py`.

### 1. Launch the Local Web Server & Dashboard
Starts the FastAPI application and hosts the interactive dashboard:
```bash
python run.py --server
```
Once started, open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 2. Run Quantitative Dataset Evaluation
Evaluates the DistilBERT QA engine on the local dataset (`data/sample_dataset.json`) and prints statistics to the console:
```bash
python run.py --evaluate
```

---

## 🔌 API Documentation

FastAPI serves backend endpoints under the following paths:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/dataset` | `GET` | Returns list of pre-configured passages and questions for scenario selector. |
| `/api/answer` | `POST` | Accepts `{passage, question}`, returns extracted answer, span coordinates, confidence score, and text analytics. |
| `/api/metrics` | `GET` | Evaluates the model on the entire dataset and returns performance summary per category. |

---

## 🛡️ License
This project is open-source and available under the MIT License.
