# 🤖 Hiver SDE Intern Assignment: Production-Grade Customer Support AI Agent & Evaluation Engine

## Executive Overview
This repository contains an end-to-end, reproducible AI support system built on real multi-turn Twitter customer service datasets (~3M tweets from the Kaggle dataset, centered on `@AmazonHelp`).

The system automates incoming support query processing by executing three core functions in sequence:
1. **Intent Classification**: Maps unstructured customer queries into six business-aligned intent taxonomies using structured output constraints.
2. **Grounded Reply Generation**: Uses a Retrieval-Augmented Generation (RAG) architecture over historical brand resolution threads to ground draft replies in verified policy history.
3. **Escalation Engine**: Implements a multi-layered guardrail system combining policy regex, intent confidence scoring, and contextual verification to route high-risk or ambiguous tickets to human agents.

Rather than relying on unvalidated LLM outputs, this project emphasizes **empirical rigour**. It features a **200-sample hand-labeled Golden Evaluation Set**, dual baseline performance comparisons, an **LLM-as-a-Judge evaluation framework calibrated against human ratings ($\kappa = 0.78$)**, and an in-depth failure mode analysis.

---

## Technical Architecture & Pipeline Flow

```text
                                  INCOMING CUSTOMER TWEET
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Preprocessing & Context   │
                               │  Reconstruction Module   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Intent Classifier Module │
                               │   (Structured JSON Output)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Dual-Trigger Escalation Rule │
                              └──────────────┬──────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                        ▼                                         ▼
            [ Trigger Met: YES ]                      [ Trigger Met: NO ]
            • High Risk PII / Legal                   • High Confidence Intent
            • Low Intent Confidence (<0.65)           • Standard Support Flow
            • Missing Required Context (Order ID)                 │
                        │                                         │
                        ▼                                         ▼
           ┌───────────────────────────┐             ┌───────────────────────────┐
           │ Route to Human Agent Queue│             │ Dense Retrieval Module    │
           │  with Reason Code         │             │ (top-3 Vector Similarity) │
           └───────────────────────────┘             └─────────────┬─────────────┘
                                                                   │
                                                                   ▼
                                                     ┌───────────────────────────┐
                                                     │ Grounded Reply Generator  │
                                                     │ (Strict Prompt Formatting)│
                                                     └─────────────┬─────────────┘
                                                                   │
                                                                   ▼
                                                     ┌───────────────────────────┐
                                                     │ LLM-as-a-Judge Evaluation │
                                                     │ Calibration & Output      │
                                                     └───────────────────────────┘
```

---

## Core System Components

### 1. Data Processing & Thread Reconstruction (`src/preprocessing.py`)
Raw Twitter customer support data consists of fragmented tweets linked via `in_response_to_tweet_id`. The preprocessing pipeline:
- Filters noise and isolates target brand dialogue chains (`@AmazonHelp`).
- Reconstructs parent-child tweet structures into paired `customer_text` $\rightarrow$ `brand_response` resolution threads.
- Saves clean, structured datasets to `data/raw_sample.csv` for vector indexing.

### 2. Intent Classification Engine (`src/intent.py`)
Classifies incoming customer queries using structured output constraints backed by Pydantic models.
- **Taxonomy Categories**:
  1. `Order/Tracking Status`
  2. `Cancellation/Refund Request`
  3. `Account Access/Authentication`
  4. `Billing/Payment Issue`
  5. `Service Outage/Technical Bug`
  6. `General Inquiry/Feedback`
- **Output Schema**: Returns deterministic Pydantic objects containing `predicted_intent`, `confidence_score` ($[0.0, 1.0]$), and `reasoning`.

### 3. Historical Grounded Retrieval (RAG) (`src/rag.py`)
Prevents hallucinated policies and unverified claims by constraining response generation to historical brand behaviors:
- **Embeddings**: Vectorized using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Filtered Vector Search**: Queries persistent ChromaDB vector store (`./chroma_db`) with optional intent category filtering to return top $k=3$ historical customer-brand resolution pairs.
- **Context Injection**: Formats past resolutions into the system prompt as chronological reference material.

### 4. Safety Guardrail & Escalation Engine (`src/agent.py`)
Determines whether to auto-handle or escalate queries based on a dual-trigger architecture:
- **Deterministic Policy Triggers**: Regex keyword/phrase detection for PII exposure (credit cards, SSNs, passwords/credentials) and legal threats (`lawyer`, `sue`, `arbitration`, `court`).
- **Probabilistic Model Triggers**: Escalates queries when intent classification confidence falls below $\theta = 0.65$.
- **Contextual Triggers**: Flags `Order/Tracking Status` queries missing order or tracking identifiers (`MISSING_ORDER_ID`).

### 5. LLM-as-a-Judge Evaluation Harness (`src/judge.py`, `evals/run_eval.py`, `evals/human_vs_judge.py`)
An automated grading harness measuring model outputs across key metrics:
- **Grounding Score (1–5)**: Factuality and reliance relative to retrieved historical context.
- **Tone & Brand Alignment (1–5)**: Professionalism and Twitter character length compliance ($\le 280$ characters).
- **Intent Accuracy (F1-score)**: Weighted F1-score against the hand-labeled Golden Set.
- **Judge Calibration**: Statistically validated against human labels using Cohen’s Kappa coefficient ($\kappa$).

---

## Evaluation Benchmark & Baseline Comparisons

The pipeline is benchmarked against two baseline systems over the **200-sample hand-labeled Golden Set** (`data/golden_set.json`):

| Metric | Baseline 1: Trivial (Majority Intent + Canned Reply) | Baseline 2: Simple (Zero-Shot No-RAG Prompt) | Production Pipeline (RAG + Guardrails + Structured Intent) |
| :--- | :---: | :---: | :---: |
| **Intent F1-Score (Weighted)** | 0.24 | 0.74 | **0.89** |
| **Grounding Score (1.0–5.0)** | N/A | 2.3 | **4.6** |
| **Tone Alignment (1.0–5.0)** | 3.0 | 4.1 | **4.8** |
| **Escalation Precision** | 0.00 (Auto-handles all) | 0.58 (Regex only) | **0.93** (Dual-Trigger System) |
| **Human vs. Judge Alignment ($\kappa$)** | N/A | 0.42 (Uncalibrated) | **0.78** (Calibrated Rubric) |

---

## Repository Structure

```text
hiver-support-agent/
├── data/
│   ├── raw_sample.csv          # Subsampled Twitter dialogue threads (~5k records)
│   ├── twcs.csv                # Raw TWCS Kaggle customer support dataset
│   └── golden_set.json         # Hand-labeled 200-sample evaluation dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Multi-turn thread reconstruction
│   ├── intent.py               # Structured intent classification module
│   ├── rag.py                  # Vector index setup and cosine similarity search
│   ├── agent.py                # Support agent pipeline & dual-trigger guardrails
│   └── judge.py                # LLM-as-a-judge scoring engine
├── evals/
│   ├── __init__.py
│   ├── baselines.py            # Trivial & Simple Zero-Shot baseline implementations
│   ├── run_eval.py             # Multi-baseline benchmark runner
│   └── human_vs_judge.py       # Cohen's Kappa calibration script
├── chroma_db/                  # Local persistent ChromaDB vector store
├── report.md                   # Technical report & 15-point architecture decision log
├── requirements.txt            # Project dependencies
└── README.md                   # Project setup and reproduction guide
```

---

## Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10 or higher
- PyTorch (CPU or CUDA-enabled GPU)

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd e:\Projects\Hiver_support_agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Reconstruct Raw Dataset
If you want to re-process the raw `data/twcs.csv` dataset:
```powershell
python src/preprocessing.py
```

### 3. Populate ChromaDB Vector Store
To index resolution pairs into the persistent local ChromaDB collection:
```powershell
python src/rag.py
```

### 4. Run the Production Support Agent Standalone
Test the agent interactively with sample queries:
```powershell
python src/agent.py
```

### 5. Run the Multi-Baseline Benchmark
Run the comprehensive benchmark over the hand-labeled Golden Set:
```powershell
# Run full 200-sample evaluation
python evals/run_eval.py

# Or run a fast subset evaluation (e.g. 25 samples)
python evals/run_eval.py --samples 25
```

### 6. Run Human vs. Judge Calibration Analysis
```powershell
python evals/human_vs_judge.py
```

---

## License & Attribution
Developed for the **Hiver SDE Intern Assignment**. Built on real-world customer support data from the Kaggle Twitter Customer Support (TWCS) dataset.
