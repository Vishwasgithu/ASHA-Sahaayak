# 🩺 ASHA-Sahaayak
###  Agentic AI-Powered Multilingual Maternal Healthcare Assistant for Early Risk Detection and Evidence-Based Clinical Decision Support.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg">
  <img src="https://img.shields.io/badge/Flask-Web%20API-green.svg">
  <img src="https://img.shields.io/badge/Whisper-Speech%20Recognition-orange.svg">
  <img src="https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-red.svg">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Database-purple.svg">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-black.svg">
</p>

---

# 📌 Overview

**ASHA-Sahaayak** is an AI-powered healthcare assistant designed to support **ASHA (Accredited Social Health Activist)** workers in rural and low-resource settings.

The system enables ASHA workers to provide patient information through **voice or text**, automatically identifies maternal symptoms, estimates the maternal risk level using a rule-based clinical engine, and retrieves evidence from trusted healthcare guidelines using a **Retrieval-Augmented Generation (RAG)** pipeline.

The objective is to provide **evidence-based clinical decision support** while reducing dependence on internet connectivity through local AI models.

---

# 🎯 Objectives

- Multilingual voice and text-based healthcare assistance
- Early maternal risk detection
- Evidence-based clinical recommendations
- Clinical knowledge retrieval using RAG
- Offline-capable AI architecture
- Decision support for frontline healthcare workers

---

# 🚀 Features

- 🎤 Voice-to-Text using OpenAI Whisper
- 🌍 English & Hindi symptom detection
- 🤰 Pregnancy month extraction
- ⚠️ Maternal risk assessment
- 📚 WHO, NHM & ASHA knowledge base
- 🔍 Semantic search using ChromaDB
- 🧠 Retrieval-Augmented Generation (RAG)
- 🤖 Local LLM inference using Ollama
- 🖥 Flask REST API
- 📊 Interactive frontend dashboard

---

# 🏗 System Architecture

```text
ASHA Worker
        │
        ▼
 Voice / Text Input
        │
        ▼
 OpenAI Whisper
 (Speech → Text)
        │
        ▼
 Healthcare Engine
        │
        ├── Symptom Detection
        ├── Pregnancy Month Extraction
        └── Rule-Based Risk Assessment
        │
        ▼
 Clinical Query Generation
        │
        ▼
 Retrieval-Augmented Generation (RAG)
        │
        ▼
 WHO + NHM + ASHA Guidelines
        │
        ▼
 Document Loader
        │
        ▼
 Chunking
        │
        ▼
 Embeddings
 (BAAI/bge-small-en-v1.5)
        │
        ▼
 ChromaDB
        │
        ▼
 Top-K Relevant Clinical Evidence
        │
        ▼
 Local LLM (Ollama + Qwen)
        │
        ▼
 Evidence-Based Recommendation
        │
        ▼
 Frontend Dashboard
```

---

# 🛠 Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Speech Recognition | OpenAI Whisper |
| Knowledge Retrieval | RAG |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Local LLM | Ollama + Qwen |
| Prompt Framework | LangChain |
| Environment | Python Virtual Environment |

---

# 📂 Project Structure

```
ASHA-Sahaayak
│
├── Backend
│   ├── app.py
│   ├── healthcare_engine.py
│   ├── rag/
│   ├── knowledge_base/
│   ├── vector_db/
│   ├── tests/
│   ├── build_index.py
│   └── requirements.txt
│
├── Frontend
│
├── README.md
└── .gitignore
```

---

# 🔄 Workflow

1. ASHA worker enters symptoms using voice or text.
2. Whisper converts speech into text.
3. Healthcare engine extracts:
   - Symptoms
   - Pregnancy month
   - Maternal risk level
4. A clinical query is generated.
5. RAG retrieves relevant evidence from WHO, NHM, and ASHA guidelines.
6. The retrieved evidence is provided to the local LLM.
7. The LLM generates an evidence-grounded recommendation.
8. Results are displayed on the dashboard.

---

# 🧠 AI Components

### Speech Recognition

- OpenAI Whisper

### Maternal Risk Detection

Current implementation:

- Rule-Based Clinical Engine

Future scope:

- Machine Learning-based risk prediction

### Retrieval-Augmented Generation (RAG)

Knowledge Sources:

- WHO Guidelines
- NHM India Guidelines
- ASHA Manuals
- Maternal Healthcare Research Papers

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ASHA-Sahaayak.git
cd ASHA-Sahaayak
```

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r Backend/requirements.txt
```

---

## Configure Environment Variables

Copy

```
Backend/.env.example
```

to

```
Backend/.env
```

Configure your preferred LLM provider.

Example:

```env
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5-coder:7b
```

or

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Build Vector Database

```bash
python Backend/build_index.py
```

---

## Test RAG Pipeline

```bash
python Backend/test_rag.py
```

---

## Run Backend

```bash
python Backend/app.py
```

Backend runs at

```
http://127.0.0.1:5000
```

---

# 📈 Current Status

### ✅ Implemented

- Voice & text input
- Whisper transcription
- English & Hindi symptom extraction
- Pregnancy month detection
- Rule-based maternal risk assessment
- Flask backend APIs
- Healthcare knowledge base
- Document loading & chunking
- Embedding generation
- ChromaDB indexing
- Semantic retrieval
- Local LLM integration

### 🔄 In Progress

- Enhanced frontend integration of RAG recommendations
- Additional multilingual support
- Response validation improvements
- Advanced clinical reasoning

---

# 🔮 Future Scope

- Marathi language support
- Telugu language support
- Offline deployment
- EMR generation
- Explainable AI
- Hospital integration
- ML-based maternal risk prediction
- Follow-up and reminder system

---

# 📚 What I Learned

- Retrieval-Augmented Generation (RAG)
- Speech Recognition using Whisper
- Vector Databases
- Embedding Models
- ChromaDB
- LangChain
- Flask API Development
- Ollama & Local LLM Deployment
- Prompt Engineering
- Semantic Search
- Modular AI System Design
- AI Integration for Healthcare

---

# 📄 License

This project was developed for academic and research purposes.
