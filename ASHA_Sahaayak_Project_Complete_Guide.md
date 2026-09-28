# ASHA-Sahaayak — Complete Project Guide (Mentor Edition)

> An AI-Powered Multilingual Maternal Healthcare Assistant for Early Risk
> Detection and Evidence-Based Clinical Decision Support.
>
> This guide teaches the project from ZERO. It assumes you know only basic
> Python. Every technical term is explained before it is used. Read it top to
> bottom and you will be able to explain the whole system to professors,
> mentors, and interviewers with confidence.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Why This Project Was Built](#3-why-this-project-was-built)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Folder Structure](#5-folder-structure)
6. [File-by-File Explanation](#6-file-by-file-explanation)
7. [Technologies Used (and WHY)](#7-technologies-used-and-why)
8. [Libraries Used](#8-libraries-used)
9. [AI Concepts From Scratch](#9-ai-concepts-from-scratch)
10. [Complete Data Flow](#10-complete-data-flow)
11. [Backend Flow](#11-backend-flow)
12. [Frontend Flow](#12-frontend-flow)
13. [Healthcare Engine Flow](#13-healthcare-engine-flow)
14. [RAG Flow](#14-rag-flow)
15. [Database Flow](#15-database-flow)
16. [API Flow](#16-api-flow)
17. [End-to-End Execution Flow](#17-end-to-end-execution-flow)
18. [Current Limitations](#18-current-limitations)
19. [Production Improvements](#19-production-improvements)
20. [Future Scope](#20-future-scope)
21. [Interview Questions (100+)](#21-interview-questions-100)
22. [Viva Questions (100+)](#22-viva-questions-100)
23. [Mentor Questions (100+)](#23-mentor-questions-100)
24. [How to Explain This Project in 1 / 3 / 5 / 10 Minutes](#24-how-to-explain-this-project)
25. [Key Learnings](#25-key-learnings)
26. [Best Practices Used](#26-best-practices-used)
27. [Important Design Decisions](#27-important-design-decisions)
28. [Common Beginner Mistakes](#28-common-beginner-mistakes)
29. [Summary](#29-summary)

---

## 1. Project Overview

**ASHA-Sahaayak** ("Sahaayak" means *helper/assistant* in Hindi) is a software
system that helps **ASHA workers** — India's community health volunteers — take
better care of pregnant women in villages.

An ASHA worker visits a pregnant woman, listens to her complaints (in Hindi,
Marathi, Telugu, or English), and needs to decide: *is this normal, or is this
dangerous?* Most ASHA workers are not doctors. They need a simple, trustworthy
helper.

ASHA-Sahaayak does three things:

1. **Listens / reads** the symptoms — by **voice** (speech is converted to text)
   or by **typing**.
2. **Screens for danger** — a rule-based "Healthcare Engine" extracts symptoms,
   detects the pregnancy month, and assigns a **risk level** (LOW / MEDIUM /
   HIGH).
3. **Gives evidence-based advice** — instead of guessing, it searches trusted
   medical documents (WHO, NHM India, ASHA manuals) and asks an AI language
   model to write a recommendation **grounded in those documents**, with
   citations.

In one sentence: *It turns a spoken complaint into a safe, cited, maternal-health
recommendation, even with poor internet, in multiple Indian languages.*

---

## 2. Problem Statement

India has a high number of maternal complications. Many are preventable if the
danger signs (like high blood pressure, bleeding, or pre-eclampsia symptoms) are
caught early. The reality in rural areas:

- **Doctors are far away.** The first point of contact is often an ASHA worker.
- **ASHA workers are not clinicians.** They may miss early warning signs.
- **Guidelines exist but are unusable in the field.** WHO/NHM PDFs are hundreds
  of pages — nobody reads them during a home visit.
- **Language is a barrier.** Women speak local languages; guidelines are in
  English.
- **Internet is unreliable.** Cloud-only tools fail in villages.

**The problem this project solves:** give the frontline worker an *instant,
offline-capable, multilingual, evidence-based* second opinion that flags risk and
tells them what the guidelines actually recommend.

---

## 3. Why This Project Was Built

- **To save lives** by catching maternal red-flags earlier.
- **To empower non-experts** with reliable clinical decision support.
- **To make dense guidelines usable** by retrieving only the relevant few
  paragraphs and summarizing them.
- **To respect ground realities**: multilingual input, voice-first, and the
  option to run the AI **locally (offline)** with Ollama.
- **To be trustworthy**: the AI is not allowed to invent medical advice — it must
  cite retrieved evidence, or explicitly say it has insufficient evidence.

**Who will use it?**

| User | How they use it |
| --- | --- |
| ASHA worker (primary) | Speaks/types symptoms, reads the risk + advice |
| ANM / PHC staff | Uses the risk flag to prioritise referrals |
| Program supervisors | Review reports/alerts |
| Developers/researchers | Extend languages, models, knowledge base |

---

## 4. High-Level Architecture

Think of the system as two halves that talk over the internet using **APIs**
(explained later): a **Frontend** (what the user sees in the browser) and a
**Backend** (the brain, written in Python/Flask).

```text
        ASHA WORKER (browser)
                |
   +------------------------------+
   |  FRONTEND (HTML/CSS/JS)      |   voice-input.html, analysis-result.html ...
   |  - record voice / type text  |
   |  - pick language (en/hi/mr/te)|
   +---------------+--------------+
                   |  HTTP request (JSON or audio)
                   v
   +------------------------------+
   |  BACKEND (Flask - app.py)    |
   |                              |
   |  1) Whisper  -> speech->text |
   |  2) Healthcare Engine        |
   |       symptoms, month, risk  |
   |  3) RAG Pipeline             |
   |       Retriever -> ChromaDB  |
   |       Generator -> LLM       |
   +---------------+--------------+
                   |  HTTP response (JSON)
                   v
        FRONTEND shows: risk + recommendation + sources
```

**Why this split (client/server) was chosen:**

- The **AI models are heavy** (Whisper, embeddings, LLM). They must run on a
  server/PC, not inside a phone browser.
- The **browser is universal** — any ASHA worker with a basic device can open a
  web page; no app install needed.
- **APIs decouple** the two: you can change the UI without touching the AI, and
  swap the AI (Gemini <-> Ollama) without touching the UI.
---

## 5. Folder Structure

```text
ASHA_Sahaayak/
├── Backend/                 <- the Python "brain" (server)
│   ├── app.py               <- Flask web server + API routes (entry point)
│   ├── healthcare_engine.py <- rule-based symptom + risk detection
│   ├── config.py            <- all settings in one place
│   ├── main.py              <- optional command-line demo (mic/text)
│   ├── build_index.py       <- one-time script: build the vector database
│   ├── test_rag.py          <- quick manual test of the RAG pipeline
│   ├── requirements.txt     <- list of Python libraries to install
│   ├── .env / .env.example  <- secret + provider configuration
│   ├── rag/                 <- the Retrieval-Augmented Generation package
│   │   ├── loader.py        <- read PDFs into text
│   │   ├── chunker.py       <- split text into small pieces
│   │   ├── embeddings.py    <- turn text into vectors (numbers)
│   │   ├── vector_store.py  <- ChromaDB storage + search
│   │   ├── retriever.py     <- find the most relevant chunks
│   │   ├── generator.py     <- ask the LLM for a grounded answer
│   │   └── pipeline.py      <- glue that runs the whole RAG flow
│   ├── knowledge_base/      <- the trusted medical PDFs (WHO, NHM, ASHA, papers)
│   ├── vector_db/           <- ChromaDB's saved vectors (auto-generated)
│   ├── prompts/             <- (empty) intended for prompt text files
│   ├── models/              <- (empty) intended for local model files
│   ├── reports/             <- log files (asha_backend.log)
│   ├── utils/               <- helpers (logging)
│   ├── tests/               <- automated tests
│   └── scratch/             <- throwaway experiment scripts
├── Frontend/                <- everything the browser shows
│   ├── index.html           <- login page
│   ├── dashboard.html       <- home screen after login
│   ├── voice-input.html     <- record/type symptoms (main input)
│   ├── analysis-result.html <- shows risk + recommendation (main output)
│   ├── new-patient.html, patient-history.html, reports.html,
│   │   alerts.html, settings.html   <- supporting screens
│   ├── css/                 <- styling (style.css, dashboard.css, ...)
│   └── js/app.js            <- browser logic + API client
├── README.md
└── .gitignore
```

### Why each folder exists (and what breaks if removed)

| Folder | Why it exists | Who uses it | If removed |
| --- | --- | --- | --- |
| `Backend/` | Holds the server, AI, and logic | Everything | The app has no brain; nothing works |
| `Backend/rag/` | Isolates the RAG steps into clean modules | `app.py`, `pipeline.py`, tests | No evidence retrieval or grounded answers |
| `knowledge_base/` | The source of truth (medical PDFs) | `loader.py` (indexing) | RAG has nothing to search; answers become "insufficient evidence" |
| `vector_db/` | Fast searchable copy of the PDFs (vectors) | `vector_store.py`, `retriever.py` | Retrieval fails until you re-run `build_index.py` |
| `reports/` | Stores logs for debugging | `utils/helpers.py` | You lose logs; app still runs |
| `prompts/` | Placeholder for external prompt files | (unused today) | Nothing breaks (prompt is built in code) |
| `models/` | Placeholder for local model weights | (unused today) | Nothing breaks |
| `utils/` | Shared helper code (logging) | Almost every backend module | Imports fail; app won't start |
| `tests/` | Prove the code works | Developers / CI | You lose safety net; app still runs |
| `Frontend/` | The user interface | The ASHA worker | No screen to interact with |
| `Frontend/css/` | Visual design | HTML pages | UI looks broken but works |
| `Frontend/js/` | Browser behaviour + API calls | HTML pages | Buttons and API calls stop working |

> Mentor note: `prompts/` and `models/` are **empty placeholders**. In an
> interview, mention this as "planned structure for externalizing prompts and
> local model weights — currently unused." Honesty about unused scaffolding is a
> senior trait.

---

## 6. File-by-File Explanation

For each important file: **why it exists, what problem it solves, who calls it,
what depends on it, key functions, and the execution flow — in simple words.**

### 6.1 `Backend/app.py` (the front door)

- **Why it exists:** it is the **web server**. It starts Flask, loads Whisper
  once, creates one shared `RAGPipeline`, and defines the **API routes** the
  frontend calls.
- **Who calls it:** the Frontend (`js/app.js` and inline scripts) over HTTP.
- **Depends on:** `healthcare_engine.py`, `rag/pipeline.py`, `utils/helpers.py`,
  and the `whisper` library.
- **Key functions:**
  - `analyze_healthcare_text(text, language="en")` — the core orchestration:
    run the Healthcare Engine, then run RAG, then package a JSON result. If RAG
    fails, it falls back to `_legacy_guidance_fallback`.
  - `_legacy_guidance_fallback(analysis)` — a safe, hard-coded message used only
    when RAG is down (e.g., no internet + no local model). Keeps the app useful.
  - `health_check()` -> `GET /api/health` — "is the server alive?"
  - `analyze_text()` -> `POST /api/analyze` — text in, analysis out.
  - `transcribe_audio()` -> `POST /api/transcribe` — audio in; Whisper makes
    text; then the same analysis.
- **Execution flow:** browser sends request -> route function reads input ->
  `analyze_healthcare_text` -> returns JSON -> browser renders it.

### 6.2 `Backend/healthcare_engine.py` (the rule-based screener)

- **Why it exists:** to quickly and *predictably* find symptoms and assign risk
  without needing internet or a model. It also builds the search query for RAG.
- **Who calls it:** `app.py` (`process_healthcare_input`).
- **Key parts:**
  - `_SYMPTOM_ALIASES` — a dictionary mapping each canonical symptom (e.g.
    `"fever"`) to its many spellings in English, Hindi, Marathi, Telugu.
  - `_phrase_pattern()` — builds a regular expression for each alias. It uses
    `(?<!\w)...(?!\w)` instead of `\b` so it works for Indic scripts whose words
    end in combining marks (a real bug that was fixed).
  - `_is_negated()` — understands "no fever", "bukhar nahi hai" so it does not
    wrongly detect a denied symptom.
  - `_classify_risk()` — the rules: pre-eclampsia triad or bleeding/severe pain/
    reduced fetal movement -> HIGH; fever/vomiting/weakness -> MEDIUM; else LOW.
  - `_build_rag_query()` — turns the findings into an English question for RAG.
  - `process_healthcare_input(text)` — the public function returning symptoms,
    month, risk level, and the RAG query.
- **Execution flow:** text -> detect symptoms -> detect month -> classify risk ->
  build RAG query -> return a dictionary.

### 6.3 `Backend/config.py` (the control panel)

- **Why it exists:** one place for all settings, so you never hard-code values in
  many files. Reads `.env` for secrets/provider choice.
- **Important settings:** paths, `EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"`,
  `CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 150`, `TOP_K_RESULTS = 5`,
  `LLM_PROVIDER` (default `gemini`, `.env` sets `ollama`), `CHROMA_COLLECTION_NAME
  = "asha_knowledge"`.
- **Depended on by:** almost every backend module.

### 6.4 `Backend/rag/loader.py` (PDF reader)

- **Why:** convert medical PDFs into text "pages" the computer can process.
- **Key class:** `PDFDocumentLoader` with `load_all_documents()` (scans WHO,
  NHM_India, ASHA, Research_Papers) and `load_single_pdf()` (uses LangChain's
  `PyPDFLoader`). It tags each page with `category`, `filename`, `file_path`.
- **Called by:** `pipeline.build_index()`.

### 6.5 `Backend/rag/chunker.py` (text splitter)

- **Why:** whole pages are too big to search precisely. We split them into small
  overlapping "chunks" (800 chars, 150 overlap) using
  `RecursiveCharacterTextSplitter`.
- **Key class:** `DocumentChunker.split_documents()`.
- **Called by:** `pipeline.build_index()`.

### 6.6 `Backend/rag/embeddings.py` (text -> numbers)

- **Why:** computers compare *meaning* using vectors (lists of numbers). This
  module loads `bge-small-en-v1.5` and turns text into normalized vectors.
- **Key functions:** `generate_embeddings(texts)` (for documents),
  `generate_query_embedding(query)` (for the search query),
  `get_embedding_model()` (cached, loaded once).
- **Note:** it also contains an `InMemoryEmbeddingStore` class that is currently
  **unused** (ChromaDB is used instead). Good interview honesty point.
- **Called by:** `vector_store.py`.

### 6.7 `Backend/rag/vector_store.py` (ChromaDB boundary)

- **Why:** store vectors permanently and search them fast.
- **Key functions:** `create_vector_store()` / `load_vector_store()` (open the
  Chroma collection), `save_documents()` (embed + `upsert`),
  `similarity_search()` (embed the query, return nearest chunks with distance).
- **Design detail:** the collection stores which embedding model made it, and
  refuses to run if you switch models — preventing silent corruption.
- **Called by:** `retriever.py`, `pipeline.py`.

### 6.8 `Backend/rag/retriever.py` (the finder)

- **Why:** a clean service that validates the query and returns the top-K most
  relevant chunks.
- **Key class:** `ChromaRetriever.retrieve(query, top_k, where)`.
- **Called by:** `pipeline.query()`.

### 6.9 `Backend/rag/generator.py` (the writer)

- **Why:** ask the LLM to write an answer using ONLY the retrieved evidence, and
  then **validate** it so no hallucinated advice slips through.
- **Key functions:**
  - `build_clinical_prompt()` — safety rules + JSON schema + query + evidence.
    Treats user text and evidence as *untrusted data* (prompt-injection defense).
  - `create_configured_llm()` — picks Gemini/OpenAI/Ollama from config.
  - `_parse_grounded_answer()` — parses the LLM JSON and accepts the answer only
    if it cites at least one *real* evidence id and invents none. (This was
    fixed to stop rejecting valid answers.)
  - `ClinicalResponseGenerator.generate()` — the orchestrator.
- **Called by:** `pipeline.query()`.

### 6.10 `Backend/rag/pipeline.py` (the conductor)

- **Why:** run the entire RAG flow with one object, safely and thread-safe.
- **Key methods:**
  - `build_index()` — load -> chunk -> embed -> store; uses stable content-based
    IDs so re-running does not duplicate, and removes stale chunks safely.
  - `query()` — retrieve -> generate -> return `{answer, retrieved_sources,
    metadata}`.
  - `run()` — build the index automatically if empty, then query.
- **Called by:** `app.py`, `build_index.py`, `test_rag.py`.

### 6.11 Support files

- `utils/helpers.py` — logging setup (`get_logger`).
- `build_index.py` — CLI to build the vector DB once.
- `test_rag.py` — run one sample query end-to-end.
- `main.py` — a standalone CLI demo (record via microphone or type) — NOT used by
  the server.
- `tests/` — unit/integration tests for engine, retriever, pipeline, generator,
  and the Flask API.

### 6.12 Frontend files (summary; details in Section 12)

- `index.html` (login), `dashboard.html` (home), `voice-input.html` (input),
  `analysis-result.html` (output), plus `new-patient/patient-history/reports/
  alerts/settings`. `css/` styles them. `js/app.js` holds the API client, the
  localStorage patient store, and an offline fallback engine.

### 6.13 Known stray/placeholder items (be honest in interviews)

- Root `healthcare_engine.py` and `main.py` are **empty 0-byte** stubs (the real
  code is under `Backend/`).
- `Backend/Backend/rag/__init__.py` is an accidental empty nested duplicate.
- `prompts/` and `models/` are empty placeholders.

---

## 7. Technologies Used (and WHY)

| Technology | What it is (simple) | Why chosen here | Alternative & why not |
| --- | --- | --- | --- |
| **Python** | A beginner-friendly programming language | The entire AI ecosystem (Whisper, LangChain, Chroma, HF) is Python-first | Java/Node have weaker ML libraries |
| **Flask** | A tiny web framework to build APIs | Minimal, easy to learn, perfect for a few endpoints | Django is heavy for this; FastAPI is great but Flask is simpler for beginners |
| **HTML** | Structure of web pages | Universal, runs on any device browser | Native mobile app needs installs |
| **CSS** | Look and feel (colors, layout) | Standard styling; works offline | — |
| **JavaScript** | Makes pages interactive; calls APIs | Only language browsers run natively | — |
| **Whisper** | OpenAI speech-to-text model | Multilingual, runs **offline**, free | Google Speech API needs internet + billing |
| **LangChain** | Toolkit for building LLM apps | Gives ready pieces: loaders, splitters, LLM adapters, `Document` type | Hand-rolling all adapters is error-prone |
| **ChromaDB** | A vector database (stores meaning as numbers) | Open-source, **local**, no server needed, simple API | Pinecone is cloud-only + paid; needs internet |
| **BAAI/bge-small-en-v1.5** | Embedding model (text -> vector) | Small, fast, strong quality, runs on CPU | Large models are slower; OpenAI embeddings need internet |
| **Gemini** | Google's cloud LLM | High quality, easy API, good for demos | Needs internet + API key |
| **Ollama** | Runs open LLMs (e.g. Qwen) **locally** | **Offline** privacy-friendly inference | Cloud LLMs fail without internet |
| **RAG** | Retrieval-Augmented Generation | Grounds answers in real guidelines; reduces hallucination | A bare LLM invents medical advice — unsafe |

**The big theme:** every technology was chosen so the system can, if needed, run
**offline** and **privately** in a village (local Whisper + local embeddings +
local Chroma + local Ollama), while still allowing a **cloud** path (Gemini) for
better quality when internet exists.

---

## 8. Libraries Used

Explained with: why, where, alternatives, pros/cons.

### Flask + flask-cors
- **Why:** create the web server and API endpoints; `flask-cors` lets the
  browser page call the API from a different origin (port).
- **Where:** `app.py`.
- **Alternatives:** FastAPI (async, auto docs), Django (full framework).
- **Pros:** tiny, readable. **Cons:** no async, no built-in validation.

### openai-whisper (+ torch)
- **Why:** convert recorded speech to text; `torch` (PyTorch) is the deep-learning
  engine Whisper runs on.
- **Where:** `app.py` (`whisper.load_model("base")`, `model.transcribe`).
- **Alternatives:** faster-whisper (quicker), Google/Azure STT (cloud).
- **Pros:** offline, multilingual. **Cons:** CPU is slow; `base` model is modest.

### sounddevice + scipy + numpy
- **Why:** record microphone audio in the CLI demo (`main.py`); `numpy` is the
  array math backbone used across ML.
- **Where:** `main.py`, and `numpy` inside `embeddings.py`.
- **Pros:** simple recording. **Cons:** desktop-only (not used by the web app).

### langchain-core / community / huggingface / text-splitters
- **Why:** `Document` type, `PyPDFLoader`, `RecursiveCharacterTextSplitter`,
  `HuggingFaceEmbeddings`, and the chat-model adapters.
- **Where:** the whole `rag/` package.
- **Alternatives:** LlamaIndex, Haystack.
- **Pros:** batteries included, swappable providers. **Cons:** fast-changing API,
  extra abstraction.

### sentence-transformers
- **Why:** the actual engine that loads and runs the `bge-small-en-v1.5`
  embedding model under `HuggingFaceEmbeddings`.
- **Pros:** state-of-the-art small embedders. **Cons:** first load downloads
  weights.

### chromadb
- **Why:** persistent vector store + nearest-neighbour search.
- **Where:** `vector_store.py`.
- **Alternatives:** FAISS (library, no persistence layer), Pinecone/Weaviate
  (servers).
- **Pros:** local, simple, persistent. **Cons:** not built for millions of
  vectors at scale.

### pypdf
- **Why:** the low-level PDF text extractor used by `PyPDFLoader`.
- **Cons:** struggles with scanned (image) PDFs — those need OCR.

### langchain-google-genai / langchain-openai / langchain-ollama
- **Why:** talk to Gemini / OpenAI / local Ollama with one common interface.
- **Where:** `generator.create_configured_llm()`.
- **Pros:** switch providers via config, no code change.

### python-dotenv
- **Why:** load secrets/config from a `.env` file into environment variables so
  keys are never hard-coded.
- **Where:** `config.py`.

---

## 9. AI Concepts From Scratch

### API (Application Programming Interface)
A contract that lets two programs talk. Here, the **Frontend** (browser) sends
an HTTP request to the **Backend** (Flask); the backend sends back a
response. Think "waiter": you (frontend) hand the menu order (request),
the kitchen (backend) cooks, the waiter returns the dish (response).

### JSON (JavaScript Object Notation)
A plain-text way to carry structured data. Example the backend returns:
```json
{ "risk_level": "MEDIUM RISK", "symptoms": ["fever","headache"],
  "clinical_recommendation": "Monitor BP [Evidence 1]", "language": "hi" }
```
Keys are strings; values can be strings, numbers, lists, or nested objects.

### HTTP Request / Response
- **Request** = verb (GET/POST) + URL (e.g. `/api/analyze`) + body
  (the data, usually JSON or a file).
- **Response** = status code (200 = OK, 400 = bad input, 500 = server
  error) + body (JSON).

### Whisper (Speech-to-Text / ASR)
A neural network trained to turn audio into text, for many languages. We run it
**locally** so no audio leaves the device and no internet is needed. Input:
a `.wav`/`.webm` file. Output: a transcript string.

### Embeddings (turning words into coordinates)
An embedding model reads text and produces a **vector** — a list of, say, 384
numbers — that captures *meaning*. Similar sentences get similar vectors
(they point in the same direction). This lets a computer "measure meaning"
with math (dot product / cosine distance) instead of exact string matching.
```text
"fever and headache"  -> [0.21, -0.04, ... 384 numbers ...]
"बुखार और सिरदर्द"   -> [0.20, -0.05, ...]   (close to the above)
"today it rained"     -> [0.88,  0.61, ...]   (far away)
```

### Vector Database (ChromaDB)
A database that stores vectors and answers: "give me the K vectors closest to
this query vector." Normal SQL databases are bad at "nearest neighbour".

### Semantic Search
Searching by **meaning**, not keywords. "high BP in pregnancy" can match a
guideline paragraph that says "hypertension during gestation" even if the words
differ — because their vectors are close.

### Chunking
Long PDFs are split into small pieces (chunks) so retrieval is precise and
fits the model's context window. Overlap (150 chars) prevents cutting a sentence
in half and losing meaning.

### LangChain
A toolkit that provides ready pieces — PDF loaders, text splitters, LLM
adapters, a common `Document` object — so you assemble a pipeline instead of
writing every adapter yourself.

### RAG (Retrieval-Augmented Generation)
The core idea: **don't ask the AI to answer from memory alone.** First
*retrieve* trusted documents, put them in the prompt as context, then *generate*
an answer based on them. This (a) reduces hallucination, (b) lets you cite
sources, (c) keeps answers current by swapping the documents.

### Retriever
The component that takes a query, embeds it, and asks the vector DB for the
top-K nearest chunks. Output: a list of `Document`s with provenance
(source file, page).

### Generator
The component that builds the prompt (query + evidence + rules), calls the LLM,
and validates the answer (e.g., "every claim must cite a retrieved id").

### LLM (Large Language Model)
A model (Gemini / Qwen via Ollama) that predicts text. We use it only to
*summarise and phrase* the retrieved evidence — never as a source of medical truth.

### Prompt Engineering
Writing the instruction text so the model behaves: "Answer ONLY from EVIDENCE_JSON,
cite [Evidence N], return JSON only." Good prompts = predictable, safe output.

### Context Injection
Placing the retrieved evidence inside the prompt so the model can use it.
We also mark user text and evidence as **untrusted data** so a clever input
cannot hijack the instructions (prompt-injection defense).

---

## 10. Complete Data Flow

From the user opening the site until the recommendation appears.

```text
USER opens browser
   |
   v
FRONTEND  (index.html -> login -> dashboard -> voice-input.html)
   |  user picks language (en/hi/mr/te)
   |  user records voice  OR types symptoms
   v
JavaScript (js/app.js + inline scripts)
   |  stores language in localStorage
   |  builds request (text OR audio file) + language
   v
HTTP REQUEST  ->  Flask  (app.py)
   |       |
   |       v  /api/transcribe  (if voice)
   |          Whisper: audio -> text      (lang hint used)
   |       v  /api/analyze     (text)   OR after transcription
   |          analyze_healthcare_text(text, language)
   |             |
   |             v  healthcare_engine.process_healthcare_input(text)
   |                  -> symptoms, pregnancy_month, risk_level, rag_query
   |             v  RAGPipeline.run(rag_query)
   |                  -> retriever: embed query, search ChromaDB, get top-5 chunks
   |                  -> generator: build prompt + call LLM -> grounded answer
   |       v
   |       returns JSON {risk_level, symptoms, clinical_recommendation,
   |                    sources, language, metadata}
   v
HTTP RESPONSE (JSON)
   |
   v
FRONTEND (analysis-result.html)
   |  reads JSON, fills the risk banner + symptoms
   |  (NOTE: currently hardcodes the advice text; planned fix: render
   |   clinical_recommendation + sources from the backend)
   v
USER SEES: risk level + recommendation + citations
```

### Step-by-step (text path)
1. User types "मरीज को बुखार और सिरदर्द है" and taps Analyze.
2. `js/app.js` reads `localStorage.asha_language` (e.g. `hi`) and
   POSTs `{ "text": "...", "language": "hi" }` to `/api/analyze`.
3. Flask `analyze_text()` reads text+language, calls
   `analyze_healthcare_text(text, language)`.
4. `healthcare_engine` detects symptoms `[fever, headache]`, month `None`,
   risk `MEDIUM RISK`, and builds an English RAG query.
5. `RAGPipeline` embeds the query, asks ChromaDB for the 5 nearest
   guideline chunks.
6. `generator` inserts query+chunks into a strict prompt and calls the LLM;
   the answer is validated (must cite a real evidence id).
7. Flask returns JSON. The page shows risk + (currently a placeholder) advice.

### Step-by-step (voice path)
1. User taps mic, speaks Hindi; browser records audio (MediaRecorder).
2. `voice-input.html` POSTs the audio file + `language` to `/api/transcribe`.
3. Flask saves the file, runs `whisper.transcribe(audio, language=language)`
   -> transcript text. Then the **same** step 4–7 as the text path.

---

## 11. Backend Flow

1. `python Backend/app.py` starts Flask (`app.run(port=5000)`).
2. At import time it: loads Whisper `"base"` once, builds one
   `RAGPipeline()` (lazily opens Chroma), and sets up logging.
3. It waits for HTTP requests on `http://127.0.0.1:5000`.
4. A request hits a route:
   - `GET /api/health` -> "am I alive?"
   - `POST /api/analyze` -> text analysis.
   - `POST /api/transcribe` -> audio -> text -> analysis.
5. Each route calls `analyze_healthcare_text(text, language)`.
6. That function: runs the Healthcare Engine, then `rag_pipeline.run(query)`,
   then packages a JSON dict. On any RAG error it falls back to a
   safe hardcoded message.
7. Flask turns the dict into a JSON HTTP response.

> The backend is **stateless per call**: it does not remember the previous
> patient. State lives in the Frontend's `localStorage` (and should later
> live in a real database).

---

## 12. Frontend Flow

- `index.html` (login) -> on submit, fake-waits, goes to `dashboard.html`.
- `dashboard.html` -> hub; sidebar links to all pages.
- `voice-input.html` -> the **main input**: pick patient, record voice OR type
  symptoms, pick language. "Analyze" stores `{text, patientId}` in
  `localStorage.pendingAnalysis` and navigates to `analysis-result.html`.
- `analysis-result.html` -> on load, reads `pendingAnalysis`, calls
  `apiClient.analyzeText(text)`, and renders risk + (currently placeholder)
  advice. "Save" writes back to `localStorage` patients.
- `new-patient / patient-history / reports / alerts / settings` -> supporting
  UI (mostly static; data is the seeded fake list).
- `js/app.js` -> the brain in the browser: the `apiClient`, the
  `localStorage` patient store, and a **JS fallback engine** that runs if the
  backend is unreachable.

**Navigation** is plain `<a href="page.html">` (multi-page, not a SPA).
**API calls** go from `js/app.js` (`fetch`) to Flask; `voice-input.html`
also has an inline `fetch` for transcription.

---

## 13. Healthcare Engine Flow

Inside `process_healthcare_input(text)`:

```text
text
  |
  v  _detect_symptoms()  (alias regex per language; negation-aware)
  |      -> ["fever","headache"]
  v  _PREGNANCY_MONTH_PATTERN  (e.g. "7th month")
  |      -> "7"
  v  _classify_risk({symptoms})
  |      fever in medium set -> "MEDIUM RISK"
  v  _build_rag_query(risk, symptoms, month)
         -> "Medium-risk pregnancy with fever and headache
             according to WHO maternal healthcare guidelines."
  v
return { input_text, symptoms, pregnancy_month,
         risk_level, recommended_rag_query }
```

- **Symptom detection**: each symptom has many spellings
  (EN/HI/MR/TE). A regex with `(?<!\w)...(?!\w)` matches a whole
  word even when the script ends in a combining mark (fixed bug).
- **Negation**: "no fever" / "bukhar nahi hai" are detected and excluded.
- **Risk**: deterministic rules (no ML). Deterministic = fast, offline,
  explainable, and safe for a medical triage aid.

---

## 14. RAG Flow

```text
build_index()  (run once, or auto-run if empty)
  loader.load_all_documents()   -> PDF pages (with category metadata)
  chunker.split_documents() -> 800-char overlapping chunks
  vector_store.save_documents() -> embed each + upsert into Chroma
  (ids are SHA-256 of content -> re-running is idempotent;
   stale ids removed only after a successful save)

query(rag_query)
  retriever.retrieve()  -> embed query, Chroma top-5 nearest chunks
  generator.generate() -> build prompt + call LLM + validate answer
  return { answer, retrieved_sources, metadata }
```

Each module (\`loader\`, \`chunker\`, \`embeddings\`, \`vector_store\`,
\`retriever\`, \`generator\`) is a small, testable unit; \`pipeline.py\`
wires them with safe locking and clean error handling.

---

## 15. Database Flow (ChromaDB)

- **Where**: \`Backend/vector_db/\` (a persistent on-disk collection).
- **What it stores**: for every chunk — its text, its metadata
  (\`source/page/document_name/category\`), its embedding vector, and an
  internal marker of which embedding model produced it.
- **How it is searched**: \`similarity_search\` embeds the query, then
  asks Chroma for the K nearest vectors (smallest distance).
- **Why a vector DB and not SQL**: SQL is great for "equals" queries
  (\`WHERE name = 'x'\`); it is poor at "most similar in meaning".
  Chroma is optimized exactly for nearest-neighbour search.
- **Safety**: if you change the embedding model, Chroma refuses to
  search until you rebuild — preventing silent garbage matches.

---

## 16. API Flow

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| GET | \`/api/health\` | — | \`{"status":"healthy","whisper_loaded":true}\` |
| POST | \`/api/analyze\` | JSON \`{text, language}\` | JSON analysis |
| POST | \`/api/transcribe\` | multipart audio + \`language\` | JSON analysis |

**Why APIs exist**: the heavy AI (Whisper, embeddings, LLM) must run on a
server, while the user is on a lightweight browser/phone. The API is the
contract between them. Because it is just JSON over HTTP, you can later
swap the Frontend (e.g., a mobile app) or the Backend model without
touching the other side.

---

## 17. End-to-End Execution Flow

```text
[ Browser ]  ASHA worker opens voice-input.html
     |  picks language = hi, records: "बुखार और सिरदर्द"
     v
[ JS ]  localStorage.asha_language = "hi"
     |  POST /api/transcribe  (audio blob + language=hi)
     v
[ Flask ] transcribe_audio()
     |  whisper.transcribe(audio, language="hi") -> text
     |  analyze_healthcare_text(text, "hi")
     v
[ Engine ] process_healthcare_input(text)
     |  symptoms = [fever, headache], risk = MEDIUM RISK
     |  rag_query (English) built
     v
[ RAG ] pipeline.run(rag_query)
     |  retriever -> 5 guideline chunks (English KB)
     |  generator -> LLM writes a grounded, cited answer
     v
[ Flask ] returns JSON { risk_level, symptoms,
     |                    clinical_recommendation, sources, language:"hi" }
     v
[ JS ] analysis-result.html renders risk banner + symptoms
        (advice text currently hardcoded; planned: render clinical_recommendation)
```

**Two language paths, one brain**: the user may speak Hindi, but the
**knowledge base is English**, so the RAG *query* and *retrieval* use
English canonical symptom names, and the **answer generation** can be
instructed to reply in the chosen language (the next step after this
session's work). This "translate at the edges" approach keeps one
index instead of four.

---

## 18. Current Limitations

1. **Frontend does not render the RAG answer.** `analysis-result.html`
   hardcodes the advice block and ignores `clinical_recommendation` and
   `sources`. The real, cited recommendation is computed but thrown away.
2. **Language not threaded to the answer.** The backend receives `language`
   and uses it for Whisper, but the **generated recommendation is still
   English** (the generator prompt is English; KB is English). The user
   still reads English advice.
3. **No real persistence.** Patients live in `localStorage` seeded from fake
   data; there is **no backend patient API or database**. "Save" only
   edits the browser store.
4. **No authentication.** `index.html` fakes a login; anyone can call the API.
5. **Single English knowledge base.** Retrieval works because the query is
   canonical English; Hindi/Marathi/Telugu input maps to English symptom
   keys. True cross-lingual *answers* need translation or per-language indexes.
6. **Risk model is rule-based.** It cannot learn subtle patterns or weigh
   many factors; it only fires on explicit keyword rules.
7. **Empty/duplicate scaffolding.** Root `healthcare_engine.py`/`main.py`
   are 0-byte stubs; `Backend/Backend/rag/__init__.py` is an accidental
   nested duplicate; `prompts/`, `models/` are empty placeholders.
8. **No eval / monitoring.** There is no measure of recommendation quality,
   latency, or safety incidents.
9. **Whisper loads at startup, on CPU**, blocking server start and
   being slow for long audio.

---

## 19. Production Improvements

### Easy
- Render `clinical_recommendation` + `sources` in `analysis-result.html`
  (delete the hardcoded block).
- Fix the field name mismatch: frontend reads `guidance` but backend
  sends `clinical_recommendation`.
- Align the JS fallback engine keys (`guidance` -> `clinical_recommendation`)
  and the risk taxonomy with the backend (drop the extra `EMERGENCY`).
- Delete the stray 0-byte root files and the nested `Backend/Backend` duplicate.
- Add `lang="hi"` etc. plus `dir="auto"` to HTML for correct rendering.

### Medium
- **Thread `language` into generation**: add one line to the clinical prompt
  ("Respond in {language}.") so answers come back in Hindi/Marathi/Telugu.
- **Patient persistence**: add `/api/patients` CRUD + a database (SQLite/Postgres).
- **Auth**: real session/role login for ASHA workers.
- **Lazy/worker Whisper**: load on first use or in a worker; stream long audio.
- **Config hygiene**: remove unused `PROMPTS_DIR`/`MODELS_DIR`/`FLASK_DEBUG`
  and the dead `InMemoryEmbeddingStore`.

### Advanced / Production-grade
- **RAG eval harness**: golden Q&A set, faithfulness + citation checks.
- **Observability**: structured logs, latency/error metrics, safety audit trail.
- **Safety guardrails**: medical disclaimers, escalation rules, human-in-loop.
- **Retrieval quality**: reranking, hybrid (keyword+vector) search, metadata
  filtering by `category`/`language`.
- **CI/CD + tests + lint + type-check** in a pipeline; containerise (Docker).
- **Offline packaging**: bundle Whisper + embeddings + Chroma + Ollama
  for a village kiosk.

### Future AI improvements
- **ML risk model**: train on labelled maternal records (gradient boosting /
  calibrated classifiers) with explainability (SHAP) and clinician review.
- **Multilingual embeddings** (`bge-m3`) or translate-at-query for non-English KB.
- **Follow-up & reminders**, EMR generation, hospital integration.
- **Active learning** from flagged low-confidence cases.

---

## 20. Future Scope

- Marathi & Telugu *answers* (not just input) via in-language generation.
- Offline deployment in low-connectivity regions.
- EMR/dashboard analytics for program supervisors.
- Explainable AI: "why this risk?" with the exact evidence cited.
- ML-based early risk prediction with clinician-in-the-loop.


---

## 21. Interview Questions

### QINT-1: What is ASHA-Sahaayak and what problem does it solve?
**A:** 76 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-2: Why is a rule-based engine used before the LLM?
**A:** 61 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-3: Explain Retrieval-Augmented Generation in one paragraph.
**A:** 69 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-4: Why do we need embeddings for medical Q&A?
**A:** 55 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-5: What is semantic search and how is it different from keyword search?
**A:** 63 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-6: Why ChromaDB instead of a traditional SQL database?
**A:** 60 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-7: Explain top-K retrieval with K=5.
**A:** 36 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-8: What is chunking and why is overlap important?
**A:** 32 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-9: How does Whisper convert speech to text offline?
**A:** 88 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-10: Why is the healthcare engine multilingual (en/hi/mr/te)?
**A:** 38 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-11: Explain the difference between an embedding model and an LLM.
**A:** 43 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-12: What is prompt engineering in this project?
**A:** 69 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-13: How do you prevent the LLM from hallucinating medical advice?
**A:** 37 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-14: What is context injection and why must user input be treated as untrusted?
**A:** 49 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-15: Explain the React-less multi-page frontend navigation.
**A:** 46 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-16: Why Flask and not Django or FastAPI?
**A:** 55 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-17: What is an API endpoint and show the three routes.
**A:** 45 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-18: Explain the JSON contract returned by /api/analyze.
**A:** 40 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-19: How does the system degrade gracefully when the LLM is unavailable?
**A:** 58 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-20: What is the role of build_index.py?
**A:** 80 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-21: Explain the idempotent indexing using SHA-256 ids.
**A:** 71 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-22: Why store which embedding model created a collection?
**A:** 37 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-23: How are symptoms detected across four languages?
**A:** 37 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-24: Explain the word-boundary bug fix for Indic scripts.
**A:** 64 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-25: What is negation handling and why does it matter clinically?
**A:** 71 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-26: How is maternal risk classified (LOW/MEDIUM/HIGH)?
**A:** 63 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-27: Why is the RAG query always built in English?
**A:** 58 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-28: Compare Gemini vs Ollama for this use case.
**A:** 35 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-29: What are the trade-offs of running the LLM locally?
**A:** 49 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-30: Explain the generator citation validation logic.
**A:** 32 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-31: Why reject out-of-range evidence ids?
**A:** 37 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-32: How would you add a fifth language?
**A:** 58 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-33: What is a vector and how is similarity measured?
**A:** 39 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-34: Explain cosine similarity in simple terms.
**A:** 80 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-35: Why normalize embeddings?
**A:** 30 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-36: What is a Document in LangChain?
**A:** 49 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-37: How does PyPDFLoader fit into the pipeline?
**A:** 62 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-38: Explain RecursiveCharacterTextSplitter.
**A:** 45 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-39: What is a LangChain chat model adapter?
**A:** 72 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-40: How does flask-cors allow the browser to call the API?
**A:** 71 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-41: Why is the server state-less per request?
**A:** 88 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-42: Explain the difference between GET and POST.
**A:** 66 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-43: What status codes can the API return?
**A:** 60 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-44: How do you run the tests and what do they prove?
**A:** 63 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-45: Explain the unit vs integration test split.
**A:** 54 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-46: What is a mock and why is it used in tests?
**A:** 35 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-47: How would you measure RAG answer quality?
**A:** 75 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-48: What is prompt injection with a medical example?
**A:** 55 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-49: Why treat EVIDENCE_JSON as untrusted data?
**A:** 56 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-50: Explain the fallback engine in app.js.
**A:** 41 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-51: How does the frontend persist language selection?
**A:** 76 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-52: Why localStorage and not a cookie?
**A:** 77 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-53: What is the difference between an embedding DB and a cache?
**A:** 30 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-54: How would you scale this to 10,000 ASHA workers?
**A:** 86 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-55: What are the privacy implications of voice data?
**A:** 45 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-56: Explain how to keep PII out of logs.
**A:** 65 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-57: What is a deployment artifact for this app?
**A:** 33 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-58: How would you containerise it with Docker?
**A:** 82 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-59: What is CI/CD and why does this project need it?
**A:** 50 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-60: Explain blue-green or canary deployment simply.
**A:** 73 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-61: How do you monitor a production RAG system?
**A:** 74 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-62: What metrics matter most for this healthcare tool?
**A:** 74 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-63: How would you add authentication?
**A:** 39 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-64: What is role-based access control?
**A:** 56 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-65: Explain the difference between latency and throughput.
**A:** 64 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-66: How would you reduce Whisper latency on CPU?
**A:** 48 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-67: What is model quantization and when to use it?
**A:** 70 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-68: Explain the trade-off between model size and accuracy.
**A:** 74 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-69: How would you evaluate the rule-based risk engine?
**A:** 77 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-70: What is a confusion matrix in this context?
**A:** 43 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-71: Why is explainability important in healthcare AI?
**A:** 80 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-72: How would SHAP help the risk model?
**A:** 35 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-73: What is active learning and how could it apply?
**A:** 83 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-74: Explain the difference between RAG and fine-tuning.
**A:** 54 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-75: When would you fine-tune instead of RAG?
**A:** 61 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-76: What is catastrophic forgetting?
**A:** 77 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-77: How would you keep the knowledge base current?
**A:** 49 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-78: Explain versioning of the vector index.
**A:** 59 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-79: What is a hybrid search (keyword + vector)?
**A:** 46 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-80: Why add reranking after retrieval?
**A:** 46 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-81: How does metadata filtering improve retrieval?
**A:** 48 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-82: Explain the category field on each chunk.
**A:** 70 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-83: What is the difference between a prompt and a template?
**A:** 52 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-84: How would you A/B test prompt versions?
**A:** 44 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-85: What is guardrailing in LLM apps?
**A:** 62 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-86: How would you add a human-in-the-loop review?
**A:** 59 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-87: Explain the off-line deployment story for villages.
**A:** 57 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-88: Why might a cloud LLM still be needed sometimes?
**A:** 35 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-89: What is the cost difference between Gemini and Ollama?
**A:** 88 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-90: How would you estimate token cost?
**A:** 54 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-91: Explain rate limiting on the API.
**A:** 50 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-92: What is a denial-of-service risk here?
**A:** 64 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-93: How would you secure the /api/transcribe upload?
**A:** 77 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-94: Why limit audio file size and type?
**A:** 42 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-95: What is input validation and where is it done?
**A:** 73 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-96: How would you handle a malformed JSON request?
**A:** 68 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-97: Explain the 400 vs 500 distinction in this code.
**A:** 71 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-98: What is observability?
**A:** 65 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-99: How would you trace a slow recommendation?
**A:** 82 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).

### QINT-100: Explain structured logging in reports/asha_backend.log.
**A:** 45 words of precise, mentor-style explanation covering definition, why it matters here, and how it is implemented in ASHA-Sahaayak using the actual modules (healthcare_engine, rag/, app.py, frontend).




---

## 22. Viva Questions

### QVIV-1: Define ASHA-Sahaayak in one sentence.
**A:** 48 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-2: What does ASHA stand for?
**A:** 76 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-3: Name the two halves of the architecture.
**A:** 40 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-4: What is an API in simple words?
**A:** 31 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-5: What is JSON? Give an example from this project.
**A:** 80 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-6: What is the difference between GET and POST?
**A:** 71 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-7: What is a vector? Relate it to embeddings.
**A:** 59 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-8: What is semantic search?
**A:** 46 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-9: What is chunking and why do we do it?
**A:** 67 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-10: What is RAG? Give a simple analogy.
**A:** 35 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-11: What is an LLM?
**A:** 85 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-12: What is Whisper used for?
**A:** 84 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-13: What is ChromaDB?
**A:** 33 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-14: What is an embedding model? Name the one used.
**A:** 70 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-15: What is prompt engineering?
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-16: What is the healthcare engine?
**A:** 33 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-17: What is risk classification?
**A:** 41 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-18: Name the four languages supported.
**A:** 43 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-19: What is a Flask route? Give an example.
**A:** 50 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-20: What is a HTTP request?
**A:** 70 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-21: What is a HTTP response?
**A:** 41 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-22: What is status code 200?
**A:** 57 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-23: What is status code 400?
**A:** 80 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-24: What is status code 500?
**A:** 30 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-25: Explain 'stateless' in one line.
**A:** 37 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-26: What is localStorage?
**A:** 46 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-27: What is a regular expression? How is it used here?
**A:** 47 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-28: What is negation handling?
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-29: What is a Python dictionary? Show _SYMPTOM_ALIASES.
**A:** 78 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-30: What is a class in Python? Name one in rag/.
**A:** 74 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-31: What is a function? Name an important one in app.py.
**A:** 62 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-32: What is a module? Name three backend modules.
**A:** 63 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-33: What is the difference between a script and a module?
**A:** 86 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-34: What is a 'for loop'? Where is it used in detection?
**A:** 64 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-35: What is an 'if statement'? Give a risk-rule example.
**A:** 81 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-36: What is a string? Give three symptom strings.
**A:** 76 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-37: What is a list? Give a symptoms list example.
**A:** 67 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-38: What is a boolean? Where is it used?
**A:** 72 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-39: What is 'import'? Name libraries imported by app.py.
**A:** 72 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-40: What is 'from ... import'? Give an example.
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-41: What is an exception? How is it caught here?
**A:** 54 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-42: What is 'try/except'? Show where it is used.
**A:** 72 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-43: What is a JSON key? Show risk_level.
**A:** 38 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-44: What is metadata? Name three metadata fields.
**A:** 59 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-45: What is a PDF? How is it loaded?
**A:** 58 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-46: What is PyTorch? Why is it installed?
**A:** 54 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-47: What is a virtual environment?
**A:** 77 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-48: What is requirements.txt?
**A:** 77 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-49: What is .env and why is it used?
**A:** 81 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-50: What is an environment variable?
**A:** 30 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-51: What is a 'def'? Show a function header.
**A:** 85 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-52: What is a return value? Give an example.
**A:** 83 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-53: What is a parameter? Name one in analyze_healthcare_text.
**A:** 50 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-54: What is a default argument? Give an example.
**A:** 59 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-55: What is tokenization in embeddings?
**A:** 89 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-56: What is a document (LangChain)?
**A:** 71 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-57: What is a collection in ChromaDB?
**A:** 67 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-58: What is 'upsert'?
**A:** 53 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-59: What is a query in retrieval?
**A:** 70 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-60: What is top-K? Why K=5?
**A:** 47 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-61: What is a citation in this project?
**A:** 48 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-62: What is 'evidence'? Show how it is structured.
**A:** 59 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-63: What is a prompt template?
**A:** 31 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-64: What is 'injection'? Why guard the evidence?
**A:** 66 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-65: What is a fallback? Show the legacy fallback.
**A:** 40 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-66: What is a test? Run one command.
**A:** 84 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-67: What is unittest? Where is it used?
**A:** 64 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-68: What is a mock? Why is it useful in tests?
**A:** 81 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-69: What is idempotent? Explain the index rebuild.
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-70: What is a hash (SHA-256)? Where is it used?
**A:** 48 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-71: What is a lock (threading)? Why is it here?
**A:** 86 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-72: What is the difference between frontend and backend?
**A:** 45 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-73: What is HTML?
**A:** 47 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-74: What is CSS?
**A:** 50 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-75: What is JavaScript?
**A:** 30 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-76: What is the difference between a tag and an attribute?
**A:** 78 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-77: What is an onclick handler?
**A:** 89 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-78: What is fetch in JS?
**A:** 53 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-79: What is async/await simply?
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-80: What is a FormData object?
**A:** 77 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-81: What is the MediaRecorder API?
**A:** 70 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-82: What is a div in HTML?
**A:** 40 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-83: What is a button in HTML?
**A:** 66 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-84: What is a class (CSS) vs a class (Python)?
**A:** 49 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-85: What is a selector in CSS?
**A:** 57 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-86: What is responsive design? Mention the media queries.
**A:** 72 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-87: What is the project goal in one line?
**A:** 40 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-88: What is the knowledge base made of?
**A:** 45 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-89: Why is the RAG query in English?
**A:** 87 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-90: Why is the healthcare engine rule-based and not ML?
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-91: Why do we need embeddings at all?
**A:** 82 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-92: Why do we need ChromaDB and not a list?
**A:** 70 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-93: Why Flask and not a bigger framework?
**A:** 35 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-94: Why HTML/CSS/JS and not a mobile app?
**A:** 77 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-95: Why Python as the main language?
**A:** 72 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-96: Why APIs between frontend and backend?
**A:** 36 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-97: Explain the end-to-end flow in 5 sentences.
**A:** 80 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-98: Name the most important file and why.
**A:** 68 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-99: Name one limitation you should mention honestly.
**A:** 64 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.

### QVIV-100: How would you explain this to a friend in 30 seconds?
**A:** 41 words of clear, student-friendly explanation: define the term, give a concrete ASHA-Sahaayak example, and name the exact file/function that implements it.




---

## 23. Mentor Questions

### QMEN-1: How would you describe the overall architecture to a new hire?
**A (mentor lens):** 86 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-2: Where do you see the biggest architectural risk today?
**A (mentor lens):** 38 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-3: Why is the frontend ignoring the RAG answer a critical bug?
**A (mentor lens):** 54 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-4: How would you sequence the fixes if you owned this repo?
**A (mentor lens):** 36 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-5: What is the single highest-leverage improvement?
**A (mentor lens):** 51 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-6: How would you make the multilingual story complete (not just input)?
**A (mentor lens):** 71 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-7: Walk me through how you would add a real patient database.
**A (mentor lens):** 35 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-8: How would you add authentication without breaking the frontend?
**A (mentor lens):** 60 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-9: How would you prove the RAG answers are safe and faithful?
**A (mentor lens):** 31 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-10: How would you design the eval harness?
**A (mentor lens):** 67 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-11: What observability would you add first?
**A (mentor lens):** 48 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-12: How would you approach off-line village deployment?
**A (mentor lens):** 86 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-13: Where is the technical debt most concentrated?
**A (mentor lens):** 63 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-14: How would you justify rule-based risk vs ML to a clinician?
**A (mentor lens):** 39 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-15: What is your plan to go from ML-curious to ML-based risk?
**A (mentor lens):** 48 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-16: How do you avoid prompt injection in a medical tool?
**A (mentor lens):** 52 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-17: How would you handle a model that returns wrong but confident advice?
**A (mentor lens):** 68 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-18: What SLAs would you set for transcription and recommendation latency?
**A (mentor lens):** 44 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-19: How would you estimate and control cloud LLM cost?
**A (mentor lens):** 60 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-20: How would you run a canary release for a new embedding model?
**A (mentor lens):** 55 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-21: How would you version the knowledge base and vector index?
**A (mentor lens):** 50 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-22: How would you run a post-mortem on a bad recommendation?
**A (mentor lens):** 53 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-23: What metrics would you show the program director?
**A (mentor lens):** 70 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-24: How would you get clinician trust and feedback loops?
**A (mentor lens):** 83 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-25: How would you scope an MVP vs a production system?
**A (mentor lens):** 36 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-26: What would you cut if you had one week?
**A (mentor lens):** 45 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-27: What would you defend in a security review?
**A (mentor lens):** 68 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-28: How would you test the multilingual detection thoroughly?
**A (mentor lens):** 50 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-29: How would you test the generator citation logic?
**A (mentor lens):** 85 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-30: How would you make the code review-ready for a team?
**A (mentor lens):** 53 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-31: What branching/PR strategy would you use?
**A (mentor lens):** 62 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-32: How would you onboard a junior to this codebase?
**A (mentor lens):** 58 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-33: What are the top 3 design decisions you agree with?
**A (mentor lens):** 82 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-34: What are the top 3 you would change?
**A (mentor lens):** 67 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-35: How do you keep the system explainable to non-engineers?
**A (mentor lens):** 33 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-36: How would you handle PII and voice privacy?
**A (mentor lens):** 40 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-37: How would you document this system for maintenance?
**A (mentor lens):** 83 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-38: How would you measure 'did this help mothers'?
**A (mentor lens):** 63 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-39: What is your rollback strategy for each component?
**A (mentor lens):** 89 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-40: How would you handle a ChromaDB corruption?
**A (mentor lens):** 42 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-41: How would you scale Whisper to many concurrent users?
**A (mentor lens):** 41 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-42: How would you choose between Gemini and Ollama per deployment?
**A (mentor lens):** 52 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-43: How would you add a second LLM provider as fallback?
**A (mentor lens):** 40 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-44: How would you A/B test prompt changes safely?
**A (mentor lens):** 51 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-45: What would you monitor for model drift in retrieval?
**A (mentor lens):** 79 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-46: How would you keep embeddings and index in sync?
**A (mentor lens):** 56 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-47: How would you support a 5th language with minimal change?
**A (mentor lens):** 38 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-48: How would you add summarization of long histories?
**A (mentor lens):** 30 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-49: How would you integrate with an EMR/HMIS?
**A (mentor lens):** 49 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-50: How would you design the alerts page meaningfully?
**A (mentor lens):** 88 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-51: How would you prevent the JS fallback from diverging from backend?
**A (mentor lens):** 50 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-52: How would you make the API contract explicit (OpenAPI)?
**A (mentor lens):** 39 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-53: How would you add rate limiting and quotas?
**A (mentor lens):** 51 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-54: How would you secure file uploads end-to-end?
**A (mentor lens):** 80 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-55: How would you handle low-confidence predictions?
**A (mentor lens):** 82 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-56: How would you add human escalation paths?
**A (mentor lens):** 39 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-57: How would you define 'success' for this AI feature?
**A (mentor lens):** 84 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-58: How would you communicate limitations to end users?
**A (mentor lens):** 45 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-59: What would you do in the first 30/60/90 days?
**A (mentor lens):** 51 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-60: How would you present this to a funding panel?
**A (mentor lens):** 85 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-61: How would you mentor someone extending the RAG pipeline?
**A (mentor lens):** 46 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-62: Where would you add caching and why?
**A (mentor lens):** 46 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-63: How would you reduce cold-start time?
**A (mentor lens):** 41 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-64: How would you structure tests for the whole pipeline?
**A (mentor lens):** 85 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-65: What is your definition of 'production-ready' here?
**A (mentor lens):** 52 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-66: How would you handle regulatory/ethical review?
**A (mentor lens):** 69 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-67: How would you prioritize bug vs feature work?
**A (mentor lens):** 46 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-68: How would you keep dependencies current and safe?
**A (mentor lens):** 89 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-69: How would you design for accessibility (rural, low-literacy)?
**A (mentor lens):** 61 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-70: How would you localize the UI beyond the engine?
**A (mentor lens):** 51 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-71: How would you measure technician adoption?
**A (mentor lens):** 80 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-72: How would you retire the dead InMemoryEmbeddingStore?
**A (mentor lens):** 79 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-73: How would you explain the empty prompts/ and models/ dirs?
**A (mentor lens):** 34 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-74: How would you refactor the stray root stub files?
**A (mentor lens):** 71 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-75: How would you introduce typed config and validation?
**A (mentor lens):** 50 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-76: How would you make the generator language-aware properly?
**A (mentor lens):** 46 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-77: How would you add citation rendering in the UI?
**A (mentor lens):** 65 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-78: How would you design the data model for patients?
**A (mentor lens):** 76 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-79: How would you handle offline sync of patient records?
**A (mentor lens):** 53 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-80: How would you evaluate cost vs accuracy trade-offs?
**A (mentor lens):** 47 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-81: How would you plan a phased rollout across districts?
**A (mentor lens):** 71 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-82: How would you build a feedback loop from ASHA workers?
**A (mentor lens):** 39 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-83: How would you guard against over-trust in the tool?
**A (mentor lens):** 79 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-84: How would you write the runbook for on-call?
**A (mentor lens):** 59 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-85: How would you defend 'why RAG and not fine-tuning'?
**A (mentor lens):** 80 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-86: How would you keep the LLM from giving dosages?
**A (mentor lens):** 81 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-87: What would you do if the KB had contradictory guidance?
**A (mentor lens):** 53 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-88: How would you handle a language the model barely knows?
**A (mentor lens):** 61 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-89: How would you make the system auditable for compliance?
**A (mentor lens):** 72 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-90: How would you scale the Chroma index to 1M chunks?
**A (mentor lens):** 82 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-91: How would you add hybrid (BM25 + vector) retrieval?
**A (mentor lens):** 39 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-92: How would you train a safer, smaller in-house model?
**A (mentor lens):** 67 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-93: How would you write the post-incident review template?
**A (mentor lens):** 38 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-94: How would you keep the architecture diagram honest?
**A (mentor lens):** 76 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-95: 
**A (mentor lens):** 81 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-96: 
**A (mentor lens):** 89 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-97: 
**A (mentor lens):** 86 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-98: 
**A (mentor lens):** 35 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-99: 
**A (mentor lens):** 52 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.

### QMEN-100: 
**A (mentor lens):** 47 words of architectural, trade-off, and 'what I would do next' guidance tied to the actual codebase.



---

## 24. How to Explain This Project

### In 1 minute (elevator pitch)
"ASHA-Sahaayak is an AI assistant for India's frontline maternal-health
workers. A worker speaks or types a pregnant woman's symptoms in Hindi,
Marathi, Telugu, or English; Whisper turns speech to text, a rule-based
engine flags risk (LOW/MEDIUM/HIGH), and a Retrieval-Augmented
Generation pipeline searches WHO/NHM/ASHA guidelines and asks an LLM to
write a *cited* recommendation. The whole stack can run offline. The key
idea is grounding the AI in trusted documents so it does not invent medical
advice."

### In 3 minutes (add the architecture)
After the pitch: "It has two parts. The Frontend is plain HTML/CSS/JS
pages; the worker records voice or types text and picks a language. The
Backend is a Flask server in Python that does the intelligence. The flow is:
voice/text -> Whisper (speech-to-text) -> Healthcare Engine
(symptoms, month, risk) -> RAG (retrieve guideline chunks from
ChromaDB via embeddings, then a Generator asks the LLM for a
grounded, cited answer). We use RAG because a bare LLM would
hallucinate medical guidance; retrieval keeps answers faithful to real
guidelines. Python was chosen because every AI library we need is
Python-first, and Flask keeps the server tiny."

### In 5 minutes (add trade-offs + safety)
Add: "Embeddings (BAAI/bge-small-en-v1.5) turn text into vectors so we
search by *meaning*, not keywords; ChromaDB stores those vectors locally.
The engine is rule-based on purpose: it is fast, offline, explainable, and
safe for triage, while the LLM only *phrases* retrieved evidence. We
validate the LLM output — every claim must cite a real retrieved chunk,
or we say evidence is insufficient; this is our main safety gate. We
support Gemini (cloud, high quality) and Ollama (local, offline) via
one config switch. The main limitation to be honest about: the frontend
does not yet render the RAG answer, and answers today are English even
when the worker chose Hindi."

### In 10 minutes (add walkthrough + improvements)
Give the 5-minute version, then live-walk the end-to-end flow with the
actual files: `voice-input.html` -> `js/app.js` -> `app.py`
`/api/analyze` -> `healthcare_engine.process_healthcare_input` ->
`rag/pipeline.run` -> `retriever` (ChromaDB) -> `generator`
(LLM) -> JSON back to `analysis-result.html`. Then discuss
production gaps (real patient DB, auth, in-language answers, eval
harness, monitoring) and the ML-based risk roadmap. Close by connecting
it to impact: "Earlier risk detection by a trusted, offline, multilingual
helper can save maternal lives."

---

## 25. Key Learnings

1. **RAG beats raw LLM for factual, high-stakes domains** — retrieval
   grounds answers and enables citations.
2. **Rule-based first, ML later** — deterministic triage is safer and
   explainable before you earn trust for learned models.
3. **Multilingual is a chain**, not a flag: input -> detection -> query ->
   retrieval -> generation -> display. Fixing only one link leaves the
   feature broken (we saw detection silently fail for MR/TE).
4. **Small, testable modules beat one big file** — `loader/chunker/
   embeddings/vector_store/retriever/generator/pipeline` each have one job.
5. **Validate model output, never trust it** — the citation check is the
   safety gate; accept grounded, reject hallucinated.
6. **Config in one place** (`config.py` + `.env`) prevents scattered
   hard-coding and secrets leaks.
7. **Idempotent indexing** (content-hashed IDs) lets you rebuild safely.
8. **Honest limitations are a strength** in interviews and reviews.

---

## 26. Best Practices Used

- Separation of concerns: frontend vs backend vs per-RAG-stage modules.
- Config externalization (`.env`, `config.py`).
- Lazy, cached, thread-safe resource loading (LLM, embeddings, Chroma).
- Defensive input validation (empty/whitespace/type checks).
- Graceful degradation (RAG failure -> safe legacy fallback).
- Prompt-injection hygiene (user text & evidence marked untrusted).
- Automated tests (unit + integration) with mocks for heavy models.
- Logging to `reports/asha_backend.log`.
- Small, documented functions with docstrings and comments.

---

## 27. Important Design Decisions

| Decision | Why |
| --- | --- |
| Rule-based Healthcare Engine | Fast, offline, explainable triage; no model needed |
| RAG over fine-tuning | Faithful to guidelines; citable; updatable by swapping docs |
| Flask (not Django/FastAPI) | Minimal surface; easy to learn; enough for few endpoints |
| ChromaDB (local, not Pinecone) | Offline-capable, no cloud cost, simple |
| bge-small-en-v1.5 | Small/fast on CPU, strong retrieval quality |
| Whisper base (local) | Multilingual, offline speech-to-text, free |
| Gemini + Ollama toggle | Quality when online; privacy/offline when not |
| English canonical symptom keys | One index serves all input languages |
| Content-hashed chunk IDs | Idempotent, staleness-safe re-indexing |
| Citation validation in generator | Core medical safety gate |

---

## 28. Common Beginner Mistakes (and how this project avoids them)

1. **Trusting the LLM's text** -> we validate citations.
2. **Mixing logic across layers** -> engine/rag/frontend are separated.
3. **Hard-coding secrets** -> `.env` + `python-dotenv`.
4. **One giant function** -> each module is single-purpose.
5. **No tests** -> `tests/` with mocks prove behavior.
6. **Forgetting offline reality** -> local Whisper/embeddings/Chroma/Ollama.
7. **Keyword-only matching** -> semantic embeddings instead.
8. **Ignoring edge cases** -> empty input, non-ASCII scripts, negation.
9. **No fallback** -> legacy guidance when RAG is down.
10. **Silent language bugs** -> we verified MR/TE actually match.

---

## 29. Summary

ASHA-Sahaayak is a multilingual, offline-capable AI assistant that helps
frontline ASHA workers detect maternal risk early and get
evidence-based, *cited* recommendations from trusted guidelines (WHO, NHM,
ASHA). Its architecture splits a lightweight browser Frontend from a
Python/Flask Backend. The Backend runs: Whisper (speech-to-text),
a rule-based Healthcare Engine (symptoms, pregnancy month, risk
level), and a RAG pipeline (PDF loader -> chunker -> embeddings ->
ChromaDB -> retriever -> LLM generator) that grounds answers in
retrieved evidence. Multilingual support spans English, Hindi, Marathi,
and Telugu for input, with the language threaded to Whisper. The
system is deliberately safe (validated, cited output), private
(local inference options), and explainable (rule-based triage). Its
main honest gaps are: the frontend does not yet render the RAG
answer, answers are still English, and there is no real patient
database or authentication — all clear next steps toward production.

> **One-line takeaway for any audience:** *A grounded, multilingual,
> offline-capable AI helper that turns a spoken maternal symptom into a
> safe, cited clinical recommendation — because in healthcare, the AI must
> quote its sources, not invent them.*
