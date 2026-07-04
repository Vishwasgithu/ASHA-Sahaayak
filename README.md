# ASHA Sahaayak

ASHA Sahaayak includes a Flask backend, Whisper transcription, maternal-risk
screening, and an evidence-grounded RAG pipeline over the PDFs in
`Backend/knowledge_base`.

## Windows Setup

Run these commands from PowerShell in the repository root.

### 1. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

If the virtual environment does not exist yet:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r Backend\requirements.txt
```

The first embedding run downloads `BAAI/bge-small-en-v1.5`, so an internet
connection and sufficient disk space are required.

### 3. Configure Gemini

Create the local environment file from the tracked example:

```powershell
Copy-Item Backend\.env.example Backend\.env
notepad Backend\.env
```

Set your Gemini API key in `Backend/.env`:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=replace_with_your_key
LLM_MODEL_NAME=gemini-2.0-flash
```

`Backend/.env` is ignored by Git and must not be committed.

### 4. Build the ChromaDB index

```powershell
python Backend\build_index.py
```

This loads the PDFs, creates chunks and embeddings, and stores the vectors in
`Backend/vector_db`.

### 5. Test RAG

```powershell
python Backend\test_rag.py
```

The script prints the answer, retrieved evidence chunks, and retrieval
metadata for a sample maternal-health query.

### 6. Run the Flask backend

```powershell
python Backend\app.py
```

The API is available at `http://127.0.0.1:5000`.
