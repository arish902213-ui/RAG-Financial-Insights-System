# RAG-Based Knowledge & Financial Insights System

End-to-end Streamlit application that supports:

- **PDF RAG Q&A** with citations (`file`, `page`, `chunk_id`)
- **PDF financial signal detection + visualization** (for financial-like PDF content)
- **Unified grounded chat from current uploaded PDF(s)**
- **Persistent FAISS index** between runs

## Project Structure

```text
rag_knowledge_finance/
├── app.py
├── ingestion/
│   ├── pdf_loader.py
│   ├── csv_loader.py
├── processing/
│   ├── chunker.py
│   ├── finance_features.py
│   ├── summaries.py
├── retrieval/
│   ├── embedder.py
│   ├── faiss_store.py
│   ├── retriever.py
├── generation/
│   ├── prompt_builder.py
│   ├── llm_client.py
│   ├── answerer.py
├── assets/
├── data/
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) enable LLM generation with strict grounding:

```bash
set OPENAI_API_KEY=your_key_here
```

If no API key is configured, the app still works using extractive grounded responses from retrieved chunks.

## Run

```bash
streamlit run app.py
```

## Usage

1. Upload one or more PDFs.
2. App auto-processes and indexes current PDF(s) on upload.
3. Ask questions in the unified chat.
4. If financial patterns are detected in the PDF, charts are shown automatically.

## Grounding and Hallucination Control

- Retrieval uses FAISS top-k similarity search (`k=5`)
- Prompt enforces: use only retrieved context
- If insufficient evidence: return

```text
Not enough information in the uploaded documents/data.
```

- Citations are returned as `(file, page, chunk_id)`

## Persistence

Index files are saved in:

- `data/index/index.faiss`
- `data/index/records.json`

They are automatically loaded on next run.

## Acceptance Test Checklist

- Upload PDF + ask question → answer with page citations.
- Ask out-of-scope question → refusal message shown.
- Restart app → FAISS index still available.

## Deploy

### Option 1: Streamlit Cloud (Easiest - Recommended) ⭐

**No servers to manage! Deploy in minutes:**

1. Push this repository to GitHub (make it public)
2. Go to https://share.streamlit.io
3. Click "New app" and select your GitHub repository
4. Set main file to `app.py`
5. Click "Deploy!"

Your app will be live at: `https://[your-username]-rag-based-knowledge-financial-insights.streamlit.app`

**To add OpenAI API support:**
- In Streamlit Cloud dashboard, go to app settings → Secrets
- Add: `OPENAI_API_KEY = "your-key-here"`

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed GitHub & Streamlit Cloud setup instructions.

### Option 2: Docker (recommended for self-hosted)

Build image:

```bash
docker build -t rag-knowledge-finance .
```

Run container:

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here rag-knowledge-finance
```

Then open:

```text
http://localhost:8501
```

### Option 3: Azure App Service (Container)

1. Push this project to GitHub.
2. Build and push Docker image to Azure Container Registry (ACR).
3. Create Azure Web App for Containers and point it to that image.
4. Set app setting `OPENAI_API_KEY` (optional).
5. Browse the Web App URL.

### Option 4: Render / Railway / Fly.io

- Use Docker deploy from this repository.
- Expose port `8501`.
- Set environment variable `OPENAI_API_KEY` if needed.

### Important persistence note

Cloud platforms with ephemeral storage may reset `data/index` between restarts.
For persistent FAISS indexes, mount a persistent disk/volume to `/app/data`.
