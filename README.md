#  RAG-BOT

A modular chat application that indexes PDF and UTF-8 text documents in Qdrant,
routes questions to uploaded documents, general language-model answers, or
Tavily web search, and saves conversation history in MongoDB.

## Requirements

- Python 3.12 or newer
- OpenAI API key
- MongoDB and Qdrant (local services or hosted instances)
- Tavily API key for web-search answers

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Configure the remaining
service URLs and keys as needed. `.env` is excluded from version control.
`STREAMLIT_DEV_MODE=true` enables a temporary local chat session without the
external Rust login service; do not enable this mode for a public deployment.

## Install and run on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set your environment values in `.env`, then start the backend and frontend in
separate terminals using `run_backend.cmd` and `run_frontend.cmd`. The default
API is `http://127.0.0.1:8000`; interactive API documentation is at `/docs`.

The frontend production login expects the Rust authentication service to expose
`POST /initialize`, `POST /login`, and `POST /register` under
`RUST_AUTH_API_URL`. The backend query and upload APIs do not implement
authorization; protect them behind an authenticated deployment gateway before
public use.

## Deploying Streamlit and FastAPI separately

The Streamlit frontend makes backend requests from the Streamlit server. In a
hosted deployment, `127.0.0.1` refers to the Streamlit machine/container, not
your computer and not a separately hosted FastAPI service. Deploy FastAPI as a
reachable service and configure the frontend's `PYTHON_API_URL` to its public
base URL, for example `https://your-api.example.com` (do not add `/rag/query`).
The backend must be reachable over HTTPS and allow requests from the frontend
deployment.

For Streamlit Community Cloud, add this setting under **App settings >
Secrets**:

```toml
PYTHON_API_URL = "https://your-api.example.com"
```

The frontend reads `PYTHON_API_URL` from Streamlit Secrets when present, and
otherwise uses the environment variable or local default. Configure the backend
with its own OpenAI, Qdrant, MongoDB, and Tavily settings. Do not use a localhost
Qdrant or MongoDB address in the backend unless those services run alongside it.

## API

- `GET /` returns `{"status":"running"}`.
- `POST /rag/documents/upload` accepts multipart `file` and the `X-Description`
  header. PDF and TXT extensions are case-insensitive. Scanned PDFs need OCR
  before upload.
- `POST /rag/query` accepts `{"query":"...","session_id":"..."}` and returns
  the generated answer in `result`.

Uploaded document descriptions are stored with their Qdrant chunks, so they
remain available across backend restarts. All documents use the configured
shared collection; ownership and per-user document isolation are not provided.
