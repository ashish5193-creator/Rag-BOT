from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.services.clients import get_embeddings, get_qdrant_client, require_openai_key


class DocumentError(ValueError):
    """An upload the application cannot parse or index."""


def _extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise DocumentError("Unsupported file type. Upload a PDF or TXT file.")
    if not content:
        raise DocumentError("The uploaded file is empty.")
    try:
        if extension == ".txt":
            return content.decode("utf-8-sig").strip()
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise DocumentError(
            "Could not read this file. Confirm it is a valid UTF-8 text file or PDF."
        ) from exc


def _enhance_description(description: str) -> str:
    require_openai_key()
    model = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    response = model.invoke(
        "Rewrite this document description as a concise, accurate guide for a "
        "retrieval system. Do not add facts not present in the description.\n\n"
        f"Description: {description.strip()}"
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise RuntimeError("The language model returned an empty document description.")
    return response.content.strip()


def index_document(filename: str, content: bytes, description: str) -> None:
    if not description.strip():
        raise DocumentError("Provide a short description for the uploaded document.")
    text = _extract_text(filename, content)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    if not chunks:
        raise DocumentError(
            "No text could be extracted from this file. Scanned PDFs require OCR "
            "before they can be indexed."
        )
    enhanced_description = _enhance_description(description)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": Path(filename).name,
                "description": enhanced_description,
            },
        )
        for chunk in chunks
    ]
    client = get_qdrant_client()
    if client.collection_exists(settings.qdrant_collection):
        store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=get_embeddings(),
        )
        store.add_documents(documents)
    else:
        QdrantVectorStore.from_documents(
            documents,
            embedding=get_embeddings(),
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
        )
