from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from qdrant_client import QdrantClient

from app.config import settings


def require_openai_key() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")


def require_tavily_key() -> None:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    require_openai_key()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client


def get_chat_collection():
    return get_mongo_client()[settings.mongodb_db_name]["chat_history"]
