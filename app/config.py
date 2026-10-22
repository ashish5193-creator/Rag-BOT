import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    tavily_api_key: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    mongodb_url: str
    mongodb_db_name: str
    python_api_url: str
    streamlit_dev_mode: bool
    rust_auth_api_url: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
            qdrant_collection=os.getenv("QDRANT_CODE_COLLECTION", "codebase").strip(),
            mongodb_url=os.getenv("MONGODB_URL", "mongodb://localhost:27017").strip(),
            mongodb_db_name=os.getenv("MONGODB_DB_NAME", "adaptive_rag").strip(),
            python_api_url=os.getenv(
                "PYTHON_API_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
            streamlit_dev_mode=_env_bool("STREAMLIT_DEV_MODE"),
            rust_auth_api_url=os.getenv("RUST_AUTH_API_URL", "").rstrip("/"),
        )


settings = Settings.from_environment()
