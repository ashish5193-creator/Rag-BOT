from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="Adaptive RAG API",
    description="Document-aware chat with general and web-search answering.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "running"}
