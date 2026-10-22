from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING

from app.services.clients import get_chat_collection


def save_message(session_id: str, role: str, content: str) -> ObjectId:
    collection: Collection = get_chat_collection()
    result = collection.insert_one(
        {
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }
    )
    return result.inserted_id


def load_history(session_id: str, exclude_id: ObjectId | None = None) -> list[dict[str, str]]:
    collection: Collection = get_chat_collection()
    cursor = (
        collection.find({"session_id": session_id})
        .sort([("created_at", DESCENDING), ("_id", DESCENDING)])
        .limit(1000)
    )
    messages: list[dict[str, str]] = []
    for item in reversed(list(cursor)):
        if exclude_id is not None and item["_id"] == exclude_id:
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"human", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages
