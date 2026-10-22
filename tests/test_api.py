import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "running"})

    @patch("app.api.routes.answer_question", return_value="Answer")
    def test_query_returns_result(self, answer_question) -> None:
        response = self.client.post(
            "/rag/query",
            json={"query": "What is in the file?", "session_id": "session-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"result": "Answer"})
        answer_question.assert_called_once_with("What is in the file?", "session-1")

    @patch("app.api.routes.index_document")
    def test_upload_accepts_document_and_description(self, index_document) -> None:
        response = self.client.post(
            "/rag/documents/upload",
            files={"file": ("notes.TXT", b"Text to index", "text/plain")},
            headers={"X-Description": "Meeting notes"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True})
        index_document.assert_called_once_with(
            filename="notes.TXT",
            content=b"Text to index",
            description="Meeting notes",
        )

    def test_upload_rejects_missing_description(self) -> None:
        response = self.client.post(
            "/rag/documents/upload",
            files={"file": ("notes.txt", b"Text to index", "text/plain")},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
