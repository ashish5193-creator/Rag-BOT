import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.workflow import graph as rag_graph


class WorkflowTests(unittest.TestCase):
    @patch.object(rag_graph, "require_openai_key")
    @patch.object(rag_graph, "ChatOpenAI")
    def test_general_question_uses_general_answer_path(
        self, chat_model, _require_openai_key
    ) -> None:
        model = chat_model.return_value
        model.with_structured_output.return_value.invoke.return_value = (
            SimpleNamespace(route="general")
        )
        model.invoke.return_value = AIMessage(content="A general answer.")

        workflow = rag_graph.RAGWorkflow()
        workflow._search_documents = lambda _: []

        result = workflow.answer("What is a vector database?", [])

        self.assertEqual(result, "A general answer.")
        model.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
