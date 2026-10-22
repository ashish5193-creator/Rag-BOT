from typing import Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.services.clients import (
    get_embeddings,
    get_qdrant_client,
    require_openai_key,
    require_tavily_key,
)


class RouteDecision(BaseModel):
    route: Literal["documents", "general", "web"] = Field(
        description="Best source for answering the user's latest question"
    )


class RelevanceDecision(BaseModel):
    relevant: bool = Field(description="Whether the retrieved text answers the question")


class WorkflowState(TypedDict):
    query: str
    search_query: str
    history: list[dict[str, str]]
    candidate_context: str
    route: str
    context: str
    retry_count: int
    relevant: bool
    result: str


def _message_history(history: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history:
        message_type = HumanMessage if item["role"] == "human" else AIMessage
        messages.append(message_type(content=item["content"]))
    return messages


def _format_documents(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'uploaded document')}\n"
        f"Description: {doc.metadata.get('description', '')}\n"
        f"{doc.page_content}"
        for doc in documents
    )


class RAGWorkflow:
    def __init__(self) -> None:
        require_openai_key()
        self.model = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
        self.graph = self._build_graph()

    def _search_documents(self, query: str) -> list[Document]:
        client = get_qdrant_client()
        if not client.collection_exists(settings.qdrant_collection):
            return []
        store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=get_embeddings(),
        )
        return store.similarity_search(query, k=5)

    def _build_graph(self):
        def gather_candidate(state: WorkflowState) -> dict:
            return {
                "candidate_context": _format_documents(
                    self._search_documents(state["query"])
                )
            }

        def classify(state: WorkflowState) -> dict:
            history = _message_history(state["history"][-20:])
            messages = [
                SystemMessage(
                    content=(
                        "Select documents only when the question appears answerable "
                        "from uploaded-document context. Select web for current facts, "
                        "news, or information that needs external sources. Otherwise "
                        "select general. Candidate document context:\n"
                        f"{state['candidate_context'] or '(no matching documents)'}"
                    )
                ),
                *history,
                HumanMessage(content=state["query"]),
            ]
            decision = self.model.with_structured_output(RouteDecision).invoke(messages)
            return {"route": decision.route}

        def retrieve(state: WorkflowState) -> dict:
            docs = self._search_documents(state["search_query"])
            return {"context": _format_documents(docs)}

        def check_relevance(state: WorkflowState) -> dict:
            if not state["context"]:
                return {"relevant": False}
            decision = self.model.with_structured_output(RelevanceDecision).invoke(
                [
                    SystemMessage(
                        content="Decide whether the supplied context contains information "
                        "that can answer the question."
                    ),
                    HumanMessage(
                        content=f"Question: {state['query']}\n\nContext:\n{state['context']}"
                    ),
                ]
            )
            return {"relevant": decision.relevant}

        def rewrite(state: WorkflowState) -> dict:
            response = self.model.invoke(
                [
                    SystemMessage(
                        content="Rewrite the user's question as a concise semantic-search "
                        "query. Keep its meaning and output only the rewritten query."
                    ),
                    HumanMessage(content=state["query"]),
                ]
            )
            if not isinstance(response.content, str) or not response.content.strip():
                raise RuntimeError("The language model returned an empty rewritten query.")
            return {
                "search_query": response.content.strip(),
                "retry_count": state["retry_count"] + 1,
            }

        def answer_documents(state: WorkflowState) -> dict:
            if not state["context"] or not state["relevant"]:
                return {
                    "result": "I couldn't find relevant information in the uploaded documents."
                }
            response = self.model.invoke(
                [
                    SystemMessage(
                        content="Answer the question using only the uploaded-document "
                        "context. If it does not contain the answer, say so. Cite source "
                        "filenames when available."
                    ),
                    *(_message_history(state["history"][-20:])),
                    HumanMessage(
                        content=f"Question: {state['query']}\n\nContext:\n{state['context']}"
                    ),
                ]
            )
            return {"result": self._text(response.content)}

        def answer_general(state: WorkflowState) -> dict:
            response = self.model.invoke(
                [
                    SystemMessage(content="Answer helpfully and accurately."),
                    *_message_history(state["history"][-20:]),
                    HumanMessage(content=state["query"]),
                ]
            )
            return {"result": self._text(response.content)}

        def search_web(state: WorkflowState) -> dict:
            require_tavily_key()
            search = TavilySearchResults(max_results=5)
            results = search.invoke({"query": state["query"]})
            if not results:
                raise RuntimeError("The web search returned no results.")
            response = self.model.invoke(
                [
                    SystemMessage(
                        content="Answer using the web-search results. Distinguish facts "
                        "from uncertainty and include relevant source URLs."
                    ),
                    *_message_history(state["history"][-20:]),
                    HumanMessage(
                        content=f"Question: {state['query']}\n\n"
                        f"Web-search results:\n{results}"
                    ),
                ]
            )
            return {"result": self._text(response.content)}

        def after_relevance(state: WorkflowState) -> str:
            if state["relevant"] or state["retry_count"] >= 1:
                return "answer_documents"
            return "rewrite"

        builder = StateGraph(WorkflowState)
        builder.add_node("gather_candidate", gather_candidate)
        builder.add_node("classify", classify)
        builder.add_node("retrieve", retrieve)
        builder.add_node("check_relevance", check_relevance)
        builder.add_node("rewrite", rewrite)
        builder.add_node("answer_documents", answer_documents)
        builder.add_node("answer_general", answer_general)
        builder.add_node("search_web", search_web)
        builder.add_edge(START, "gather_candidate")
        builder.add_edge("gather_candidate", "classify")
        builder.add_conditional_edges(
            "classify",
            lambda state: state["route"],
            {
                "documents": "retrieve",
                "general": "answer_general",
                "web": "search_web",
            },
        )
        builder.add_edge("retrieve", "check_relevance")
        builder.add_conditional_edges(
            "check_relevance",
            after_relevance,
            {"answer_documents": "answer_documents", "rewrite": "rewrite"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("answer_documents", END)
        builder.add_edge("answer_general", END)
        builder.add_edge("search_web", END)
        return builder.compile()

    @staticmethod
    def _text(content: object) -> str:
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The language model returned an empty answer.")
        return content.strip()

    def answer(self, query: str, history: list[dict[str, str]]) -> str:
        state = self.graph.invoke(
            {
                "query": query,
                "search_query": query,
                "history": history,
                "candidate_context": "",
                "route": "",
                "context": "",
                "retry_count": 0,
                "relevant": False,
                "result": "",
            }
        )
        return state["result"]
