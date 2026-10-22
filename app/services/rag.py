from app.services.history import load_history, save_message
from app.workflow.graph import RAGWorkflow


def answer_question(query: str, session_id: str) -> str:
    user_message_id = save_message(session_id, "human", query)
    history = load_history(session_id, exclude_id=user_message_id)
    answer = RAGWorkflow().answer(query, history)
    save_message(session_id, "assistant", answer)
    return answer
