import uuid

import requests
import streamlit as st

from app.config import settings


st.set_page_config(page_title="DocMind", page_icon="💬", layout="wide")


def _request_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail or response.text or f"Request failed ({response.status_code}).")


def _login_page() -> None:
    st.title("DocMind")
    st.caption("Sign in to chat with uploaded documents, general knowledge, or web search.")
    if settings.streamlit_dev_mode:
        st.info("Local development mode is enabled. Authentication is bypassed.")
        if st.button("Start a local chat", type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.authenticated = True
            st.rerun()
        return

    if not settings.rust_auth_api_url:
        st.error("Set RUST_AUTH_API_URL to connect the Rust authentication service.")
        return

    initialize, sign_in, create_account = st.tabs(
        ["Initialize service", "Sign in", "Create account"]
    )
    with initialize:
        if st.button("Initialize authentication service"):
            try:
                response = requests.post(
                    f"{settings.rust_auth_api_url}/initialize", timeout=15
                )
                response.raise_for_status()
                st.success("Authentication service initialized.")
            except requests.RequestException as exc:
                st.error(f"Could not initialize authentication service: {exc}")
    with sign_in:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            _authenticate("login", username, password)
    with create_account:
        with st.form("register_form"):
            username = st.text_input("Username", key="register_username")
            password = st.text_input("Password", type="password", key="register_password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            _authenticate("register", username, password)


def _authenticate(endpoint: str, username: str, password: str) -> None:
    if not username.strip() or not password:
        st.error("Enter both a username and password.")
        return
    try:
        response = requests.post(
            f"{settings.rust_auth_api_url}/{endpoint}",
            json={"username": username, "password": password},
            timeout=15,
        )
        if not response.ok:
            st.error(_request_error(response))
            return
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.authenticated = True
        st.session_state.username = username
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Authentication service request failed: {exc}")


def _chat_page() -> None:
    st.title("DocMind by Ashish")
    with st.sidebar:
        st.subheader("Upload a document")
        uploaded_file = st.file_uploader(
            "Choose one PDF or TXT file", type=["pdf", "txt"]
        )
        description = st.text_input("Document description")
        if st.button("Upload and index", disabled=uploaded_file is None):
            if not description.strip():
                st.error("Add a short description before uploading.")
            elif uploaded_file is not None:
                try:
                    response = requests.post(
                        f"{settings.python_api_url}/rag/documents/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "application/octet-stream",
                            )
                        },
                        headers={"X-Description": description.strip()},
                        timeout=180,
                    )
                    if response.ok:
                        st.success("Document uploaded and indexed.")
                    else:
                        st.error(_request_error(response))
                except requests.RequestException as exc:
                    st.error(f"Document upload failed: {exc}")
        st.divider()
        if st.button("Sign out"):
            for key in ("authenticated", "session_id", "username", "messages"):
                st.session_state.pop(key, None)
            st.rerun()

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{settings.python_api_url}/rag/query",
                        json={
                            "query": prompt,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=180,
                    )
                    if response.ok:
                        answer = response.json()["result"]
                        st.markdown(answer)
                        messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(_request_error(response))
                except (requests.RequestException, ValueError, KeyError) as exc:
                    st.error(f"Could not get a response from the API: {exc}")


if st.session_state.get("authenticated"):
    _chat_page()
else:
    _login_page()
