import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="EduRAG - Academic Assistant", page_icon="📚", layout="wide")
st.title("📚 EduRAG: Educational Document Assistant")

# Sidebar for uploading documents
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            resp = requests.post(f"{BACKEND_URL}/upload", files=files)
            if resp.status_code == 200:
                st.success(f"Ingested '{uploaded_file.name}' successfully!")
            else:
                st.error("Error uploading file.")

    st.divider()
    st.header("Knowledge Base")
    if st.button("Refresh Document List"):
        res = requests.get(f"{BACKEND_URL}/documents").json()
        st.json(res)

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(f"{BACKEND_URL}/query", json={"question": prompt, "retrieval": "hybrid"}).json()
            answer = res.get("answer", "No response.")
            sources = res.get("sources", [])
            
            st.markdown(answer)
            if sources:
                with st.expander("View Retrieved Sources"):
                    for s in sources:
                        st.caption(f"**Document:** {s['source']} | **Page:** {s['page']} | **Score:** {s['score']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})