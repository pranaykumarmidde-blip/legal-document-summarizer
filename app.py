
import streamlit as st
from pypdf import PdfReader
from transformers import pipeline
import torch

st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Document Summarization Assistant")
st.caption("AI-powered assistance for understanding legal contracts")

st.info(
    "Upload a legal document to generate a concise summary. "
    "Always verify the summary against the original contract."
)

@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    return pipeline(
        "summarization",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")

def summarize_text(text):
    summarizer = load_summarizer()
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= 450:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    summaries = []
    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        summaries.append(result[0]["summary_text"])

    if len(summaries) > 1:
        combined = " ".join(summaries)
        final = summarizer(
            combined,
            max_length=180,
            min_length=40,
            do_sample=False,
            truncation=True
        )
        return final[0]["summary_text"]

    return summaries[0] if summaries else "No text to summarize."

uploaded_file = st.file_uploader(
    "Upload your legal document",
    type=["pdf", "txt"]
)

if uploaded_file:
    if st.button("📄 Extract Document Text"):
        text = extract_text(uploaded_file)
        st.session_state["document_text"] = text

    if "document_text" in st.session_state:
        text = st.session_state["document_text"]

        if not text.strip():
            st.error("No readable text found in this document.")
        else:
            st.success(f"Document loaded: {len(text.split())} words")

            with st.expander("Preview extracted text"):
                st.text_area(
                    "Document content",
                    text[:10000],
                    height=250
                )

            if st.button("✨ Generate Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        st.session_state["summary"] = summarize_text(text)
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")

            if "summary" in st.session_state:
                st.subheader("📝 Generated Summary")
                st.write(st.session_state["summary"])
                st.download_button(
                    "Download Summary",
                    st.session_state["summary"],
                    file_name="legal_summary.txt",
                    mime="text/plain"
                )

st.divider()
st.caption(
    "Educational prototype only. This tool does not provide legal advice "
    "or guarantee that every contractual obligation is captured."
)

