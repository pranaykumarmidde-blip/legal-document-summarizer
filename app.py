
import streamlit as st
from pypdf import PdfReader
from transformers import pipeline
import torch
import io

st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Document Summarization Assistant")
st.write("Upload a legal contract and generate an AI-powered summary.")

st.info("Supported formats: PDF and TXT")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

def extract_text(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file.getvalue()))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    return file.getvalue().decode("utf-8", errors="ignore")

def split_text(text, max_words=350):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

uploaded = st.file_uploader(
    "Choose your legal document",
    type=["pdf", "txt"],
    key="legal_upload"
)

if uploaded is not None:
    st.success(f"Uploaded: {uploaded.name}")

    try:
        text = extract_text(uploaded)

        if not text.strip():
            st.error(
                "No readable text found. This may be a scanned PDF. "
                "Try a text-based PDF or TXT file."
            )
        else:
            st.success(f"Extracted {len(text.split())} words.")

            with st.expander("View extracted document text"):
                st.text_area(
                    "Document text",
                    text[:12000],
                    height=250
                )

            if st.button("✨ Generate Summary", type="primary"):
                try:
                    with st.spinner(
                        "Loading AI model and generating summary..."
                    ):
                        summarizer = load_model()
                        chunks = split_text(text)
                        summaries = []

                        for chunk in chunks:
                            result = summarizer(
                                chunk,
                                max_length=110,
                                min_length=25,
                                do_sample=False,
                                truncation=True
                            )
                            summaries.append(
                                result[0]["summary_text"]
                            )

                        combined = " ".join(summaries)

                        if len(summaries) > 1:
                            final_result = summarizer(
                                combined,
                                max_length=180,
                                min_length=35,
                                do_sample=False,
                                truncation=True
                            )
                            summary = final_result[0]["summary_text"]
                        else:
                            summary = combined

                    st.subheader("📝 Generated Summary")
                    st.write(summary)

                    st.download_button(
                        "⬇️ Download Summary",
                        data=summary,
                        file_name="legal_summary.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error("The AI model could not generate a summary.")
                    st.write(f"Error details: {str(e)[:1000]}")
                    st.info(
                        "If this is a memory or model-loading error, "
                        "we may need a smaller model or a different host."
                    )

    except Exception as e:
        st.error("Could not read this document.")
        st.write(str(e)[:1000])

st.divider()
st.caption(
    "Educational prototype only. Verify all summaries against "
    "the original contract. This application does not provide legal advice."
)
