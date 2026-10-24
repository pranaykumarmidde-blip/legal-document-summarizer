
import io
import re

import streamlit as st
import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Legal Document Summarization Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        color: #4F8BF9;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">⚖️ Legal Document Summarization Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-powered assistance for reviewing legal contracts and agreements</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload a PDF or TXT contract to generate a concise summary. "
    "Always verify the generated summary against the original document."
)


# --------------------------------------------------
# LOAD TRANSFORMER MODEL
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# --------------------------------------------------
# EXTRACT TEXT FROM PDF OR TXT
# --------------------------------------------------
def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)

        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    if filename.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8", errors="replace"
        )

    raise ValueError("Unsupported file type. Upload a PDF or TXT file.")


# --------------------------------------------------
# SPLIT TEXT INTO MODEL-SIZED CHUNKS
# --------------------------------------------------
def split_into_chunks(text, tokenizer, max_tokens=700):
    words = text.split()
    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        word_tokens = tokenizer.encode(
            word,
            add_special_tokens=False
        )
        token_length = len(word_tokens)

        if current_words and current_length + token_length > max_tokens:
            chunks.append(" ".join(current_words))
            current_words = []
            current_length = 0

        current_words.append(word)
        current_length += token_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# --------------------------------------------------
# GENERATE SUMMARY
# --------------------------------------------------
def summarize_document(text):
    tokenizer, model = load_model()
    chunks = split_into_chunks(text, tokenizer)

    summaries = []

    for index, chunk in enumerate(chunks, start=1):
        st.write(f"Processing section {index} of {len(chunks)}...")

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=2,
                do_sample=False,
                early_stopping=True,
            )

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        if summary.strip():
            summaries.append(summary.strip())

    if not summaries:
        return "The model did not generate a summary."

    combined_summary = " ".join(summaries)

    # If the document has multiple sections, summarize the combined
    # summaries only when they fit safely within the model's input.
    encoded = tokenizer(
        combined_summary,
        return_tensors="pt",
        truncation=True,
        max_length=900,
    )

    if len(summaries) > 1:
        with torch.inference_mode():
            final_ids = model.generate(
                **encoded,
                max_new_tokens=160,
                num_beams=2,
                do_sample=False,
                early_stopping=True,
            )

        combined_summary = tokenizer.decode(
            final_ids[0],
            skip_special_tokens=True,
        )

    return combined_summary


# --------------------------------------------------
# SIMPLE KEY-POINT EXTRACTION
# --------------------------------------------------
def extract_key_points(text):
    patterns = {
        "Payment": r"[^.!?\n]*(?:payment|pay|fee|price|amount|invoice|penalty)[^.!?\n]*[.!?]?",
        "Termination": r"[^.!?\n]*(?:terminate|termination|cancel|cancellation)[^.!?\n]*[.!?]?",
        "Confidentiality": r"[^.!?\n]*(?:confidential|confidentiality|disclosure)[^.!?\n]*[.!?]?",
        "Duration and delivery": r"[^.!?\n]*(?:days|months|years|deadline|delivery|duration|term of this agreement)[^.!?\n]*[.!?]?",
        "Intellectual property": r"[^.!?\n]*(?:intellectual property|ownership|copyright|license|licence)[^.!?\n]*[.!?]?",
        "Disputes": r"[^.!?\n]*(?:dispute|arbitration|jurisdiction|governing law|court)[^.!?\n]*[.!?]?",
        "Liability": r"[^.!?\n]*(?:liability|liable|indemnity|indemnify|damages)[^.!?\n]*[.!?]?",
    }

    results = {}

    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        cleaned = []
        for match in matches[:3]:
            sentence = " ".join(match.split())
            if sentence and sentence not in cleaned:
                cleaned.append(sentence)

        if cleaned:
            results[label] = cleaned

    return results


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("About the Project")
    st.write(
        "This application uses a pretrained transformer model "
        "to create draft summaries of legal documents."
    )

    st.divider()
    st.subheader("Features")
    st.write("• PDF and TXT upload")
    st.write("• Text extraction")
    st.write("• AI-generated summary")
    st.write("• Contract key-point extraction")
    st.write("• Downloadable summary")

    st.warning(
        "This prototype does not replace legal advice. "
        "Key-point extraction uses keyword matching and may miss clauses."
    )


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your legal document",
    type=["pdf", "txt"],
    help="Choose a text-based PDF or a plain-text TXT document.",
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if uploaded_file.size > 10 * 1024 * 1024:
        st.error(
            "This prototype limits uploads to 10 MB. "
            "Please upload a smaller document."
        )
        st.stop()

    try:
        document_text = extract_text(uploaded_file)

        if not document_text.strip():
            st.error(
                "No readable text was found. This PDF may contain scanned "
                "images; OCR would be needed to extract its text."
            )
            st.stop()

        st.success(
            f"Text extracted successfully: "
            f"{len(document_text.split())} words."
        )

        with st.expander("👁️ Preview extracted document text"):
            st.text_area(
                "Extracted text",
                value=document_text[:15000],
                height=250,
                disabled=True,
            )

        tab_summary, tab_points = st.tabs(
            ["📝 Summary", "🔍 Key Points"]
        )

        with tab_summary:
            if st.button(
                "✨ Generate Summary",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.status(
                        "Preparing the AI model and summarizing...",
                        expanded=True,
                    ) as status:
                        summary = summarize_document(document_text)
                        status.update(
                            label="Summary generated",
                            state="complete",
                            expanded=False,
                        )

                    st.session_state["legal_summary"] = summary
                    st.session_state["legal_filename"] = uploaded_file.name

                except Exception as error:
                    st.error("Summary generation failed.")
                    st.code(str(error))
                    st.info(
                        "Check the Streamlit deployment logs if the error "
                        "continues. The model may have failed to download "
                        "or the hosting environment may have run out of memory."
                    )

            if st.session_state.get("legal_summary"):
                st.subheader("Generated Summary")
                st.write(st.session_state["legal_summary"])

                st.download_button(
                    "⬇️ Download Summary",
                    data=st.session_state["legal_summary"],
                    file_name="legal_document_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with tab_points:
            st.subheader("Potentially Important Clauses")

            key_points = extract_key_points(document_text)

            if key_points:
                for category, sentences in key_points.items():
                    with st.expander(category):
                        for sentence in sentences:
                            st.write(f"• {sentence}")
            else:
                st.write(
                    "No matching clauses were identified by the "
                    "keyword-based extractor."
                )

            st.caption(
                "These are keyword-based suggestions, not a complete "
                "legal analysis. Review the original document."
            )

    except Exception as error:
        st.error("The document could not be processed.")
        st.code(str(error))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Educational prototype only. Verify summaries and extracted clauses "
    "against the original contract. This application does not provide "
    "legal advice and does not guarantee factual completeness."
)
