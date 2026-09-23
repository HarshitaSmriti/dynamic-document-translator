"""
Dynamic Document Translator
Production Streamlit Application for Translating DOCX Documents
using fine-tuned IndicTrans2 Models.
"""

from pathlib import Path
from io import BytesIO
import streamlit as st
from docx import Document

from translator.language_detection import detect_document_language
from translator.translation import TranslationEngine, UnsupportedRouteError
from translator.document import translate_docx_document
from translator.model import get_device

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dynamic Document Translator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for polished production UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .stat-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-primary {
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
    .badge-success {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }
    .card-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

SUPPORTED_LANGUAGES = ["English", "Hindi", "Bengali"]


@st.cache_resource(show_spinner=False)
def get_translation_engine() -> TranslationEngine:
    """
    Streamlit cached resource loader for the translation engine.
    Ensures model pipelines are loaded only once in memory.
    """
    return TranslationEngine(models_root=MODELS_DIR)


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">📄 Dynamic Document Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Accurate DOCX document translation with structure and format preservation '
    'powered by fine-tuned IndicTrans2 models.</div>',
    unsafe_allow_html=True,
)

device = get_device()
engine = get_translation_engine()

# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Select a Word document (.docx)",
        type=["docx"],
        help="Upload a DOCX file in English, Hindi, or Bengali.",
    )

if uploaded_file is not None:
    # Read document into memory
    file_bytes = uploaded_file.getvalue()
    doc_buffer = BytesIO(file_bytes)

    try:
        doc = Document(doc_buffer)
    except Exception as e:
        st.error(f"Failed to parse DOCX file: {e}")
        st.stop()

    # Detect Language
    detection = detect_document_language(doc)

    with col_left:
        st.markdown("---")
        st.subheader("2. Document Analysis")
        st.write(f"**Filename:** `{uploaded_file.name}` ({len(file_bytes) / 1024:.1f} KB)")

        if detection.detected_language is None:
            st.error(
                "❌ **Language Detection Failed**: "
                + (detection.warning_message or "Insufficient recognizable text found in the document.")
            )
            st.stop()

        # Display detection result
        detected_lang = detection.detected_language
        conf_pct = detection.confidence * 100

        st.markdown(
            f'<div style="margin: 0.8rem 0;">'
            f'<span class="stat-badge badge-primary">Detected Language: <strong>{detected_lang}</strong></span> &nbsp; '
            f'<span class="stat-badge badge-success">Confidence: <strong>{conf_pct:.1f}%</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if detection.is_ambiguous and detection.warning_message:
            st.warning(f"⚠️ **Language Ambiguity Notice**: {detection.warning_message}")

        # Script character distribution breakdown
        with st.expander("📊 View Script Character Breakdown"):
            st.write(f"**Total Alphabetic Characters Analyzed:** {detection.total_valid_chars}")
            st.json(detection.char_counts)

    with col_right:
        st.subheader("3. Select Target Language")

        # Available targets (exclude detected source language)
        available_targets = [lang for lang in SUPPORTED_LANGUAGES if lang != detected_lang]

        target_lang = st.selectbox(
            "Translate to:",
            options=available_targets,
            index=0 if available_targets else None,
            help="Source language is detected automatically and locked.",
        )

        # Check route validity
        is_route_valid = True
        route_error_message = ""

        try:
            engine.validate_route(detected_lang, target_lang)
        except UnsupportedRouteError as err:
            is_route_valid = False
            route_error_message = str(err)

        if not is_route_valid:
            st.error(f"⛔ **Unsupported Route**: {route_error_message}")
            st.info(
                "ℹ️ **Note**: Direct translation between Indic languages (Hindi ↔ Bengali) is disabled. "
                "Only English ↔ Indic routes are supported."
            )

        st.markdown("---")
        st.subheader("4. Translate & Download")

        translate_button = st.button(
            "🚀 Translate Document",
            type="primary",
            disabled=not is_route_valid,
            use_container_width=True,
        )

        if translate_button and is_route_valid:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(ratio: float, message: str):
                progress_bar.progress(min(max(ratio, 0.0), 1.0))
                status_text.info(message)

            try:
                # Reset buffer position for translation
                doc_buffer.seek(0)

                translated_buffer = translate_docx_document(
                    doc_source=doc_buffer,
                    source_lang=detected_lang,
                    target_lang=target_lang,
                    engine=engine,
                    progress_callback=update_progress,
                    batch_size=16,
                )

                update_progress(1.0, "✅ Translation complete! Ready for download.")

                # Output filename
                orig_stem = Path(uploaded_file.name).stem
                output_filename = f"{orig_stem}_{target_lang.lower()}.docx"

                st.download_button(
                    label=f"📥 Download Translated Document ({output_filename})",
                    data=translated_buffer,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary",
                    use_container_width=True,
                )

            except Exception as e:
                status_text.empty()
                st.error(f"❌ **Translation Error occurred:** {str(e)}")
else:
    with col_right:
        st.info("👈 Please upload a `.docx` document to start translation.")
