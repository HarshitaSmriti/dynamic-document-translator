"""
DOCX Document translation and structure preservation engine.
Deeply preserves headings, paragraphs, bold/italic runs, alignments, and tables.
"""

import re
from typing import List, Callable, Optional, BinaryIO, Union, Tuple, Dict
from io import BytesIO
from docx import Document
from docx.text.paragraph import Paragraph

from .translation import TranslationEngine

# Regex to detect sentence boundaries while preserving decimals, acronyms, and quotes
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.?!।॥])\s+(?=[A-Z0-9\u0900-\u097F\u0980-\u09FF"“\'\(])')

# Regex to detect bullet points and numbered list markers
PREFIX_REGEX = re.compile(r'^(\s*(?:[\u2022\u2023\u25E6\u2043\u2219\*\-\+]|\d+[\.\)]|[a-zA-Z][\.\)]))\s+')


def extract_prefix_and_body(text: str) -> Tuple[str, str]:
    """
    Extracts leading bullet/numbering marker from text so formatting markers
    are preserved identically without sending bullet symbols to the translation model.
    """
    m = PREFIX_REGEX.match(text)
    if m:
        prefix = m.group(0)
        body = text[len(prefix):]
        return prefix, body
    return "", text


def split_into_sentences(text: str) -> List[str]:
    """
    Splits long paragraphs into individual sentences for higher translation quality.
    Leaves short headings, titles, table cells, and single sentences intact.
    """
    if not text or not text.strip():
        return [text]

    cleaned = text.strip()
    # Don't split short text segments or single sentences
    if len(cleaned) < 100 or cleaned.count(".") + cleaned.count("।") <= 1:
        return [cleaned]

    sentences = SENTENCE_SPLIT_REGEX.split(cleaned)
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    return valid_sentences if valid_sentences else [cleaned]


def _apply_translation_to_paragraph(paragraph: Paragraph, translated_text: str) -> None:
    """
    Updates paragraph text while deeply preserving run formatting (bold, italic, color).
    Intelligently splits labeled prefixes (e.g. **Requirement:** Description).
    """
    if not paragraph.runs:
        paragraph.text = translated_text
        return

    if len(paragraph.runs) == 1:
        paragraph.runs[0].text = translated_text
        return

    first_run = paragraph.runs[0]
    second_run = paragraph.runs[1] if len(paragraph.runs) > 1 else None

    # Check if first run is bold label and subsequent run is normal body text
    if first_run.bold and second_run and not second_run.bold:
        if ":" in translated_text:
            parts = translated_text.split(":", 1)
            first_run.text = parts[0].strip() + ": "
            second_run.text = parts[1].strip()
            for run in paragraph.runs[2:]:
                run.text = ""
            return
        elif " - " in translated_text:
            parts = translated_text.split(" - ", 1)
            first_run.text = parts[0].strip() + " - "
            second_run.text = parts[1].strip()
            for run in paragraph.runs[2:]:
                run.text = ""
            return

    # Fallback: assign translated text to first run and clear subsequent runs
    first_run.text = translated_text
    for run in paragraph.runs[1:]:
        run.text = ""


class DocxTranslator:
    """
    Translates DOCX documents while preserving structure and styling.
    """

    def __init__(self, engine: TranslationEngine) -> None:
        self.engine = engine

    def translate_document(
        self,
        doc_source: Union[str, BinaryIO, BytesIO],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        batch_size: int = 4,
        num_beams: int = 1,
    ) -> Document:
        """
        Translates a DOCX document, returning the modified Document object.
        Optimized with text deduplication, sentence splitting, and fast batching.
        """
        doc = Document(doc_source)

        # 1. Collect all target paragraphs from body, tables, headers, footers
        target_paragraphs: List[Paragraph] = []

        # Body paragraphs
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                target_paragraphs.append(p)

        # Tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        if p.text and p.text.strip():
                            target_paragraphs.append(p)

        # Headers and Footers
        for section in doc.sections:
            if section.header:
                for p in section.header.paragraphs:
                    if p.text and p.text.strip():
                        target_paragraphs.append(p)
            if section.footer:
                for p in section.footer.paragraphs:
                    if p.text and p.text.strip():
                        target_paragraphs.append(p)

        total_paragraphs = len(target_paragraphs)
        if total_paragraphs == 0:
            if progress_callback:
                progress_callback(1.0, "Document has no translatable text.")
            return doc

        # 2. Extract raw texts and analyze segment structure
        raw_texts = [p.text for p in target_paragraphs]
        unique_texts = list(dict.fromkeys(raw_texts))

        # Separate prefix markers and split into sentences
        # Map: unique_raw_text -> list of (prefix, [sentences])
        text_structure: Dict[str, Tuple[str, List[str]]] = {}
        all_sentences_to_translate: List[str] = []

        for text in unique_texts:
            prefix, body = extract_prefix_and_body(text)
            sentences = split_into_sentences(body)
            text_structure[text] = (prefix, sentences)
            for s in sentences:
                if s.strip():
                    all_sentences_to_translate.append(s.strip())

        unique_sentences = list(dict.fromkeys(all_sentences_to_translate))
        total_unique_sents = len(unique_sentences)

        if progress_callback:
            progress_callback(
                0.05,
                f"Extracted {total_paragraphs} paragraphs ({total_unique_sents} unique sentences)...",
            )

        # 3. Translate unique sentences in batches
        def batch_progress(done_items: int, total_items: int):
            if progress_callback:
                ratio = 0.08 + 0.82 * (done_items / max(1, total_items))
                progress_callback(
                    ratio,
                    f"Translating: {done_items}/{total_items} sentences ({ratio:.0%})",
                )

        translated_sentences = self.engine.translate_batch(
            unique_sentences,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=batch_size,
            num_beams=num_beams,
            progress_callback=batch_progress,
        )

        # Sentence translation lookup map
        sent_translation_map = dict(zip(unique_sentences, translated_sentences))

        # Reconstruct full paragraph translations from translated sentences
        translation_map: Dict[str, str] = {}
        for text, (prefix, sentences) in text_structure.items():
            if not sentences:
                translation_map[text] = prefix
            else:
                translated_sents = [sent_translation_map.get(s, s) for s in sentences]
                joined_text = " ".join(translated_sents)
                translation_map[text] = prefix + joined_text

        # 4. Reconstruct document
        if progress_callback:
            progress_callback(0.95, "Applying translations and preserving document formatting...")

        for paragraph in target_paragraphs:
            translated_text = translation_map.get(paragraph.text, paragraph.text)
            _apply_translation_to_paragraph(paragraph, translated_text)

        if progress_callback:
            progress_callback(1.0, "Translation completed successfully!")

        return doc


def translate_docx_document(
    doc_source: Union[str, BinaryIO, BytesIO],
    source_lang: str,
    target_lang: str,
    engine: TranslationEngine,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    batch_size: int = 4,
    num_beams: int = 1,
) -> BytesIO:
    """
    Convenience function that translates a DOCX and returns a BytesIO buffer of the saved file.
    """
    translator = DocxTranslator(engine)
    translated_doc = translator.translate_document(
        doc_source=doc_source,
        source_lang=source_lang,
        target_lang=target_lang,
        progress_callback=progress_callback,
        batch_size=batch_size,
        num_beams=num_beams,
    )

    output_buffer = BytesIO()
    translated_doc.save(output_buffer)
    output_buffer.seek(0)
    return output_buffer

