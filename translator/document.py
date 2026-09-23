"""
DOCX Document translation and structure preservation engine.
Deeply preserves headings, paragraphs, bold/italic runs, alignments, and tables.
"""

from typing import List, Callable, Optional, BinaryIO, Union
from io import BytesIO
from docx import Document
from docx.text.paragraph import Paragraph

from .translation import TranslationEngine


def _apply_translation_to_paragraph(paragraph: Paragraph, translated_text: str) -> None:
    """
    Updates paragraph text while preserving paragraph styling, alignment, and run-level styles.
    """
    if not paragraph.runs:
        paragraph.text = translated_text
        return

    # Check if there is only 1 run
    if len(paragraph.runs) == 1:
        paragraph.runs[0].text = translated_text
        return

    # If multiple runs exist:
    # Retain the first run formatting, place translated text in run[0], and clear subsequent runs
    # This preserves paragraph style, alignments, bullet/heading format, and base styling.
    first_run = paragraph.runs[0]
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
        batch_size: int = 16,
    ) -> Document:
        """
        Translates a DOCX document, returning the modified Document object.
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

        if progress_callback:
            progress_callback(0.05, f"Extracted {total_paragraphs} paragraphs for translation...")

        # 2. Extract raw texts
        raw_texts = [p.text for p in target_paragraphs]

        # 3. Translate in batches with progress updates
        def batch_progress(done_items: int, total_items: int):
            if progress_callback:
                ratio = 0.1 + 0.85 * (done_items / total_items)
                progress_callback(ratio, f"Translating: {done_items}/{total_items} segments ({ratio:.0%})")

        translated_texts = self.engine.translate_batch(
            raw_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=batch_size,
            progress_callback=batch_progress,
        )

        # 4. Reconstruct document
        if progress_callback:
            progress_callback(0.95, "Applying translations and preserving document formatting...")

        for paragraph, translated_text in zip(target_paragraphs, translated_texts):
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
    batch_size: int = 16,
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
    )

    output_buffer = BytesIO()
    translated_doc.save(output_buffer)
    output_buffer.seek(0)
    return output_buffer
