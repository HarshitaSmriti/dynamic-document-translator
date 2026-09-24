"""
Test script to translate sample BRD document to Hindi and Bengali.
"""

import sys
import time
from pathlib import Path
from docx import Document

sys.stdout.reconfigure(encoding="utf-8")

from translator.language_detection import detect_document_language
from translator.translation import TranslationEngine
from translator.document import DocxTranslator

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BRD_PATH = BASE_DIR / "sample_brd_document.docx"

print("1. Language Detection", flush=True)
doc = Document(str(BRD_PATH))
detection = detect_document_language(doc)
print(f"Detected: {detection.detected_language} (Confidence: {detection.confidence*100:.1f}%)", flush=True)

engine = TranslationEngine(models_root=MODELS_DIR)
translator = DocxTranslator(engine)

print("\n2. Translating BRD to Hindi", flush=True)
def progress_cb(ratio, msg):
    print(f"[{ratio*100:.0f}%] {msg}", flush=True)

t0 = time.time()
hi_doc = translator.translate_document(
    doc_source=str(BRD_PATH),
    source_lang="English",
    target_lang="Hindi",
    progress_callback=progress_cb,
    batch_size=4,
    num_beams=1,
)
t1 = time.time()
hi_out_path = BASE_DIR / "sample_brd_hindi.docx"
hi_doc.save(str(hi_out_path))
print(f"Saved: {hi_out_path} ({t1-t0:.2f}s)", flush=True)

print("\n3. Translating BRD to Bengali", flush=True)
t2 = time.time()
bn_doc = translator.translate_document(
    doc_source=str(BRD_PATH),
    source_lang="English",
    target_lang="Bengali",
    progress_callback=progress_cb,
    batch_size=4,
    num_beams=1,
)
t3 = time.time()
bn_out_path = BASE_DIR / "sample_brd_bengali.docx"
bn_doc.save(str(bn_out_path))
print(f"Saved: {bn_out_path} ({t3-t2:.2f}s)", flush=True)
print("\nBRD translation tests completed successfully.", flush=True)


