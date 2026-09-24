import sys, time
from pathlib import Path
from docx import Document

sys.stdout.reconfigure(encoding="utf-8")

from translator.language_detection import detect_document_language
from translator.translation import TranslationEngine
from translator.document import DocxTranslator

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BRD_PATH = BASE_DIR / "sample_brd_document.docx"

print("--- Step 1: Loading & Detecting BRD Language ---", flush=True)
doc = Document(str(BRD_PATH))
detection = detect_document_language(doc)
print(f"Detected Language: {detection.detected_language} (Confidence: {detection.confidence*100:.1f}%)", flush=True)
print(f"Character counts: {detection.char_counts}", flush=True)

engine = TranslationEngine(models_root=MODELS_DIR)
translator = DocxTranslator(engine)

print("\n--- Step 2: Translating BRD to Hindi ---", flush=True)
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
print(f"\nSaved translated Hindi BRD to: {hi_out_path} (Total Time: {t1-t0:.2f}s)", flush=True)

print("\n--- Step 3: Inspecting First 8 Paragraphs in Hindi ---", flush=True)
for idx, p in enumerate(hi_doc.paragraphs[:12]):
    if p.text.strip():
        print(f"P{idx} [Style: {p.style.name}]: {p.text}", flush=True)

print("\n--- Step 4: Inspecting Functional Requirements Table in Hindi ---", flush=True)
if len(hi_doc.tables) >= 3:
    req_table = hi_doc.tables[2]
    for r_idx, row in enumerate(req_table.rows[:5]):
        row_vals = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        print(f"Row {r_idx}: {' | '.join(row_vals)}", flush=True)

print("\n--- Step 5: Run-Level Formatting Checks ---", flush=True)
for idx, p in enumerate(hi_doc.paragraphs):
    if len(p.runs) > 1 and p.runs[0].bold and not p.runs[1].bold:
        print(f"P{idx}: Run 0 (Bold): '{p.runs[0].text}' | Run 1 (Normal): '{p.runs[1].text}'", flush=True)

print("\n--- Step 6: Translating BRD to Bengali ---", flush=True)
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
print(f"\nSaved translated Bengali BRD to: {bn_out_path} (Total Time: {t3-t2:.2f}s)", flush=True)

print("\n--- Step 7: Inspecting Bengali Sample Paragraphs & Tables ---", flush=True)
for idx, p in enumerate(bn_doc.paragraphs[:6]):
    if p.text.strip():
        print(f"BN P{idx}: {p.text}", flush=True)

print("\nALL BRD TRANSLATION TESTS COMPLETED SUCCESSFULLY!", flush=True)

