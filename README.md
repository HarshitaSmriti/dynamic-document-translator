# Dynamic Document Translator
 
DOCX document translation platform between **English**, **Hindi**, and **Bengali** with formatting and layout preservation, powered by fine-tuned **IndicTrans2** sequence-to-sequence models.
 
---
 
## Key Capabilities
 
- **Automated Language Detection**: Unicode script-distribution analysis across Latin, Devanagari, and Bengali scripts with confidence scoring.
- **DOCX Structure Preservation**: Retains headings, paragraph alignments, run-level styles (bold, italic, colors), tables, bullet markers, and headers/footers.
- **Fine-Tuned IndicTrans2 Engine**: Pure-Python normalization and tokenizer shims with `use_cache=False` to prevent token stuttering.
- **Route Validation**: Supports English <-> Hindi and English <-> Bengali translation routes.
 
---
 
## Project Structure
 
```text
dynamic-document-translator/
├── .streamlit/
│   └── config.toml               # Streamlit server & CORS configuration
├── models/
│   └── fine_tuned/
│       ├── en_indic_fine_tuned/   # Tokenizers, configs & weights for EN -> Indic
│       └── indic_en_fine_tuned/   # Tokenizers, configs & weights for Indic -> EN
├── translator/
│   ├── __init__.py
│   ├── document.py               # Structure-preserving DOCX parser & reconstructor
│   ├── language_detection.py     # Script-distribution language detector
│   ├── model.py                  # Model loader & Transformers 4.32.1 compatibility shims
│   ├── processor.py              # Pure-Python normalization & transliteration processor
│   └── translation.py           # Batched inference engine with route validation
├── app.py                        # Streamlit web application
├── generate_brd.py               # Script to generate sample BRD test document
├── test_brd_translation.py       # Comprehensive BRD translation test suite
├── test_translation.py           # Quick verification script
├── requirements.txt              # Pinned dependencies
└── README.md
```
 
---
 
## Setup & Execution
 
1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```
 
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
 
3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```
 
---
 
## Testing
 
Run translation verification:
```bash
python test_translation.py
python test_brd_translation.py
```
 
---
 
## License
 
MIT

