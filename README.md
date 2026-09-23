# 📄 Dynamic Document Translator

Production Streamlit application for translating DOCX documents between **English**, **Hindi**, and **Bengali** with full structure, format, and typography preservation, powered by fine-tuned **IndicTrans2** models.

---

## 🚀 Features

- **Automated Language Detection**: Uses Unicode script-distribution analysis (Latin, Devanagari, Bengali) for deterministic language identification and confidence scoring.
- **Deep DOCX Structure Preservation**: Retains document headings, paragraphs, run-level styling (bold, italic, underline), alignments, and tables across translations.
- **Fine-Tuned IndicTrans2 Engine**: Custom inference pipeline with `use_cache=False` for zero word/token repetition artifacts.
- **Strict Route Enforcement**: Seamlessly handles English $\leftrightarrow$ Hindi and English $\leftrightarrow$ Bengali routes while disabling direct Hindi $\leftrightarrow$ Bengali routes with clear user notices.

---

## 📁 Repository Structure

```text
dynamic-document-translator/
├── .streamlit/
│   └── config.toml               # Streamlit server & CORS configuration
├── models/
│   └── fine_tuned/
│       ├── en_indic_fine_tuned/   # Tokenizers, configs & scripts for EN -> Indic
│       └── indic_en_fine_tuned/   # Tokenizers, configs & scripts for Indic -> EN
├── translator/
│   ├── __init__.py
│   ├── document.py               # Structure-preserving DOCX parser & reconstructor
│   ├── language_detection.py     # Script-distribution language detector
│   ├── model.py                  # Model loader & Transformers 4.32.1 compatibility shims
│   ├── processor.py              # Pure-Python normalization & transliteration processor
│   └── translation.py           # Batched inference engine with route validation
├── app.py                        # Streamlit web application
├── requirements.txt              # Pinned dependencies
├── test_translation.py           # Verification and regression tests
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <REPO_URL>
   cd dynamic-document-translator
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Model Weights**:
   Place the fine-tuned `model.safetensors` files inside their respective directories:
   - `models/fine_tuned/en_indic_fine_tuned/model.safetensors`
   - `models/fine_tuned/indic_en_fine_tuned/model.safetensors`

5. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 🧪 Running Tests

```bash
python test_translation.py
```

---

## 📄 License

MIT License.
