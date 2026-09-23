"""
Quick verification script for translation using the modular translator package.
"""

import sys
from pathlib import Path

# Ensure utf-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from translator.translation import TranslationEngine
from translator.language_detection import detect_language_from_text

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

engine = TranslationEngine(models_root=MODELS_DIR)

text = "Hello, how are you today?"
detection = detect_language_from_text(text)
print(f"Input text: '{text}' (Detected: {detection.detected_language})")

# Translate to Hindi
hi_output = engine.translate_text(text, source_lang="English", target_lang="Hindi")
print(f"Hindi Output: '{hi_output}'")

# Translate to Bengali
bn_output = engine.translate_text(text, source_lang="English", target_lang="Bengali")
print(f"Bengali Output: '{bn_output}'")

# Translate back from Hindi to English
en_output = engine.translate_text(hi_output, source_lang="Hindi", target_lang="English")
print(f"Back to English: '{en_output}'")
