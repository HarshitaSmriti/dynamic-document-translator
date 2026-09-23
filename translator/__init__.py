"""
Dynamic Document Translator Core Package.
"""

from .language_detection import detect_document_language, DetectionResult
from .translation import TranslationEngine, UnsupportedRouteError
from .document import translate_docx_document
from .model import load_translation_pipeline, get_device

__all__ = [
    "detect_document_language",
    "DetectionResult",
    "TranslationEngine",
    "UnsupportedRouteError",
    "translate_docx_document",
    "load_translation_pipeline",
    "get_device",
]
