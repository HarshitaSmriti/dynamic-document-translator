"""
Text preprocessing and postprocessing for IndicTrans2 models.
"""

import re
import html
from typing import List, Optional
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer

LANG_TO_FLORES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
}

FLORES_TO_LANG = {v: k for k, v in LANG_TO_FLORES.items()}

FLORES_TO_ISO = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "ben_Beng": "bn",
}


def clean_translation_formatting(text: str, original_text: str = "", tgt_lang: str = "hin_Deva") -> str:
    """
    Cleans up common formatting and punctuation artifacts in translation outputs:
    - HTML and XML entities (&lt;, &gt;, &amp;, etc.)
    - Spacing around punctuation, brackets, quotes, and colons
    - Hyphenated identifiers, numbers, and version strings
    - Terminal colon preservation for headers and labels
    """
    if not text:
        return ""

    t = text.strip()

    # Unescape HTML/XML artifacts
    t = re.sub(r'&lt\s*_?\s*-\s*_?\s*&gt\s*_?', '<->', t)
    t = re.sub(r'&lt\s*;\s*', '<', t)
    t = re.sub(r'&gt\s*;\s*', '>', t)
    t = re.sub(r'&amp\s*;\s*', '&', t)
    t = re.sub(r'&quot\s*;\s*', '"', t)
    t = re.sub(r'&apos\s*;\s*', "'", t)
    t = html.unescape(t)
    t = re.sub(r'&\s*(?:एम्प|এম্প|एम्पीयर|amp)\b', '&', t, flags=re.IGNORECASE)

    # Spacing around punctuation
    t = re.sub(r'\s+([,;:!?।॥\.\%\)\]\}\>])', r'\1', t)
    t = re.sub(r'([\(\[\{\<“"\'])\s+', r'\1', t)

    # Numbers, versions, and percentages
    t = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', t)
    t = re.sub(r'([A-Za-z0-9]+)\s*-\s*([A-Za-z0-9]+)', r'\1-\2', t)
    t = re.sub(r'\b([vV])\s+(\d+)', r'\1\2', t)
    t = re.sub(r'(\d+)\s*%', r'\1%', t)
    t = re.sub(r'([\$€£₹])\s*(\d+)', r'\1\2', t)

    # Quotes
    t = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', t)
    t = re.sub(r'“\s+([^”]+?)\s+”', r'“\1”', t)

    # Parenthesis balancing
    if original_text:
        for open_ch, close_ch in [('(', ')'), ('[', ']'), ('{', '}')]:
            orig_close_count = original_text.count(close_ch)
            trans_close_count = t.count(close_ch)
            if orig_close_count < trans_close_count and (close_ch * 2) in t:
                t = t.replace(close_ch * 2, close_ch)

    # Deduplicate punctuation
    t = re.sub(r'[:ः]{2,}', ':', t)
    t = re.sub(r'।{2,}', '।', t)
    t = re.sub(r'\.{2,}', '.', t)
    t = re.sub(r'([।\.])\s*:', ':', t)

    # Preserve terminal colon if original had colon
    if original_text and original_text.rstrip().endswith(":") and not t.endswith(":"):
        if t.endswith("।") or t.endswith(".") or t.endswith("॥") or t.endswith("ः"):
            t = t[:-1].rstrip() + ":"
        else:
            t = t + ":"

    t = re.sub(r'[ \t]{2,}', ' ', t)
    return t


class DocumentIndicProcessor:
    """
    Handles normalization, tokenization, transliteration, and detokenization.
    """

    def __init__(self) -> None:
        self.en_normalizer = MosesPunctNormalizer("en")
        self.en_tokenizer = MosesTokenizer("en")
        self.en_detokenizer = MosesDetokenizer("en")
        self.normalizer_factory = IndicNormalizerFactory()

    def preprocess_sentence(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or not text.strip():
            return f"{src_lang} {tgt_lang} "

        cleaned_text = text.strip()

        if src_lang == "eng_Latn":
            norm = self.en_normalizer.normalize(cleaned_text)
            tok_list = self.en_tokenizer.tokenize(norm)
            tokenized = " ".join(tok_list)
        else:
            iso = FLORES_TO_ISO.get(src_lang, "hi")
            normalizer = self.normalizer_factory.get_normalizer(iso)
            norm = normalizer.normalize(cleaned_text)
            toks = indic_tokenize.trivial_tokenize(norm, iso)
            tokenized = " ".join(toks)
            if iso != "hi":
                tokenized = UnicodeIndicTransliterator.transliterate(tokenized, iso, "hi")

        return f"{src_lang} {tgt_lang} {tokenized}"

    def preprocess_batch(self, sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        return [self.preprocess_sentence(s, src_lang, tgt_lang) for s in sentences]

    def postprocess_sentence(self, text: str, tgt_lang: str, original_text: str = "") -> str:
        if not text or not text.strip():
            return ""

        cleaned_text = text.strip()
        iso = FLORES_TO_ISO.get(tgt_lang, "en")

        if tgt_lang == "eng_Latn":
            detok = self.en_detokenizer.detokenize(cleaned_text.split())
        else:
            if iso != "hi":
                cleaned_text = UnicodeIndicTransliterator.transliterate(cleaned_text, "hi", iso)
            detok = indic_detokenize.trivial_detokenize(cleaned_text, iso)

        return clean_translation_formatting(detok, original_text=original_text, tgt_lang=tgt_lang)

    def postprocess_batch(
        self, sentences: List[str], tgt_lang: str, original_sentences: Optional[List[str]] = None
    ) -> List[str]:
        if original_sentences and len(original_sentences) == len(sentences):
            return [
                self.postprocess_sentence(s, tgt_lang, orig)
                for s, orig in zip(sentences, original_sentences)
            ]
        return [self.postprocess_sentence(s, tgt_lang) for s in sentences]


