"""
Text preprocessing and postprocessing for IndicTrans2 models.
Pure-Python implementation avoiding native compiler dependencies.
"""

from typing import List
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesPunctNormalizer, MosesTokenizer, MosesDetokenizer

# Supported Language Code Mappings
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


class DocumentIndicProcessor:
    """
    Handles normalization, tokenization, transliteration, and detokenization
    for IndicTrans2 models without requiring Cython/C-compiler toolchains.
    """

    def __init__(self) -> None:
        self.en_normalizer = MosesPunctNormalizer("en")
        self.en_tokenizer = MosesTokenizer("en")
        self.en_detokenizer = MosesDetokenizer("en")
        self.normalizer_factory = IndicNormalizerFactory()

    def preprocess_sentence(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Preprocesses a single sentence for IndicTrans2 model input.
        """
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
            # IndicTrans2 unified script: transliterate non-Devanagari Indic scripts to Devanagari
            if iso != "hi":
                tokenized = UnicodeIndicTransliterator.transliterate(tokenized, iso, "hi")

        return f"{src_lang} {tgt_lang} {tokenized}"

    def preprocess_batch(self, sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Preprocesses a batch of sentences for the model.
        """
        return [self.preprocess_sentence(s, src_lang, tgt_lang) for s in sentences]

    def postprocess_sentence(self, text: str, tgt_lang: str) -> str:
        """
        Postprocesses a decoded sentence from IndicTrans2 output back to readable format.
        """
        if not text or not text.strip():
            return ""

        cleaned_text = text.strip()
        iso = FLORES_TO_ISO.get(tgt_lang, "en")

        if tgt_lang == "eng_Latn":
            return self.en_detokenizer.detokenize(cleaned_text.split())
        else:
            # Transliterate from internal Devanagari back to target script if necessary
            if iso != "hi":
                cleaned_text = UnicodeIndicTransliterator.transliterate(cleaned_text, "hi", iso)
            return indic_detokenize.trivial_detokenize(cleaned_text, iso)

    def postprocess_batch(self, sentences: List[str], tgt_lang: str) -> List[str]:
        """
        Postprocesses a batch of decoded sentences.
        """
        return [self.postprocess_sentence(s, tgt_lang) for s in sentences]
