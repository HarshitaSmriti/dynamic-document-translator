"""
Translation engine for batched inference and route validation.
"""

import os
from pathlib import Path
from typing import List, Callable, Optional, Dict
import torch

from .model import load_translation_pipeline, get_device
from .processor import LANG_TO_FLORES, DocumentIndicProcessor

if not torch.cuda.is_available():
    cpu_cores = min(4, os.cpu_count() or 4)
    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(min(2, cpu_cores))


class UnsupportedRouteError(ValueError):
    pass


class TranslationEngine:
    """
    Manages translation pipelines, route validation, and batched generation.
    """

    def __init__(self, models_root: Path) -> None:
        self.models_root = Path(models_root)
        self.en_indic_path = self.models_root / "fine_tuned" / "en_indic_fine_tuned"
        self.indic_en_path = self.models_root / "fine_tuned" / "indic_en_fine_tuned"

        self._pipelines: Dict[str, tuple] = {}
        self.device = get_device()

    def _get_pipeline(self, route_type: str):
        if route_type not in self._pipelines:
            model_path = self.en_indic_path if route_type == "en_indic" else self.indic_en_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint directory not found at: {model_path}")
            self._pipelines[route_type] = load_translation_pipeline(model_path)
        return self._pipelines[route_type]

    def validate_route(self, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            raise UnsupportedRouteError(f"Source and target languages are identical ({source_lang}).")

        if source_lang == "English" and target_lang in ["Hindi", "Bengali"]:
            return "en_indic"
        elif source_lang in ["Hindi", "Bengali"] and target_lang == "English":
            return "indic_en"
        elif source_lang in ["Hindi", "Bengali"] and target_lang in ["Hindi", "Bengali"]:
            raise UnsupportedRouteError(
                f"Direct translation between {source_lang} and {target_lang} is disabled. "
                "Only English <-> Indic routes are supported."
            )
        else:
            raise UnsupportedRouteError(f"Translation route from '{source_lang}' to '{target_lang}' is not supported.")

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 4,
        max_length: int = 256,
        num_beams: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        if not texts:
            return []

        route_type = self.validate_route(source_lang, target_lang)
        tokenizer, model, processor = self._get_pipeline(route_type)

        src_flores = LANG_TO_FLORES[source_lang]
        tgt_flores = LANG_TO_FLORES[target_lang]

        results = [""] * len(texts)
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]

        if not non_empty_indices:
            return results

        total_items = len(non_empty_indices)
        processed_count = 0

        for chunk_start in range(0, total_items, batch_size):
            chunk_indices = non_empty_indices[chunk_start : chunk_start + batch_size]
            chunk_texts = [texts[i] for i in chunk_indices]

            preprocessed = processor.preprocess_batch(chunk_texts, src_lang=src_flores, tgt_lang=tgt_flores)

            inputs = tokenizer(
                preprocessed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    use_cache=False,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            postprocessed = processor.postprocess_batch(
                decoded, tgt_lang=tgt_flores, original_sentences=chunk_texts
            )

            for idx, trans_text in zip(chunk_indices, postprocessed):
                results[idx] = trans_text

            processed_count += len(chunk_indices)
            if progress_callback:
                progress_callback(processed_count, total_items)

        return results

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip():
            return text
        return self.translate_batch([text], source_lang, target_lang)[0]


