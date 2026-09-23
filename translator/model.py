"""
Model and Tokenizer loader with runtime environment compatibility shims.
Provides Streamlit resource-cached loaders with GPU/CPU auto-detection.
"""

import sys
import types
from pathlib import Path
from typing import Tuple, Any
import importlib.util
import torch
import transformers
import transformers.utils
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForSeq2SeqLM

from .processor import DocumentIndicProcessor

# ---------------------------------------------------------------------------
# Runtime Compatibility Shims for Transformers 4.32.1 & IndicTrans Checkpoints
# ---------------------------------------------------------------------------

def _apply_transformers_compatibility_shims() -> None:
    """
    Applies non-invasive shims to ensure older/custom IndicTrans2 modules
    function seamlessly with Transformers 4.32.1.
    """
    # 1. Provide _attn_implementation property on PretrainedConfig
    if not hasattr(PretrainedConfig, "_attn_implementation"):
        PretrainedConfig._attn_implementation = property(
            lambda self: getattr(self, "attn_implementation", "eager")
        )

    # 2. Provide transformers.modeling_attn_mask_utils if missing
    if "transformers.modeling_attn_mask_utils" not in sys.modules:
        mod_attn = types.ModuleType("transformers.modeling_attn_mask_utils")

        def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        def _prepare_4d_causal_attention_mask(
            attention_mask: torch.Tensor,
            input_shape,
            inputs_embeds: torch.Tensor,
            past_key_values_length: int = 0,
            sliding_window = None,
        ):
            bsz, tgt_len = input_shape
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((tgt_len, tgt_len), fill_value=min_dtype, dtype=dtype, device=device)
            if tgt_len > 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)
            if past_key_values_length > 0:
                causal_mask = torch.cat(
                    [torch.zeros((bsz, 1, tgt_len, past_key_values_length), dtype=dtype, device=device), causal_mask],
                    dim=-1,
                )
            if attention_mask is not None:
                expanded_mask = _prepare_4d_attention_mask(attention_mask, dtype, tgt_len=tgt_len)
                causal_mask = causal_mask + expanded_mask
            return causal_mask

        mod_attn._prepare_4d_attention_mask = _prepare_4d_attention_mask
        mod_attn._prepare_4d_causal_attention_mask = _prepare_4d_causal_attention_mask
        mod_attn._prepare_4d_attention_mask_for_sdpa = _prepare_4d_attention_mask
        mod_attn._prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask
        sys.modules["transformers.modeling_attn_mask_utils"] = mod_attn

    # 3. Provide transformers.integrations.deepspeed if missing
    if "transformers.integrations.deepspeed" not in sys.modules:
        import transformers.deepspeed as ds
        mod_integrations = types.ModuleType("transformers.integrations")
        mod_ds = types.ModuleType("transformers.integrations.deepspeed")
        mod_ds.is_deepspeed_zero3_enabled = ds.is_deepspeed_zero3_enabled
        mod_integrations.deepspeed = mod_ds
        sys.modules["transformers.integrations"] = mod_integrations
        sys.modules["transformers.integrations.deepspeed"] = mod_ds

    # 4. Flash Attention availability flags
    if not hasattr(transformers.utils, "is_flash_attn_2_available"):
        transformers.utils.is_flash_attn_2_available = lambda: False
    if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):
        transformers.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False


# Apply shims on module import
_apply_transformers_compatibility_shims()


def get_device() -> torch.device:
    """
    Returns CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_custom_indictrans_tokenizer(model_dir: Path) -> Any:
    """
    Safely loads the IndicTransTokenizer from the checkpoint directory,
    resolving token recursion and missing added_tokens_decoder attributes.
    """
    tokenization_path = model_dir / "tokenization_indictrans.py"
    if not tokenization_path.exists():
        raise FileNotFoundError(f"Missing custom tokenizer file: {tokenization_path}")

    spec = importlib.util.spec_from_file_location("tokenization_indictrans", tokenization_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Resolve token recursion bug and attribute initialization
    mod.IndicTransTokenizer.added_tokens_encoder = {}
    mod.IndicTransTokenizer.added_tokens_decoder = {}
    mod.IndicTransTokenizer._convert_token_to_id = (
        lambda self, token: self.encoder.get(token, self.encoder.get(self.unk_token, 3))
    )

    tokenizer = mod.IndicTransTokenizer(
        src_vocab_fp=str(model_dir / "dict.SRC.json"),
        tgt_vocab_fp=str(model_dir / "dict.TGT.json"),
        src_spm_fp=str(model_dir / "model.SRC"),
        tgt_spm_fp=str(model_dir / "model.TGT"),
    )
    return tokenizer


def load_translation_pipeline(model_dir: Path) -> Tuple[Any, Any, DocumentIndicProcessor]:
    """
    Loads tokenizer, Seq2SeqLM model, and DocumentIndicProcessor for a given checkpoint.
    """
    _apply_transformers_compatibility_shims()
    device = get_device()

    tokenizer = load_custom_indictrans_tokenizer(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), trust_remote_code=True)
    model.to(device)
    model.eval()

    processor = DocumentIndicProcessor()
    return tokenizer, model, processor
