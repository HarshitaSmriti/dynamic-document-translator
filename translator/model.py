"""
Model and tokenizer loaders with runtime compatibility shims for IndicTrans2.
"""

import sys
import types
import importlib.util
from pathlib import Path
from typing import Tuple, Any
import torch
import transformers
import transformers.utils
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoModelForSeq2SeqLM

from .processor import DocumentIndicProcessor


def _apply_transformers_compatibility_shims() -> None:
    """
    Shims to ensure custom IndicTrans2 architecture modules
    work with modern Transformers releases.
    """
    if not hasattr(PretrainedConfig, "_attn_implementation"):
        PretrainedConfig._attn_implementation = property(
            lambda self: getattr(self, "attn_implementation", "eager")
        )

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

    if "transformers.integrations.deepspeed" not in sys.modules:
        import transformers.deepspeed as ds
        mod_integrations = types.ModuleType("transformers.integrations")
        mod_ds = types.ModuleType("transformers.integrations.deepspeed")
        mod_ds.is_deepspeed_zero3_enabled = ds.is_deepspeed_zero3_enabled
        mod_integrations.deepspeed = mod_ds
        sys.modules["transformers.integrations"] = mod_integrations
        sys.modules["transformers.integrations.deepspeed"] = mod_ds

    if "transformers.onnx" not in sys.modules:
        mod_onnx = types.ModuleType("transformers.onnx")
        mod_onnx_utils = types.ModuleType("transformers.onnx.utils")

        class OnnxConfig:
            pass

        class OnnxSeq2SeqConfigWithPast(OnnxConfig):
            pass

        mod_onnx.OnnxConfig = OnnxConfig
        mod_onnx.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
        mod_onnx_utils.compute_effective_axis_dimension = lambda *args, **kwargs: 1

        sys.modules["transformers.onnx"] = mod_onnx
        sys.modules["transformers.onnx.utils"] = mod_onnx_utils

    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "_tie_or_clone_weights"):
        def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
            output_embeddings.weight = input_embeddings.weight
            if getattr(output_embeddings, "bias", None) is not None:
                pad_len = output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]
                if pad_len > 0:
                    output_embeddings.bias.data = torch.nn.functional.pad(
                        output_embeddings.bias.data,
                        (0, pad_len),
                        "constant",
                        0,
                    )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings

        PreTrainedModel._tie_or_clone_weights = _tie_or_clone_weights

    if not hasattr(transformers.utils, "is_flash_attn_2_available"):
        transformers.utils.is_flash_attn_2_available = lambda: False
    if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):
        transformers.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

    try:
        import transformers.dynamic_module_utils as dmu
        if not getattr(dmu, "_indic_patched", False):
            orig_get_class = dmu.get_class_in_module

            def _patched_get_class(class_name, module, **kwargs):
                cls = orig_get_class(class_name, module, **kwargs)
                if hasattr(cls, "tie_weights"):
                    orig_tw = cls.tie_weights

                    def _wrapped_tw(self, *a, **kw):
                        try:
                            return orig_tw(self, *a, **kw)
                        except TypeError:
                            return orig_tw(self)

                    cls.tie_weights = _wrapped_tw
                return cls

            dmu.get_class_in_module = _patched_get_class
            dmu._indic_patched = True
    except Exception:
        pass


_apply_transformers_compatibility_shims()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_custom_indictrans_tokenizer(model_dir: Path) -> Any:
    """
    Loads IndicTransTokenizer from local checkpoint directory.
    """
    tokenization_path = model_dir / "tokenization_indictrans.py"
    if not tokenization_path.exists():
        raise FileNotFoundError(f"Missing custom tokenizer file: {tokenization_path}")

    spec = importlib.util.spec_from_file_location("tokenization_indictrans", tokenization_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.IndicTransTokenizer.added_tokens_encoder = {}
    mod.IndicTransTokenizer.added_tokens_decoder = {}
    mod.IndicTransTokenizer._added_tokens_encoder = {}
    mod.IndicTransTokenizer._added_tokens_decoder = {}
    mod.IndicTransTokenizer._special_tokens_map = {}
    mod.IndicTransTokenizer._convert_token_to_id = (
        lambda self, token: self.encoder.get(token, self.encoder.get(self.unk_token, 3))
    )

    orig_init = mod.IndicTransTokenizer.__init__

    def _patched_init(self, *args, **kwargs):
        self.__dict__["_special_tokens_map"] = {}
        self.__dict__["_added_tokens_encoder"] = {}
        self.__dict__["_added_tokens_decoder"] = {}
        self.__dict__["added_tokens_encoder"] = {}
        self.__dict__["added_tokens_decoder"] = {}
        orig_init(self, *args, **kwargs)

    mod.IndicTransTokenizer.__init__ = _patched_init

    tokenizer = mod.IndicTransTokenizer(
        src_vocab_fp=str(model_dir / "dict.SRC.json"),
        tgt_vocab_fp=str(model_dir / "dict.TGT.json"),
        src_spm_fp=str(model_dir / "model.SRC"),
        tgt_spm_fp=str(model_dir / "model.TGT"),
    )
    return tokenizer


def load_translation_pipeline(model_dir: Path) -> Tuple[Any, Any, DocumentIndicProcessor]:
    """
    Loads tokenizer, Seq2Seq model, and text processor.
    """
    _apply_transformers_compatibility_shims()
    device = get_device()

    tokenizer = load_custom_indictrans_tokenizer(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), trust_remote_code=True)
    model.to(device)
    model.eval()

    processor = DocumentIndicProcessor()
    return tokenizer, model, processor

