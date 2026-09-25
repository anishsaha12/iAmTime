import copy
import logging
from dataclasses import dataclass
import math
from typing import List, cast, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerNorm,
    T5Config
)
from transformers.utils import ModelOutput

from .layers import (
    ResidualBlock,
    Patch,
    InstanceNorm,
    TokenReadCrossAttention,
    TokenSelfAttention,
    TokenWrite,
    PatchDecoder,
)
from .blocks import (
    CompositeICLEncoderStack,
    PatchCrossAttentionDecoder,
    init_tokens_for_blocks, 
    exog_exists_from_masks, 
    token_slots, 
    build_token_valid_mask, 
    build_allow_patchsplit
)

logger = logging.getLogger(__file__)


@dataclass
class iAmTimeConfig:
    context_length: int
    patch_size: int
    patch_stride: int
    quantiles: List[float]
    num_experts: int
    use_arcsinh: bool
    max_output_steps: int
    use_reg_token: bool = False


@dataclass
class ICLConfig:
    context_num_examples: int # max number of examples before query
    context_example_history_length: int # max length of each history-target/exogenous/query-history-target time-series (before [MID])
    context_example_future_length: int # max length of each future-target/future-exogenous/query-future-target time-series (after [MID])
    use_future_exog_token: bool
    start_token_id: int
    target_series_token_id: int
    exogenous_token_id: int
    mid_token_id: int
    future_exogenous_token_id: int
    end_token_id: int


@dataclass
class iAmTimeOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    quantile_preds: Optional[torch.Tensor] = None


class iAmTimeModel(PreTrainedModel):
    config_class = T5Config  # type: ignore[assignment]
    _supports_long_horizon: bool = True
    _supports_future_covariates: bool = True
    _supports_sdpa: bool = True
    supports_gradient_checkpointing = True

    def __init__(self, config: T5Config):
        assert hasattr(config, "iamtime_config"), "Not a valid config"

        super().__init__(config)
        self.config: T5Config
        self.model_dim = config.d_model

        self.iamtime_config = iAmTimeConfig(**config.iamtime_config)
        self.icl_config = ICLConfig(**config.icl_config)

        # Only [PAD] token (and [REG] token)
        if self.iamtime_config.use_reg_token:
            config.reg_token_id = 1

        config.vocab_size = 2 if self.iamtime_config.use_reg_token else 1
        self.shared_emb = nn.Embedding(config.vocab_size, config.d_model)

        config.icl_vocab_size = 6 if self.icl_config.use_future_exog_token else 5
        self.shared_token_emb = nn.Embedding(config.icl_vocab_size, config.d_model)

        self.input_patch_embedding = ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim=self.iamtime_config.patch_size * 3,
            h_dim=config.d_ff,
            out_dim=config.d_model,
            act_fn_name=config.dense_act_fn,
            dropout_p=config.dropout_rate,
        )

        self.patch = Patch(
            patch_size=self.iamtime_config.patch_size, 
            patch_stride=self.iamtime_config.patch_stride
        )

        self.instance_norm = InstanceNorm(use_arcsinh=self.iamtime_config.use_arcsinh)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = CompositeICLEncoderStack(encoder_config)

        self.num_quantiles = len(self.iamtime_config.quantiles)
        quantiles = torch.tensor(self.iamtime_config.quantiles, dtype=self.dtype)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.decoder = PatchCrossAttentionDecoder(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_quantiles=self.num_quantiles,
            output_patch_size=self.iamtime_config.patch_size,
            act_fn_name=config.dense_act_fn,
            num_heads=self.iamtime_config.num_experts,
            dropout_p=config.dropout_rate,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize weights."""
        factor = self.config.initializer_factor

        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)

        elif isinstance(module, T5Attention):
            # Mesh TF attention init (absorbs 1/√dk into weights)
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))

        elif isinstance(module, T5DenseActDense):
            # FFN input/output projections
            module.wi.weight.data.normal_(mean=0.0, std=factor * (self.config.d_model ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * (self.config.d_ff ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()

        elif isinstance(module, T5DenseGatedActDense):
            # Gated FFN variant
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * (self.config.d_model ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * (self.config.d_model ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * (self.config.d_ff ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()

        elif isinstance(module, (TokenReadCrossAttention, TokenSelfAttention)):
            # Same convention as T5Attention for q/k/v/o projections
            d = module.d
            dk = module.dk
            h = module.h
            module.q_proj.weight.data.normal_(mean=0.0, std=factor * ((d * dk) ** -0.5))
            module.k_proj.weight.data.normal_(mean=0.0, std=factor * (d ** -0.5))
            module.v_proj.weight.data.normal_(mean=0.0, std=factor * (d ** -0.5))
            module.o_proj.weight.data.normal_(mean=0.0, std=factor * ((h * dk) ** -0.5))

        elif isinstance(module, TokenWrite):
            # FiLM projections: d → 2d
            d_model = self.config.d_model
            module.global_proj.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            if module.global_proj.bias is not None:
                module.global_proj.bias.data.zero_()
            module.future_proj.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            if module.future_proj.bias is not None:
                module.future_proj.bias.data.zero_()

        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.hidden_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()
            module.residual_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.residual_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()
            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

        elif isinstance(module, PatchDecoder):
            module.hidden_proj.weight.data.normal_(
                mean=0.0, std=factor * (module.hidden_proj.weight.size(-1) ** -0.5)
            )
            if hasattr(module.hidden_proj, "bias") and module.hidden_proj.bias is not None:
                module.hidden_proj.bias.data.zero_()
            module.output_proj.weight.data.normal_(
                mean=0.0, std=factor * (module.output_proj.weight.size(-1) ** -0.5)
            )
            if hasattr(module.output_proj, "bias") and module.output_proj.bias is not None:
                module.output_proj.bias.data.zero_()
            module.skip_proj.weight.data.normal_(
                mean=0.0, std=factor * (module.skip_proj.weight.size(-1) ** -0.5)
            )
            if hasattr(module.skip_proj, "bias") and module.skip_proj.bias is not None:
                module.skip_proj.bias.data.zero_()

        elif isinstance(module, PatchCrossAttentionDecoder):
            # Query projection: 3d → d
            module.q_proj.weight.data.normal_(
                mean=0.0, std=factor * ((3 * self.config.d_model) ** -0.5)
            )
            # Expert keys: unit normal
            module.expert_keys.data.normal_(mean=0.0, std=factor * 1.0)

        elif isinstance(module, iAmTimeModel):
            module.shared_emb.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.shared_token_emb.weight.data.normal_(mean=0.0, std=factor * 1.0)

    def _check_or_create_mask_for_tensor(
            self, input_tensor: torch.Tensor, 
            input_tensor_mask: Optional[torch.Tensor] = None
    ):
        input_tensor_mask = (
            input_tensor_mask.to(input_tensor.dtype)
            if input_tensor_mask is not None
            else torch.isnan(input_tensor).logical_not().to(input_tensor.dtype)
        )
        return input_tensor_mask

    def _pack_context(
        self,
        _context_target: Optional[torch.Tensor] = None,         # (batch, num_examples, ctx_len) or None
        _context_exog: Optional[torch.Tensor] = None,           # (batch, num_examples, num_exog, ctx_len) or None
        _context_target_mask: Optional[torch.Tensor] = None,    # (batch, num_examples, ctx_len) or None
        _context_exog_mask: Optional[torch.Tensor] = None,      # (batch, num_examples, num_exog, ctx_len) or None
    ):
        assert not(
            (_context_target is None) and (_context_exog is None)
        ), "Context Target and Exogenous cannot both be None simultaneously"
        
        # Encode all example context (target + exog)
        # Stack all context: (batch, num_examples, 1+num_exog, ctx_len)
        if _context_target is not None:
            batch_size, num_examples, ctx_len = _context_target.shape
            num_exog = _context_exog.shape[2] if _context_exog is not None else 0
            example_contexts = [_context_target.unsqueeze(2)]
            if num_exog > 0:
                example_contexts.append(_context_exog)
        elif _context_exog is not None:
            batch_size, num_examples, num_exog, ctx_len = _context_exog.shape
            example_contexts = [_context_exog,]
        example_contexts = torch.cat(example_contexts, dim=2)  # (batch, num_examples, 1+num_exog, ctx_len)
        
        # Flatten for encoding
        flat_ex_context = rearrange(example_contexts, 'b n m l -> (b n m) l')  # (batch*num_examples*(1+num_exog), ctx_len)
        flat_ex_context_mask = None

        ex_ctx_masks = None
        if _context_target is not None:
            if _context_target_mask is not None:
                ex_ctx_masks = [_context_target_mask.unsqueeze(2)]
                if num_exog > 0:
                    ex_ctx_masks.append(_context_exog_mask)
        elif _context_exog is not None:
            if _context_exog_mask is not None:
                ex_ctx_masks = [_context_exog_mask,]
        if ex_ctx_masks is not None:
            ex_ctx_masks = torch.cat(ex_ctx_masks, dim=2)
            flat_ex_context_mask = rearrange(ex_ctx_masks, 'b n m l -> (b n m) l')
        
        return flat_ex_context, flat_ex_context_mask, (batch_size, num_examples, num_exog, ctx_len)
    
    def _patch_and_embed(self, raw, mask, is_history, loc_scale=None):
        """
        Args:
            raw: input tensor (history or future)
            mask: mask tensor (history_mask or future_mask)
            is_history: bool, True for history, False for future
            loc_scale: tuple, required for future (for scaling)
        Returns:
            embeds: output embeddings
            attention_mask: mask for attention
            loc_scale: scaling tuple (only for history)
            patched_mask: mask after patching (only for future)
        """
        patch_size = self.iamtime_config.patch_size
        max_length = self.iamtime_config.context_length if is_history else None
        
        # # If raw is None, create zero raw and mask
        # if not is_history and raw is None:
        #     raw = torch.zeros(batch_size, num_future_patches * patch_size, device=self.device, dtype=self.dtype)
        #     mask = torch.zeros_like(raw)
        
        # Setup mask
        mask = mask.to(raw.dtype) if mask is not None else torch.isnan(raw).logical_not().to(raw.dtype)
        
        batch_size = raw.shape[0]
        seq_length = raw.shape[1]
        if not is_history:
            num_future_patches = math.ceil(seq_length / patch_size)
        
        # Truncate if context
        if is_history and seq_length > max_length:
            raw = raw[..., -max_length:]
            mask = mask[..., -max_length:]
        # Normalize
        if is_history:
            raw, loc_scale = self.instance_norm(raw)
        else:
            raw, _ = self.instance_norm(raw, loc_scale)
        raw = raw.to(self.dtype)
        mask = mask.to(self.dtype)
        # Pad if future and needed
        if not is_history and num_future_patches * patch_size > raw.shape[-1]:
            padding_shape = (*raw.shape[:-1], num_future_patches * patch_size - raw.shape[-1])
            raw = torch.cat([raw, torch.zeros(padding_shape).to(raw)], dim=-1)
            mask = torch.cat([mask, torch.zeros(padding_shape).to(mask)], dim=-1)
        # Guard: pad to at least one patch so .unfold() doesn't crash on
        # empty dummy examples (e.g. query-only episodes with 0 support).
        # Padded with zeros + zero mask → produces 1 fully-masked-out patch.
        if raw.shape[-1] < patch_size:
            pad_len = patch_size - raw.shape[-1]
            raw = torch.cat([torch.zeros(*raw.shape[:-1], pad_len, device=raw.device, dtype=raw.dtype), raw], dim=-1)
            mask = torch.cat([torch.zeros(*mask.shape[:-1], pad_len, device=mask.device, dtype=mask.dtype), mask], dim=-1)
        # Patch
        patched = self.patch(raw)
        patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
        patched = torch.where(patched_mask > 0.0, patched, 0.0)
        attention_mask = patched_mask.sum(dim=-1) > 0
        num_patches = attention_mask.shape[-1]
        # Time encoding
        if is_history:
            time_enc = torch.arange(start=-num_patches * patch_size, end=0, device=self.device, dtype=torch.float32)
        else:
            time_enc = torch.arange(start=0, end=num_patches * patch_size, device=self.device, dtype=torch.float32)
        time_enc = repeat(time_enc, "(n p) -> b n p", b=batch_size, n=num_patches, p=patch_size)
        time_enc = time_enc.div(cast(int, self.iamtime_config.context_length)).to(self.dtype)
        
        patched = torch.cat([time_enc, patched, patched_mask], dim=-1)
        
        embeds = self.input_patch_embedding(patched)
        if is_history:
            # append [REG] special token embedding, if needed
            if self.iamtime_config.use_reg_token:
                reg_input_ids = torch.full((batch_size, 1), self.config.reg_token_id, device=embeds.device)
                reg_embeds = self.shared_emb(reg_input_ids)
                embeds = torch.cat([embeds, reg_embeds], dim=-2)
                attention_mask = torch.cat([
                    attention_mask.to(self.dtype), 
                    (
                        attention_mask.sum(dim=-1, keepdim=True) > 0
                    ).to(self.dtype)
                    # torch.ones_like(reg_input_ids).to(self.dtype)
                ], dim=-1)
        
        return embeds, attention_mask, loc_scale, patched_mask
        
    def _reshape_tensor_and_mask(self, tensor, mask, shape):
        """
        Reshape both tensor and mask from flat to multi-dimensional form.
        Args:
            tensor: input tensor, shape (batch_size*num_examples*(1+num_exog), num_patches, ...)
            mask: input mask, shape (batch_size*num_examples*(1+num_exog), num_patches)
            shape: tuple (batch_size, num_examples, num_exog, ctx_len)
        Returns:
            reshaped_tensor: (batch_size, num_examples, 1+num_exog, num_patches, ...)
            reshaped_mask: (batch_size, num_examples, 1+num_exog, num_patches)
        """
        batch_size, num_examples, num_exog, ctx_len = shape
        n_m = 1 + num_exog
        if tensor is not None:
            tensor = rearrange(tensor, '(b n m) p ... -> b n m p ...', b=batch_size, n=num_examples, m=n_m)
        if mask is not None:
            mask = rearrange(mask, '(b n m) p -> b n m p', b=batch_size, n=num_examples, m=n_m)
        return tensor, mask
    
    def _concat_hist_fut(self, enc_hist, mask_hist, enc_fut, mask_fut):
        # enc_*: (..., P, d), mask_*: (..., P)
        X = torch.cat([enc_hist, enc_fut], dim=-2)          # concat along patch axis
        M = torch.cat([mask_hist, mask_fut], dim=-1).bool()
        return X, M
    
    def _get_token_embedding(self, token_id):
        """Look up a single token embedding via self.shared_token_emb forward pass."""
        ids = torch.tensor([token_id], device=self.device)
        return self.shared_token_emb(ids).squeeze(0)  # (d_model,)

    def _init_tokens_for_blocks(self, num_blocks, E_max):
        # Initialize token embeddings for multiple blocks in the time series model.
        return init_tokens_for_blocks(
            num_blocks=num_blocks,
            E_max=E_max,
            embed_fn=self._get_token_embedding,
            icl_config=self.icl_config,
            model_dim=self.model_dim,
            device=self.device,
        )
    
    @staticmethod
    def _pad_exog_exists(exists: torch.Tensor, E_max: int) -> torch.Tensor:
        """Pad exog_exists to E_max columns so token_slots(E_max) works."""
        N, Ecur = exists.shape
        if Ecur == E_max: return exists
        out = torch.zeros((N, E_max), dtype=torch.bool, device=exists.device)
        out[:, :Ecur] = exists
        return out
    
    def _resolve_prediction_length(
        self,
        prediction_length: Optional[int],
        target: Optional[torch.Tensor],
        query_exog_future: Optional[torch.Tensor],
    ) -> int:
        """Infer or validate the prediction horizon length."""
        if prediction_length is None:
            if target is not None:
                return target.shape[1]
            elif query_exog_future is not None:
                return query_exog_future.shape[-1]
            else:
                return self.iamtime_config.patch_size * self.iamtime_config.max_output_steps
        assert prediction_length > 0, "prediction_length must be positive"
        assert (target is None) or (target.shape[1] == prediction_length), \
            "target shape does not match prediction_length"
        assert (query_exog_future is None) or (query_exog_future.shape[-1] == prediction_length), \
            "query_exog_future shape does not match prediction_length"
        return prediction_length
    
    def _embed_component(
        self,
        target: torch.Tensor,
        exog: Optional[torch.Tensor],
        target_mask: Optional[torch.Tensor],
        exog_mask: Optional[torch.Tensor],
        is_history: bool,
        loc_scale=None,
    ):
        """Pack, patch-embed, and reshape a single component (examples or query, history or future).

        Args:
            target:      (B, E, seq_len)
            exog:        (B, E, num_exog, seq_len) or None
            target_mask: (B, E, seq_len) or None
            exog_mask:   (B, E, num_exog, seq_len) or None
            is_history:  True for history, False for future
            loc_scale:   scaling tuple from a paired history encoding

        Returns:
            enc:       (B, E, 1+num_exog, num_patches, d_model)
            mask:      (B, E, 1+num_exog, num_patches)
            loc_scale: scaling tuple (meaningful only for history)
        """
        flat, flat_mask, shape = self._pack_context(
            _context_target=target,
            _context_exog=exog,
            _context_target_mask=target_mask,
            _context_exog_mask=exog_mask,
        )
        if (not is_history) and loc_scale is not None:
            loc_scale = (
                rearrange(loc_scale[0], 'b n m 1 -> (b n m) 1'),
                rearrange(loc_scale[1], 'b n m 1 -> (b n m) 1'),
            )
        enc, attention_mask, loc_scale_out, _ = self._patch_and_embed(
            raw=flat, mask=flat_mask, is_history=is_history, loc_scale=loc_scale,
        )
        if is_history and loc_scale_out is not None:
            loc_scale_out = (
                self._reshape_tensor_and_mask(loc_scale_out[0], None, shape)[0],
                self._reshape_tensor_and_mask(loc_scale_out[1], None, shape)[0],
            )
        enc, attention_mask = self._reshape_tensor_and_mask(enc, attention_mask, shape)
        return enc, attention_mask, loc_scale_out
    
    def _build_encoder_inputs(
        self,
        enc_ex_hist, ex_hist_attention_mask, enc_ex_fut, ex_fut_attention_mask,
        enc_query_hist, query_hist_attention_mask, enc_query_fut, query_fut_attention_mask,
        batch_size, num_examples, num_exog, num_query_exog
    ):
        """Concatenate history/future, build tokens, validity masks, and allow masks.

        Returns:
            X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
            X_q,  M_q,  Tok_q,  Mtok_q,  allow_q,
            start_idx, mid_idx, P_h_q
        """
        # 1) concat hist+fut
        X_ex, M_ex = self._concat_hist_fut(
            enc_ex_hist, ex_hist_attention_mask,
            enc_ex_fut,  ex_fut_attention_mask
        ) # (B,E,Sx,Pex,d), (B,E,Sx,Pex)

        X_q,  M_q  = self._concat_hist_fut(
            enc_query_hist, query_hist_attention_mask,
            enc_query_fut,  query_fut_attention_mask
        ) # (B,1,Sq,Pq,d), (B,1,Sq,Pq)

        P_h_ex = enc_ex_hist.shape[-2]
        P_h_q  = enc_query_hist.shape[-2]

        num_exog_ex = X_ex.shape[2] - 1
        num_exog_q  = X_q.shape[2]  - 1

        assert num_exog==num_exog_ex
        assert num_query_exog==num_exog_q

        E_max = max(num_exog_ex, num_exog_q)

        # 2) exog existence (for token validity)
        exog_exists_ex = exog_exists_from_masks(ex_hist_attention_mask, ex_fut_attention_mask)   # (B*E,num_exog_ex)
        exog_exists_q  = exog_exists_from_masks(query_hist_attention_mask, query_fut_attention_mask)  # (B,num_exog_q)

        # pad exog_exists to E_max so token_slots(E_max) works
        exog_exists_ex_p = self._pad_exog_exists(exog_exists_ex, E_max)   # (B*E,E_max)
        exog_exists_q_p  = self._pad_exog_exists(exog_exists_q,  E_max)   # (B,E_max)

        # 3) init tokens
        T, start_idx, _, _, mid_idx, _, _, _ = token_slots(E_max)
        Tok_ex = self._init_tokens_for_blocks(batch_size*num_examples, E_max).reshape(batch_size,num_examples,T,self.model_dim)
        Tok_q  = self._init_tokens_for_blocks(batch_size, E_max).reshape(batch_size,1,T,self.model_dim)

        # 4) token validity masks
        Mtok_ex = build_token_valid_mask(exog_exists_ex_p, E_max, is_query=False).reshape(batch_size,num_examples,T)
        Mtok_q  = build_token_valid_mask(exog_exists_q_p,  E_max, is_query=True).reshape(batch_size,1,T)

        # 5) allow masks from patch split
        allow_ex = build_allow_patchsplit(M_ex, P_h_ex, E_max, is_query=False)  # (B*E,T,Sx*Pex)
        allow_q  = build_allow_patchsplit(M_q,  P_h_q,  E_max, is_query=True)   # (B,T,Sq*Pq)

        # 6) allow future target patches attention in encoder for query
        M_q[:, :, 0, P_h_q:] = True

        return (
            X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
            X_q, M_q, Tok_q, Mtok_q, allow_q,
            start_idx, mid_idx, P_h_q,
        )
    
    def encode(
        self,
        example_target_histories: torch.Tensor,                  # (batch, num_examples, hist_len)
        example_target_futures: torch.Tensor,                    # (batch, num_examples, future_len)
        example_exog_histories: Optional[torch.Tensor] = None,   # (batch, num_examples, num_exog, hist_len) or None
        example_exog_futures: Optional[torch.Tensor] = None,     # (batch, num_examples, num_exog, future_len) or None
        query_target_history: Optional[torch.Tensor] = None,     # (batch, query_hist_len)
        query_exog_history: Optional[torch.Tensor] = None,       # (batch, num_exog, query_hist_len) or None
        query_exog_future: Optional[torch.Tensor] = None,        # (batch, num_exog, future_len) or None
        example_target_hist_mask: Optional[torch.Tensor] = None, # (batch, num_examples, hist_len) or None
        example_target_fut_mask: Optional[torch.Tensor] = None,  # (batch, num_examples, future_len) or None
        example_exog_hist_mask: Optional[torch.Tensor] = None,   # (batch, num_examples, num_exog, hist_len) or None
        example_exog_fut_mask: Optional[torch.Tensor] = None,    # (batch, num_examples, num_exog, future_len) or None
        query_target_hist_mask: Optional[torch.Tensor] = None,   # (batch, query_hist_len) or None
        query_exog_hist_mask: Optional[torch.Tensor] = None,     # (batch, num_exog, query_hist_len) or None
        query_exog_fut_mask: Optional[torch.Tensor] = None,      # (batch, num_exog, future_len) or None
        prediction_length: Optional[int] = None,                 # future_len: int, required if both target and query_exog_fut_mask are None
    ):
        batch_size, num_examples, _ = example_target_histories.shape
        
        num_exog = example_exog_histories.shape[2] if example_exog_histories is not None else 0
        num_query_exog = query_exog_history.shape[1] if query_exog_history is not None else 0

        # Prepare query future target and mask as zeros since they are to be predicted
        query_target_future = torch.zeros(batch_size, prediction_length, device=self.device, dtype=self.dtype)
        query_target_fut_mask = torch.zeros_like(query_target_future)

        # Check and Create Masks
        example_target_hist_mask = self._check_or_create_mask_for_tensor(example_target_histories, example_target_hist_mask)
        example_target_fut_mask = self._check_or_create_mask_for_tensor(example_target_futures, example_target_fut_mask)
        example_exog_hist_mask = self._check_or_create_mask_for_tensor(example_exog_histories, example_exog_hist_mask)
        example_exog_fut_mask = self._check_or_create_mask_for_tensor(example_exog_futures, example_exog_fut_mask)
        query_target_hist_mask = self._check_or_create_mask_for_tensor(query_target_history, query_target_hist_mask)
        query_exog_hist_mask = self._check_or_create_mask_for_tensor(query_exog_history, query_exog_hist_mask)
        query_exog_fut_mask = self._check_or_create_mask_for_tensor(query_exog_future, query_exog_fut_mask)

        # Encode all four components: pack → patch → embed → reshape
        enc_ex_hist, ex_hist_attention_mask, ex_hist_loc_scale = self._embed_component(
            example_target_histories, example_exog_histories,
            example_target_hist_mask, example_exog_hist_mask,
            is_history=True,
        ) # (batch_size, num_examples, (1+num_exog), num_ex_hist_patches, d_model)
        enc_ex_fut, ex_fut_attention_mask, _ = self._embed_component(
            example_target_futures, example_exog_futures,
            example_target_fut_mask, example_exog_fut_mask,
            is_history=False, loc_scale=ex_hist_loc_scale,
        ) # (batch_size, num_examples, (1+num_exog), num_ex_fut_patches, d_model)
        enc_query_hist, query_hist_attention_mask, query_hist_loc_scale = self._embed_component(
            query_target_history.unsqueeze(1),
            query_exog_history.unsqueeze(1) if query_exog_history is not None else None,
            query_target_hist_mask.unsqueeze(1) if query_target_hist_mask is not None else None,
            query_exog_hist_mask.unsqueeze(1) if query_exog_hist_mask is not None else None,
            is_history=True,
        ) # (batch_size, 1, (1+num_query_exog), num_query_hist_patches, d_model)
        enc_query_fut, query_fut_attention_mask, _ = self._embed_component(
            query_target_future.unsqueeze(1),
            query_exog_future.unsqueeze(1) if query_exog_future is not None else None,
            query_target_fut_mask.unsqueeze(1),
            query_exog_fut_mask.unsqueeze(1) if query_exog_fut_mask is not None else None,
            is_history=False, loc_scale=query_hist_loc_scale,
        )  # (batch_size, 1, (1+num_query_exog), num_query_fut_patches, d_model)

        # Build encoder inputs: concat hist/fut, tokens, validity & allow masks
        (
            X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
            X_q, M_q, Tok_q, Mtok_q, allow_q,
            start_idx, mid_idx, P_h_q,
        ) = self._build_encoder_inputs(
            enc_ex_hist, ex_hist_attention_mask, enc_ex_fut, ex_fut_attention_mask,
            enc_query_hist, query_hist_attention_mask, enc_query_fut, query_fut_attention_mask,
            batch_size, num_examples, num_exog, num_query_exog
        )

        # Run the encoder stack
        X_ex, Tok_ex, X_q, Tok_q = self.encoder(
            X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
            X_q,  M_q,  Tok_q,  Mtok_q,  allow_q,
            P_h_q=P_h_q, start_idx=start_idx, mid_idx=mid_idx,
        )
        
        return (
            (X_ex, Tok_ex, X_q, Tok_q), 
            (M_ex, Mtok_ex, M_q, Mtok_q), 
            (query_hist_loc_scale, P_h_q, start_idx, mid_idx)
        )
    
    def _compute_loss(self, quantile_preds, target, target_mask, query_hist_loc_scale):
        # normalize target
        target, _ = self.instance_norm(target, query_hist_loc_scale)
        target = target.unsqueeze(1)  # type: ignore
        assert quantile_preds.shape[-1] >= target.shape[-1]

        target = target.to(quantile_preds.device)
        target_mask = (
            target_mask.unsqueeze(1).to(quantile_preds.device)
            if target_mask is not None
            else ~torch.isnan(target)
        )
        target = torch.where(target_mask, target, torch.zeros_like(target))

        # pad target and target_mask if they are shorter than model's prediction_length
        if quantile_preds.shape[-1] > target.shape[-1]:
            padding_shape = (
                *target.shape[:-1],
                quantile_preds.shape[-1] - target.shape[-1]
            )
            target = torch.cat(
                [target, torch.zeros(padding_shape).to(target)], dim=-1
            )
            target_mask = torch.cat(
                [target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1
            )

        loss = (
            2
            * torch.abs(
                (target - quantile_preds)
                * (
                    (target <= quantile_preds).float()
                    - rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
                )
            )
            * target_mask.float()
        )
        loss = loss.mean(dim=-1)  # Mean over prediction horizon
        loss = loss.sum(dim=-1)  # Sum over quantile levels
        loss = loss.mean()  # Mean over batch
        return loss
    
    def forward(
        self,
        example_target_histories: torch.Tensor,                  # (batch, num_examples, hist_len)
        example_target_futures: torch.Tensor,                    # (batch, num_examples, future_len)
        example_exog_histories: Optional[torch.Tensor] = None,   # (batch, num_examples, num_exog, hist_len) or None
        example_exog_futures: Optional[torch.Tensor] = None,     # (batch, num_examples, num_exog, future_len) or None
        query_target_history: Optional[torch.Tensor] = None,     # (batch, query_hist_len)
        query_exog_history: Optional[torch.Tensor] = None,       # (batch, num_exog, query_hist_len) or None
        query_exog_future: Optional[torch.Tensor] = None,        # (batch, num_exog, future_len) or None
        example_target_hist_mask: Optional[torch.Tensor] = None, # (batch, num_examples, hist_len) or None
        example_target_fut_mask: Optional[torch.Tensor] = None,  # (batch, num_examples, future_len) or None
        example_exog_hist_mask: Optional[torch.Tensor] = None,   # (batch, num_examples, num_exog, hist_len) or None
        example_exog_fut_mask: Optional[torch.Tensor] = None,    # (batch, num_examples, num_exog, future_len) or None
        query_target_hist_mask: Optional[torch.Tensor] = None,   # (batch, query_hist_len) or None
        query_exog_hist_mask: Optional[torch.Tensor] = None,     # (batch, num_exog, query_hist_len) or None
        query_exog_fut_mask: Optional[torch.Tensor] = None,      # (batch, num_exog, future_len) or None
        target: Optional[torch.Tensor] = None,                   # (batch, future_len) for loss
        target_mask: Optional[torch.Tensor] = None,
        prediction_length: Optional[int] = None,                 # future_len: int, required if both target and query_exog_fut_mask are None
    ) -> iAmTimeOutput:
        """
        Forward pass for ICL-style input. See docstring for input shapes.
        This method processes a batch of example time series (with optional exogenous variables) and a query time series,
        encoding their histories and futures, fusing representations, and decoding to produce quantile forecasts. It also
        computes the quantile loss if target values are provided.

        Args:
            example_target_histories (torch.Tensor): Historical target series for examples, shape (batch, num_examples, hist_len).
            example_target_futures (torch.Tensor): Future target series for examples, shape (batch, num_examples, future_len).
            example_exog_histories (Optional[torch.Tensor]): Historical exogenous variables for examples, shape (batch, num_examples, num_exog, hist_len) or None.
            example_exog_futures (Optional[torch.Tensor]): Future exogenous variables for examples, shape (batch, num_examples, num_exog, future_len) or None.
            query_target_history (Optional[torch.Tensor]): Historical target series for the query, shape (batch, query_hist_len) or None.
            query_exog_history (Optional[torch.Tensor]): Historical exogenous variables for the query, shape (batch, num_exog, query_hist_len) or None.
            query_exog_future (Optional[torch.Tensor]): Future exogenous variables for the query, shape (batch, num_exog, future_len) or None.
            example_target_hist_mask (Optional[torch.Tensor]): Mask for example target histories, shape (batch, num_examples, hist_len) or None.
            example_target_fut_mask (Optional[torch.Tensor]): Mask for example target futures, shape (batch, num_examples, future_len) or None.
            example_exog_hist_mask (Optional[torch.Tensor]): Mask for example exogenous histories, shape (batch, num_examples, num_exog, hist_len) or None.
            example_exog_fut_mask (Optional[torch.Tensor]): Mask for example exogenous futures, shape (batch, num_examples, num_exog, future_len) or None.
            query_target_hist_mask (Optional[torch.Tensor]): Mask for query target history, shape (batch, query_hist_len) or None.
            query_exog_hist_mask (Optional[torch.Tensor]): Mask for query exogenous history, shape (batch, num_exog, query_hist_len) or None.
            query_exog_fut_mask (Optional[torch.Tensor]): Mask for query exogenous future, shape (batch, num_exog, future_len) or None.
            target (Optional[torch.Tensor]): Ground truth target values for the query, shape (batch, future_len), used for loss computation.
            target_mask (Optional[torch.Tensor]): Mask for the target values, shape (batch, future_len) or None.
            prediction_length (Optional[int]): Length of the prediction horizon (future_len), required if both target and query_exog_fut_mask are None.

        Returns:
            iAmTimeOutput: An object containing:
                - loss (Optional[torch.Tensor]): The computed quantile loss if target is provided, else None.
                - quantile_preds (torch.Tensor): Predicted quantiles for the query, shape (batch_size, num_quantiles, prediction_length).

        Notes:
            - All input tensors should be properly shaped and masked as described.
            - The method normalizes inputs, encodes context, fuses representations, and decodes to produce forecasts.
            - If target is provided, the quantile loss is computed and returned.
        """
        batch_size = example_target_histories.shape[0]
        prediction_length = self._resolve_prediction_length(
            prediction_length, target, query_exog_future,
        )
        (_, _, X_q, Tok_q), _, (query_hist_loc_scale, P_h_q, start_idx, mid_idx) = self.encode(
            example_target_histories=example_target_histories, example_target_futures=example_target_futures, 
            example_exog_histories=example_exog_histories, example_exog_futures=example_exog_futures, 
            query_target_history=query_target_history, query_exog_history=query_exog_history, 
            query_exog_future=query_exog_future, 
            example_target_hist_mask=example_target_hist_mask, example_target_fut_mask=example_target_fut_mask, 
            example_exog_hist_mask=example_exog_hist_mask, example_exog_fut_mask=example_exog_fut_mask, 
            query_target_hist_mask=query_target_hist_mask, query_exog_hist_mask=query_exog_hist_mask, 
            query_exog_fut_mask=query_exog_fut_mask, 
            prediction_length=prediction_length
        )

        # Decoder uses X_q and Tok_q ([START], [MID]) conditioning
        query_target_hist_loc_scale = (
            query_hist_loc_scale[0][:, 0, 0, :], 
            query_hist_loc_scale[1][:, 0, 0, :]
        )
        quantile_preds = self.decoder(
            future_patch_h=X_q[:, 0, 0, P_h_q:, :], # (B, N_future_patches, d_model)
            start_token=Tok_q[:, 0, start_idx, :],
            mid_token=Tok_q[:, 0, mid_idx, :],
            prediction_length=prediction_length,
            return_attn_weights=True,
        ) 
        quantile_preds = quantile_preds[0] # shape: (B, num_quantiles, prediction_length)

        if target is not None:
            loss = self._compute_loss(quantile_preds, target, target_mask, query_target_hist_loc_scale) 
        else: loss = None

        # Unscale predictions
        quantile_preds = rearrange(
            quantile_preds, "b q h -> b (q h)",
            b=batch_size, q=self.num_quantiles, h=prediction_length,
        )
        quantile_preds = self.instance_norm.inverse(quantile_preds, query_target_hist_loc_scale)
        quantile_preds = rearrange(
            quantile_preds, "b (q h) -> b q h",
            q=self.num_quantiles, h=prediction_length,
        )

        return iAmTimeOutput(loss=loss, quantile_preds=quantile_preds)