import torch
from torch import nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerNorm


class RoPE(nn.Module):
    """Applies rotary position embeddings (RoPE) to input tensors.

    Implementation adapted from:
    https://github.com/amazon-science/chronos-forecasting/blob/f951d9aefa06f5389b2ed6b0e51fd5a1a4cf194b/src/chronos/chronos2/layers.py#L18
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/models/llama/modeling_llama.py#L95
    """

    def __init__(self, dim: int, base: float = 10000):
        super().__init__()

        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.inv_freq: torch.Tensor  # type hint for type checker
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_embed, k_embed


class T5AttentionWithRoPE(T5Attention):
    def __init__(self, config, layer_idx=None, use_rope=False, rope_theta=10000.0):
        super().__init__(config, has_relative_attention_bias=False, layer_idx=layer_idx)
        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.key_value_proj_dim, base=rope_theta)

    def forward(
        self,
        hidden_states,
        mask,
        key_value_states=None,
        past_key_value=None,
        layer_head_mask=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        position_ids=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if self.use_rope:
                # --- RoPE application ---
                assert position_ids is not None, "position_ids is required but found None"
                cos, sin = self.rope_embed(value_states, position_ids)
                query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores = scores + mask

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Self-attention layer using T5AttentionWithRoPE
class T5LayerSelfAttentionWithRoPE(nn.Module):
    def __init__(self, config, layer_idx=None, use_rope=False, rope_theta=10000.0):
        super().__init__()
        self.use_rope = use_rope
        if use_rope:
            self.SelfAttention = T5AttentionWithRoPE(config, layer_idx=layer_idx, use_rope=True, rope_theta=rope_theta)
        else:
            self.SelfAttention = T5AttentionWithRoPE(config, layer_idx=layer_idx, use_rope=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
        position_ids=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        if self.use_rope:
            assert position_ids is not None, "position_ids is required but found None"
            attention_output = self.SelfAttention(
                normed_hidden_states,
                mask=attention_mask,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_ids=position_ids,
            )
        else:
            attention_output = self.SelfAttention(
                normed_hidden_states,
                mask=attention_mask,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs
    

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class Patch(nn.Module):
    """
    Implementation adapted from:
    https://github.com/amazon-science/chronos-forecasting/blob/f951d9aefa06f5389b2ed6b0e51fd5a1a4cf194b/src/chronos/chronos_bolt.py#L50
    """

    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride) # batch_size x num_patches x patch_size
        return x


class InstanceNorm(nn.Module):
    """
    Implementation adapted from:
    https://github.com/amazon-science/chronos-forecasting/blob/f951d9aefa06f5389b2ed6b0e51fd5a1a4cf194b/src/chronos/chronos_bolt.py#L71
    """

    def __init__(self, eps: float = 1e-5, use_arcsinh: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale

        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)

        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale

        if self.use_arcsinh:
            x = torch.sinh(x)

        x = x * scale + loc

        return x.to(orig_dtype)
    

class ICLMaskedMeanPooling(nn.Module):
    """
    Mean Pools over the given dimension utilizing the mask.
    Input: (batch, ..., N, d_model), (batch, ..., N)
    Output: (batch, ..., 1, d_model), (batch, ..., 1)
    """
    def __init__(self, pooling_dimension: int):
        super().__init__()
        self.pooling_dimension = pooling_dimension

    def forward(self, x, mask):
        # x: (batch, num_examples, total_patches, d_model)
        # mask: (batch, num_examples, total_patches)
        sum_x = (x * mask.unsqueeze(-1)).sum(dim=self.pooling_dimension, keepdim=True) 
        # (batch, num_examples, 1, d_model)
        pooled_mask = mask.sum(dim=self.pooling_dimension, keepdim=True)  
        # (batch, num_examples, 1)
        pooled = sum_x / pooled_mask.clamp(min=1e-6).unsqueeze(-1)
        pooled_mask = (pooled_mask > 0).to(mask.dtype)  
        # (batch, num_examples, 1)
        return pooled, pooled_mask


class TokenReadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, Tok, Xflat, allow_mask, tok_valid_mask=None):
        """
        Tok:        [N, T, d]
        Xflat:      [N, K, d]   where K = S*P
        allow_mask: [N, T, K]   bool; True = allowed attention i.e., token t may attend key k
        tok_valid_mask: [N, T] bool; invalid tokens get zero update

        Returns: deltaTok [N, T, d]
        """
        N, T, d = Tok.shape
        _, K, _ = Xflat.shape

        Q = self.q_proj(Tok)     # [N,T,d]
        Kp = self.k_proj(Xflat)  # [N,K,d]
        Vp = self.v_proj(Xflat)  # [N,K,d]

        # Split Heads: [N,h,T,dk] and [N,h,K,dk]
        # Qh: [N,h,T,dk], Kh/Vh: [N,h,K,dk]
        Qh = Q.view(N, T, self.h, self.dk).transpose(1, 2)
        Kh = Kp.view(N, K, self.h, self.dk).transpose(1, 2)
        Vh = Vp.view(N, K, self.h, self.dk).transpose(1, 2)

        # Convert allow_mask to additive mask broadcastable to [N,h,T,K]
        attn_mask = torch.where(
            allow_mask[:, None, :, :],  # [N,1,T,K]
            torch.tensor(0.0, device=Tok.device, dtype=Tok.dtype),
            torch.tensor(torch.finfo(Tok.dtype).min, device=Tok.device, dtype=Tok.dtype)
        )

        out = nn.functional.scaled_dot_product_attention(
            Qh, Kh, Vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [N,h,T,dk]

        # Merge heads -> [N,T,d]
        out = out.transpose(1, 2).contiguous().view(N, T, d)
        out = self.o_proj(out)

        # Zero out invalid tokens (e.g., query's END, query's target_fut header)
        if tok_valid_mask is not None:
            out = out * tok_valid_mask[:, :, None].to(out.dtype)

        return out    
    

class TokenSelfAttention(nn.Module):
    """Multi-head self-attention over token slots.

    Takes pre-normed tokens and a boolean validity mask,
    returns a delta to be added as a residual.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, Tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Self-attention over tokens with key-padding mask.

        Args:
            Tok:  [N, T, d]  pre-normed token embeddings
            mask: [N, T]     bool; True = valid token, False = padded/invalid

        Returns:
            delta: [N, T, d]  residual update (invalid tokens zeroed)
        """
        N, T, d = Tok.shape

        Q = self.q_proj(Tok)   # [N, T, d]
        K = self.k_proj(Tok)   # [N, T, d]
        V = self.v_proj(Tok)   # [N, T, d]

        # Split heads: [N, h, T, dk]
        Qh = Q.view(N, T, self.h, self.dk).transpose(1, 2)
        Kh = K.view(N, T, self.h, self.dk).transpose(1, 2)
        Vh = V.view(N, T, self.h, self.dk).transpose(1, 2)

        # Key-padding → additive mask [N, 1, 1, T]
        attn_mask = torch.where(
            mask[:, None, None, :],
            torch.tensor(0.0, device=Tok.device, dtype=Tok.dtype),
            torch.tensor(torch.finfo(Tok.dtype).min, device=Tok.device, dtype=Tok.dtype),
        )

        out = nn.functional.scaled_dot_product_attention(
            Qh, Kh, Vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [N, h, T, dk]

        # Merge heads → [N, T, d]
        out = out.transpose(1, 2).contiguous().view(N, T, d)
        out = self.o_proj(out)

        # Zero out invalid tokens
        out = out * mask[:, :, None].to(out.dtype)

        return out


class CrossExampleAttn(nn.Module):
    """
    Query uses [START], [MID]
    Examples provide [START], [MID] per example

    q_tokens:  (B, 2, d)
    ex_tokens: (B, E, 2, d)

    returns:
        delta_q:  (B, 2, d)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = TokenReadCrossAttention(d_model, n_heads, dropout)

    def forward(self, q_tokens: torch.Tensor, ex_tokens: torch.Tensor) -> torch.Tensor:
        B, E, two, d = ex_tokens.shape
        assert two == 2

        ex_tokens_flat = rearrange(ex_tokens, 'B E two d -> B (E two) d')  # (B, 2E, d)
        allow = torch.ones((B, 2, E * 2), dtype=torch.bool, device=q_tokens.device)

        return self.attn(
            Tok=q_tokens,
            Xflat=ex_tokens_flat,
            allow_mask=allow,
            tok_valid_mask=torch.ones((B, 2), dtype=torch.bool, device=q_tokens.device),
        )


class TokenWrite(nn.Module):
    """
    [START] -> global FiLM over all query patches
    [MID]   -> future-only FiLM over query future patches
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.global_proj = nn.Linear(d_model, 2 * d_model)
        self.future_proj = nn.Linear(d_model, 2 * d_model)

    def forward(
        self,
        X_q: torch.Tensor,          # (B,1,Sq,Pq,d)
        start_token: torch.Tensor,  # (B,d)
        mid_token: torch.Tensor,    # (B,d)
        num_hist_patches: int,
    ) -> torch.Tensor:
        # Global write from START
        g_start, b_start = self.global_proj(start_token).chunk(2, dim=-1)
        X_q = X_q * (1.0 + g_start[:, None, None, None, :]) + b_start[:, None, None, None, :]

        # Future-only write from MID
        g_mid, b_mid = self.future_proj(mid_token).chunk(2, dim=-1)
        X_future = X_q[:, :, :, num_hist_patches:, :]
        X_future = X_future * (1.0 + g_mid[:, None, None, None, :]) + b_mid[:, None, None, None, :]
        X_q = torch.cat([X_q[:, :, :, :num_hist_patches, :], X_future], dim=3)

        return X_q


class PatchDecoder(nn.Module):
    """
    Lightweight patch decoder.
    Input:
        (B, N_future_patches, d_model)
    Output:
        (B, N_future_patches, num_quantiles * output_patch_size)
    """
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_proj = nn.Linear(in_dim, h_dim)
        self.output_proj = nn.Linear(h_dim, out_dim)
        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.act = ACT2FN[act_fn_name]

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = T5LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.hidden_proj(x))
        decoded = self.dropout(self.output_proj(hidden))
        skip = self.skip_proj(x)

        out = decoded + skip

        if self.use_layer_norm:
            out = self.layer_norm(out)
        return out