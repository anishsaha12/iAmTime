import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange
from typing import Optional
from .layers import (
    TokenReadCrossAttention, 
    TokenSelfAttention,
    CrossExampleAttn, 
    TokenWrite, 
    T5LayerSelfAttentionWithRoPE,
    PatchDecoder
)
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5LayerFF

def series_indices(E_max: int):
    """
    History side: T_hist + E_max exog histories
    Future side: T_fut (examples only) + E_max future exogs (examples + query)
    So S = 2 + 2*E_max slots.
    Returns indices into the S dimension for:
      T_hist, X_hist[k], T_fut, X_fut[k]
    """
    idx_T_hist = 0
    idx_X_hist = [1 + k for k in range(E_max)]          # 1..E_max
    idx_T_fut  = 1 + E_max
    idx_X_fut  = [idx_T_fut + 1 + k for k in range(E_max)]
    S = 2 + 2*E_max
    return S, idx_T_hist, idx_X_hist, idx_T_fut, idx_X_fut

def token_slots(E_max: int):
    """
    Returns indices into the T dimension for:
        START, TARGET_HIST, EXOG_HIST[k], MID, TARGET_FUT, FUTURE_EXOG[k], END
    Example: for E_max=2, we have T=9 slots:
        0 START
        1 TARGET_HIST
        2 EXOG1_HIST
        3 EXOG2_HIST
        4 MID
        5 TARGET_FUT
        6 FUT_EXOG1
        7 FUT_EXOG2
        8 END
    """
    start = 0
    target_hist = 1
    exog_hist = [2 + k for k in range(E_max)]
    mid = 2 + E_max
    target_fut = 3 + E_max
    fut_exog = [4 + E_max + k for k in range(E_max)]
    end = 4 + 2*E_max
    T = end + 1
    return T, start, target_hist, exog_hist, mid, target_fut, fut_exog, end

def exog_count_per_example(example_exog_hist_mask, example_exog_fut_mask):
    """
    Returns a boolean tensor indicating the existence of exogenous variables
    for each example and each exogenous variable.
    For examples:
        example_exog_hist_mask: (B, num_examples, num_exog, hist_len) (float/bool)
        exog k exists for (b,e) if it has any valid timestep in history or future.
    """
    # returns exog_exists: [B, E, num_exog] bool
    ex_hist_any = example_exog_hist_mask.bool().any(dim=-1)  # [B,E,num_exog]
    ex_fut_any  = example_exog_fut_mask.bool().any(dim=-1)   # [B,E,num_exog]
    exog_exists = ex_hist_any | ex_fut_any
    return exog_exists  # per-example/per-exog existence

def exog_exists_query(query_exog_hist_mask, query_exog_fut_mask):
    """Returns a boolean tensor indicating the existence of exogenous variables for the query.
    For the query:
        query_exog_hist_mask: (B, num_exog, q_hist_len)
        query_exog_fut_mask: (B, num_exog, fut_len)
    """
    q_hist_any = query_exog_hist_mask.bool().any(dim=-1)  # [B,num_exog]
    q_fut_any  = query_exog_fut_mask.bool().any(dim=-1)   # [B,num_exog]
    return q_hist_any | q_fut_any

def pack_series_slot_patch_mask_examples(
    tgt_hist_patch_mask,  # [B,E,1,P_ex]
    ex_hist_patch_mask,   # [B,E,num_exog,P_ex] or None
    tgt_fut_patch_mask,   # [B,E,1,P_ex]
    ex_fut_patch_mask,    # [B,E,num_exog,P_ex] or None
    E_max: int,
):
    """
    Packs the example patch masks into a single mask M of shape [B,E,S,P_ex], where S is the total number of series slots.
    The series slots are ordered as:
    - T_hist (target history)
    - X_hist[k] (exogenous histories)
    - T_fut (target future)
    - X_fut[k] (exogenous futures)
    The function fills the appropriate slots in M with the corresponding patch masks, and leaves the rest as False.
    Map them into the series slots:
        slot T_hist gets target-history patch mask
        slots X_hist[k] get exog-history patch masks
        slot T_fut gets target-future patch mask
        slots X_fut[k] get exog-future patch masks
        Same for query, except slot T_fut is all zeros (since missing).
    """
    B, E, _, P = tgt_hist_patch_mask.shape
    S, idx_T_hist, idx_X_hist, idx_T_fut, idx_X_fut = series_indices(E_max)

    M = torch.zeros((B, E, S, P), dtype=torch.bool, device=tgt_hist_patch_mask.device)
    M[:, :, idx_T_hist, :] = tgt_hist_patch_mask[:, :, 0, :].bool()
    M[:, :, idx_T_fut,  :] = tgt_fut_patch_mask[:, :, 0, :].bool()

    if ex_hist_patch_mask is not None:
        num_exog = ex_hist_patch_mask.shape[2]
        for k in range(min(num_exog, E_max)):
            M[:, :, idx_X_hist[k], :] = ex_hist_patch_mask[:, :, k, :].bool()
    if ex_fut_patch_mask is not None:
        num_exog = ex_fut_patch_mask.shape[2]
        for k in range(min(num_exog, E_max)):
            M[:, :, idx_X_fut[k], :] = ex_fut_patch_mask[:, :, k, :].bool()

    return M  # [B,E,S,P_ex]

def build_token_valid_mask(exog_exists, E_max: int, is_query: bool):
    """
    exog_exists:
        examples: [N, E_max] bool (per block)
        query:    [N, E_max] bool (per block, N=batch here)
    We'll mark token slots invalid if:
        that exog k doesn't exist for that block
        query doesn't have future target header or END
    returns M_tok: [N, T] bool
    """
    N = exog_exists.shape[0]
    T, start, target_hist, exog_hist, mid, target_fut, fut_exog, end = token_slots(E_max)

    M_tok = torch.ones((N, T), dtype=torch.bool, device=exog_exists.device)

    # Mask missing EXOG_k and FUTURE_EXOG_k tokens if the example has fewer exogs
    # exog_count[n] = m => EXOG_{m+1..E_max} invalid
    for k in range(E_max):
        M_tok[:, exog_hist[k]] = exog_exists[:, k]
        M_tok[:, fut_exog[k]]  = exog_exists[:, k]

    if is_query:
        # Query has no TARGET_SERIES after MID and no END token
        M_tok[:, target_fut] = False
        M_tok[:, end] = False

    return M_tok

def build_token_to_series_mask(
    exog_exists,        # [N,E_max] bool, per block
    M_patch,            # [N,S,P] bool, per block patch validity
    E_max: int,
    is_query: bool,
    start_reads_future_target_in_examples: bool = True,
):
    """
    Semantics:
        [TARGET_HIST] reads T_hist
        [EXOG_k] reads Xk_hist
        [MID] reads all history-side series
        [FUTURE_EXOG_k] reads Xk_fut
        [TARGET_FUT] reads T_fut (examples only)
        [START] reads “everything known” (examples: include target future; query: exclude it)
        [END] reads everything (examples only)

    Returns token_to_series_mask: [N, T, S, P] bool
    """
    N, S, P = M_patch.shape
    T, start, target_hist, exog_hist, mid, target_fut, fut_exog, end = token_slots(E_max)
    S_check, idx_T_hist, idx_X_hist, idx_T_fut, idx_X_fut = series_indices(E_max)
    assert S_check == S

    mask = torch.zeros((N, T, S, P), dtype=torch.bool, device=M_patch.device)

    # TARGET_HIST -> T_hist
    mask[:, target_hist, idx_T_hist, :] = True

    # EXOG_k -> X_hist[k]
    for k in range(E_max):
        mask[:, exog_hist[k], idx_X_hist[k], :] = True

    # MID -> history side (T_hist + existing X_hist)
    mask[:, mid, idx_T_hist, :] = True
    for k in range(E_max):
        mask[:, mid, idx_X_hist[k], :] = exog_exists[:, k][:, None]

    # TARGET_FUT -> T_fut (examples only)
    if not is_query:
        mask[:, target_fut, idx_T_fut, :] = True

    # FUTURE_EXOG_k -> X_fut[k]
    for k in range(E_max):
        mask[:, fut_exog[k], idx_X_fut[k], :] = True

    # START -> everything known
    mask[:, start, idx_T_hist, :] = True
    for k in range(E_max):
        mask[:, start, idx_X_hist[k], :] = exog_exists[:, k][:, None]
        mask[:, start, idx_X_fut[k],  :] = exog_exists[:, k][:, None]

    if (not is_query) and start_reads_future_target_in_examples:
        mask[:, start, idx_T_fut, :] = True  # demos can include labeled future target

    # END -> everything (examples only)
    if not is_query:
        mask[:, end, :, :] = True

    # Intersect with patch validity so tokens never read padded patches
    mask = mask & M_patch[:, None, :, :]  # [N,1,S,P] broadcast

    return mask  # [N,T,S,P]

def init_tokens_for_blocks(num_blocks, E_max, embed_fn, icl_config, model_dim, device):
    """
    Initialize token embeddings for multiple blocks in the time series model.

    Args:
        num_blocks (int): Number of blocks to initialize tokens for.
        E_max (int): Maximum number of exogenous variables, used to determine token slot positions.
        embed_fn (callable): A function that takes a token_id (int) and returns a (model_dim,) embedding.
        icl_config: ICL config with token IDs and use_future_exog_token flag.
        model_dim (int): Model embedding dimension.
        device: Torch device.

    Returns:
        torch.Tensor: Tensor of shape (num_blocks, T, model_dim) containing initialized
                     token embeddings.
    """
    T, start, target_hist, exog_hist, mid, target_fut, fut_exog, end = token_slots(E_max)

    # Build each token slot as a (num_blocks, 1, model_dim) tensor, then cat.
    # Avoids in-place slice assignment which breaks torch.compile autograd.
    # IMPORTANT: each slot must be a distinct tensor (use .clone() for repeated
    # embeddings) so torch.cat produces contiguous, non-aliased storage.
    slots = [None] * T
    slots[start]        = embed_fn(icl_config.start_token_id).expand(num_blocks, -1).unsqueeze(1).clone()
    slots[target_hist]  = embed_fn(icl_config.target_series_token_id).expand(num_blocks, -1).unsqueeze(1).clone()
    exog_emb_base = embed_fn(icl_config.exogenous_token_id).expand(num_blocks, -1).unsqueeze(1)
    for s in exog_hist:
        slots[s]        = exog_emb_base.clone()
    slots[mid]          = embed_fn(icl_config.mid_token_id).expand(num_blocks, -1).unsqueeze(1).clone()
    slots[target_fut]   = embed_fn(icl_config.target_series_token_id).expand(num_blocks, -1).unsqueeze(1).clone()

    fut_exog_id = icl_config.future_exogenous_token_id if icl_config.use_future_exog_token else icl_config.exogenous_token_id
    fut_exog_emb_base = embed_fn(fut_exog_id).expand(num_blocks, -1).unsqueeze(1)
    for s in fut_exog:
        slots[s]        = fut_exog_emb_base.clone()
    slots[end]          = embed_fn(icl_config.end_token_id).expand(num_blocks, -1).unsqueeze(1).clone()

    # Fill any remaining None slots with zeros (shouldn't happen, but safe)
    zero = torch.zeros((num_blocks, 1, model_dim), device=device)
    slots = [s if s is not None else zero for s in slots]

    Tok = torch.cat(slots, dim=1)  # (num_blocks, T, model_dim)
    return Tok


def exog_exists_from_masks(hist_mask, fut_mask):
    """
    Determine which exogenous series exist (have any valid patch) per block.
    Works for both examples and query.
    Args:
        hist_mask: (B, E, Sx, P_h) , where Sx = 1+num_exog (target + exog)
        fut_mask:  (B, E, Sx, P_f)
    Returns:
        exog_exists: (B*E, num_exog) bool
        For examples with E > 1, this is (B*E, num_exog).
        For query with E = 1, this is (B, num_exog).
    """
    B, E, Sx, _ = hist_mask.shape
    num_exog = Sx - 1
    hist_any = hist_mask[:, :, 1:, :].bool().any(dim=-1)  # (B, E, num_exog)
    fut_any  = fut_mask[:,  :, 1:, :].bool().any(dim=-1)  # (B, E, num_exog)
    exog_exists = (hist_any | fut_any).reshape(B * E, num_exog)  # (B*E, num_exog)
    return exog_exists

def build_allow_patchsplit(M, P_h, E_max, is_query: bool):
    """
    Build a token-to-patch allow mask for examples or query blocks.

    For examples (is_query=False):
        M: (B, E, Sx, P) bool, where Sx = 1+num_exog, P = P_h + P_f
        Returns allow: (B*E, T, Sx*P) bool
        - TARGET_HIST reads target series (idx 0) history patches
        - EXOG_HIST[k] reads exog series k (idx 1+k) history patches
        - MID reads all series history patches
        - TARGET_FUT reads target series (idx 0) future patches
        - FUTURE_EXOG[k] reads exog series k (idx 1+k) future patches
        - START reads everything (all series, all patches incl. target future)
        - END reads everything

    For query (is_query=True):
        M: (B, 1, Sq, P) bool, where Sq = 1+num_query_exog, P = P_h + P_f
        Returns allow: (B, T, Sq*P) bool
        - TARGET_HIST reads target series (idx 0) history patches
        - EXOG_HIST[k] reads exog series k (idx 1+k) history patches
        - MID reads all series history patches
        - TARGET_FUT is all False (query has no known target future)
        - FUTURE_EXOG[k] reads exog series k (idx 1+k) future patches
        - START reads everything that exists (history + future exogs, NOT target future)
        - END is all False (query has no END token)

    In both cases the result is intersected with patch validity from M.

    Args:
        M: (B, E, Sx, P) bool patch validity mask
        P_h: number of history patches (first P_h along last dim are history)
        E_max: max number of exogenous variables (determines token slot layout)
        is_query: if True, apply query semantics; if False, apply example semantics

    Returns:
        allow: (N, T, Sx*P) bool, where N = B*E for examples, N = B for query
    """
    B, E, Sx, P = M.shape
    num_exog = Sx - 1
    N = B * E
    T, start, target_hist, exog_hist, mid, target_fut, fut_exog, end = token_slots(E_max)

    mask = torch.zeros((N, T, Sx, P), dtype=torch.bool, device=M.device)
    Mblk = M.reshape(N, Sx, P)

    hist_slice = slice(0, P_h)
    fut_slice = slice(P_h, P)

    # TARGET_HIST reads target series (0) history patches
    mask[:, target_hist, 0, hist_slice] = True

    # EXOG_HIST[k] reads series k (1..num_exog) history patches
    for k in range(min(num_exog, E_max)):
        mask[:, exog_hist[k], 1 + k, hist_slice] = True

    # MID reads all series history patches
    mask[:, mid, :, hist_slice] = True

    if not is_query:
        # TARGET_FUT reads target series future patches (examples have target future)
        mask[:, target_fut, 0, fut_slice] = True

    # FUTURE_EXOG[k] reads series k future patches
    for k in range(min(num_exog, E_max)):
        mask[:, fut_exog[k], 1 + k, fut_slice] = True

    if is_query:
        # START reads everything except target future (query doesn't know it)
        mask[:, start, :, hist_slice] = True
        for k in range(min(num_exog, E_max)):
            mask[:, start, 1 + k, fut_slice] = True
        # END is invalid for query; leave all False
    else:
        # START reads everything known (examples include target future)
        mask[:, start, :, :] = True
        # END reads everything
        mask[:, end, :, :] = True

    # Intersect with patch validity
    mask = mask & Mblk[:, None, :, :]

    # Flatten (Sx, P) -> Sx*P
    allow = mask.reshape(N, T, Sx * P)
    return allow


class CompositeICLEncoderBlock(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.time_series_attn = T5LayerSelfAttentionWithRoPE(config, layer_idx=layer_idx, use_rope=True, rope_theta=config.rope_theta)
        self.example_fusion_attn = T5LayerSelfAttentionWithRoPE(config, layer_idx=layer_idx, use_rope=False)
        self.token_read_cross_attn = TokenReadCrossAttention(config.d_model, config.num_heads, config.dropout_rate)
        self.token_self_attn = TokenSelfAttention(config.d_model, config.num_heads, config.dropout_rate)
        self.example_fusion_cross_attn = CrossExampleAttn(config.d_model, config.num_heads, config.dropout_rate)
        self.token_write = TokenWrite(config.d_model)
        self.ffn_patches = T5LayerFF(config)
        self.ffn_tokens = T5LayerFF(config)
        self.ln = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def _as_key_padding_additive_mask(self, mask_2d_bool, d_type):
        # as T5-style blocks want additive masks shaped (N,1,1,L)
        assert mask_2d_bool.ndim == 2, "attention_mask must have shape (batch, seq_len)"
        # Add new dims for attention heads and q_len
        mask_4d_bool = mask_2d_bool[:, None, None, :]
        # Invert binary mask to float mask which can be added to attention scores
        mask_4d_float = mask_4d_bool.to(dtype=d_type)
        mask_4d_float = (1.0 - mask_4d_float) * torch.finfo(d_type).min
        return mask_4d_float

    def forward(self,
        X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,   # examples
        X_q,  M_q,  Tok_q,  Mtok_q,  allow_q,    # query
        start_idx: int,
        mid_idx: int,
        P_h_q: int,
        output_attentions: bool = False,
    ):
        """
        X_ex: (B,E,Sx,Pex,d)   M_ex: (B,E,Sx,Pex)
        Tok_ex: (B,E,T,d)      Mtok_ex: (B,E,T)
        allow_ex: (B*E, T, Sx*Pex)

        X_q:  (B,1,Sq,Pq,d)    M_q: (B,1,Sq,Pq)
        Tok_q: (B,1,T,d)       Mtok_q: (B,1,T)
        allow_q: (B, T, Sq*Pq)
        """

        B,E,Sx,Pex,d = X_ex.shape
        _,_,Sq,Pq,_  = X_q.shape
        T = Tok_ex.shape[2]

        # ---- 1) Temporal self-attn over patches (within-series) ----
        # Examples: (B,E,Sx,Pex,d) -> (B*E*Sx, Pex, d)
        X = rearrange(X_ex, 'B E Sx Pex d -> (B E Sx) Pex d')
        Mp = rearrange(M_ex, 'B E Sx Pex -> (B E Sx) Pex').bool()
        position_ids = torch.arange(0, Pex, dtype=torch.long, device=X.device).unsqueeze(0)
        X = self.time_series_attn(
            X,
            position_ids=position_ids,
            attention_mask=self._as_key_padding_additive_mask(Mp, X.dtype),
            output_attentions=output_attentions,
        )[0]
        X_ex = rearrange(X, '(B E Sx) Pex d -> B E Sx Pex d', B=B, E=E, Sx=Sx)

        # Query: (B,1,Sq,Pq,d) -> (B*Sq, Pq, d)
        X = rearrange(X_q, 'B 1 Sq Pq d -> (B Sq) Pq d')
        Mp = rearrange(M_q, 'B 1 Sq Pq -> (B Sq) Pq').bool()
        position_ids = torch.arange(0, Pq, dtype=torch.long, device=X.device).unsqueeze(0)
        X = self.time_series_attn(
            X,
            position_ids=position_ids,
            attention_mask=self._as_key_padding_additive_mask(Mp, X.dtype),
            output_attentions=output_attentions,
        )[0]
        X_q = rearrange(X, '(B Sq) Pq d -> B 1 Sq Pq d', B=B, Sq=Sq)

        # ---- 2) Example fusion at each patch index ----
        # Examples: (B,E,Sx,Pex,d) -> (B*Pex,E*Sx,d)
        # Query: (B,1,Sq,Pq,d) -> (B*Pq,Sq,d)
        assert Pex == Pq, "For simplicity of implementation, we require Pex == Pq in this version. We will relax this requirement in the future."
        X = torch.cat(
            [
                rearrange(X_ex, 'B E Sx Pex d -> (B Pex) (E Sx) d'), 
                rearrange(X_q, 'B 1 Sq Pq d -> (B Pq) Sq d')
            ], 
            dim=1
        )  # (B*Pexq, E*Sx + Sq, d)
        Ms = torch.cat(
            [
                rearrange(M_ex, 'B E Sx Pex -> (B Pex) (E Sx)').bool(), 
                rearrange(M_q, 'B 1 Sq Pq -> (B Pq) Sq').bool()
            ], 
            dim=1
        )  # (B*Pexq, E*Sx + Sq)
        X = self.example_fusion_attn(
            X, 
            attention_mask=self._as_key_padding_additive_mask(Ms, X.dtype),
            output_attentions=output_attentions,
        )[0]
        X_ex, X_q = X.split([E*Sx, Sq], dim=1)
        X_ex = rearrange(X_ex, '(B Pex) (E Sx) d -> B E Sx Pex d', B=B, E=E, Sx=Sx, Pex=Pex)
        X_q = rearrange(X_q, '(B Pq) Sq d -> B 1 Sq Pq d', B=B, Sq=Sq, Pq=Pq)

        # ---- 3.1) Token READ (tokens attend to patches) ----
        # Examples blocks
        X_ex_flat = rearrange(X_ex, 'B E Sx Pex d -> (B E) (Sx Pex) d')
        Tok_ex_blk = rearrange(Tok_ex, 'B E T d -> (B E) T d')
        Mtok_ex_blk = rearrange(Mtok_ex, 'B E T -> (B E) T').bool()

        Tok_ex_blk = Tok_ex_blk + self.token_read_cross_attn(
            Tok=self.ln(Tok_ex_blk),
            Xflat=self.ln(X_ex_flat),
            allow_mask=allow_ex,
            tok_valid_mask=Mtok_ex_blk,
        )
        Tok_ex = rearrange(Tok_ex_blk, '(B E) T d -> B E T d', B=B, E=E)

        # Query blocks
        X_q_flat = rearrange(X_q, 'B 1 Sq Pq d -> B (Sq Pq) d')
        Tok_q_blk = Tok_q[:,0,:,:]          # (B,T,d)
        Mtok_q_blk = Mtok_q[:,0,:].bool()   # (B,T)

        Tok_q_blk = Tok_q_blk + self.token_read_cross_attn(
            Tok=self.ln(Tok_q_blk),
            Xflat=self.ln(X_q_flat),
            allow_mask=allow_q,
            tok_valid_mask=Mtok_q_blk,
        )
        Tok_q = Tok_q_blk.unsqueeze(1)  # (B,1,T,d) — avoid in-place write

        # ---- 3.2) Token Self-Attention within each block ----
        # Examples blocks
        Tok_ex_blk = rearrange(Tok_ex, 'B E T d -> (B E) T d')
        Mtok_ex_blk = rearrange(Mtok_ex, 'B E T -> (B E) T').bool()
        Tok_ex_blk = Tok_ex_blk + self.token_self_attn(
            self.ln(Tok_ex_blk),
            mask=Mtok_ex_blk,
        )
        Tok_ex = rearrange(Tok_ex_blk, '(B E) T d -> B E T d', B=B, E=E)

        # Query blocks
        Tok_q_blk = Tok_q[:,0,:,:]           # (B, T, d)
        Mtok_q_blk = Mtok_q[:,0,:].bool()    # (B, T)
        Tok_q_blk = Tok_q_blk + self.token_self_attn(
            self.ln(Tok_q_blk),
            mask=Mtok_q_blk,
        )
        Tok_q = Tok_q_blk.unsqueeze(1)  # (B,1,T,d) — avoid in-place write

        # ---- 4) Cross-example attn on [START] + [MID] (query reads examples) ----
        Tok_q_blk = Tok_q[:, 0, :, :]  # (B,T,d)
        q_start_mid = torch.stack(
            [Tok_q_blk[:, start_idx, :], Tok_q_blk[:, mid_idx, :]],
            dim=1
        )  # (B,2,d)
        ex_start_mid = torch.stack(
            [Tok_ex[:,:,start_idx,:], Tok_ex[:,:,mid_idx,:]],
            dim=2
        )  # (B,E,2,d)
        q_start_mid = q_start_mid + self.example_fusion_cross_attn(
            q_tokens=self.ln(q_start_mid),
            ex_tokens=self.ln(ex_start_mid),
        )  # (B,2,d)
        # Scatter START and MID tokens back without in-place ops
        # Build index tensors for scatter
        B_cur = Tok_q_blk.shape[0]
        idx_start = torch.full((B_cur, 1, d), start_idx, device=Tok_q_blk.device, dtype=torch.long)
        idx_mid = torch.full((B_cur, 1, d), mid_idx, device=Tok_q_blk.device, dtype=torch.long)
        Tok_q_blk = Tok_q_blk.scatter(1, idx_start, q_start_mid[:, 0:1, :])
        Tok_q_blk = Tok_q_blk.scatter(1, idx_mid, q_start_mid[:, 1:2, :])
        Tok_q = Tok_q_blk.unsqueeze(1)  # (B,1,T,d) — avoid in-place write

        # ---- 5) Token WRITE (FiLM) into query patches ----
        X_q = self.token_write(
            X_q=X_q,
            start_token=Tok_q[:,0,start_idx,:],
            mid_token=Tok_q[:,0,mid_idx,:],
            num_hist_patches=P_h_q,
        )

        # ---- 6) FFNs ----
        X_ex = self.ffn_patches(X_ex)
        Tok_ex = self.ffn_tokens(Tok_ex)

        X_q = self.ffn_patches(X_q)
        Tok_q = self.ffn_tokens(Tok_q)

        return X_ex, Tok_ex, X_q, Tok_q
    

class CompositeICLEncoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.ModuleList(
            [CompositeICLEncoderBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.ln = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gradient_checkpointing = False

    def forward(
        self,
        X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
        X_q,  M_q,  Tok_q,  Mtok_q,  allow_q,
        start_idx: int,
        mid_idx: int,
        P_h_q: int,
        output_attentions: bool = False,
    ):
        """
        Run the full CompositeICL encoder stack.

        Args:
            X_ex:      (B, E, Sx, Pex, d)  example patch embeddings
            M_ex:      (B, E, Sx, Pex)      example patch masks
            Tok_ex:    (B, E, T, d)         example token embeddings
            Mtok_ex:   (B, E, T)            example token validity masks
            allow_ex:  (B*E, T, Sx*Pex)     example allow masks
            X_q:       (B, 1, Sq, Pq, d)    query patch embeddings
            M_q:       (B, 1, Sq, Pq)       query patch masks
            Tok_q:     (B, 1, T, d)         query token embeddings
            Mtok_q:    (B, 1, T)            query token validity masks
            allow_q:   (B, T, Sq*Pq)        query allow masks
            P_h_q:     int, number of history patches in the query
            start_idx: int, index of the START token slot
            mid_idx:   int, index of the MID token slot
            output_attentions: bool

        Returns:
            X_ex, Tok_ex, X_q, Tok_q after all encoder layers,
            with final layer norm and dropout applied to patch embeddings.
        """
        X_ex = self.dropout(X_ex)
        X_q  = self.dropout(X_q)

        for layer in self.block:
            if self.gradient_checkpointing and self.training:
                X_ex, Tok_ex, X_q, Tok_q = torch_checkpoint(
                    layer,
                    X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
                    X_q,  M_q,  Tok_q,  Mtok_q,  allow_q,
                    start_idx, mid_idx, P_h_q, output_attentions,
                    use_reentrant=False,
                )
            else:
                X_ex, Tok_ex, X_q, Tok_q = layer(
                    X_ex, M_ex, Tok_ex, Mtok_ex, allow_ex,
                    X_q,  M_q,  Tok_q,  Mtok_q,  allow_q,
                    start_idx=start_idx,
                    mid_idx=mid_idx,
                    P_h_q=P_h_q,
                    output_attentions=output_attentions,
                )

        # Final layer norm + dropout on patch embeddings
        X_ex = self.dropout(self.ln(X_ex))
        X_q  = self.dropout(self.ln(X_q))

        return X_ex, Tok_ex, X_q, Tok_q


class PatchCrossAttentionDecoder(nn.Module):
    """
    Decoder block structured as cross-attention over learned expert memory.

    Decoder cross-attention:
      1. Query:  derived from context tokens ([START], [MID], future summary)
      2. Keys:   learned expert prototype embeddings (one per head)
      3. Values: expert-specific FFN decoders applied to future patches
      4. Output: attention-weighted aggregation of expert values

    Mixture-of-experts decoder: W ≡ K·W_q/√d, where
    linear projection W ∈ R^{Hx3d} is factored into a query
    and expert embeddings is the key K ∈ R^{Hxd}.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_quantiles: int,
        output_patch_size: int,
        act_fn_name: str,
        num_heads: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert num_heads >= 1

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_quantiles = num_quantiles
        self.output_patch_size = output_patch_size

        # --- Cross-attention components ---
        # Query projection: [START; MID; future_summary] → d-dim query
        self.q_proj = nn.Linear(3 * d_model, d_model, bias=False)

        # Learned expert keys (memory slots the context query attends to)
        self.expert_keys = nn.Parameter(torch.randn(num_heads, d_model))

        # Expert value projections (each head decodes patches independently)
        self.expert_values = nn.ModuleList([
            PatchDecoder(
                in_dim=d_model,
                h_dim=d_ff,
                out_dim=num_quantiles * output_patch_size,
                act_fn_name=act_fn_name,
                dropout_p=dropout_p,
            )
            for _ in range(num_heads)
        ])

    def _compute_query(self, future_patch_h, start_token, mid_token):
        """
        Compute the cross-attention query vector from the context tokens and future patch summary.
        Args:
            future_patch_h: (B, N, d) tensor of future patch embeddings
            start_token: (B, d) tensor of START token embeddings
            mid_token: (B, d) tensor of MID token embeddings
        Returns:
            query: (B, d) tensor of query vectors for cross-attention
        """        
        future_summary = future_patch_h.mean(dim=1)                            # (B, d)
        context = torch.cat([start_token, mid_token, future_summary], dim=-1)  # (B, 3d)
        query = self.q_proj(context)                                           # (B, d)
        return query

    def _cross_attention_weights(
        self,
        query: torch.Tensor,        # (B, d)
        key: torch.Tensor = None,   # (H, d) expert key
        scale: float = None,        # optional scaling factor for attention scores
    ) -> torch.Tensor:
        """
        Cross-attention score computation:
          query  = W_q · [START; MID; mean(future_patches)]
          scores = query · expert_keys^T / √d
          weights = softmax(scores)
        """
        scores = torch.matmul(query, key.T) / (self.d_model ** 0.5)  # (B, H)
        return torch.softmax(scores, dim=-1)                         # (B, H)

    def _reshape_to_sequence(
        self,
        patch_out: torch.Tensor,
        prediction_length: int,
    ) -> torch.Tensor:
        """Reshape (B, N, Q*P) → (B, Q, prediction_length)."""
        return rearrange(
            patch_out,
            'B N (Q P) -> B Q (N P)',
            Q=self.num_quantiles,
            P=self.output_patch_size,
        )[:, :, :prediction_length]
    
    def _compute_value(self, future_patch_h, prediction_length):
        """
        Compute expert-specific values by applying each expert's FFN decoder to the future patch embeddings.
        Args:
            future_patch_h: (B, N, d) tensor of future patch embeddings
        Returns:
            values: (B, H, Q, L) tensor of expert-specific decoded patches,
                    where H=num_heads, Q=num_quantiles, L=output_patch_size
        """        
        return torch.stack([
            self._reshape_to_sequence(expert(future_patch_h), prediction_length)
            for expert in self.expert_values
        ], dim=1)  # (B, H, Q, L)

    def forward(
        self,
        future_patch_h: torch.Tensor,   # (B, N_future_patches, d_model)
        start_token: torch.Tensor,      # (B, d_model)
        mid_token: torch.Tensor,        # (B, d_model)
        prediction_length: int,
        return_attn_weights: bool = False,
    ):
        # Compute cross-attention query from context tokens and future patch summary
        query = self._compute_query(
            future_patch_h, start_token, mid_token
        )  # (B, d_model)
        # Cross-attention weights (context query → expert keys)
        attn_weights = self._cross_attention_weights(
            query, key=self.expert_keys
        )  # (B, H)
        # Expert value computation (each expert decodes patches)
        value = self._compute_value(future_patch_h, prediction_length)  # (B, H, Q, L)
        # Attention aggregation
        output = (value * attn_weights[:, :, None, None]).sum(dim=1)  # (B, Q, L)

        if return_attn_weights:
            return output, attn_weights, value
        return output