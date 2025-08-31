#######################
# CODE BASED ON https://github.com/hyunwoongko/transformer/blob/master/README.md
#######################


import torch
import torch.nn as nn
import math, inspect
from contextlib import suppress
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from rotary_embedding_torch import RotaryEmbedding

import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from flash_attn.utils.distributed import get_dim_for_local_rank

try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
    )
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None
    flash_attn_with_kvcache = None

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None


# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = Attention(d_model // n_head)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, stft_emb=None, return_attn=False):
        """
        Forward pass for MultiHeadAttention with optional stft_emb.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_length, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_length, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_length, d_model).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, 1, seq_length, seq_length).
            stft_emb (torch.Tensor, optional): STFT embeddings of shape (batch_size, stft_dim).

        Returns:
            torch.Tensor: Output tensor after attention and concatenation.
        """
        # 1. Apply Linear layers with conditional LoRA if applicable
#        if isinstance(self.w_q, ConditionalLoRALinear):
#            q = self.w_q(q, stft_emb=stft_emb)  # ConditionalLoRALinear handles stft_emb
#        else:
        q = self.w_q(q)

#        if isinstance(self.w_v, ConditionalLoRALinear):
#            v = self.w_v(v, stft_emb=stft_emb)
#        else:
        v = self.w_v(v)

        # w_k remains a standard Linear layer
        k = self.w_k(k)

        # 2. Split into multiple heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. Apply scaled dot-product attention
        out, attention = self.attention(q, k, v, mask, need_weights=return_attn)

        # 4. Concatenate heads and apply final Linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        if return_attn:
            return (out, attention)
        else:
            return out

    def split(self, tensor):
        """
        Splits the tensor into multiple heads.

        Args:
            tensor (torch.Tensor): Tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, n_head, seq_length, d_tensor).
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    @staticmethod
    def concat(tensor):
        """
        Concatenates the multiple heads back into a single tensor.

        Args:
            tensor (torch.Tensor): Tensor of shape (batch_size, n_head, seq_length, d_tensor).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


## 147hrs per epoch 1 B model 4 GPUs - definitely faster on all types than vanilla O(n^3) matmul att calc 
class Attention(nn.Module):
    def __init__(self, d_head, dropout=0.1):
        super().__init__()
        self.rotary_embed = RotaryEmbedding(d_head // 2)
        self.p            = dropout                 # dropout prob

    def forward(self, q, k, v, mask=None, need_weights=False):
        # ─── RoPE ──────────────────────────────────────────────────────
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
        causal = mask is not None

        # ─── Fast path – kernels, no weights requested ────────────────
        if not need_weights:
            with sdpa_kernel(backends=[
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.p if self.training else 0.0,
                        is_causal=causal)
            return out, None

        # ─── Probe: try the keyword (works on ≤ 2.5) ──────────────────
        with suppress(TypeError):
            out, attn = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=causal,
                    need_attn_weights=True)     # raises on 2.6
            return out, attn                   # (B,H,S,S)

        # ─── Manual SDPA fallback for 2.6+ ────────────────────────────
        d = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        if causal:
            scores = scores.masked_fill(
                torch.triu(torch.ones_like(scores), 1).bool(), float('-inf'))
        attn  = torch.softmax(scores, dim=-1)
        if self.training and self.p:
            attn = F.dropout(attn, p=self.p)
        out = attn @ v
        return out, attn

#class Attention(nn.Module):
#    def __init__(self, d_head, dropout=0.1):
#        super().__init__()
#        self.dropout = torch.nn.Dropout(dropout)
#        self.softmax = nn.Softmax(dim=-1)
#        self.rotary_embed = RotaryEmbedding(d_head//2)
#        self.first = True
#        self.dropout_value = dropout
#
#    def forward(self, q, k, v, mask=None):
#        # apply RoPE
#        q = self.rotary_embed.rotate_queries_or_keys(q)
#        k = self.rotary_embed.rotate_queries_or_keys(k)
#
#        d_k = k.size(-1)
##        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
#        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]): #[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
#            scores = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_value if self.training else 0, is_causal=True if mask is not None else False)
#
#        return scores, None

# this one is faster version of MHA attention but it works in low/mixed precision mode only
# cant conclude this yet - for 1B, it is faster
#class Attention(nn.Module):
#    """Implement the scaled dot product attention with softmax.
#    Arguments
#    ---------
#        softmax_scale: The temperature to use for the softmax attention.
#                      (default: 1/sqrt(d_keys) where d_keys is computed at
#                      runtime)
#        attention_dropout: The dropout rate to apply to the attention
#                           (default: 0.0)
#    """
#
#    def __init__(
#        self,
#        d_head,
#        dropout=0.1,
#        causal=False,
#        softmax_scale=None,
#        attention_dropout=0.1,
#        window_size=(-1, -1),
#        alibi_slopes=None,
#        deterministic=False,
#    ):
#        super().__init__()
#        assert flash_attn_varlen_qkvpacked_func is not None, "FlashAttention is not installed"
#        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
#        self.causal = causal
#        self.softmax_scale = softmax_scale
#        self.drop = nn.Dropout(attention_dropout)
#        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
#        self.window_size = window_size
#        self.deterministic = deterministic
#        self.rotary_embed = RotaryEmbedding(d_head//2)
#
#    def forward(self, q, k, v, mask=None, causal=None, cu_seqlens=None, max_seqlen=None):
#        """Implements the multihead softmax attention.
#        Arguments
#        ---------
#            qkv: The tensor containing the query, key, and value.
#                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
#                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
#                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
#            causal: if passed, will override self.causal
#            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
#                of the sequences in the batch, used to index into qkv.
#            max_seqlen: int. Maximum sequence length in the batch.
#        Returns:
#        --------
#            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
#                else (B, S, H, D).
#        """
#        # q.shape = B, H, S, D
#        causal = True if mask is not None else False
###        with torch.cuda.amp.autocast():
#        q = self.rotary_embed.rotate_queries_or_keys(q)
#        k = self.rotary_embed.rotate_queries_or_keys(k)
###        q = q.to(v.dtype)
###        k = k.to(v.dtype)
#        qkv = torch.concatenate([q.transpose(1,2).unsqueeze(2), k.transpose(1,2).unsqueeze(2), v.transpose(1,2).unsqueeze(2)], axis=2)
#        assert qkv.dtype in [torch.float16, torch.bfloat16], f'{type(qkv)=}'
#        assert qkv.is_cuda
#        causal = self.causal if causal is None else causal
#        unpadded = cu_seqlens is not None
#        if self.alibi_slopes is not None:
#            self.alibi_slopes = self.alibi_slopes.to(torch.float32)
#        if unpadded:
#            assert cu_seqlens.dtype == torch.int32
#            assert max_seqlen is not None
#            assert isinstance(max_seqlen, int)
#            out = flash_attn_varlen_qkvpacked_func(
#                qkv,
#                cu_seqlens,
#                max_seqlen,
#                self.drop.p if self.training else 0.0,
#                softmax_scale=self.softmax_scale,
#                causal=causal,
#                alibi_slopes=self.alibi_slopes,
#                window_size=self.window_size,
#                deterministic=self.deterministic,
#            )
#        else:
#            out = flash_attn_qkvpacked_func(
#                qkv,
#                self.drop.p if self.training else 0.0,
#                softmax_scale=self.softmax_scale,
#                causal=causal,
#                alibi_slopes=self.alibi_slopes,
#                window_size=self.window_size,
#                deterministic=self.deterministic,
#            )
#        return out.transpose(1,2), None


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
