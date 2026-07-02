"""GPT2 model implementation for mini-vLLM.

GPT2 is a decoder-only transformer with:
- Learned absolute positional embeddings (no offset)
- Pre-LayerNorm architecture
- GeLU activation
- Fused QKV projection (Conv1D in HuggingFace)
- Tied input/output embeddings

HuggingFace GPT2 uses Conv1D whose weights are stored as (in, out), i.e. the
transpose of PyTorch nn.Linear (out, in). load_weights handles this transpose.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

from minivllm.models.layers import (
    Attention,
    ColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = ["GPT2ForCausalLM"]

_LAYER_RE = re.compile(r"transformer\.h\.(\d+)\.(.*)")


class GPT2Attention(nn.Module):
    """GPT2 multi-head self-attention (no RoPE)."""

    def __init__(self, config, use_buffered_page_attention: bool = False) -> None:
        super().__init__()
        n_embd: int = config.n_embd
        n_head: int = config.n_head
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(n_embd, self.head_dim, n_head, bias=True)
        self.out_proj = RowParallelLinear(n_embd, n_embd, bias=True)
        self.attn = Attention(
            n_head,
            self.head_dim,
            self.scale,
            n_head,
            use_buffered_page_attention=use_buffered_page_attention,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(total_tokens, self.n_head, self.head_dim)
        k = k.view(total_tokens, self.n_head, self.head_dim)
        v = v.view(total_tokens, self.n_head, self.head_dim)
        o = self.attn(q, k, v)
        o = o.reshape(total_tokens, self.n_head * self.head_dim)
        return self.out_proj(o)


class GPT2MLP(nn.Module):
    """GPT2 feed-forward network with GeLU activation."""

    def __init__(self, config) -> None:
        super().__init__()
        n_embd: int = config.n_embd
        n_inner: int = config.n_inner if config.n_inner is not None else 4 * n_embd
        self.fc1 = ColumnParallelLinear(n_embd, n_inner, bias=True)
        self.fc2 = RowParallelLinear(n_inner, n_embd, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GPT2Block(nn.Module):
    """GPT2 decoder block: Pre-LayerNorm → Attention → Pre-LayerNorm → MLP."""

    def __init__(self, config, use_buffered_page_attention: bool = False) -> None:
        super().__init__()
        n_embd: int = config.n_embd
        eps: float = getattr(config, "layer_norm_epsilon", 1e-5)
        self.ln_1 = nn.LayerNorm(n_embd, eps=eps)
        self.self_attn = GPT2Attention(config, use_buffered_page_attention)
        self.ln_2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = hidden_states + self.self_attn(positions, self.ln_1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, None


class GPT2Model(nn.Module):
    """GPT2 backbone: embeddings + stacked blocks + final LayerNorm."""

    def __init__(self, config) -> None:
        super().__init__()
        n_embd: int = config.n_embd
        eps: float = getattr(config, "layer_norm_epsilon", 1e-5)
        use_buf = getattr(config, "use_buffered_page_attention", False)

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, n_embd)
        self.embed_positions = nn.Embedding(config.n_positions, n_embd)
        self.layers = nn.ModuleList(
            [GPT2Block(config, use_buf) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd, eps=eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None)
        return self.ln_f(hidden_states)


class GPT2ForCausalLM(nn.Module):
    """GPT2 causal language model for mini-vLLM."""

    packed_modules_mapping: dict = {}

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = GPT2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.n_embd, bias=False)
        # GPT2 ties lm_head weights to input token embeddings
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def set_kv_cache(self, kv_cache: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        for idx, layer in enumerate(self.model.layers):
            if idx < len(kv_cache):
                layer.self_attn.attn.k_cache = kv_cache[idx][0]
                layer.self_attn.attn.v_cache = kv_cache[idx][1]

    def load_weights(self, weights: dict[str, Any]) -> None:
        """Load HuggingFace GPT2 weights.

        Key transformations applied:
        - Conv1D weights are transposed: (in, out) → (out, in) for nn.Linear
        - c_attn (fused QKV) is split into q / k / v shards
        - HF name prefix ``transformer.*`` is remapped to ``model.*``
        """
        params = dict(self.named_parameters())
        n_embd = self.config.n_embd

        for name, w in weights.items():
            if "rotary_emb.inv_freq" in name:
                continue
            # lm_head is tied to embed_tokens — skip to avoid double-copy
            if name == "lm_head.weight":
                continue

            m = _LAYER_RE.match(name)
            if m:
                idx = m.group(1)
                rest = m.group(2)
                pre = f"model.layers.{idx}."

                if rest in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"):
                    dst = pre + rest

                elif rest == "attn.c_attn.weight":
                    # Conv1D: (n_embd, 3*n_embd) → transpose → (3*n_embd, n_embd)
                    w_t = w.T
                    for shard, sw in [
                        ("q", w_t[:n_embd, :]),
                        ("k", w_t[n_embd : 2 * n_embd, :]),
                        ("v", w_t[2 * n_embd :, :]),
                    ]:
                        p_name = pre + "self_attn.qkv_proj.weight"
                        if p_name in params:
                            p = params[p_name]
                            if hasattr(p, "weight_loader"):
                                p.weight_loader(p, sw, shard)
                            else:
                                p.data.copy_(sw)
                    continue

                elif rest == "attn.c_attn.bias":
                    # bias: (3*n_embd,) — split q / k / v
                    for shard, sb in [
                        ("q", w[:n_embd]),
                        ("k", w[n_embd : 2 * n_embd]),
                        ("v", w[2 * n_embd :]),
                    ]:
                        p_name = pre + "self_attn.qkv_proj.bias"
                        if p_name in params:
                            p = params[p_name]
                            if hasattr(p, "weight_loader"):
                                p.weight_loader(p, sb, shard)
                            else:
                                p.data.copy_(sb)
                    continue

                elif rest == "attn.c_proj.weight":
                    w = w.T  # (n_embd, n_embd)
                    dst = pre + "self_attn.out_proj.weight"

                elif rest == "attn.c_proj.bias":
                    dst = pre + "self_attn.out_proj.bias"

                elif rest == "mlp.c_fc.weight":
                    w = w.T  # (n_inner, n_embd)
                    dst = pre + "mlp.fc1.weight"

                elif rest == "mlp.c_fc.bias":
                    dst = pre + "mlp.fc1.bias"

                elif rest == "mlp.c_proj.weight":
                    w = w.T  # (n_embd, n_inner)
                    dst = pre + "mlp.fc2.weight"

                elif rest == "mlp.c_proj.bias":
                    dst = pre + "mlp.fc2.bias"

                else:
                    continue

            elif name == "transformer.wte.weight":
                dst = "model.embed_tokens.weight"
            elif name == "transformer.wpe.weight":
                dst = "model.embed_positions.weight"
            elif name in ("transformer.ln_f.weight", "transformer.ln_f.bias"):
                dst = name.replace("transformer.", "model.")
            else:
                continue

            if dst not in params:
                continue
            p = params[dst]
            if hasattr(p, "weight_loader"):
                p.weight_loader(p, w, None)
            else:
                p.data.copy_(w)
