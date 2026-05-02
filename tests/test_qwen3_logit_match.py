"""Compare mini-vllm vs HuggingFace logits layer-by-layer for Qwen3.

Usage:
    python tests/test_qwen3_logit_match.py

Requires a local Qwen3-0.6B model at the path specified by MINIVLLM_MODEL
or /Users/robin/hfhub/models/Qwen/Qwen3-0.6B by default.
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MINIVLLM_DEVICE'] = 'cpu'

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # noqa: E402
from minivllm.models import create_model  # noqa: E402
from minivllm.utils.context import reset_context, set_context  # noqa: E402

MODEL_PATH = os.environ.get(
    'MINIVLLM_MODEL',
    '/Users/robin/hfhub/models/Qwen/Qwen3-0.6B',
)
PROMPT = ('<|im_start|>user\nHello, who are you?'
          '<|im_end|>\n<|im_start|>assistant\n')


def load_minivllm_model():
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = create_model(config)
    weights = load_file(os.path.join(MODEL_PATH, 'model.safetensors'))
    model.load_weights(weights)
    model.eval()
    return model


def load_hf_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
    ).eval()


def compare_logits(mv_model, hf_model, tokenizer):
    """Step 1: Compare final logits."""
    input_ids = tokenizer.encode(PROMPT, return_tensors='pt')
    seq_len = input_ids.shape[1]

    set_context(
        is_prefill=True,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        cum_seqlens_q=torch.tensor([0, seq_len]),
        cum_seqlens_k=torch.tensor([0, seq_len]),
        slot_mapping=torch.arange(seq_len),
    )

    with torch.no_grad():
        hidden = mv_model(input_ids.view(-1), torch.arange(seq_len))
        mv_logits = mv_model.compute_logits(hidden)

        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits

    cos_sim = F.cosine_similarity(
        mv_logits[-1].unsqueeze(0),
        hf_logits[0, -1].unsqueeze(0),
    ).item()

    print('=' * 60)
    print('Step 1: Final Logits Comparison')
    print('=' * 60)
    print(f'  Cosine similarity: {cos_sim:.6f}')

    mv_probs = torch.softmax(mv_logits[-1], dim=-1)
    hf_probs = torch.softmax(hf_logits[0, -1], dim=-1)
    mv_top5 = torch.topk(mv_probs, 5)
    hf_top5 = torch.topk(hf_probs, 5)

    print('  mini-vllm top-5:')
    for i in range(5):
        tok = tokenizer.decode([mv_top5.indices[i].item()])
        print(f'    {tok!r}: {mv_top5.values[i].item():.4f}')
    print('  HuggingFace top-5:')
    for i in range(5):
        tok = tokenizer.decode([hf_top5.indices[i].item()])
        print(f'    {tok!r}: {hf_top5.values[i].item():.4f}')

    reset_context()
    return cos_sim


def compare_layer_by_layer(mv_model, hf_model, tokenizer):
    """Step 2: Hook HF model, step mini-vllm manually, compare each layer."""
    input_ids = tokenizer.encode(PROMPT, return_tensors='pt')
    seq_len = input_ids.shape[1]

    set_context(
        is_prefill=True,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        cum_seqlens_q=torch.tensor([0, seq_len]),
        cum_seqlens_k=torch.tensor([0, seq_len]),
        slot_mapping=torch.arange(seq_len),
    )

    hf_layer_outputs = {}

    def make_hook(idx):

        def hook(module, args, kwargs, output):  # noqa: ARG001
            o = output[0] if isinstance(output, tuple) else output
            hf_layer_outputs[idx] = o.detach()

        return hook

    handles = [
        layer.register_forward_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(hf_model.model.layers)
    ]

    with torch.no_grad():
        _ = hf_model(input_ids)

        positions = torch.arange(seq_len)
        hidden = mv_model.model.embed_tokens(input_ids.view(-1))
        residual = None

        print()
        print('=' * 60)
        print('Step 2: Layer-by-Layer Comparison')
        print('=' * 60)

        for i in range(len(mv_model.model.layers)):
            layer = mv_model.model.layers[i]
            hidden, residual = layer(positions, hidden, residual)

            if i in hf_layer_outputs:
                effective = hidden + residual
                hf_h = hf_layer_outputs[i][0, -1]
                mv_h = effective[-1]

                cos_sim = F.cosine_similarity(
                    mv_h.unsqueeze(0),
                    hf_h.unsqueeze(0),
                ).item()
                max_diff = (mv_h - hf_h).abs().max().item()
                status = 'OK' if cos_sim > 0.99 else '*** DIVERGE ***'
                print(
                    f'  Layer {i:2d}: cos_sim={cos_sim:.6f}, '
                    f'max_diff={max_diff:.4f}  {status}', )

                if cos_sim < 0.99 and i <= 3:
                    compare_subcomponents(
                        mv_model,
                        hf_model,
                        input_ids,
                        seq_len,
                        layer_idx=i,
                    )
                    break

    for h in handles:
        h.remove()
    reset_context()


def compare_subcomponents(mv_model, hf_model, input_ids, seq_len, layer_idx=0):
    """Step 3: Dig into a specific layer's subcomponents."""
    print()
    print('=' * 60)
    print(f'Step 3: Subcomponent Breakdown (Layer {layer_idx})')
    print('=' * 60)

    set_context(
        is_prefill=True,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        cum_seqlens_q=torch.tensor([0, seq_len]),
        cum_seqlens_k=torch.tensor([0, seq_len]),
        slot_mapping=torch.arange(seq_len),
    )

    layer = mv_model.model.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]
    hf_attn = hf_layer.self_attn
    positions = torch.arange(seq_len)

    with torch.no_grad():
        emb = mv_model.model.embed_tokens(input_ids.view(-1))
        emb_hf = hf_model.model.embed_tokens(input_ids)

        # Embedding
        match = torch.allclose(emb, emb_hf.view_as(emb), atol=1e-5)
        status = 'PASS' if match else 'FAIL'
        print(f"  Embedding: {status}")

        # Input LayerNorm
        ln_out, _ = layer.input_layernorm(emb, None)
        ln_hf = hf_layer.input_layernorm(emb_hf.view_as(emb))
        match = torch.allclose(ln_out, ln_hf, atol=1e-4)
        status = 'PASS' if match else 'FAIL'
        print(f"  Input LayerNorm: {status}")

        # QKV Projection
        qkv = layer.self_attn.qkv_proj(ln_out)
        q_mv, k_mv, v_mv = qkv.split(
            [
                layer.self_attn.q_size, layer.self_attn.kv_size,
                layer.self_attn.kv_size
            ],
            dim=-1,
        )
        q_hf = F.linear(ln_hf.view_as(ln_out), hf_attn.q_proj.weight)
        k_hf = F.linear(ln_hf.view_as(ln_out), hf_attn.k_proj.weight)
        v_hf = F.linear(ln_hf.view_as(ln_out), hf_attn.v_proj.weight)

        status = 'PASS' if torch.allclose(q_mv, q_hf, atol=1e-4) else 'FAIL'
        print(f"  Q Projection: {status}")
        status = 'PASS' if torch.allclose(k_mv, k_hf, atol=1e-4) else 'FAIL'
        print(f"  K Projection: {status}")
        status = 'PASS' if torch.allclose(v_mv, v_hf, atol=1e-4) else 'FAIL'
        print(f"  V Projection: {status}")

        # Q/K Norm
        num_heads = layer.self_attn.num_heads
        num_kv_heads = layer.self_attn.num_kv_heads
        head_dim = layer.self_attn.head_dim

        q_mv_r = q_mv.view(seq_len, num_heads, head_dim)
        k_mv_r = k_mv.view(seq_len, num_kv_heads, head_dim)

        q_normed, _ = layer.self_attn.q_norm(q_mv_r)
        k_normed, _ = layer.self_attn.k_norm(k_mv_r)

        q_hf_r = q_hf.view(seq_len, num_heads, head_dim)
        k_hf_r = k_hf.view(seq_len, num_kv_heads, head_dim)
        q_hf_normed = hf_attn.q_norm(q_hf_r.float()).to(q_hf_r.dtype)
        k_hf_normed = hf_attn.k_norm(k_hf_r.float()).to(k_hf_r.dtype)

        status = ('PASS' if torch.allclose(q_normed, q_hf_normed, atol=1e-4)
                  else 'FAIL')
        print(f"  Q Norm: {status}")
        status = ('PASS' if torch.allclose(k_normed, k_hf_normed, atol=1e-4)
                  else 'FAIL')
        print(f"  K Norm: {status}")

        # RoPE
        q_rope, k_rope = layer.self_attn.rotary_emb(
            positions,
            q_normed.clone(),
            k_normed.clone(),
        )

        inv_freq = 1.0 / (1000000.0**(
            torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        emb_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
        emb_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)

        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_hf_rope = (q_hf_normed.float() *
                     emb_cos) + (rotate_half(q_hf_normed.float()) * emb_sin)
        k_hf_rope = (k_hf_normed.float() *
                     emb_cos) + (rotate_half(k_hf_normed.float()) * emb_sin)

        status = ('PASS'
                  if torch.allclose(q_rope, q_hf_rope, atol=1e-4) else 'FAIL')
        print(f"  RoPE (Q): {status}")
        status = ('PASS'
                  if torch.allclose(k_rope, k_hf_rope, atol=1e-4) else 'FAIL')
        print(f"  RoPE (K): {status}")

        # Full Attention
        attn_out = layer.self_attn(positions, ln_out)
        print(f"  Attention output: computed (norm={attn_out.norm():.4f})")

        # RMSNorm mutation test
        from minivllm.models.layers.layernorm import RMSNorm  # noqa: E402

        norm = RMSNorm(4)
        x_orig = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x_before = x_orig.clone()
        _, _ = norm(x_orig)
        mutated = not torch.allclose(x_orig, x_before)
        status = 'BUG: input mutated!' if mutated else 'OK: input preserved'
        print(f"  RMSNorm mutation: {status}")

    reset_context()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print('Loading models...')
    mv_model = load_minivllm_model()
    hf_model = load_hf_model()

    cos_sim = compare_logits(mv_model, hf_model, tokenizer)
    compare_layer_by_layer(mv_model, hf_model, tokenizer)

    print()
    if cos_sim > 0.99:
        print('RESULT: Logits match (cos_sim > 0.99)')
    else:
        print(f'RESULT: Logits DIVERGE (cos_sim={cos_sim:.6f})')

    return 0 if cos_sim > 0.99 else 1


if __name__ == '__main__':
    sys.exit(main())
