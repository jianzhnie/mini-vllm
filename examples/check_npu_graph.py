"""
NPU Graph quick check — verify NPU runtime and flash attention availability.

Usage:
    python examples/check_npu_graph.py
"""

import sys

import torch

from minivllm.utils.device import is_torch_npu_available


def check_npu_graph() -> None:
    print("Checking NPU environment...", flush=True)

    if not is_torch_npu_available():
        print("NPU not available — skipping NPU checks.", flush=True)
        return

    print(f"NPU devices: {torch.npu.device_count()}", flush=True)
    print(f"Device name: {torch.npu.get_device_name(0)}", flush=True)

    # Check NPU Flash Attention APIs
    print("\nNPU Flash Attention API check:", flush=True)
    try:
        import torch_npu

        apis = {
            "npu_fusion_attention": "Training + basic prefill",
            "npu_incre_flash_attention": "Incremental decode (legacy)",
            "npu_prompt_flash_attention": "Prefill (legacy)",
            "npu_fused_infer_attention_score": "Unified inference (recommended)",
        }
        for api, desc in apis.items():
            available = hasattr(torch_npu, api)
            status = "AVAILABLE" if available else "not available"
            print(f"  {api}: {status}  ({desc})", flush=True)
    except ImportError:
        print("  torch_npu not importable", flush=True)

    # Quick functional test
    print("\nFunctional check: attention forward pass...", flush=True)
    device = torch.device("npu:0")
    B, H, S, D = 2, 4, 128, 64

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (B, H, S, D), f"Unexpected shape: {out.shape}"
        print(f"  SDPA output shape: {out.shape} — PASSED", flush=True)
    except Exception as e:
        print(f"  SDPA functional check FAILED: {e}", flush=True)
        sys.exit(1)

    print("\nNPU environment: OK", flush=True)


if __name__ == "__main__":
    check_npu_graph()
