"""
NPU Flash Attention 使用示例

本示例展示了如何在mini-vllm中使用NPU Flash Attention功能，
包括预填充和解码阶段的使用方法。
"""

import torch

from minivllm.models.layers.attention import Attention
from minivllm.utils.context import InferenceContext


def example_prefill_phase():
    """
    预填充阶段使用NPU Flash Attention的示例
    """
    print('=== 预填充阶段 NPU Flash Attention 示例 ===')

    # 初始化Attention层
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32
    scale = 1.0 / (head_dim**0.5)

    attention_layer = Attention(num_heads=num_heads,
                                head_dim=head_dim,
                                scale=scale,
                                num_kv_heads=num_kv_heads)

    # 创建模拟输入张量 (T, num_heads, head_dim)
    batch_sizes = [2, 3]  # 两个序列，长度分别为2和3
    total_tokens = sum(batch_sizes)

    q = torch.randn(total_tokens, num_heads, head_dim, device='npu:0')
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device='npu:0')
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device='npu:0')

    # 构建累积长度张量
    cum_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32,
                                 device='npu:0')  # [0, 2, 5] 表示两个序列长度为2和3
    cum_seqlens_k = torch.tensor([0, 2, 5], dtype=torch.int32, device='npu:0')

    # 创建推理上下文
    context = InferenceContext(is_prefill=True,
                               max_seqlen_q=max(batch_sizes),
                               max_seqlen_k=max(batch_sizes),
                               cum_seqlens_q=cum_seqlens_q,
                               cum_seqlens_k=cum_seqlens_k,
                               slot_mapping=torch.arange(total_tokens,
                                                         device='npu:0'),
                               context_lens=None,
                               block_tables=None)

    # 设置上下文到全局环境
    from minivllm.utils.context import set_context
    set_context(context)

    # 执行注意力计算
    output = attention_layer(q, k, v)

    print(f'输入形状: q={q.shape}, k={k.shape}, v={v.shape}')
    print(f'输出形状: {output.shape}')
    print(f'使用NPU设备: {output.device}')
    print()


def example_decode_phase():
    """
    解码阶段使用NPU Flash Attention的示例
    """
    print('=== 解码阶段 NPU Flash Attention 示例 ===')

    # 初始化Attention层
    num_heads = 32
    head_dim = 128
    num_kv_heads = 32
    scale = 1.0 / (head_dim**0.5)

    attention_layer = Attention(num_heads=num_heads,
                                head_dim=head_dim,
                                scale=scale,
                                num_kv_heads=num_kv_heads)

    # 创建模拟输入张量 (batch_size, num_heads, head_dim)
    batch_size = 2
    q = torch.randn(batch_size, num_heads, head_dim, device='npu:0')

    # 创建KV缓存 (假设最大序列长度为10，块大小为4，总共3个块)
    max_blocks = 3
    block_size = 4
    kv_cache_shape = (max_blocks, block_size, num_kv_heads, head_dim)

    k_cache = torch.zeros(kv_cache_shape, device='npu:0')
    v_cache = torch.zeros(kv_cache_shape, device='npu:0')

    # 初始化KV缓存
    attention_layer.k_cache = k_cache
    attention_layer.v_cache = v_cache

    # 模拟已有的序列长度
    context_lens = torch.tensor([3, 5], dtype=torch.int32,
                                device='npu:0')  # 两个序列分别有3和5个token

    # 块映射表 (每个序列使用哪些块)
    block_tables = torch.tensor(
        [
            [0, 1, -1, -1],  # 序列1使用块0和1
            [1, 2, -1, -1]  # 序列2使用块1和2
        ],
        dtype=torch.int32,
        device='npu:0')

    # 创建虚拟的k和v（这些会在prefill阶段被存储到缓存中）
    k = torch.randn(batch_size, num_kv_heads, head_dim, device='npu:0')
    v = torch.randn(batch_size, num_kv_heads, head_dim, device='npu:0')

    # 创建推理上下文
    context = InferenceContext(
        is_prefill=False,  # 解码阶段
        max_seqlen_q=None,
        max_seqlen_k=None,
        cum_seqlens_q=None,
        cum_seqlens_k=None,
        slot_mapping=torch.arange(batch_size, device='npu:0'),  # 当前批次的槽位映射
        context_lens=context_lens,
        block_tables=block_tables)

    # 设置上下文到全局环境
    from minivllm.utils.context import set_context
    set_context(context)

    # 执行注意力计算（这将使用缓存的KV）
    output = attention_layer(q, k, v)

    print(f'查询形状: {q.shape}')
    print(f'KV缓存形状: k_cache={k_cache.shape}, v_cache={v_cache.shape}')
    print(f'输出形状: {output.shape}')
    print(f'使用NPU设备: {output.device}')
    print()


def check_npu_availability():
    """
    检查NPU可用性并显示相关信息
    """
    print('=== NPU 可用性检查 ===')

    if torch.npu.is_available():
        print('✓ NPU 可用')
        device_count = torch.npu.device_count()
        print(f'  - 可用设备数: {device_count}')

        for i in range(device_count):
            name = torch.npu.get_device_name(i)
            capability = torch.npu.get_device_capability(i)
            print(f'  - 设备 {i}: {name}, 计算能力: {capability}')
    else:
        print('✗ NPU 不可用')
        return False

    # 检查NPU Flash Attention API可用性
    try:
        import torch_npu
        if hasattr(torch_npu, 'npu_fusion_attention'):
            print('✓ torch_npu.npu_fusion_attention 可用')
        else:
            print('✗ torch_npu.npu_fusion_attention 不可用')

        if hasattr(torch_npu, 'npu_incre_flash_attention'):
            print('✓ torch_npu.npu_incre_flash_attention 可用')
        else:
            print('✗ torch_npu.npu_incre_flash_attention 不可用')
    except ImportError:
        print('✗ torch_npu 模块不可用')
        return False

    print()
    return True


def main():
    """
    主函数：运行所有示例
    """
    print('华为昇腾NPU Flash Attention 使用示例')
    print('=' * 50)

    # 检查NPU可用性
    if not check_npu_availability():
        print('NPU不可用，无法运行示例')
        return

    # 设置NPU设备
    torch.npu.set_device(0)

    # 运行预填充阶段示例
    try:
        example_prefill_phase()
    except Exception as e:
        print(f'预填充阶段示例出错: {e}')
        import traceback
        traceback.print_exc()
        print()

    # 运行解码阶段示例
    try:
        example_decode_phase()
    except Exception as e:
        print(f'解码阶段示例出错: {e}')
        import traceback
        traceback.print_exc()
        print()


if __name__ == '__main__':
    main()
