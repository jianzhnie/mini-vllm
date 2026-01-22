"""
---
title: Advanced Sampling Techniques Examples
summary: Examples demonstrating various sampling techniques for language models.
---

# Advanced Sampling Techniques Examples

This file shows how to use various sampling strategies for text generation.
Each strategy offers different tradeoffs between diversity and coherence.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minivllm.sampling import (FrequencyPenaltySampler, GreedySampler,
                               MinPSampler, PresencePenaltySampler,
                               RandomSampler, RepetitionPenaltySampler,
                               TemperatureMinPTopKSampler, TemperatureSampler,
                               TopKSampler, TopKTopPSampler, TopPSampler,
                               TypicalSampler)


def demonstrate_basic_samplers():
    """
    Demonstrate basic sampling techniques: greedy, random, and temperature-based.
    """
    print('=' * 60)
    print('Basic Sampling Techniques')
    print('=' * 60)

    # Sample logits
    logits = torch.randn(1, 50257)  # Typical vocabulary size

    # 1. Greedy Sampling - Always select highest probability token
    greedy_sampler = GreedySampler()
    token = greedy_sampler(logits)
    print(f'Greedy sampling result: {token.item()}')

    # 2. Random Sampling - Uniform probability for all tokens
    random_sampler = RandomSampler()
    token = random_sampler(logits)
    print(f'Random sampling result: {token.item()}')

    # 3. Temperature Sampling - Control randomness with temperature
    temp_sampler_low = TemperatureSampler(temperature=0.5)
    token = temp_sampler_low(logits)
    print(f'Temperature=0.5 result: {token.item()}')

    temp_sampler_high = TemperatureSampler(temperature=2.0)
    token = temp_sampler_high(logits)
    print(f'Temperature=2.0 result: {token.item()}')


def demonstrate_filtering_samplers():
    """
    Demonstrate filtering-based sampling: top-k, top-p, min-p, typical.
    """
    print('\n' + '=' * 60)
    print('Filtering-Based Sampling Techniques')
    print('=' * 60)

    logits = torch.randn(1, 50257)

    # 1. Top-K Sampling - Restrict to top-k tokens
    inner_sampler = TemperatureSampler(temperature=1.0)
    topk_sampler = TopKSampler(k=50, sampler=inner_sampler)
    token = topk_sampler(logits)
    print(f'Top-K (k=50) sampling result: {token.item()}')

    # 2. Top-P (Nucleus) Sampling - Select tokens with cumulative prob >= p
    topp_sampler = TopPSampler(p=0.95, temperature=1.0)
    token = topp_sampler(logits)
    print(f'Top-P (p=0.95) sampling result: {token.item()}')

    # 3. Min-P Sampling - Select tokens with prob >= p * max_prob
    minp_inner_sampler = TemperatureSampler(temperature=1.0)
    minp_sampler = MinPSampler(p=0.05, sampler=minp_inner_sampler)
    token = minp_sampler(logits)
    print(f'Min-P (p=0.05) sampling result: {token.item()}')

    # 4. Typical Sampling - Select typical tokens based on entropy
    typical_inner_sampler = TemperatureSampler(temperature=1.0)
    typical_sampler = TypicalSampler(tau=1.0, sampler=typical_inner_sampler)
    token = typical_sampler(logits)
    print(f'Typical (tau=1.0) sampling result: {token.item()}')


def demonstrate_combined_samplers():
    """
    Demonstrate combined sampling strategies.
    """
    print('\n' + '=' * 60)
    print('Combined Sampling Techniques')
    print('=' * 60)

    logits = torch.randn(1, 50257)

    # 1. Top-K + Top-P Combined
    combined_kp_sampler = TopKTopPSampler(k=50, p=0.95, temperature=1.0)
    token = combined_kp_sampler(logits)
    print(f'Top-K (50) + Top-P (0.95) result: {token.item()}')

    # 2. Temperature + Min-P + Top-K Combined
    combined_temp_sampler = TemperatureMinPTopKSampler(temperature=1.0,
                                                       min_p=0.05,
                                                       top_k=100)
    token = combined_temp_sampler(logits)
    print(f'Temp (1.0) + Min-P (0.05) + Top-K (100) result: {token.item()}')


def demonstrate_penalty_samplers():
    """
    Demonstrate penalty-based samplers for diversity control.
    """
    print('\n' + '=' * 60)
    print('Penalty-Based Samplers')
    print('=' * 60)

    logits = torch.randn(1, 50257)
    prev_tokens = torch.tensor([100, 200, 300])  # Previously generated tokens
    sequence = torch.tensor([100, 200, 100, 300, 100])  # Full sequence so far

    # 1. Repetition Penalty - Penalize previously generated tokens
    rep_penalty_sampler = RepetitionPenaltySampler(penalty=1.2)
    token = rep_penalty_sampler(logits, prev_tokens)
    print(f'Repetition penalty sampling result: {token.item()}')

    # 2. Frequency Penalty - Penalize by token frequency
    freq_penalty_sampler = FrequencyPenaltySampler(alpha=0.5)
    token = freq_penalty_sampler(logits, sequence)
    print(f'Frequency penalty sampling result: {token.item()}')

    # 3. Presence Penalty - Penalize any previously seen token
    pres_penalty_sampler = PresencePenaltySampler(penalty=0.5)
    token = pres_penalty_sampler(logits, sequence)
    print(f'Presence penalty sampling result: {token.item()}')


def demonstrate_full_generation():
    """
    Demonstrate complete text generation with various samplers.
    Note: This requires downloading a model from HuggingFace.
    """
    print('\n' + '=' * 60)
    print('Full Text Generation Example')
    print('=' * 60)

    try:
        # Load a small model for demonstration
        model_name = 'gpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        prompt = 'The future of AI is'
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Generate with greedy sampling
        print(f"\nPrompt: '{prompt}'")
        print('\n1. Greedy Sampling:')
        with torch.no_grad():
            greedy_output = model.generate(
                input_ids,
                max_length=50,
                do_sample=False,
                num_beams=1,
            )
        print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

        # Generate with temperature sampling
        print('\n2. Temperature Sampling (T=0.7):')
        with torch.no_grad():
            temp_output = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                top_k=0,
            )
        print(tokenizer.decode(temp_output[0], skip_special_tokens=True))

        # Generate with top-p sampling
        print('\n3. Top-P Sampling (p=0.9):')
        with torch.no_grad():
            topp_output = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=0,
            )
        print(tokenizer.decode(topp_output[0], skip_special_tokens=True))

        # Generate with top-k sampling
        print('\n4. Top-K Sampling (k=50):')
        with torch.no_grad():
            topk_output = model.generate(
                input_ids,
                max_length=50,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            )
        print(tokenizer.decode(topk_output[0], skip_special_tokens=True))

    except Exception as e:
        print(
            f'Note: Full generation example requires model download. Error: {e}'
        )
        print(
            'You can uncomment this section and run it separately if needed.')


def main():
    """
    Run all demonstrations.
    """
    print('\n' + '=' * 60)
    print('Mini-vLLM Sampling Techniques Demonstration')
    print('=' * 60)

    demonstrate_basic_samplers()
    demonstrate_filtering_samplers()
    demonstrate_combined_samplers()
    demonstrate_penalty_samplers()
    demonstrate_full_generation()

    print('\n' + '=' * 60)
    print('Demonstration Complete')
    print('=' * 60)


if __name__ == '__main__':
    main()
