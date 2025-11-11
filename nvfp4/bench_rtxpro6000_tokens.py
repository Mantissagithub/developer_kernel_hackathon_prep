from vllm import LLM, SamplingParams
import time
import os

# RTX Pro 6000 48GB benchmark
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

benchmark_prompts = [
    "Explain quantum computing and its applications in cryptography in detail.",
    "Write a Python function to implement binary search on a sorted array with error handling.",
    "What are the key differences between machine learning and deep learning? Provide examples.",
    "Describe the process of photosynthesis and its importance to the ecosystem.",
    "How does a transformer architecture work in natural language processing models?",
    "Explain the theory of relativity and its impact on modern physics.",
    "Write a step-by-step guide for building a REST API using FastAPI.",
    "What are the main causes of climate change and potential solutions to mitigate it?",
    "Describe the differences between SQL and NoSQL databases with use cases.",
    "Explain how blockchain technology works and its applications beyond cryptocurrency."
]


def benchmark_model(model_name):
    llm_kwargs = {
        "model": model_name,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "max_num_seqs": 16
    }

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    print(f"Warming up {model_name}...")
    _ = llm.generate([benchmark_prompts[0]], sampling_params)

    total_tokens_generated = 0
    total_time_taken = 0.0

    for prompt in benchmark_prompts:
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()

        tokens_generated = len(outputs[0].outputs[0].token_ids)
        time_taken = end_time - start_time

        total_tokens_generated += tokens_generated
        total_time_taken += time_taken
        print(f"Prompt: {prompt[:50]}... | Tokens: {tokens_generated} | Time: {time_taken:.2f}s")

    tokens_per_second = total_tokens_generated / total_time_taken if total_time_taken > 0 else 0
    print(f"\n{'='*80}")
    print(f"GPU: RTX Pro 6000 48GB")
    print(f"Model: {model_name}")
    print(f"Total Tokens Generated: {total_tokens_generated}")
    print(f"Total Time Taken: {total_time_taken:.2f} seconds")
    print(f"Tokens per Second: {tokens_per_second:.2f}")
    print(f"{'='*80}\n")

    return {
        "model": model_name,
        "total_tokens": total_tokens_generated,
        "total_time": total_time_taken,
        "tokens_per_sec": tokens_per_second
    }


def main():
    models_to_benchmark = [
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    ]

    results = []
    for model_name in models_to_benchmark:
        result = benchmark_model(model_name)
        results.append(result)

    print("\n" + "="*80)
    print("RTX PRO 6000 48GB COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<50} {'Tokens/sec':<15} {'Total Time':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<50} {r['tokens_per_sec']:<15.2f} {r['total_time']:<15.2f}s")
    print("="*80)


if __name__ == "__main__":
    main()

