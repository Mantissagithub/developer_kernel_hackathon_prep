from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
import os

# H100 80GB GSM8K accuracy test
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def extract_answer(text):
    matches = re.findall(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if matches:
        return matches[-1].replace(',', '')
    return None


def extract_model_answer(text):
    text = str(text).strip()
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def evaluate_model(model_name, num_samples=1319):
    llm_kwargs = {
        "model": model_name,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": True,
        "max_num_seqs": 8
    }

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0

    print(f"Evaluating {model_name} on GSM8K...")
    print(f"Total samples: {len(dataset)}")

    for idx, item in enumerate(dataset):
        question = item['question']
        answer = item['answer']

        correct_answer = extract_answer(answer)
        if correct_answer is None:
            continue

        prompt = f"Question: {question}\nAnswer: Let's think step by step."

        outputs = llm.generate([prompt], sampling_params)
        model_output = outputs[0].outputs[0].text

        model_answer = extract_model_answer(model_output)

        is_correct = model_answer == correct_answer
        if is_correct:
            correct += 1

        total += 1

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} | Accuracy: {correct/total*100:.2f}%")

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*80}")
    print(f"GPU: H100 80GB")
    print(f"Model: {model_name}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")

    return {
        "model": model_name,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


def main():
    models_to_evaluate = [
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    ]

    results = []
    for model_name in models_to_evaluate:
        result = evaluate_model(model_name)
        results.append(result)

    print("\n" + "="*80)
    print("H100 80GB GSM8K ACCURACY COMPARISON")
    print("="*80)
    print(f"{'Model':<50} {'Accuracy':<15} {'Correct/Total':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<50} {r['accuracy']:<15.2f}% {r['correct']}/{r['total']}")
    print("="*80)


if __name__ == "__main__":
    main()

