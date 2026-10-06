# Inference-Time Reasoning Strategies for Small LLMs

Compares six inference-time strategies on **Qwen/Qwen3-0.6B** using a fixed 64-problem subset of GSM8K (seed 42):

1. Direct answer (`/no_think`)
2. Chain-of-thought (`/no_think`)
3. Chain-of-thought (`/think`)
4. Self-consistency (N = 5, majority vote)
5. Few-shot prompting (k = 1)
6. Self-refinement

Each method is scored on accuracy, invalid-output rate, average generated tokens, and per-problem latency, all through one shared batched generation helper.

## Usage

```bash
pip install -r requirements.txt
# Download the GSM8K test split (JSON lines) as gsm8k_test.json:
curl -o gsm8k_test.json https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
jupyter notebook inference_techniques.ipynb
```

A GPU is recommended (batch size 8 fits in ~15 GB VRAM; lower `BATCH_SIZE` if you run out of memory).

## Outputs

Running the notebook produces `gsm8k_64_clean.json`, `predictions_<method>.csv` per method, `summary_results.csv`, and `accuracy_vs_tokens.png`. The cleaned subset used here is in `data/`.


