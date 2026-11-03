# LLM Inference Profiling

Measuring how LLM inference behaves in the prefill and decode phases with vLLM, and explaining the results with a memory-bound vs. compute-bound analysis.

## Motivation

LLM inference has two phases:

- **Prefill**: all input tokens are processed in parallel → expected to be compute-bound.
- **Decode**: output tokens are generated one at a time, and each step reads the full model weights and KV cache from memory → expected to be memory-bound.

This repo checks that expectation with real measurements on a consumer GPU.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB, Blackwell, sm_120) |
| OS | Ubuntu 26.04.1 LTS |
| NVIDIA driver / CUDA | 595.91.07 / 13.2 |
| vLLM | 0.33.0 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |

Full setup steps and the errors I ran into are in [setup.md](setup.md).

## Experiments

1. **Prefill**: vary input length, measure time to first token.
2. **Decode**: vary output length, measure time per output token.

Each setting is repeated 3 times and averaged.

## Results

(TBD)

## Analysis

(TBD: why decode is slower, with a theoretical estimate from memory bandwidth)

## Repository structure

```
.
├── README.md
├── setup.md       # environment setup log, errors and fixes
├── scripts/       # measurement scripts
├── results/       # plots and summarized results
└── notes/         # paper summaries (vLLM, SqueezeLLM)
```

## Notes

- [vLLM (PagedAttention) summary](notes/vllm.md)
- [SqueezeLLM summary](notes/squeezellm.md)