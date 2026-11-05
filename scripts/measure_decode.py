"""Experiment 2: decode speed vs. output length.

Input length is fixed and short. We generate exactly N output tokens and
subtract the prefill time (N = 1), so what is left is the decode time.
"""
import csv
import random
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------- 1. Settings ----------
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_TOKENS = 128                          # fixed, short input
OUTPUT_LENGTHS = [64, 128, 256, 512, 1024, 2048]  # doubling each step
WARMUP = 1
REPEATS = 3
OUT_CSV = Path("results/decode.csv")

# For the theoretical ceiling (see README):
# each decode step reads all weights once.
WEIGHT_GB = 0.494 * 2       # ~494M params x 2 bytes (bf16) ~= 0.99 GB
MEM_BW_GBPS = 896           # RTX 5070 Ti spec memory bandwidth (GB/s)
CEILING_TOK_S = MEM_BW_GBPS / WEIGHT_GB

# ---------- 2. Load the model ----------
llm = LLM(
    model=MODEL,
    max_model_len=4096,
    enable_prefix_caching=False,
)


def make_prompt(n_tokens: int) -> dict:
    ids = [random.randint(1000, 20000) for _ in range(n_tokens)]
    return {"prompt_token_ids": ids}


def time_one_run(n_out: int) -> float:
    """Generate exactly n_out tokens; return wall-clock time in ms."""
    # ignore_eos=True: do not stop early even if the model emits an
    # end-of-text token, so every run produces exactly n_out tokens.
    params = SamplingParams(max_tokens=n_out, ignore_eos=True, temperature=0.0)
    prompt = make_prompt(INPUT_TOKENS)

    start = time.perf_counter()
    out = llm.generate([prompt], params, use_tqdm=False)
    end = time.perf_counter()

    assert len(out[0].outputs[0].token_ids) == n_out  # sanity check
    return (end - start) * 1000


def measure(n_out: int) -> list[float]:
    for _ in range(WARMUP):
        time_one_run(n_out)
    return [time_one_run(n_out) for _ in range(REPEATS)]


# ---------- 3. Baseline: prefill + first token only ----------
prefill_runs = measure(1)
prefill_ms = statistics.mean(prefill_runs)
print(f"prefill (input={INPUT_TOKENS}, output=1): {prefill_ms:.2f} ms")
print(f"theoretical decode ceiling: {CEILING_TOK_S:.0f} tok/s "
      f"({MEM_BW_GBPS} GB/s / {WEIGHT_GB:.2f} GB)")

# ---------- 4. Measure decode ----------
rows = []
for n in OUTPUT_LENGTHS:
    times = measure(n)
    total_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)

    # Everything after the first token is decode.
    decode_ms = total_ms - prefill_ms
    ms_per_token = decode_ms / (n - 1)
    tok_per_s = 1000 / ms_per_token
    pct = 100 * tok_per_s / CEILING_TOK_S

    print(f"output={n:5d} | total {total_ms:8.2f} ms (± {std_ms:.2f}) | "
          f"{ms_per_token:.3f} ms/token | {tok_per_s:6.0f} tok/s | {pct:5.1f}% of ceiling")
    rows.append({
        "output_tokens": n,
        "total_ms": round(total_ms, 3),
        "std_ms": round(std_ms, 3),
        "decode_ms_per_token": round(ms_per_token, 4),
        "decode_tokens_per_s": round(tok_per_s, 1),
        "pct_of_ceiling": round(pct, 1),
        "runs_ms": ";".join(f"{t:.3f}" for t in times),
    })

# ---------- 5. Save ----------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {OUT_CSV}")