"""Experiment 1: prefill latency vs. input length.

Output is fixed to 1 token, so the measured time is (almost) all prefill.
"""
import csv
import random
import statistics
import time
from pathlib import Path

#VLLM 엔진 사용을 위한 패키지
from vllm import LLM, SamplingParams

# ---------- 1. Settings ----------
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_LENGTHS = [128, 256, 512, 1024, 2048, 4096]  # number of input tokens to test
WARMUP = 1                               # runs thrown away per length
REPEATS = 3                              # runs averaged per length
OUT_CSV = Path("results/prefill.csv")

# ---------- 2. Load the model ----------
# enable_prefix_caching=False: vLLM normally reuses KV cache for prompts it has
# already seen. We send similar prompts repeatedly, so caching would skip the
# prefill and make the numbers meaningless.
llm = LLM(
    model=MODEL,
    max_model_len=8192,
    enable_prefix_caching=False, # 같은 입력을 재사용시 prefill을 건너뛰므로 의미없는 수치가 됨
)
# 답을 1토큰만 만들고 멈춤, 무작위성 없이 항상 확률 젤 높은 토큰 선택
params = SamplingParams(max_tokens=1, temperature=0.0)


def make_prompt(n_tokens: int) -> dict:
    """Build a prompt of exactly n_tokens token ids.

    Random ids (from the normal-word range, avoiding special tokens) give an
    exact length without depending on how a sentence happens to tokenize.
    """
    ids = [random.randint(1000, 20000) for _ in range(n_tokens)]
    # vLLM에 넣을 형태로 만듬
    return {"prompt_token_ids": ids}


#토큰 수를 받아서 걸린 시간을 밀리초 단위로 반환
def time_one_run(n_tokens: int) -> float:
    """Run one request and return wall-clock time in milliseconds."""
    prompt = make_prompt(n_tokens)
    start = time.perf_counter()
    #use_tqdm=False: 진행률 표시줄을 표시하지 않음 -> 속도 측정에 방해되지 않도록
    llm.generate([prompt], params, use_tqdm=False)
    end = time.perf_counter()
    return (end - start) * 1000


# ---------- 3. Measure ----------
rows = []
for n in INPUT_LENGTHS:
    for _ in range(WARMUP):
        time_one_run(n)  # thrown away
    #시간 3개를 리스트로 담음
    times = [time_one_run(n) for _ in range(REPEATS)]
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)
    #초당 처리한 입력 토큰 수
    tok_per_s = n / (mean_ms / 1000)

    print(f"input={n:5d} tokens | {mean_ms:8.2f} ms (± {std_ms:.2f}) | {tok_per_s:10.0f} tok/s")
    rows.append({
        "input_tokens": n,
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "tokens_per_s": round(tok_per_s, 1),
        "runs_ms": ";".join(f"{t:.3f}" for t in times),
    })

# ---------- 4. Save ----------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {OUT_CSV}")