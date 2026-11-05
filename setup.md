# Setup Log

How I set up vLLM on a consumer Blackwell GPU (RTX 5070 Ti, sm_120), including the errors I hit and how I fixed them.

## 0. What is being set up

```
⑥ Model (Qwen2.5-0.5B-Instruct)   ← weights, downloaded from Hugging Face to ~/.cache
⑤ vLLM                            ← LLM serving engine (PagedAttention, batching)
④ PyTorch                         ← tensor library; must include sm_120 kernels for Blackwell
③ CUDA                            ← NVIDIA's platform for running computation on the GPU
② NVIDIA driver                   ← lets the OS talk to the GPU
① GPU (RTX 5070 Ti)               ← hardware; 16 GB VRAM holds weights + KV cache
─────────────────────────────────
   conda env "vllm"               ← isolated box for ④ and ⑤
```

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (16 GB, Blackwell, compute capability 12.0 / sm_120) |
| NVIDIA driver | 595.91.07 |
| CUDA (driver-supported) | 13.2 |
| OS | Ubuntu 26.04.1 LTS |
| conda | 26.9.4 (Miniforge) |
| Python | 3.12 |
| PyTorch | 2.14.0+cu130 |
| vLLM | 0.33.0|
| Model | Qwen/Qwen2.5-0.5B-Instruct |

## 2. Steps

### 2.1 Check the GPU and driver
```bash
nvidia-smi
```
Driver 595 / CUDA 13.2 — already new enough for Blackwell (needs driver 570+ and CUDA 12.8+). No update needed.

### 2.2 Install conda (Miniforge) and create an environment
```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh      # accept license, default location, "yes" to conda init
source ~/.bashrc
conda --version                      # conda 26.9.4

conda create -n vllm python=3.12 -y
conda activate vllm
```

### 2.3 Install vLLM
```bash
pip install vllm      # also installs a matching PyTorch build
```

### 2.4 Verify PyTorch sees sm_120
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```
Output:
```
2.14.0+cu130
NVIDIA GeForce RTX 5070 Ti
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```
`sm_120` is in the list, so this PyTorch build has kernels for Blackwell.

### 2.5 First run
```python
# scripts/first_run.py
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
out = llm.generate(["What is the capital of Korea?"], SamplingParams(max_tokens=50))
print(out[0].outputs[0].text)
```
Output:
```
(paste output here)
```

## 3. Errors and fixes

### Error 1: `conda: command not found` right after installing
- **When:** 2.2, right after the Miniforge installer finished
- **Error:** `명령어 'conda' 을(를) 찾을 수 없습니다` (command not found)
- **Cause:** `conda init` writes to `~/.bashrc`, but the already-open terminal had not re-read it.
- **Fix:** `source ~/.bashrc` (or open a new terminal). Note: `sudo conda` does not work either, because conda is installed under the user's home, not system-wide.

<!-- Copy the block above for each new error -->

## 4. What I learned

- The CUDA version shown by `nvidia-smi` is the maximum the driver supports; PyTorch ships its own CUDA runtime (here cu130).
- Blackwell GPUs (sm_120) need a PyTorch build compiled with sm_120 kernels; checking `torch.cuda.get_arch_list()` is the fastest way to confirm.