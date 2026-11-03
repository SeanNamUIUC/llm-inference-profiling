from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
out = llm.generate(["What is the capital of Korea?"], SamplingParams(max_tokens=50))
print(out[0].outputs[0].text)