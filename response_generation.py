import json
import os
from vllm import LLM, SamplingParams



path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

local_model_path = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/Qwen2.5-7B"

queries = json.load(
    open(os.path.join(path, "data/queries.factscore_bio.jsonl"), "r", encoding="utf-8")
)
queries = [query["prompt"] for query in queries]
retrieved_texts = json.load(
    open(os.path.join(path, "data/retrieved.jsonl"), "r", encoding="utf-8")
)


system_prompt = """
You are an AI assistant that provides accurate and concise answers based on the provided context. Do not 
include any other information in your response.
"""

full_prompts = []
for query, retrieved_text in zip(queries, retrieved_texts):
    full_prompt = (
        f"Retrieved context:\n{retrieved_text}\n\n Question to answer: {query}\nAnswer:"
    )
    full_prompts.append(full_prompt)

model = "Qwen/Qwen2.5-7B"

llm = LLM(
    model=model,
    download_dir=local_model_path,
    enable_prefix_caching=True,
    max_num_batched_tokens=16384,
    max_num_seqs=32,
)

# number of responses to generate
n = 5
sampling_params = SamplingParams(temperature=1.0, n=n)

outputs = llm.generate(full_prompts, sampling_params=sampling_params)

responses = [output.outputs[0].text for output in outputs]

# save responses to jsonl file
with open(os.path.join(path, "data/responses.jsonl"), "w") as f:
    for response, prompt in zip(responses, full_prompts):
        f.write(json.dumps({"response": response, "full_prompt": prompt}) + "\n")
