import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Load model with automatic device mapping across all GPUs
model_name = "Qwen/Qwen2.5-7B"
local_model_path = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/Qwen2.5-7B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=local_model_path,
    trust_remote_code=False
)
# GPT-2 doesn't have a pad token, set it to eos_token
tokenizer.pad_token = tokenizer.eos_token
# Use left-padding for decoder-only models (better for generation)
tokenizer.padding_side = 'left'

print("Loading Qwen2.5-7B model (~15GB download if not cached)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=local_model_path,
    device_map="balanced",  
    dtype=torch.float16,
    trust_remote_code=False,
    low_cpu_mem_usage=True
)

# Load queries
with open(os.path.join(path, "../factscore_bio.jsonl"), "r", encoding="utf-8") as f:
    queries = [json.loads(line) for line in f]

with open(os.path.join(path, "data/query_retrieval_top4.json"), "r", encoding="utf-8") as f:
    retrieval = json.loads(f.read())


queries = [query["prompt"] for query in queries]
# queries = queries[:5]  # Uncomment for quick testing
retrievals = [[doc for doc in retrieval[x]] for x in queries]

system_prompt = """You are an AI assistant that provides biographical information based ONLY on the provided Wikipedia context.

IMPORTANT RULES:
- Base your answer EXCLUSIVELY on the retrieved documents
- If the retrieved documents do not contain information about the requested person, then you should clearly state that the retrieved documents do not contain information about the requested person.
- Do NOT use your general knowledge or make up information
- Only mention facts that are explicitly stated in the provided context
- You may provide detailed answers if relevant information is available in the retrieved documents
- The retrieved documents are held within the <CONTEXT> tags.
"""


# Prepare prompts
full_prompts = []
m = 0
n = None
for query, retrievals in zip(queries[m:n], retrievals[m:n]):
    user_prompt = f"Retrieved context:\n <CONTEXT>"
    for i, retrieval in enumerate(retrievals):
        title = retrieval["title"]
        content = retrieval["contents"]
        user_prompt += f"START OF DOCUMENT {i} WITH TITLE: {title}\n{content}\n"
        user_prompt += f"\n ================END OF DOCUMENT {i} ================"
    user_prompt += f"<CONTEXT>\nBased on the documents above, i now want you to: {query}, ANSWER:"
    full_prompts.append(tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], 
        tokenize=False,
        add_generation_prompt=True  # Adds <|im_start|>assistant\n at the end
    ))
# Generation parameters
n_responses = 5
batch_size = 5  # Process multiple prompts at once (must be multiple of n_responses!) - Reduced for 2 GPU + 8-bit model

# Ensure batch_size is a multiple of n_responses for clean writes
assert batch_size % n_responses == 0, f"batch_size ({batch_size}) must be a multiple of n_responses ({n_responses})"

n_full_prompts = [p for p in full_prompts for _ in range(n_responses)]

# Generate responses
os.makedirs(os.path.join(path, "data"), exist_ok=True)

# Clear the output file (start fresh)
output_file = os.path.join(path, "data/responses_hf.jsonl")
with open(output_file, "w") as f:
    pass  # Just create/clear the file

all_responses = {}
print(f"Generating {n_responses} responses for {len(full_prompts)} prompts...")
print(f"Batch size: {batch_size} (covers {batch_size // n_responses} prompts per batch)")

for i in tqdm(range(0, len(n_full_prompts), batch_size)):
    batch_prompts = n_full_prompts[i:i+batch_size]
    
    # Tokenize
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1536,  # Reduced from 2048 to save GPU memory
    ).to(model.device)
    
    # Generate multiple responses per prompt
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,  # Lower temp = more focused, less hallucination
        top_p=0.9,  # Nucleus sampling for better quality
        repetition_penalty=1.1,  # Discourage repetition
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Get the padded input length (same for all prompts in batch)
    input_length = inputs['input_ids'].shape[1]
    
    for j, output in enumerate(outputs):
        # Slice from the padded input length (not the real token count!)
        # With left-padding, output = [PAD, PAD, ..., real_tokens, generated_tokens]
        new_tokens = output[input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        prompt = batch_prompts[j]
        if prompt in all_responses:
            all_responses[prompt].append(response)
        else:
            all_responses[prompt] = [response]
    
    # After each batch, write completed prompts to file
    # A prompt is "complete" when it has n_responses responses
    completed_prompts = []
    for prompt, responses in list(all_responses.items()):
        if len(responses) == n_responses:
            completed_prompts.append(prompt)
    
    # Write completed prompts and remove from dict
    if completed_prompts:
        with open(output_file, "a") as f:
            for prompt in completed_prompts:
                item = {"prompt": prompt, "responses": all_responses[prompt]}
                f.write(json.dumps(item) + "\n")
                del all_responses[prompt]

# Write any remaining responses (shouldn't happen if batch_size is multiple of n_responses)
if all_responses:
    print(f"Warning: {len(all_responses)} incomplete prompt(s) remaining")
    with open(output_file, "a") as f:
        for prompt, responses in all_responses.items():
            item = {"prompt": prompt, "responses": responses}
            f.write(json.dumps(item) + "\n")

print(f"Saved responses to {output_file}")
print(f"Total prompts processed: {len(full_prompts)}")

