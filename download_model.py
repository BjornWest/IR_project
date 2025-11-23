from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Force HuggingFace to use scratch space, not home directory
os.environ["HF_HOME"] = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/hf_cache"

model_name = "Qwen/Qwen2.5-7B"
cache_dir = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/Qwen2.5-7B"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

print("Downloading model (~15GB)...")
print("This may take 20-30 minutes on slow network...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype="auto",  # Don't load into GPU yet
    low_cpu_mem_usage=True
)

print("Download complete!")
print(f"Model cached at: {cache_dir}")

