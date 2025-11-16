import json
import os
import pathlib
import sys
import asyncio
import threading
from threading import Thread
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams

# --- 1. Setup Paths ---
path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
LFF_ROOT = PROJECT_ROOT / "long_form_factuality"
sys.path.append(str(LFF_ROOT))

from long_form_factuality.eval.safe import get_atomic_facts
# Assuming VLLMQwenModel is defined in modeling.py or imported here
# from long_form_factuality.common.modeling import VLLMQwenModel 
from long_form_factuality.eval.safe.rate_atomic_fact import check_atomic_fact

# --- 2. Create the Async Loop & Background Thread ---
# We create the loop *before* the engine so we can pass it around
loop = asyncio.new_event_loop()

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start the daemon thread that will handle all async vLLM tasks
t = threading.Thread(target=start_background_loop, args=(loop,), daemon=True)
t.start()

# --- 3. Initialize Engine Correctly ---
local_model_path = "/vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/Qwen2.5-7B"

# ERROR FIXED: Use AsyncEngineArgs, don't pass args directly to AsyncLLMEngine
engine_args = AsyncEngineArgs(
    model=local_model_path, # Use the actual path variable
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_log_lens=1024,      # Note: it is usually max_log_lens, not max_log_tokens
    seed=42,
    disable_log_requests=True # Optional: reduces console spam
)

# Initialize engine within the context of the loop
# (Ideally, we pass the loop if the API supports it, or we rely on the fact 
# that we aren't awaiting it here directly).
engine = AsyncLLMEngine.from_engine_args(engine_args)

# --- 4. Initialize the Wrapper ---
sampling_params = SamplingParams(temperature=0.0) # Define params needed for the wrapper

# This is the object that holds the bridge logic
model_wrapper = VLLMQwenModel(
    engine=engine, 
    loop=loop, 
    sampling_params=sampling_params
)

# --- 5. Run Threads ---
responses = json.load(open(os.path.join(path, "data/responses.jsonl"), "r", encoding="utf-8"))

threads = []
for response in responses:
    # CRITICAL CHANGE:
    # 1. We pass 'model_wrapper' instead of raw engine/loop. 
    #    The check_atomic_fact.main function likely expects an object with a .generate() method.
    # 2. We passed 'response' and the model.
    thread = Thread(
        target=check_atomic_fact.main, 
        args=(response, model_wrapper) 
    )
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()