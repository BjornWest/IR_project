import asyncio
import time
import torch
import warnings
import itertools
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import torch.nn.functional as F
# 1. CONFIGURATION
# ---------------------------------------------------------
# REPLACE THIS with your actual model path or ID
NLI_MODEL_PATH = "microsoft/deberta-large-mnli" #cross-encoder/nli-deberta-v3-large" 
EMBED_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
MAX_BATCH_SIZE = 128
MAX_WAIT_TIME = 0.05  # Lower latency window for GPU
TOP_PRIORITY = 0
# ---------------------------------------------------------

app = FastAPI()
warnings.filterwarnings("ignore")

# Global State
nli_model = None
nli_tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
request_queue = asyncio.PriorityQueue()
counter = itertools.count()


embed_tokenizer = None
embed_model = None

class EntailmentRequest(BaseModel):
    premise: str
    hypothesis: str
    priority: int

class VectorRequest(BaseModel):
    texts: List[str]

def load_models():
    print(f"Loading model from: {NLI_MODEL_PATH}")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Unexpected)'}")
    global embed_tokenizer, nli_tokenizer, nli_model, embed_model
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH).to(device)
    nli_model.eval()

    print(f"Loading Embedder: {EMBED_MODEL_PATH}...")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_PATH, trust_remote_code=True).half().to(device)
    embed_model.eval()
    
    # Warmup to initialize CUDA buffers
    print("Warming up GPU...")
    dummy_input = nli_tokenizer(
        ["Warmup premise"], ["Warmup hypothesis"], 
        return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        nli_model(**dummy_input)
    print("Model ready.")

def run_inference_batch(inputs: List[dict]):
    """
    Tokenizes and classifies a batch of inputs on the GPU.
    """
    premises = [i['premise'] for i in inputs]
    hypotheses = [i['hypothesis'] for i in inputs]

    # Tokenize the batch
    # padding=True is crucial here: it pads to the longest sequence *in this specific batch*
    encoded = nli_tokenizer(
        premises, 
        hypotheses, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    ).to(device)
    decoded = nli_tokenizer.decode(encoded.input_ids[0], skip_special_tokens=False)
    with torch.no_grad():
        outputs = nli_model(**encoded)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    id2label = nli_model.config.id2label
    results = []
    for sample_probs, pred in zip(probabilities, predictions):
        prob_dict = {
            id2label[class_idx]: sample_probs[class_idx].item()
            for class_idx in range(sample_probs.shape[-1])
        }
        results.append({
            "label": id2label[pred.item()],
            "probabilities": prob_dict
        })

    return results

# --- BACKGROUND WORKER ---

async def batch_processor():
    global TOP_PRIORITY
    while True:
        batch_items = []
        batch_futures = []
        
        # 1. Wait for first item
        priority_wrapper = await request_queue.get()
        item = priority_wrapper[2]
        TOP_PRIORITY = -priority_wrapper[0]

        batch_items.append(item[0])
        batch_futures.append(item[1])
        
        # 2. Collect more items within the time window
        start_time = time.time()
        while len(batch_items) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_time
            remaining = MAX_WAIT_TIME - elapsed
            
            if remaining <= 0:
                break
            
            try:
                # specific wait to fill the buffer
                priority_wrapper = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                item = priority_wrapper[2]
                batch_items.append(item[0])
                batch_futures.append(item[1])
            except asyncio.TimeoutError:
                break
        
        # 3. Process Batch
        print(f"[{time.strftime('%X')}] Pending: {request_queue.qsize()} | Batch Size: {len(batch_items)} | Queue Top: {TOP_PRIORITY}")
        try:
            # We run the blocking GPU call in a separate thread so we don't block the async event loop
            results = await asyncio.to_thread(run_inference_batch, batch_items)
            
            for future, result in zip(batch_futures, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            print(f"Batch Error: {e}")
            for future in batch_futures:
                if not future.done():
                    future.set_exception(e)
# --- LIFESPAN ---

async def monitor_queue_status():
    print("Queue monitor started.")
    while True:
        q_size = request_queue.qsize()
        
        # 1. Check Waiting Line
        queue_top = 0
        if q_size > 0:
            try:
                top_item = request_queue._queue[0]
                queue_top = -top_item[0]
            except: pass
            
        # 2. Check Active GPU Work
        active_top = TOP_PRIORITY
        
        # 3. True System Max
        true_max = max(queue_top, active_top)
        
        # Only log if there is work (waiting OR active)
        if q_size > 0 or active_top > 0:
            # print(f"[{time.strftime('%X')}] Pending: {q_size} | Queue Top: {active_top} | Active Batch: {active_top} | Max Priority: {true_max}")
            pass
        
        await asyncio.sleep(2)





@app.on_event("startup")
async def startup_event():
    load_models()
    asyncio.create_task(batch_processor())
    asyncio.create_task(monitor_queue_status())
# --- ENDPOINT ---

@app.post("/classify")
async def classify(req: EntailmentRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    queue_item = (req.priority, next(counter), (req.model_dump(), future))
    await request_queue.put(queue_item)
    result = await future
    return {"status": "success", **result}




# Helper function for Qwen Pooling (Last Token Pooling)
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Updated /vectors endpoint
@app.post("/vectors")
async def get_vectors(req: VectorRequest):
    # 1. Format inputs with the required Qwen instruction
    # Note: For clustering facts, we treat them as documents (no heavy instruction needed),
    # but for the queries you compare against, use the query instruction.
    # For simplicity in clustering (Fact vs Fact), standard raw text is often okay, 
    # but adding a generic prefix is safer.
    formatted_texts = [f"{t}" for t in req.texts] 

    # 2. Tokenize
    batch_dict = embed_tokenizer(
        formatted_texts, 
        max_length=256, # You can go up to 32k, but 8k is usually plenty
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    ).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = embed_model(**batch_dict)
        # Qwen uses LAST TOKEN pooling, not Mean pooling like BERT
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # 4. Normalize (Cosine Similarity requires normalized vectors)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return {"vectors": embeddings.tolist()}