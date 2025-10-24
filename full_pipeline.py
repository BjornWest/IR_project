import torch
from transformers import AutoTokenizer, AutoModel
import ollama
import faiss
import json
import os
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
LFF_ROOT = PROJECT_ROOT / "long_form_factuality"
sys.path.append(str(LFF_ROOT))

from long_form_factuality.eval.safe import get_atomic_facts
from long_form_factuality.common.modeling import OllamaQwenModel
from long_form_factuality.eval.safe.rate_atomic_fact import check_atomic_fact

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

generation_model = "qwen2.5:7b"

index = faiss.read_index(os.path.join(path, "data/index/wiki1.index"))

objs = []
with open(
    os.path.join(path, "data/chunked/wiki1_chunked.jsonl"), "r", encoding="utf-8"
) as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)  # parse one JSON object per line
            objs.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_num}: {e}")

text_collection = [obj["contents"] for obj in objs]


print(path)
print("Loading models...")
retrieval_model = AutoModel.from_pretrained("facebook/contriever")
retrieval_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
print("Models loaded.")


system_prompt = """
You are an AI assistant that provides accurate and concise answers based on the provided context. Use the context to answer the questions to the best of your ability.
If the context does not contain the answer, respond with "I don't know".
"""

queries = [
    "Why did the titanic sink?",
    "Can you tell me about the history of the internet?",
    "What are the benefits of renewable energy?",
]


def embed_query(text):
    inputs = retrieval_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = retrieval_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


def generate_response(prompt, silent=False):
    response = ollama.generate(
        model=generation_model, prompt=prompt, stream=not silent, system=system_prompt
    )
    if not silent:
        chunks = []
        for chunk in response:
            chunk_text = chunk["response"]
            print(chunk_text, end="", flush=True)
            chunks.append(chunk_text)
        print()

        return "".join(chunks)
    else:
        return response["response"]


def main(silent=False):
    for query in queries[0:1]:
        query_embedding = embed_query(query).numpy().astype("float32")
        D, indicies = index.search(query_embedding, k=10)

        retrieved_docs = []
        for idx in indicies[0]:
            retrieved_docs.append(text_collection[idx])

        context = "\n\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        print(prompt)
        answer = generate_response(prompt, silent=silent)

        # Change directory so that file importing works in long_form_factuality
        original_cwd = os.getcwd()
        os.chdir(LFF_ROOT)

        atomic_facts = get_atomic_facts.main(
            answer, OllamaQwenModel(model_name="qwen2.5:7b")
        )
        for fact in atomic_facts:
            print("Fact:", fact)

        # for fact in atomic_facts:
        #     print("Fact:", fact)
        #     atomic_fact_rating = check_atomic_fact(
        #         prompt=prompt,
        #         response=answer,
        #         atomic_fact=fact,
        #         rater=OllamaQwenModel(model_name=generation_model),
        #         )


if __name__ == "__main__":
    main()
