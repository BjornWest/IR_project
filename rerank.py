#!/usr/bin/env python3
"""Re-rank query retrievals down to top-4 hits using Contriever embeddings."""

import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer


INPUT_PATH = Path("data/query_retrieval.json")
OUTPUT_PATH = Path("data/query_retrieval_top4.json")
MODEL_NAME = "facebook/contriever"
TOP_K = 4
BATCH_SIZE = 16


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings with an attention mask and L2-normalize."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    mean_embeddings = summed / counts
    return torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)


def embed(texts: List[str], tokenizer: AutoTokenizer, model: AutoModel, device: torch.device) -> torch.Tensor:
    """Embed texts with Contriever."""
    all_embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
        embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")
    with INPUT_PATH.open() as f:
        retrievals: Dict[str, List[Dict[str, str]]] = json.load(f)

    print("Loading Contriever…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device).eval()

    reranked: Dict[str, List[Dict[str, str]]] = {}

    for idx, (query, docs) in enumerate(retrievals.items(), start=1):
        query_emb = embed([query], tokenizer, model, device)[0]
        doc_texts = [f"{doc.get('title', '')}\n{doc.get('contents', '')}" for doc in docs]
        doc_embs = embed(doc_texts, tokenizer, model, device)
        scores = torch.mv(doc_embs, query_emb)  # cosine because embeddings are normalized
        keep = min(TOP_K, len(docs))
        top_idx = torch.topk(scores, k=keep).indices.tolist()
        reranked[query] = [docs[i] for i in top_idx]
        if idx % 50 == 0 or idx == len(retrievals):
            print(f"Processed {idx}/{len(retrievals)} queries")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(reranked, f, ensure_ascii=False, indent=4)
    print(f"Wrote top-{TOP_K} docs per query to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

