import json
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MATRICES_PATH = os.path.join(DATA_DIR, "matrices.npy")
CLUSTERS_PATH = os.path.join(DATA_DIR, "query_hierarchical_clusters.jsonl")

# You can override these via environment variables if you want to tweak selection.
THRESHOLD = float(os.environ.get("EDGE_THRESHOLD", "0.65"))
TOP_K = int(os.environ.get("TOP_K", "20"))


def load_data() -> Tuple[Sequence[Any], List[Dict[str, Any]]]:
    matrices = np.load(MATRICES_PATH, allow_pickle=True)

    cluster_data: List[Dict[str, Any]] = []
    with open(CLUSTERS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            cluster_data.append(json.loads(line))

    if len(matrices) != len(cluster_data):
        print(
            f"Warning: matrices ({len(matrices)}) and cluster_data ({len(cluster_data)}) lengths differ",
            file=sys.stderr,
        )
    return matrices, cluster_data


def build_adj(matrix: Any, threshold: float) -> List[List[int]]:
    n = len(matrix)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            if i == j:
                continue
            try:
                score = row[j][2]
            except Exception:
                continue
            if score > threshold:
                adj[i].append(j)
    return adj


def longest_path(adj: List[List[int]]) -> int:
    """Return the length of the longest path in a (mostly) DAG; skips cycles defensively."""
    sys.setrecursionlimit(10000)
    seen: Dict[int, int] = {}
    visiting: set[int] = set()

    def dfs(u: int) -> int:
        if u in seen:
            return seen[u]
        visiting.add(u)
        best = 0
        for v in adj[u]:
            if v in visiting:
                # Skip cycles; the data should be a DAG but be defensive.
                continue
            best = max(best, 1 + dfs(v))
        visiting.remove(u)
        seen[u] = best
        return best

    return max((dfs(i) for i in range(len(adj))), default=0)


def extract_clusters(entry: Any) -> Tuple[List[Dict[str, Any]], str | None]:
    """Return clusters list and an optional query string for reporting."""
    if isinstance(entry, dict):
        clusters = entry.get("clusters", entry)
        query = entry.get("query") or entry.get("original_query")
    else:
        clusters = entry
        query = None
    return clusters, query


def summarize_graph(matrix: Any, entry: Any, threshold: float) -> Dict[str, Any]:
    clusters, query = extract_clusters(entry)
    adj = build_adj(matrix, threshold)
    depth = longest_path(adj)
    max_branch = max((len(nbrs) for nbrs in adj), default=0)
    avg_branch = (sum(len(nbrs) for nbrs in adj) / len(adj)) if adj else 0.0
    score = depth / (1 + max_branch)  # prefer deeper graphs with limited fan-out
    return {
        "depth": depth,
        "max_branch": max_branch,
        "avg_branch": avg_branch,
        "score": score,
        "num_nodes": len(adj),
        "query": query,
    }


def rank_graphs() -> List[Tuple[int, Dict[str, Any]]]:
    matrices, cluster_data = load_data()
    results: List[Tuple[int, Dict[str, Any]]] = []
    for idx, (matrix, entry) in enumerate(zip(matrices, cluster_data)):
        summary = summarize_graph(matrix, entry, THRESHOLD)
        results.append((idx, summary))
    results.sort(key=lambda x: (-x[1]["score"], -x[1]["depth"], x[1]["max_branch"]))
    return results


def main() -> int:
    results = rank_graphs()
    print(f"Top {TOP_K} deep-but-narrow candidates (threshold={THRESHOLD}):")
    for idx, summary in results[:TOP_K]:
        print(
            f"{idx:4d} | depth={summary['depth']:3d} | "
            f"max_branch={summary['max_branch']:2d} | avg_branch={summary['avg_branch']:.2f} | "
            f"nodes={summary['num_nodes']:3d} | score={summary['score']:.2f} | "
            f"query={summary['query']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

