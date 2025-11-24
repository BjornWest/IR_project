import os
import json
from pyserini.search.lucene import LuceneSearcher
from huggingface_hub import snapshot_download

class BM25Retriever:
    _instance = None
    _searcher = None

    def __new__(cls):
        """Singleton pattern to ensure we only load the index once."""
        if cls._instance is None:
            cls._instance = super(BM25Retriever, cls).__new__(cls)
            cls._instance._initialize_index()
        return cls._instance

    def _initialize_index(self):
        print("Initializing BM25 Index... (This might take a moment)")
        
        # 1. Define download path
        download_dir = os.path.join(os.getcwd(), "bm25_data")
        
        # 2. Download (or verify existing)
        try:
            index_path = snapshot_download(
                repo_id="HeydarS/bm25_index", 
                repo_type="dataset",
                local_dir=download_dir,
                local_dir_use_symlinks=False # Force copy to avoid Windows permission errors
            )
        except Exception as e:
            print(f"Download failed: {e}")
            raise

        # 3. Load Searcher
        try:
            self._searcher = LuceneSearcher(index_path)
            self._searcher.set_bm25(k1=1.2, b=0.75)
        except Exception as e:
            print("Failed to load Lucene index. Do you have Java 11+ installed?")
            raise e
        print("BM25 Index loaded successfully.")

    def search(self, query: str, k: int = 5) -> str:
        """
        Searches the index and returns a formatted string of results.
        """
        try:
            hits = self._searcher.search(query, k=k)
            
            formatted_results = []
            
            for i, hit in enumerate(hits):
                # Lucene stores the original text in a JSON string inside 'raw'
                # We must extract it to give the LLM readable text.
                content = ""
                try:
                    raw_json = json.loads(hit.raw)
                    # Try common field names for text content
                    content = raw_json.get('contents') or raw_json.get('text') or raw_json.get('body') or "No text content found."
                except:
                    content = "Error decoding document text."

                result_str = f"Result {i+1} (Score: {hit.score:.2f}):\n{content}"
                formatted_results.append(result_str)

            return "\n\n".join(formatted_results)

        except Exception as e:
            print(f"Search failed: {e}")
            return ""

# Global instance for easy import
bm25_engine = BM25Retriever()