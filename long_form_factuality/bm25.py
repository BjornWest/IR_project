import os
import json
from pyserini.search.lucene import LuceneSearcher
from huggingface_hub import snapshot_download
import pickle
import tqdm  # pip install tqdm

class BM25Retriever:
    _instance = None
    _searcher = None
    _wiki_download_dir = None
    _offset_map = None
    _returned_docids = None  # Track which documents have already been returned

    def __new__(cls):
        """Singleton pattern to ensure we only load the index once."""
        if cls._instance is None:
            cls._instance = super(BM25Retriever, cls).__new__(cls)
            cls._instance._returned_docids = set()  # Initialize the set
            cls._instance._initialize_index()
        return cls._instance


    def _build_index_offset(self, jsonl_path, index_path):
        offset_map = {}

        print("Building offset index...")
        with open(jsonl_path, "rb") as f:
            # Get the initial position
            offset = f.tell()
            for line in tqdm.tqdm(f):
                # We only parse the ID to save time/memory
                # Assuming the structure is {"id": "...", "contents": "..."}
                try:
                    # fast parse: decode just enough to find the ID if possible, 
                    # or parse the whole line if the structure is complex.
                    doc = json.loads(line)
                    doc_id = doc.get("id") # Ensure this matches your json key
                    
                    if doc_id:
                        offset_map[doc_id] = offset
                except Exception:
                    pass
                
                # Update offset for the NEXT line
                offset = f.tell()
        # Save the map to disk
        with open(index_path, "wb") as f:
            pickle.dump(offset_map, f)

        print(f"Index built! Mapped {len(offset_map)} documents.")




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
        self._wiki_download_dir = os.path.join(os.getcwd(), "wiki_dump")
        # check if corresponding text documents have been downloaded
        if not os.path.exists(self._wiki_download_dir):
            os.makedirs(self._wiki_download_dir)
            print("Wiki dump not found. Downloading...")
            snapshot_download(
                repo_id="HeydarS/enwiki_20251001",
                repo_type="dataset",
                local_dir=self._wiki_download_dir,
                local_dir_use_symlinks=False
            )
            index_path = os.path.join(self._wiki_download_dir, "enwiki_offsets.pkl")
            self._build_index_offset(os.path.join(self._wiki_download_dir, "enwiki_20251001.jsonl"), index_path)
        self._offset_map = pickle.load(open(os.path.join(self._wiki_download_dir, "enwiki_offsets.pkl"), "rb"))
    def search(self, query: str, k: int = 5) -> str:
        """
        Searches the index and returns a formatted string of results.
        Only returns documents that haven't been returned in previous searches.
        """
        # Request more results than needed to account for filtering out already-returned docs
        # Request k * 3 to ensure we have enough new results (adjust multiplier as needed)
        max_hits = max(k * 3, 50)  # Request at least 50 results to have enough to filter from
        hits = self._searcher.search(query, k=max_hits)

        result_str = f"Searched for \"{query}\" and found the following results:\n"
        new_results_count = 0
        
        for hit in hits:
            # Skip documents that have already been returned
            if hit.docid in self._returned_docids:
                continue
            
            # Mark this document as returned
            self._returned_docids.add(hit.docid)
            
            # Get the document content
            offset = self._offset_map[hit.docid]
            with open(os.path.join(self._wiki_download_dir, "enwiki_20251001.jsonl"), "rb") as f:
                f.seek(offset)
                line = f.readline()
                content = json.loads(line)['contents']
            
            result_str += f"\nResult {new_results_count + 1}:\n{content}\n"
            new_results_count += 1
            
            # Stop once we have k new results
            if new_results_count >= k:
                break
        
        # If we couldn't find enough new results, return what we have
        if new_results_count == 0:
            return "No new documents found. All relevant documents have already been retrieved."
        
        return result_str
    
    def reset_search_history(self):
        """
        Resets the tracking of returned documents.
        Call this when starting a new query session.
        """
        self._returned_docids.clear()

# Global instance for easy import
bm25_engine = BM25Retriever()