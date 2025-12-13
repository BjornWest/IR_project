from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List

# !! DOES NOT WORK !!

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

model_name = "microsoft/deberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def get_adjusted_score_matrix(responses: List[str]):
    n = len(responses)
    similarity_matrix = np.zeros((n, n))
    

    entailment_id = -1
    contradiction_id = -1
    
    for k, v in model.config.id2label.items():
        if v.upper() == "ENTAILMENT":
            entailment_id = k
        elif v.upper() == "CONTRADICTION":
            contradiction_id = k
            
    if entailment_id == -1 or contradiction_id == -1:
        raise ValueError("Could not find label IDs in model config.")

    # TOKENIZATION LOOP
    tokenized_inputs_list = []
    print(f"Tokenizing {n*n} pairs individually...")
    
    for i in range(n):
        for j in range(n):
            inputs = tokenizer(
                text=responses[i], 
                text_pair=responses[j],
                return_tensors="pt", 
                truncation=False, 
                max_length=256
            )
            tokenized_inputs_list.append((i, j, inputs))


    print("Running inference...")
    model.eval()
    
    for row, col, inputs in tokenized_inputs_list:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        

        ent_score = probs[0][entailment_id].item()
        cont_score = probs[0][contradiction_id].item()
        
        # PENALISATION
        adjusted_score = ent_score - cont_score
        

        if adjusted_score < 0:
            adjusted_score = 0
            
        similarity_matrix[row, col] = adjusted_score

    return similarity_matrix

def cluster_responses(responses: List[str], threshold: float = 0.5):

    symmetric_matrix = get_adjusted_score_matrix(responses)
    
    print("\nFinal Similarity Matrix (Symmetrized & Penalized):")
    for row in symmetric_matrix:
        print([f"{x:.2f}" for x in row])


    distance_matrix = 1 - symmetric_matrix
    distance_matrix[distance_matrix < 0] = 0

    clustering = AgglomerativeClustering(
        n_clusters=None, 
        metric='precomputed', 
        linkage='average', 
        distance_threshold=1 - threshold 
    )
    
    clustering.fit(distance_matrix)
    
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(responses[idx])
        
    return clusters

if __name__ == "__main__":
    my_responses = [
        "Governments must act now to stop global warming.",           # A
        "We can wait 50 years to see if weather stabilizes.",         # B
        "The stock market is fluctuating due to tech stocks.",        # C
        "Climate change demands urgent policy changes.",              # A
        "Global warming is a hoax created to hurt the economy.",      # D (Opposite of A)
        "Immediate action is required for the climate."               # A
    ]

    print("-" * 40)
    print("Analyzing and Clustering...")
    print("-" * 40)

    
    grouped_responses = cluster_responses(my_responses, threshold=0.7)

    print("\n" + "="*30)
    print(" CLUSTERING RESULTS ")
    print("="*30)
    
    for cluster_id, texts in grouped_responses.items():
        print(f"\nCluster {cluster_id}:")
        for text in texts:
            print(f"  - {text}")