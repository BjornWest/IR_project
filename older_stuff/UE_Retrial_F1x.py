from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import List

# Randomise comparison with cluster. 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")


model_name = "microsoft/deberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def get_entailment_matrix(responses: List[str]):
    n = len(responses)
    similarity_matrix = np.zeros((n, n))
    
    entailment_id = -1
    for k, v in model.config.id2label.items():
        if v.upper() == "ENTAILMENT":
            entailment_id = k
            break
            
    if entailment_id == -1:
        raise ValueError("Could not find ENTAILMENT label in model config.")

    # TOKENIZATION LOOP

    tokenized_inputs_list = []
    
    print(f"Tokenizing {n*n} pairs individually...")
    
    for i in range(n):
        for j in range(n):

            current_pair = [responses[i], responses[j]]

            inputs = tokenizer(
                text=responses[i], 
                text_pair=responses[j],
                return_tensors="pt", 
                truncation=False, 
                max_length=256
            )

            if i == 0 and j ==4:
                print("\n" + "!"*40)
                print(f" DEBUG: TOKENIZATION CHECK (Pair {i} vs {j})")
                print("!"*40)
                print(f"Original 1: {current_pair[0]}")
                print(f"Original 2: {current_pair[1]}")
                

                decoded_sequence = tokenizer.decode(inputs["input_ids"][0])
                print(f"\nModel Input View:\n{decoded_sequence}")
                print("!"*40 + "\n")
            
            if i == 4 and j == 0:
                print("\n" + "!"*40)
                print(f" DEBUG: TOKENIZATION CHECK (Pair {i} vs {j})")
                print("!"*40)
                print(f"Original 1: {current_pair[0]}")
                print(f"Original 2: {current_pair[1]}")
                

                decoded_sequence = tokenizer.decode(inputs["input_ids"][0])
                print(f"\nModel Input View:\n{decoded_sequence}")
                print("!"*40 + "\n")

            tokenized_inputs_list.append((i, j, inputs))


    print("Running inference on tokenized list...")
    
    model.eval()
    
    for row, col, inputs in tokenized_inputs_list:

        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        

        entailment_score = probs[0][entailment_id].item()
        

        similarity_matrix[row, col] = entailment_score

    return similarity_matrix

def cluster_responses(responses: List[str], threshold: float = 0.5):

    sim_matrix = get_entailment_matrix(responses)
    
    # Possible fixes: Symmetrize the matrix
    # Averaging the directional logic
    symmetric_matrix = (sim_matrix + sim_matrix.T) / 2
    # Or taking the minimum
    # symmetric_matrix = np.minimum(sim_matrix, sim_matrix.T)
    
    print("\n--- Symmetrized Matrix (Bidirectional) ---")
    for row in symmetric_matrix:
        print([f"{x:.3f}" for x in row])


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
        "Global warming is a hoax to hurt the economy.",# Immediate action is required for the climate.", # Cluster 0
        "We can wait 50 years to see if weather stabilizes.", # Cluster 3
        "The stock market is fluctuating due to tech stocks.", # Cluster 1
        "Climate change demands urgent policy changes.", # Cluster 0
        "Governments must act now to stop global warming.", # Cluster 2
        "Immediate action is required for the climate.", # Cluster 0
    ]

    print("-" * 40)
    print("Analyzing and Clustering Responses...")
    print("-" * 40)

    # Threshold 0.7 means responses must entail each other with >70% confidence to cluster
    grouped_responses = cluster_responses(my_responses, threshold=0.6)

    print("\n" + "="*30)
    print(" CLUSTERING RESULTS ")
    print("="*30)
    
    for cluster_id, texts in grouped_responses.items():
        print(f"\nCluster {cluster_id}:")
        for text in texts:
            print(f"  - {text}")