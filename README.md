# Hierarchical Fact-based Uncertainty Estimation and SAFE
Excuse the mess, we didn't really have time to organize the repo.
All important files are in the actualy_important folder (duh), with the following being the most relevant:

response_generation.ipynb - Used for generating resonses to queries in the eval set. Note that it does not perform RAG as all retrieval was done in advance due to memory constraints
safe_atomization.ipynb, safe_fact_rating - Atomizes and rates the facts using vllm_wrapper.py to call long_form_factuality with vLLM

UE.ipynb - Performs clustering of the atomic facts
cluster_analysis.ipynb - Creates the cluster hierarchies
hierarchy_analysis.ipynb - Implements and performs presence propagation and the regression
sequence_eval.ipynb - Computes the HSU scores for the sequences.
black_box.ipynb - Computes the HDU scores for the sequences.
safe_UE_corr - Combines the final scores from all 3 metrics and visualizes the result