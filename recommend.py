
from fintuner import bert_model, tokenizer, node_list, refined_emb_np, df_reindexed, recommend, pretty_recommendation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == "__main__":
    query = input("Enter your research query: ")
    ids, scores = recommend(query, top_k=8)
    print(pretty_recommendation(query, ids, scores, df_reindexed))