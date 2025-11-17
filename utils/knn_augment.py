from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def add_knn_edges_to_graph(G, node_list, embeddings_cpu, k=6):
    emb_np = embeddings_cpu.numpy()
    sims = cosine_similarity(emb_np)
    N = sims.shape[0]
    new_edges = []
    for i in range(N):
        sims[i, i] = -1.0
        kk = min(k, N-1)
        topk = np.argpartition(-sims[i], range(kk))[:kk]
        for j in topk:
            new_edges.append((node_list[i], node_list[j]))
    G_aug = G.copy()
    before = G_aug.number_of_edges()
    G_aug.add_edges_from(new_edges)
    after = G_aug.number_of_edges()
    return G_aug