import networkx as nx
from collections import defaultdict

def build_base_graph(df):
    G = nx.Graph()
    cat_papers = defaultdict(list)
    author_papers = defaultdict(list)
    for _, row in df.iterrows():
        pid = row['id']
        G.add_node(pid, text=row['text'])
        for c in row['categories']:
            cat_papers[c].append(pid)
        for a in row['authors_set']:
            author_papers[a].append(pid)
    for papers in cat_papers.values():
        for i in range(len(papers)):
            for j in range(i+1, len(papers)):
                G.add_edge(papers[i], papers[j])
    for papers in author_papers.values():
        for i in range(len(papers)):
            for j in range(i+1, len(papers)):
                G.add_edge(papers[i], papers[j])
    return G