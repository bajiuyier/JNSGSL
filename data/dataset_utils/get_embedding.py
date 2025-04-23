import random
import time
import numpy as np
from gensim.models import Word2Vec
from tqdm import trange
import torch
from data.dataset_utils.pyg_load import pyg_load_dataset
import os

def gen_dw_emb(A, shape, number_walks=10, alpha=0, walk_length=100, window=10, workers=16, size=128):
    """
    Generate structural embeddings for nodes using the DeepWalk algorithm.

    Args:
      - A: Edge index array, format [row, col], indicating edges in the graph
      - shape: Total number of nodes in the graph
      - number_walks: Number of random walks starting from each node
      - alpha: Restart probability (0 means no restart), used for walk restarting
      - walk_length: Maximum length of each random walk
      - window: Window size used in Word2Vec
      - workers: Number of threads used to train Word2Vec
      - size: Dimension of the embedding vector

    Returns:
      - A numpy array of shape (shape, size), representing node embeddings
    """
    # Extract source and target nodes from A
    row, col = A[0], A[1]
    # Concatenate edge info and convert to string (Word2Vec requires string input)
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build adj_mat")
    t1 = time.time()

    # Construct adjacency list G as a dictionary: node -> list of neighbors
    G = {}
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)

    # Remove duplicates and self-loops from neighbors, and sort the list
    for node in G:
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)

    # Sort all nodes lexicographically
    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])

    # Generate corpus where each sample is a random walk sequence of nodes
    corpus = []  # Stores all random walk sequences
    for cnt in trange(number_walks):
        # Shuffle nodes to ensure random walk start points are random
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]  # Initialize the path starting from the current node
            while len(path) < walk_length:
                cur = path[-1]  # Current node
                # If current node has neighbors, continue the walk
                if len(G[cur]) > 0:
                    # With probability (1 - alpha), choose a random neighbor
                    if random.random() >= alpha:
                        path.append(random.choice(G[cur]))
                    else:
                        path.append(path[0])  # Restart to the initial node
                else:
                    # If no neighbors, terminate the walk
                    break
            # Append the walk sequence to the corpus
            corpus.append(path)

    t2 = time.time()
    print("Training word2vec")
    # Train Word2Vec model using the random walk sequences as training corpus
    model = Word2Vec(corpus,
                     vector_size=size,  # Dimension of the embedding
                     window=window,
                     min_count=0,
                     sg=1,  # Use Skip-Gram model
                     hs=1,  # Use Hierarchical Softmax
                     workers=workers)
    print("done.., cost: {}s".format(time.time() - t2))

    # Build output embedding matrix, one vector per node
    output = []
    for i in range(shape):
        # Embeddings are stored in model.wv as a dictionary keyed by string node IDs
        if str(i) in model.wv:
            output.append(model.wv[str(i)])
        else:
            print("{} not trained".format(i))
            output.append(np.zeros(size))
    return np.array(output)


# Set dataset path and name
path = os.getcwd() + '/data/dataset/'
name = 'wisconsin'
# Load dataset (using PyG format)
data = pyg_load_dataset(name)
# Get edge index
edge = data.edge_index
# Generate DeepWalk embeddings with dimension size (default 128), using the number of nodes in the dataset
emb = gen_dw_emb(edge, data.x.shape[0])
# Convert embeddings to torch tensor
emb = torch.tensor(emb)
# Specify path to save embeddings
# save_path = path + '/deepwalk_embedding/' + name
# # Save embeddings to file
# torch.save(emb, save_path)
print('finished')
