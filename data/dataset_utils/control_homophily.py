import torch
import numpy as np


def control_homophily(adj, labels, homophily):
    np.random.seed(0)
    # change homophily through adding edges
    adj = adj.to_dense()
    n_edges = adj.sum()/2
    n_nodes = len(labels)
    homophily_orig = get_homophily(labels, adj, 'edge')
    # print(homophily_orig)
    if homophily<homophily_orig:
        # add noisy edges
        n_add_edges = int(n_edges*homophily_orig/homophily-n_edges)
        while n_add_edges>0:
            u = np.random.randint(0, n_nodes)
            vs = np.arange(0, n_nodes)[labels!=labels[u]]
            v = np.random.choice(vs)
            if adj[u, v]==0:
                adj[u,v]=adj[v,u]=1
                n_add_edges-=1
    if homophily>homophily_orig:
        # add helpful edges
        n_add_edges = int(n_edges*(1-homophily_orig)/(1-homophily)-n_edges)
        while n_add_edges > 0:
            u = np.random.randint(0, n_nodes)
            vs = np.arange(0, n_nodes)[labels==labels[u]]
            v = np.random.choice(vs)
            if u==v:
                continue
            if adj[u,v]==0:
                adj[u,v]=adj[v,u]=1
                n_add_edges -= 1
    return adj.to_sparse()

def get_node_homophily(label, adj):
    label = label.cpu().numpy()
    adj = adj.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = (np.multiply((label == label.T), adj)).sum(axis=1)
    d = adj.sum(axis=1)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1. / d[i])
    return np.mean(homos)


def get_edge_homophily(label, adj):
    num_edge = adj.sum()
    cnt = 0
    for i, j in adj.nonzero():
        if label[i] == label[j]:
            cnt += adj[i, j]
    return cnt/num_edge


def get_homophily(label, adj, type='edge', fill=None):
    if fill:
        np.fill_diagonal(adj, fill)
    return eval('get_'+type+'_homophily(label, adj)')


def get_adjusted_homophily(_label, adj):
    label = _label.long()
    labels = label.max() + 1
    d = adj.sum(1)
    E = d.sum()
    D = torch.zeros(labels)
    for i in range(adj.shape[0]):
        D[label[i]] += d[i]

    h_edge = get_edge_homophily(label, adj)
    sum_pk = ((D / E) ** 2).sum()

    return (h_edge - sum_pk) / (1 - sum_pk)

