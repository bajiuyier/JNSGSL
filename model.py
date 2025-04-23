from utils import normalize
from base_model import GNNEncoder_OpenGSL
from utils import NonLinear
from utils import InnerProduct
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
import torch.nn.functional as F

class VGAE(nn.Module):
    """
    VGAE model: can serve as GAE/VGAE for edge prediction.
    - Uses GNNEncoder_OpenGSL as encoder to generate node embeddings;
    - Applies NonLinear activation followed by InnerProduct to compute edge scores.
    """
    def __init__(self, n_feat, conf):
        """
        Args:
        - n_feat: input feature dimension
        - conf: configuration object, should include 'gsl' section with:
            - conf.gsl['gae']: whether the model is GAE (boolean)
            - conf.gsl['n_embed']: embedding dimension
        """
        super(VGAE, self).__init__()
        self.gae = conf.gsl['gae']
        self.encoder = GNNEncoder_OpenGSL(n_feat=n_feat, n_class=conf.gsl['n_embed'], bias=False,
                                          weight_initializer='glorot', **conf.gsl)
        self.nonlinear = NonLinear('relu')
        self.metric = InnerProduct()

    def reset_parameters(self):
        """Reset parameters of all submodules (if applicable)"""
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, feats, adj):
        """
        Forward pass:
        1. Use GCN encoder to compute node embeddings (mean);
        2. Apply activation; if GAE, return embeddings directly;
        3. Use inner product to get edge score matrix (adj_logits).

        Args:
          - feats: node feature matrix
          - adj: adjacency matrix (dense or sparse)

        Returns:
          - adj_logits: predicted edge score matrix
        """
        mean = self.encoder(feats, adj)
        mean = self.nonlinear(mean)
        Z = mean
        adj_logits = self.metric(Z)
        return adj_logits


class GSL(nn.Module):
    """
    GSL model:
    - Includes an edge prediction network (ep_net) and a node classification network (nc_net);
    - Modifies the adjacency matrix via edge removal and addition.
    """
    def __init__(self, n_feat, n_class, conf):
        """
        Args:
         - n_feat: input feature dimension
         - n_class: number of output classes (for node classification)
         - conf: configuration object, contains 'gsl' parameters (e.g., temperature, ratios), and device info
        """
        super(GSL, self).__init__()
        self.temperature = conf.gsl['temperature']
        self.remove_ratio = conf.gsl['remove_ratio']
        self.add_ratio = conf.gsl['add_ratio']
        self.device = conf.device

        self.ep_net = VGAE(n_feat, conf)
        self.nc_net = GNNEncoder_OpenGSL(n_feat=n_feat, n_class=n_class,
                                         weight_initializer='glorot', bias_initializer='zeros', **conf.model)

    def reset_parameters(self):
        """Reset parameters of all submodules"""
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def sample_adj_edge_remove(self, adj_logits, adj_orig, change_frac):
        """
        Remove the lowest-scoring existing edges from the original adjacency matrix.

        Args:
          - adj_logits (Tensor): predicted edge scores
          - adj_orig (Tensor or SparseTensor): original adjacency matrix
          - change_frac (float): fraction of edges to remove

        Returns:
          - Tensor: modified adjacency matrix (dense format)
        """
        adj = (adj_orig.to_dense() if adj_orig.is_sparse else adj_orig).clone().detach()
        triu_indices = torch.triu_indices(adj.size(0), adj.size(1), offset=1).to(self.device)
        edge_scores = adj_logits[triu_indices[0], triu_indices[1]]
        min_val = torch.min(edge_scores)
        max_val = torch.max(edge_scores)
        norm_edge_scores = (edge_scores - min_val) / (max_val - min_val + 1e-8)
        exist_mask = adj[triu_indices[0], triu_indices[1]] > 0
        exist_edge_scores = norm_edge_scores[exist_mask]
        num_exist = exist_edge_scores.numel()
        num_delete = int(num_exist * change_frac)
        if num_delete < 1:
            return adj
        threshold = torch.topk(exist_edge_scores, num_delete, largest=False).values[-1]
        delete_mask = (norm_edge_scores < threshold) & exist_mask
        del_rows = triu_indices[0][delete_mask]
        del_cols = triu_indices[1][delete_mask]
        adj[del_rows, del_cols] = 0
        adj[del_cols, del_rows] = 0
        return adj

    def sample_adj_edge_add(self, adj_logits, adj_orig, change_frac, adj_new):
        """
        Add new edges based on high-scoring non-existent edges.

        Args:
          - adj_logits: edge prediction score matrix
          - adj_orig: original adjacency matrix
          - change_frac: ratio of edges to add (relative to original unique edge count)
          - adj_new: adjacency matrix after removal

        Returns:
          - Tensor: adjacency matrix after edge addition
        """
        adj = (adj_orig.to_dense() if adj_orig.is_sparse else adj_orig).clone().detach()
        triu_indices = torch.triu_indices(adj.size(0), adj.size(1), offset=1)
        exist_edges = adj[triu_indices[0], triu_indices[1]]
        n_unique = exist_edges.nonzero().size(0)
        n_add = int(n_unique * change_frac)
        if n_add < 1:
            return adj_new
        edge_scores = adj_logits[triu_indices[0], triu_indices[1]]
        min_val = torch.min(edge_scores)
        max_val = torch.max(edge_scores)
        norm_scores = (edge_scores - min_val) / (max_val - min_val + 1e-8)
        non_exist_mask = (exist_edges == 0)
        candidate_scores = norm_scores[non_exist_mask]
        candidate_indices = torch.nonzero(non_exist_mask, as_tuple=False).squeeze()
        if candidate_scores.numel() < n_add:
            selected_idx = candidate_indices
        else:
            _, sorted_idx = torch.topk(candidate_scores, n_add, largest=True)
            selected_idx = candidate_indices[sorted_idx]
        add_mask = torch.zeros_like(edge_scores)
        add_mask[selected_idx] = 1.0
        add_mask_full = torch.zeros_like(adj)
        add_mask_full[triu_indices[0], triu_indices[1]] = add_mask
        add_mask_full = add_mask_full + add_mask_full.T
        adj_new = adj_new.clone().detach()
        adj_new[add_mask_full > 0] = 1
        return adj_new

    def forward(self, feats, adj, adj_orig):
        """
        Forward pass:
          1. Compute edge scores using ep_net
          2. Modify adjacency matrix by removing and adding edges
          3. Normalize the new adjacency matrix and use it in nc_net

        Returns:
          - output: node classification logits
          - adj_logits: edge prediction scores
          - adj_new: modified adjacency matrix
        """
        adj_logits = self.ep_net(feats, adj)
        adj_new = self.sample_adj_edge_remove(adj_logits, adj_orig, self.remove_ratio)
        adj_new = self.sample_adj_edge_add(adj_logits, adj_orig, self.add_ratio, adj_new)
        adj_new_normed = normalize(adj_new)
        output = self.nc_net(feats, SparseTensor.from_dense(adj_new_normed))
        return output, adj_logits, adj_new


class JNSGSL(nn.Module):
    def __init__(self, in_dim, num_targets, s_in_dim, conf):
        """
        Args:
            in_dim: input dimension of main features
            num_targets: number of classification targets
            s_in_dim: input dimension of auxiliary features
            conf: configuration parameters (includes training weights beta, gamma)
        """
        super(JNSGSL, self).__init__()
        self.feat_model = GSL(in_dim, num_targets, conf)
        self.sfeat_model = GSL(s_in_dim, num_targets, conf)
        self.beta = conf.training['beta']
        self.gamma = conf.training['gamma']
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, feats, s_feats, normalized_adj, adj_orig):
        """
        Forward pass:
            Compute outputs from both main and auxiliary branches

        Returns:
            out1, adj_logits, adj_new from feat_model;
            out2, adj_logits2, adj_new2 from sfeat_model
        """
        out1, adj_logits, adj_new = self.feat_model(feats, normalized_adj, adj_orig)
        out2, adj_logits2, adj_new2 = self.sfeat_model(s_feats, normalized_adj, adj_orig)
        return out1, adj_logits, adj_new, out2, adj_logits2, adj_new2

    def sim(self, z1, z2):
        """
        Compute cosine similarity (normalize then exponentiate dot product)
        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_num = torch.mm(z1, z2.t())
        dot_den = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_num / dot_den)
        return sim_matrix

    def cal(self, z1_proj, z2_proj):
        """
        Compute mutual information loss (contrastive loss)
        """
        sim_matrix = self.sim(z1_proj, z2_proj)
        sim_matrix_T = sim_matrix.t()
        sim_matrix_norm = sim_matrix / (torch.sum(sim_matrix, dim=1, keepdim=True) + 1e-8)
        loss1 = -torch.log(sim_matrix_norm.diag() + 1e-8).mean()
        sim_matrix_T_norm = sim_matrix_T / (torch.sum(sim_matrix_T, dim=1, keepdim=True) + 1e-8)
        loss2 = -torch.log(sim_matrix_T_norm.diag() + 1e-8).mean()
        return (loss1 + loss2) / 2

    def compute_loss(self, feats, s_feats, normalized_adj, adj_orig, labels, train_mask, pos_weight, norm_w):
        """
        Compute total loss:
            - Node classification loss (loss_class)
            - Edge prediction loss (ep_loss)
            - Mutual information loss (mi_loss)

        Returns:
            total_loss and intermediate outputs for evaluation
        """
        out1, adj_logits, adj_new, out2, adj_logits2, adj_new2 = self.forward(feats, s_feats, normalized_adj, adj_orig)
        loss_class = self.loss_fn(out1[train_mask], labels[train_mask])
        ep_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig, pos_weight=pos_weight)
        mi_loss = self.cal(out1, out2)
        total_loss = loss_class + self.beta * ep_loss + self.gamma * mi_loss
        return total_loss, out1, out2, adj_logits, adj_new
