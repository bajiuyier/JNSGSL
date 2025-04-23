import os
import time
import warnings
import numpy as np
from copy import deepcopy
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

# Import related tools and configuration
from utils import (
    normalize,
    MultipleOptimizer,
    get_lr_schedule_by_sigmoid,
    eval_edge_pred
)
from base_solver import Solver
from config.util import load_conf
from data.dataset_utils.dataset import Dataset
from model import JNSGSL

# Set CUDA environment variables and suppress warnings
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
warnings.filterwarnings("ignore")


class JNSGSLSolver(Solver):
    """
    JNSGSLSolver: Solver for Joint Graph Structure Learning
    This solver is based on the JNSGSL model, and simultaneously trains
    the main feature branch and auxiliary feature branch,
    including both the pre-training and main training phases,
    with early stopping to save the best model.
    """
    def __init__(self, conf, dataset):
        super(JNSGSLSolver, self).__init__(conf, dataset)
        # Fix the random seed to ensure reproducibility
        self.method_name = "jns_gsl"
        print("Solver Version : [jns_gsl]")

        # Normalize the adjacency matrix: convert the normalized original adjacency matrix to sparse format
        self.normalized_adj = SparseTensor.from_torch_sparse_coo_tensor(normalize(self.adj))
        # Original adjacency matrix with self-loops (converted to dense and added with identity matrix)
        self.adj_orig = self.adj.to_dense() + torch.eye(self.n_nodes).to(self.device)

        # Add device info into config and construct the JNSGSL model (with main and auxiliary branches and loss computation)
        conf.device = self.device
        self.model = JNSGSL(
            self.feats.shape[1],
            self.num_targets,
            self.sfeats.shape[1],
            conf
        ).to(self.device)

        # Initialize result dictionary
        self.init_result()

    def init_result(self):
        """Initialize the dictionary to record training, validation, and test results."""
        self.result = {
            'train': -1, 'valid': -1, 'test': -1,
            'feature_train': -1, 'feature_val': -1, 'feature_test': -1,
            'sfeature_train': -1, 'sfeature_val': -1, 'sfeature_test': -1
        }

    def set_method(self):
        """
        Sample edges according to dataset size, and set edges and labels for validation.
        Positive and negative edges are sampled to construct validation data.
        Negative edges are mainly used to validate the effect of structure learning.
        """
        # Determine sampling ratio based on dataset size
        edge_frac = 0.01 if self.labels.size(0) > 5000 else 0.1

        # Convert adjacency matrix to sparse format and ensure diagonal is all ones
        adj_matrix = sp.csr_matrix(self.adj.to_dense().cpu().numpy())
        adj_matrix.setdiag(1)
        n_edges_sample = int(edge_frac * adj_matrix.nnz / 2)

        # Randomly sample negative edges (not present in original adjacency matrix)
        neg_edges = []
        added_edges = set()
        while len(neg_edges) < n_edges_sample:
            i = np.random.randint(0, adj_matrix.shape[0])
            j = np.random.randint(0, adj_matrix.shape[0])
            if i == j or adj_matrix[i, j] > 0 or ((i, j) in added_edges):
                continue
            neg_edges.append([i, j])
            added_edges.add((i, j))
            added_edges.add((j, i))
        neg_edges = np.asarray(neg_edges)

        # Randomly sample positive edges from the upper triangle
        nz_upper = np.array(sp.triu(adj_matrix, k=1).nonzero()).T
        np.random.shuffle(nz_upper)
        pos_edges = nz_upper[:n_edges_sample]

        # Combine positive and negative edges as validation edges, and generate corresponding labels
        self.val_edges = np.concatenate((pos_edges, neg_edges), axis=0)
        self.edge_labels = np.array([1] * n_edges_sample + [0] * n_edges_sample)

        # Reset model parameters
        self.model.feat_model.reset_parameters()
        self.model.sfeat_model.reset_parameters()

    def learn_nc(self, debug=False):
        """
        Training pipeline for a single graph:
          - First pretrain: pretrain the edge predictor and node classifier for each branch;
          - Then enter main training, using early stopping to save the best model;
          - Finally evaluate the model on the test set.
        """
        self.init_result()
        self.start_time = time.time()

        # Compute normalization weight and positive weight for edge prediction loss balancing
        adj_t = self.adj_orig
        norm_w = adj_t.shape[0] ** 2 / float((adj_t.shape[0] ** 2 - adj_t.sum()) * 2)
        pos_weight = torch.FloatTensor(
            [float(adj_t.shape[0] ** 2 - adj_t.sum()) / adj_t.sum()]
        ).to(self.device)

        # Pretrain both main and auxiliary branches for edge prediction and node classification
        self.pretrain_ep_net(norm_w, pos_weight, self.conf.training['pretrain_ep'],
                             self.model.feat_model, self.feats, debug)
        self.pretrain_nc_net(self.conf.training['pretrain_nc'], self.model.feat_model, self.feats, debug)
        self.pretrain_ep_net(norm_w, pos_weight, self.conf.training['pretrain_ep'],
                             self.model.sfeat_model, self.sfeats, debug)
        self.pretrain_nc_net(self.conf.training['pretrain_nc'], self.model.sfeat_model, self.sfeats, debug,
                             is_sfeat=True)

        # Define optimizers for each branch
        optims1 = MultipleOptimizer(
            torch.optim.Adam(self.model.feat_model.ep_net.parameters(), lr=self.conf.training['lr'], foreach=False),
            torch.optim.Adam(self.model.feat_model.nc_net.parameters(), lr=self.conf.training['lr'],
                             weight_decay=self.conf.training['weight_decay'], foreach=False)
        )
        optims2 = MultipleOptimizer(
            torch.optim.Adam(self.model.sfeat_model.ep_net.parameters(), lr=self.conf.training['lr'], foreach=False),
            torch.optim.Adam(self.model.sfeat_model.nc_net.parameters(), lr=self.conf.training['lr'],
                             weight_decay=self.conf.training['weight_decay'], foreach=False)
        )

        # Learning rate warmup using sigmoid schedule
        if self.conf.training['warmup']:
            ep_lr_schedule = get_lr_schedule_by_sigmoid(
                self.conf.training['n_epochs'],
                self.conf.training['lr'],
                self.conf.training['warmup']
            )

        # Initialize best validation metric
        self.result['valid'] = self.result.get('feature_val', -1)

        # Load pretrained weights if enabled (fweights, sweights were saved in pretraining)
        if self.conf.training['use_pre_model']:
            self.model.feat_model.load_state_dict(self.fweights)
            self.model.sfeat_model.load_state_dict(self.sweights)
            self.weights = self.fweights
        else:
            self.weights = self.fweights

        patience_step = 0
        # Main training loop
        for epoch in range(self.conf.training['n_epochs']):
            t = time.time()
            improve = ''
            # Update learning rate if warmup is used
            if self.conf.training['warmup']:
                optims1.update_lr(0, ep_lr_schedule[epoch])
                optims2.update_lr(0, ep_lr_schedule[epoch])

            # Switch model to training mode
            self.model.train()
            optims1.zero_grad()
            optims2.zero_grad()

            # Compute total loss and outputs
            loss_train, out1, out2, adj_logits, adj_new = self.model.compute_loss(
                self.feats, self.sfeats, self.normalized_adj, self.adj_orig,
                self.labels, self.train_mask, pos_weight, norm_w
            )
            # Compute training accuracy (main branch only)
            acc_train = self.metric(
                self.labels[self.train_mask].cpu().numpy(),
                out1[self.train_mask].detach().cpu().numpy()
            )

            # Backpropagation and parameter updates
            loss_train.backward()
            optims1.step()
            optims2.step()

            # Validation: use main branch's node classifier
            self.model.eval()
            with torch.no_grad():
                output = self.model.feat_model.nc_net(self.feats, self.normalized_adj)
                loss_val = self.loss_fn(output[self.val_mask], self.labels[self.val_mask])
            acc_val = self.metric(
                self.labels[self.val_mask].cpu().numpy(),
                output[self.val_mask].detach().cpu().numpy()
            )
            # Compute edge prediction metrics (AUC, AP)
            adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
            ep_auc, ep_ap = eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)

            # Early stopping: update best model if validation improves
            if acc_val > self.result['valid']:
                self.total_time = time.time() - self.start_time
                improve = '*'
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.feat_model.state_dict())
                self.adjs = {'final': adj_new.clone().detach()}
                patience_step = 0
            else:
                patience_step += 1
                self.adjs = {'final': adj_new.clone().detach()}
                if patience_step == self.conf.training['patience']:
                    print('Early stop!')
                    break

            if debug:
                print("Epoch {:05d} | Time(s) {:.4f}".format(epoch + 1, time.time() - t))
                print("    Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}"
                      .format(loss_train.item(), acc_train, loss_val, acc_val, improve))

        print('Optimization Finished!')
        print('Time(s): {:.4f}'.format(self.total_time))
        # Evaluate best model on the test set
        self.model.feat_model.load_state_dict(self.weights)
        with torch.no_grad():
            output = self.model.feat_model.nc_net(self.feats, self.normalized_adj)
            loss_test = self.loss_fn(output[self.test_mask], self.labels[self.test_mask])
        acc_test = self.metric(
            self.labels[self.test_mask].cpu().numpy(),
            output[self.test_mask].detach().cpu().numpy()
        )
        self.result['test'] = acc_test
        print("Loss(test) {:.4f} | Acc(test) {:.4f} "
              .format(loss_test.item(), acc_test,))
        return self.result, self.adjs, output

    # -------------------------------------------------------------
    # The following pretraining functions are consistent with the original implementation
    # and are designed for separately pretraining different branches
    # -------------------------------------------------------------
    def pretrain_ep_net(self, norm_w, pos_weight, n_epochs, model_part, feats, debug=False):
        """
        Pretrain the Edge Prediction Network (EP Net)
        Args:
          - norm_w: Normalization weight
          - pos_weight: Positive sample weight
          - n_epochs: Number of pretraining epochs
          - model_part: The model part to pretrain (main branch or auxiliary branch)
          - feats: Input features
        """
        optimizer = torch.optim.Adam(model_part.ep_net.parameters(), lr=self.conf.training['lr'])
        model_part.train()
        for epoch in range(n_epochs):
            t = time.time()
            optimizer.zero_grad()
            # Forward pass: compute edge prediction scores
            adj_logits = model_part.ep_net(feats, self.normalized_adj)
            loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, self.adj_orig, pos_weight=pos_weight)
            # If not a GAE model, add KL divergence term
            if not self.conf.gsl['gae']:
                mu = model_part.ep_net.mean
                lgstd = model_part.ep_net.logstd
                kl_divergence = 0.5 / adj_logits.size(0) * (1 + 2 * lgstd - mu ** 2 - torch.exp(2 * lgstd)).sum(1).mean()
                loss -= kl_divergence
            loss.backward()
            optimizer.step()
            if debug:
                adj_pred = torch.sigmoid(adj_logits.detach()).cpu()
                ep_auc, ep_ap = eval_edge_pred(adj_pred, self.val_edges, self.edge_labels)
                print('EPNet pretrain, Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | auc {:.4f} | ap {:.4f}'
                      .format(epoch + 1, time.time() - t, loss.item(), ep_auc, ep_ap))

    def pretrain_nc_net(self, n_epochs, model_part, feats, debug=False, is_sfeat=False):
        """
        Pretrain the Node Classification Network (NC Net)
        Args:
          - n_epochs: Number of pretraining epochs
          - model_part: The model part to pretrain (main branch or auxiliary branch)
          - feats: Input features
          - is_sfeat: Whether the branch is the auxiliary feature branch
        """
        optimizer = torch.optim.Adam(
            model_part.nc_net.parameters(),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay']
        )
        for epoch in range(n_epochs):
            t = time.time()
            improve = ''
            model_part.train()
            optimizer.zero_grad()
            # Forward pass: compute node classification output
            output = model_part.nc_net(feats, self.normalized_adj)
            loss_train = self.loss_fn(output[self.train_mask], self.labels[self.train_mask])
            acc_train = self.metric(
                self.labels[self.train_mask].cpu().numpy(),
                output[self.train_mask].detach().cpu().numpy()
            )
            loss_train.backward()
            optimizer.step()
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                output = model_part.nc_net(feats, self.normalized_adj)
                loss_val = self.loss_fn(output[self.val_mask], self.labels[self.val_mask])
            acc_val = self.metric(
                self.labels[self.val_mask].cpu().numpy(),
                output[self.val_mask].detach().cpu().numpy()
            )
            acc_test = self.metric(
                self.labels[self.test_mask].cpu().numpy(),
                output[self.test_mask].detach().cpu().numpy()
            )
            # Save the best model based on validation accuracy
            if is_sfeat:
                if acc_val > self.result['sfeature_val']:
                    self.total_time = time.time() - self.start_time
                    self.best_val_loss = loss_val
                    self.result['sfeature_val'] = acc_val
                    self.result['sfeature_train'] = acc_train
                    self.result['sfeature_test'] = acc_test
                    improve = '*'
                    self.sweights = deepcopy(model_part.state_dict())
                    self.sbest_graph = self.adj.to_dense()
            else:
                if acc_val > self.result['feature_val']:
                    self.total_time = time.time() - self.start_time
                    self.best_val_loss = loss_val
                    self.result['feature_val'] = acc_val
                    self.result['feature_train'] = acc_train
                    self.result['feature_test'] = acc_test
                    improve = '*'
                    self.fweights = deepcopy(model_part.state_dict())
                    self.fbest_graph = self.adj.to_dense()
            if debug:
                print(
                    "NCNet pretrain, Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}"
                    .format(epoch + 1, time.time() - t, loss_train.item(), acc_train, loss_val, acc_val, improve)
                )


def main():
    data_name = ['cora', 'citeseer', 'blogcatalog', 'flickr', 'wisconsin', 'texas', 'cornell']
    for i in range(len(data_name)):
        conf = load_conf(path=None, method='jnsgsl', dataset=data_name[i])
        dataset = Dataset(data_name[i], sfeat=True, feat_norm=conf.dataset['feat_norm'])
        res = []
        for _ in range(1):
            solver = JNSGSLSolver(conf, dataset)
            acc, new_structure, _ = solver.run_exp(split=0, debug=False)
            res.append(acc['test'])
        mean_acc = np.mean(res)
        std_acc = np.std(res)
        print("dataset is:", data_name[i],
              f'Average Accuracy: {mean_acc},'
              f'Standard Deviation of Accuracy: {std_acc}')


if __name__ == "__main__":
    main()
