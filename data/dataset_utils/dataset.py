import torch
from data.dataset_utils.pyg_load import pyg_load_dataset
from data.dataset_utils.split import get_split, k_fold
from utils import normalize
import numpy as np
from data.dataset_utils.control_homophily import control_homophily
import pickle
import os
import urllib.request
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from data.dataset_utils.control_homophily import get_homophily


class Dataset:
    """
    Dataset class:
      - Used for loading and preprocessing graph data.
      - Supports loading data for single-graph tasks (e.g., node classification) and multi-graph tasks (e.g., graph classification).
      - Includes data splitting, feature normalization, auxiliary feature loading, and homophily control.
    """

    def __init__(self, data, sfeat=False, feat_norm=False, verbose=True, n_splits=1, split='public',
                 split_params=None, homophily_control=None, path='./data/', cv=None, **kwargs):
        # Fixed dataset path (hardcoded local path)
        # path = os.path.dirname(os.getcwd()) + '/data/dataset/'
        path = os.getcwd() + '/data/dataset/'
        self.name = data  # Dataset name
        self.feat_norm = feat_norm  # Whether to normalize features
        self.verbose = verbose  # Whether to print detailed info
        self.path = path  # Dataset path
        self.device = torch.device('cuda')  # Use GPU by default
        self.single_graph = True  # Load single graph by default (node classification)
        self.split_params = split_params  # Data splitting parameters
        self.n_splits = n_splits  # Number of data splits
        self.split = split  # Split method: 'public' or 'random'
        self.sfeat = sfeat  # Whether to load auxiliary features (e.g., structural embeddings)
        # Load auxiliary features if specified
        if self.sfeat == True:
            path_sfeat = path + '/deepwalk_embedding/' + self.name
            sfeat = torch.load(path_sfeat)
            self.sfeats = torch.tensor(sfeat, dtype=torch.float32).to(self.device)
        # Ensure split type is either 'public' or 'random'
        assert self.split in ['public', 'random']
        self.cv = cv  # Cross-validation folds (optional)
        self.total_splits = n_splits * cv if cv else n_splits

        # Load and preprocess the data
        self.prepare_data(data, feat_norm, verbose)
        # Split data for single-graph tasks (train/val/test)
        if self.single_graph:
            self.split_data(split, n_splits, cv, split_params, verbose)
        else:
            self.split_graphs(split, n_splits, cv, split_params, verbose)
        # Apply homophily control if specified
        if homophily_control:
            self.adj = control_homophily(self.adj, self.labels.cpu().numpy(), homophily_control)
        # Compute and store edge homophily
        self.homophily = get_homophily(self.labels.cpu(), self.adj.cpu().to_dense(), 'edge')

    def prepare_data(self, ds_name, feat_norm=False, verbose=True):
        """
        Load and preprocess dataset:
          - For homophilous datasets, load with PyG.
          - For featureless datasets (heterophilous), generate degree-based features.
          - Save features, labels, adjacency matrix, and metadata as object attributes.

        Parameters:
          - ds_name: Dataset name
          - feat_norm: Whether to row-normalize features
          - verbose: Whether to print statistics
        """
        # If the dataset is a common node classification benchmark
        if ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph',
                       'blogcatalog', 'flickr', 'wikics', 'amazon-ratings', 'questions', 'chameleon-filtered',
                       'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'tolokers', 'cora_full',
                       'cora_ml', 'citeseer_full', 'dblp', 'pubmed_full', 'airport', 'actor', 'texas', 'cornell',
                       'wisconsin', 'Polblogs', 'brazil', 'europe', 'chameleon', 'squirrel',
                       'facebook'] or 'csbm' in ds_name:
            # Load dataset using PyG
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            self.g = self.data_raw[0]  # Retrieve single graph object
            self.feats = self.g.x  # Raw node features
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense().float()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            # Construct sparse adjacency matrix (values all 1)
            self.adj = torch.sparse.FloatTensor(self.g.edge_index,
                                                torch.ones(self.g.edge_index.shape[1]),
                                                [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges / 2
            self.n_classes = self.data_raw.num_classes
            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            # Load auxiliary structural features if needed
            if self.sfeat:
                path_sfeat = 'D:/code/python/dp/base_model/model/JNSGSL/deepwalk_embedding/structure_embedding/' + ds_name
                sfeat = torch.load(path_sfeat)
                self.sfeats = torch.tensor(sfeat, dtype=torch.float32).clone().detach().to(self.device)
            # For csbm datasets, convert labels to float
            if 'csbm' in ds_name:
                self.labels = self.labels.float()
            self.adj = self.adj.to(self.device)
            # Row-normalize features if required
            if feat_norm:
                self.feats = normalize(self.feats, style='row')
        else:
            # Graph-level datasets (graph classification)
            self.single_graph = False
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            if self.data_raw.data.x is None:
                # Use degree as feature if node features are missing
                max_degree = 0
                degs = []
                for data in self.data_raw:
                    degs += [degree(data.edge_index[0], dtype=torch.long)]
                    max_degree = max(max_degree, degs[-1].max().item())

                # Use one-hot encoding if degree is small; otherwise use normalized degree
                if max_degree < 1000:
                    self.data_raw.transform = T.OneHotDegree(max_degree)
                else:
                    deg = torch.cat(degs, dim=0).to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    self.data_raw.transform = NormalizedDegree(mean, std)
            self.n_graphs = len(self.data_raw)
            self.n_classes = self.data_raw.num_classes
            self.dim_feats = self.data_raw[0].x.shape[1]

        # Print data statistics
        if verbose:
            print("""----Data statistics------
                    #dataset_name %s
                    #Nodes %d
                    #Edges %d
                    #Classes %d""" %
                    (self.name, self.n_nodes, self.n_edges, self.n_classes))

        self.num_targets = self.n_classes

    def split_data(self, split, n_splits=1, cv=None, split_params=None, verbose=True):
        """
        Split single-graph dataset into train/val/test sets.

        Parameters:
          - split: Split method ('public' or 'random')
          - n_splits: Number of splits
          - cv: Number of folds for cross-validation (optional)
          - split_params: Parameters for random splits
          - verbose: Whether to print split statistics
        """
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        if split == 'public':
            # Check if dataset supports public splits
            assert self.name in ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', 'roman-empire',
                                 'amazon-ratings',
                                 'minesweeper', 'tolokers', 'questions', 'wikics', 'airport', 'actor', 'texas',
                                 'cornell', 'wisconsin',
                                 'brazil', 'europe', 'chameleon', 'squirrel',
                                 'facehook'], 'This dataset has no public splits.'
            if self.name in ['cora', 'citeseer', 'pubmed']:
                # Use predefined train/val/test masks
                for i in range(n_splits):
                    self.train_masks.append(torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy())
                    self.val_masks.append(torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy())
                    self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
            elif self.name in ['blogcatalog', 'flickr', 'airport']:
                # Load split indices from file
                def load_obj(file_name):
                    with open(file_name, 'rb') as f:
                        return pickle.load(f)

                def download(name):
                    url = 'https://github.com/zhao-tong/GAug/raw/master/data/graphs/'
                    try:
                        print('Downloading', url + name)
                        urllib.request.urlretrieve(url + name, os.path.join(self.path, self.name, name))
                        print('Done!')
                    except:
                        raise Exception(
                            'Download failed! Make sure you have stable Internet connection and enter the right name')

                split_file = self.name + '_tvt_nids.pkl'
                if not os.path.exists(os.path.join(self.path, self.name, split_file)):
                    download(split_file)
                train_indices, val_indices, test_indices = load_obj(os.path.join(self.path, self.name, split_file))
                for i in range(n_splits):
                    self.train_masks.append(train_indices)
                    self.val_masks.append(val_indices)
                    self.test_masks.append(test_indices)
            elif self.name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'wikics',
                               'actor', 'texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel']:
                # Use predefined public splits
                assert n_splits < 10, 'n_splits > public splits'
                self.train_masks = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.train_mask.T]
                self.val_masks = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.val_mask.T]
                self.test_masks = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.test_mask.T]

        elif split == 'random':
            # Perform random split
            if cv:
                for i in range(n_splits):
                    np.random.seed(i)
                    train_indices, val_indices, test_indices = k_fold(self.labels.cpu().numpy(), cv)
                    self.train_masks.extend(train_indices)
                    self.val_masks.extend(val_indices)
                    self.test_masks.extend(test_indices)
            else:
                assert split_params is not None, 'you need to specify the split params'
                for i in range(n_splits):
                    np.random.seed(i)
                    if self.name == 'texas':
                        train_indices, val_indices, test_indices = get_split(self.data_raw.y.cpu().numpy(),
                                                                             split_params, name='texas')
                    else:
                        train_indices, val_indices, test_indices = get_split(self.data_raw.y.cpu().numpy(),
                                                                             split_params)
                    self.train_masks.append(train_indices)
                    self.val_masks.append(val_indices)
                    self.test_masks.append(test_indices)
        else:
            raise NotImplementedError

        if verbose:
            print("""----Split statistics of %d splits------
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (self.total_splits, len(self.train_masks[0]), len(self.val_masks[0]), len(self.test_masks[0])))

    def split_graphs(self, split, n_splits, cv, split_params, verbose=True):
        """
        Split datasets for graph-level tasks.
        Parameters are similar to split_data, but apply to multiple graphs.
        """
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        if split == 'public':
            raise NotImplementedError
        elif split == 'random':
            for seed in range(n_splits):
                np.random.seed(seed)
                if cv:
                    for fold, (train_idx, test_idx, val_idx) in enumerate(
                            zip(*k_fold(self.data_raw.y.cpu().numpy(), cv))):
                        self.train_masks.append(train_idx)
                        self.val_masks.append(val_idx)
                        self.test_masks.append(test_idx)
                else:
                    assert split_params is not None
                    if self.name == 'texas':
                        train_indices, val_indices, test_indices = get_split(self.data_raw.y.cpu().numpy(),
                                                                             split_params, name='texas')
                    else:
                        train_indices, val_indices, test_indices = get_split(self.data_raw.y.cpu().numpy(),
                                                                             split_params)
                    self.train_masks.append(train_indices)
                    self.val_masks.append(val_indices)
                    self.test_masks.append(test_indices)


class NormalizedDegree(object):
    """
    NormalizedDegree:
      - Normalize the degree of each node.
      - Transform degree feature into (degree - mean) / std form and assign as node features.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # Compute node degree
        deg = degree(data.edge_index[0], dtype=torch.float)
        # Normalize: (degree - mean) / std
        deg = (deg - self.mean) / self.std
        # Set normalized degree as node feature
        data.x = deg.view(-1, 1)
        return data
