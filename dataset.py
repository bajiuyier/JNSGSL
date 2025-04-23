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
    def __init__(self, data, sfeat=False, feat_norm=False, verbose=True, n_splits=1, split='public', split_params=None, homophily_control=None,
                 path='./data/', cv=None, **kwargs):
        path ='D:/code/python/dp/GNN/OpenGSL-main/opengsl/data/dataset/data'
        self.name = data
        self.feat_norm = feat_norm
        self.verbose = verbose
        self.path = path
        self.device = torch.device('cuda')
        self.single_graph = True
        self.split_params = split_params
        self.n_splits = n_splits
        self.split = split
        self.sfeat = sfeat
        if self.sfeat == True:
            path_sfeat = 'D:/code/python/dp/base_model/model/JNSGSL/deepwalk_embedding/structure_embedding/' + self.name
            sfeat = torch.load(path_sfeat)
            self.sfeats = torch.tensor(sfeat, dtype=torch.float32).to(self.device)
        assert self.split in ['public', 'random']
        self.cv = cv
        self.total_splits = n_splits * cv if cv else n_splits
        self.prepare_data(data, feat_norm, verbose)
        if self.single_graph:
            self.split_data(split, n_splits, cv, split_params, verbose)
        else:
            self.split_graphs(split, n_splits, cv, split_params, verbose)
        if homophily_control:
            self.adj = control_homophily(self.adj, self.labels.cpu().numpy(), homophily_control)
        self.homophily = get_homophily(self.labels.cpu(), self.adj.cpu().to_dense(), 'edge')

    def prepare_data(self, ds_name, feat_norm=False, verbose=True):
        '''
        Function to Load various datasets.
        Homophilous datasets are loaded via pyg, while heterophilous datasets are loaded with `hetero_load`.
        The results are saved as `self.feats, self.adj, self.labels, self.train_masks, self.val_masks, self.test_masks`.
        Note that `self.adj` is undirected and has no self loops.

        Parameters
        ----------
        ds_name : str
            The name of dataset.
        feat_norm : bool
            Whether to normalize the features.
        verbose : bool
            Whether to print statistics.
        '''
        if ds_name in ['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho', 'coauthorcs', 'coauthorph', 'blogcatalog',
                       'flickr', 'wikics', 'amazon-ratings', 'questions', 'chameleon-filtered', 'squirrel-filtered',
                       'minesweeper', 'roman-empire', 'wiki-cooc', 'tolokers', 'cora_full', 'cora_ml', 'citeseer_full',
                       'dblp', 'pubmed_full', 'airport', 'actor', 'texas', 'cornell', 'wisconsin', 'Polblogs', 'brazil',
                       'europe', 'chameleon', 'squirrel', 'facebook'] or 'csbm' in ds_name:
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            self.g = self.data_raw[0]
            self.feats = self.g.x  # unnormalized
            if ds_name == 'flickr':
                self.feats = self.feats.to_dense().float()
            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]
            self.labels = self.g.y
            self.adj = torch.sparse.FloatTensor(self.g.edge_index, torch.ones(self.g.edge_index.shape[1]),
                                                [self.n_nodes, self.n_nodes])
            self.n_edges = self.g.num_edges/2
            self.n_classes = self.data_raw.num_classes
            self.feats = self.feats.to(self.device)
            self.labels = self.labels.to(self.device)
            if self.sfeat:
                path_sfeat = 'D:/code/python/dp/base_model/model/JNSGSL/deepwalk_embedding/structure_embedding/' + ds_name
                sfeat = torch.load(path_sfeat)
                self.sfeats = torch.tensor(sfeat, dtype=torch.float32).clone().detach().to(self.device)
            if 'csbm' in ds_name:
                self.labels = self.labels.float()
            self.adj = self.adj.to(self.device)
            # normalize features
            if feat_norm:
                self.feats = normalize(self.feats, style='row')
        else:
            # graph level
            self.single_graph = False
            self.data_raw = pyg_load_dataset(ds_name, path=self.path)
            if self.data_raw.data.x is None:
                # If node features are not available, use degree as feature
                # Find maximum degree among all graphs
                max_degree = 0
                degs = []
                for data in self.data_raw:
                    degs += [degree(data.edge_index[0], dtype=torch.long)]
                    max_degree = max(max_degree, degs[-1].max().item())

                # Two approaches to handle degree features
                if max_degree < 1000:
                    self.data_raw.transform = T.OneHotDegree(max_degree)
                else:
                    deg = torch.cat(degs, dim=0).to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    self.data_raw.transform = NormalizedDegree(mean, std)
            self.n_graphs = len(self.data_raw)
            self.n_classes = self.data_raw.num_classes
            self.dim_feats = self.data_raw[0].x.shape[1]

        if verbose:
            if self.single_graph:
                print("""----Data statistics------'
                    #dataset_name %s
                    #Nodes %d
                    #Edges %d
                    #Classes %d""" %
                      (self.name, self.n_nodes, self.n_edges, self.n_classes))
            else:
                print("""----Data statistics------'
                                    #Graphs %d
                                    #Classes %d""" %
                      (self.n_graphs, self.n_classes))
        self.num_targets = self.n_classes

    def split_data(self, split, n_splits=1, cv=None, split_params=None, verbose=True):
        '''
        Function to conduct data splitting for various datasets.

        Parameters
        ----------
        n_splits : int
            Number of data splits.
        verbose : bool
            Whether to print statistics.
        '''
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        if split == 'public':
            assert self.name in ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', 'roman-empire', 'amazon-ratings',
                                 'minesweeper', 'tolokers', 'questions', 'wikics', 'airport', 'actor', 'texas', 'cornell', 'wisconsin',
                                 'brazil', 'europe', 'chameleon', 'squirrel', 'facehook'], 'This dataset has no public splits.'
            if self.name in ['cora', 'citeseer', 'pubmed']:
                for i in range(n_splits):
                    self.train_masks.append(torch.nonzero(self.g.train_mask, as_tuple=False).squeeze().numpy())
                    self.val_masks.append(torch.nonzero(self.g.val_mask, as_tuple=False).squeeze().numpy())
                    self.test_masks.append(torch.nonzero(self.g.test_mask, as_tuple=False).squeeze().numpy())
            elif self.name in ['blogcatalog', 'flickr', 'airport']:
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
                            '''Download failed! Make sure you have stable Internet connection and enter the right name''')

                split_file = self.name + '_tvt_nids.pkl'
                if not os.path.exists(os.path.join(self.path, self.name, split_file)):
                    download(split_file)
                train_indices, val_indices, test_indices = load_obj(os.path.join(self.path, self.name, split_file))
                for i in range(n_splits):
                    self.train_masks.append(train_indices)
                    self.val_masks.append(val_indices)
                    self.test_masks.append(test_indices)
            elif self.name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 'wikics', 'actor', 'texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel']:
                assert n_splits < 10, 'n_splits > public splits'
                self.train_masks = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.train_mask.T]
                self.val_masks = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.val_mask.T]
                self.test_masks = [torch.nonzero(x, as_tuple=False).squeeze().numpy() for x in self.g.test_mask.T]

        elif split == 'random':
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
            print("""----Split statistics of %d splits------'
                #Train samples %d
                #Val samples %d
                #Test samples %d""" %
                  (self.total_splits, len(self.train_masks[0]), len(self.val_masks[0]), len(self.test_masks[0])))

    def split_graphs(self, split, n_splits, cv, split_params, verbose=True):
        '''
        Function to conduct data splitting for graph-level datasets.

        Parameters
        ----------
        n_splits : int
            Number of data splits.
        verbose : bool
            Whether to print statistics.
        '''
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []
        if split == 'public':
            raise NotImplementedError
        elif split == 'random':
            for seed in range(n_splits):
                np.random.seed(seed)
                if cv:
                    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(self.data_raw.y.cpu().numpy(), cv))):
                        self.train_masks.append(train_idx)
                        self.val_masks.append(val_idx)
                        self.test_masks.append(test_idx)
                else:
                    assert split_params is not None
                    if self.name == 'texas':
                        train_indices, val_indices, test_indices = get_split(self.data_raw.y.cpu().numpy(), split_params, name='texas')
                    else:
                        train_indices, val_indices, test_indices = get_split(self.data_raw.y.cpu().numpy(), split_params)
                    self.train_masks.append(train_indices)
                    self.val_masks.append(val_indices)
                    self.test_masks.append(test_indices)

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
