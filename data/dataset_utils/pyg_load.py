'''
load data via pyg
'''

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, WikipediaNetwork, WebKB, Actor, \
    AttributedGraphDataset, TUDataset, CitationFull, HeterophilousGraphDataset, Airports, PolBlogs, LINKXDataset, \
    Twitch
from torch_geometric.data.separate import separate
import os
import copy


class TUDatasetPlus(TUDataset):
    def __init__(self, root, name, use_node_attr=True, use_edge_attr=False):
        super().__init__(root=root, name=name, use_node_attr=use_node_attr, use_edge_attr=use_edge_attr)

    def get(self, idx: int):
        if self.len() == 1:
            return copy.copy(self._data)
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        data['idx'] = idx
        self._data_list[idx] = copy.copy(data)
        return data

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def pyg_load_dataset(name, path='./data/'):
    dic = {'cora': 'Cora',
           'citeseer': 'CiteSeer',
           'pubmed': 'PubMed',
           'amazoncom': 'Computers',
           'amazonpho': 'Photo',
           'coauthorcs': 'CS',
           'coauthorph': 'Physics',
           'airport': 'USA',
           'polblogs': 'PolBlogs',
           'brazil': 'Brazil',
           'europe': 'Europe',
           'wiki': 'Wiki',
           'ppi': 'PPI',
           'facebook': 'Facebook',
           'tweibo': 'TWeibo',
           'mag': 'MAG'}
    if name in dic.keys():
        name = dic[name]
    else:
        name = name

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=path, name=name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root=path, name=name)
    elif name in ['PolBlogs']:
        dataset = PolBlogs(root=path)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=path, name=name)
    elif name in ['wikics']:
        dataset = WikiCS(root=path)
    elif name in ['chameleon', 'squirrel', 'crocodile']:
        if name == 'crocodile':
            dataset = WikipediaNetwork(root=path, name=name, geom_gcn_preprocess=False)
        else:
            dataset = WikipediaNetwork(root=path, name=name)
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=path, name=name)
    elif name == 'actor':
        dataset = Actor(root=os.path.join(path, name))
    elif name in ['blogcatalog', 'flickr', 'Wiki', 'PPI', 'Facebook', 'TWeibo', 'MAG']:
        dataset = AttributedGraphDataset(root=path, name=name)
    elif name in ['cora_full', 'cora_ml', 'citeseer_full', 'dblp', 'pubmed_full']:
        if name in ['cora_full', 'citeseer_full', 'pubmed_full']:
            name = name.split('_')[0]
        dataset = CitationFull(root=os.path.join(path, 'CitationFull'), name=name)
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root=path, name=name)
    elif name in ["IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "REDDIT-MULTI-5K", "COLLAB", "DBLP_v1", "DD",
                  "ENZYMES", "PROTEINS", "MUTAG", "NCI1", "NCI109", "Mutagenicity", "FRANKENSTEIN"]:
        dataset = TUDatasetPlus(root=path, name=name)
    elif name in ['USA', 'Brazil', 'Europe']:
        dataset = Airports(root=path, name=name)
    elif name in ['reed98', 'penn94', 'amherst41', 'cornell5']:
        dataset = LINKXDataset(root=path, name=name)
    elif name in ["DE", "EN", "ES", "FR", "PT", "RU"]:
        dataset = Twitch(root=path, name=name)
    else:
        raise NotImplementedError
    return dataset

#
# if __name__ == '__main__':
#     dataset = pyg_load_dataset('europe')
#     print(dataset)
