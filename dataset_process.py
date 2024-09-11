from abc import ABC
import os
import glob
import scipy
from scipy.io import loadmat
import numpy as np
import pandas as pd
from typing import List

import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx

import config

ABIDE_INFO_PATH = 'meta_data/Phenotypic_V1_0b_preprocessed.csv'
args = config.parse()

def get_adj_path(dataset_name: str, Atalas_name: str, person: str, roi_num: int):
    return f'{args.data_root}/{dataset_name}/{Atalas_name}_WM/adj/{person}_{Atalas_name}_weight_adj_{dataset_name}_roi_{roi_num}.mat'


def get_ts_path(dataset_name: str, Atalas_name: str, person: str, roi_num: int):
    return f'{args.data_root}/{dataset_name}/{Atalas_name}_WM/time_series/{person}_{Atalas_name}_Ts_{dataset_name}_{roi_num}.mat'


def get_H_path(dataset_name: str, Atalas_name: str, person: str, H_type: str, roi_num: int):
    return f'{args.data_root}/{dataset_name}/{Atalas_name}_WM/{H_type}/{Atalas_name}_{H_type}_{roi_num}_{person}.mat'


def threshold_adj_array(adj_array: np.ndarray, threshold: int, num_nodes: int, add_self_link=True) -> np.ndarray:
    num_to_filter: int = int((threshold / 100.0) * (num_nodes * (num_nodes - 1) / 2))

    # For threshold operations, zero out lower triangle (including diagonal)
    adj_array[np.tril_indices(num_nodes)] = 0

    # Following code is similar to bctpy
    indices = np.where(adj_array)
    sorted_indices = np.argsort(adj_array[indices])[::-1]
    adj_array[(indices[0][sorted_indices][num_to_filter:], indices[1][sorted_indices][num_to_filter:])] = 0

    # Just to get a symmetrical matrix
    adj_array = adj_array + adj_array.T

    if add_self_link:
        # Diagonals need connection of 1 for graph operations
        adj_array[np.diag_indices(num_nodes)] = 1.0

    return adj_array


def create_name_for_brain_dataset(dataset_name: str, num_nodes: int, time_length: int, threshold: int,
                                  Atlas_name: str) -> str:
    prefix_location = './datasets_mem/'
    name_combination = '_'.join(
        [dataset_name, Atlas_name, 'num_nodes', str(num_nodes), 'time_length', str(time_length), str(threshold)])
    return prefix_location + name_combination


class BrainDataset(InMemoryDataset, ABC):
    def __init__(self, root, num_nodes: int, threshold: int, time_length: int, Atlas_name: str, transform=None,
                 pre_transform=None, ):
        if threshold < 0 or threshold > 100:
            print("NOT A VALID threshold!")
            exit(-2)

        self.num_nodes: int = num_nodes
        self.time_length: int = time_length
        self.threshold: int = threshold
        self.Atlas_name: str = Atlas_name

        super(BrainDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        pass


class ABIDEDataset(BrainDataset):
    def __init__(self, root, num_nodes: int, threshold: int, Atlas_name: str, time_length: int = 150, transform=None,
                 edge_weights=True,
                 pre_transform=None,
                 ):
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.Atlas_name = Atlas_name
        self.time_length = time_length
        self.info_df = pd.read_csv(ABIDE_INFO_PATH).set_index('SUB_ID')
        self.include_edge_weights = edge_weights
        super(ABIDEDataset, self).__init__(root=root, num_nodes=num_nodes,
                                           threshold=threshold, time_length=time_length, Atlas_name=Atlas_name,
                                           transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data_abide_brain.dataset']

    def __create_data_object(self, person: str,
                             ts: np.ndarray,
                             ind: int,
                             adj: np.ndarray,
                             edge_index: torch.Tensor,
                             edge_weight: torch.Tensor,
                             site: str,
                             H_k: np.ndarray,
                             H_lasso: np.ndarray,
                             H_elastic: np.ndarray,
                             ):
        time_series = ts.T

        x = torch.tensor(time_series, dtype=torch.float)

        num_person = int(person)
        y = torch.tensor([self.info_df.loc[num_person, 'DX_GROUP'] - 1], dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)
        H_k = torch.tensor(H_k, dtype=torch.float)
        H_lasso = torch.tensor(H_lasso, dtype=torch.float)
        H_elastic = torch.tensor(H_elastic, dtype=torch.float)

        data = Data(x=x, y=y, adj=adj, edge_index=edge_index, edge_weight=edge_weight,
                    H_lasso=H_lasso, H_elastic=H_elastic, H_k=H_k
                    )

        data.id = torch.tensor([num_person])
        data.index = torch.tensor([ind])
        data.site = site

        return data

    def process(self):
        data_list: List[Data] = []
        dataset_name = 'ABIDE'

        with open('.\datasets\ABIDE_ID.txt', 'r') as file:
            smaple = file.read()
        smaple_list = smaple.split('\n')

        ind = 0
        for person in smaple_list:
            ID = person.split('_')[-1]
            site = person.split('_')[0]
            ts = loadmat(get_ts_path(dataset_name, Atalas_name=self.Atlas_name, person=person, roi_num=self.num_nodes))[
                'time_series']

            if ts.shape[0] < self.time_length:
                continue
            elif ts.shape[0] > self.time_length:
                ts = ts[:self.time_length, :]
            adj = \
            loadmat(get_adj_path(dataset_name, Atalas_name=self.Atlas_name, person=person, roi_num=self.num_nodes))[
                'adj']

            num_nodes = ts.shape[1]
            adj_thre = threshold_adj_array(adj, self.threshold, num_nodes)
            G = nx.from_numpy_array(adj_thre, create_using=nx.DiGraph)
            edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()),
                                       dtype=torch.float).unsqueeze(1)

            H_k = loadmat(get_H_path(dataset_name, self.Atlas_name, person, 'H_k', self.num_nodes))['H']
            H_lasso = loadmat(get_H_path(dataset_name, self.Atlas_name, person, 'H_lasso', self.num_nodes))['H']
            H_elastic = loadmat(get_H_path(dataset_name, self.Atlas_name, person, 'H_elastic', self.num_nodes))['H']

            data = self.__create_data_object(person=ID, ts=ts, ind=ind, adj=adj,
                                             edge_index=edge_index, edge_weight=edge_weight,
                                             H_lasso=H_lasso, H_elastic=H_elastic, H_k=H_k,
                                             site=site
                                             )

            data_list.append(data)
            ind += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SZDataset(BrainDataset):
    def __init__(self, root, num_nodes: int, threshold: int, Atlas_name: str, time_length: int = 195, transform=None,
                 edge_weights=True,
                 pre_transform=None
                 ):
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.Atlas_name = Atlas_name
        self.time_length = time_length
        self.include_edge_weights = edge_weights
        super(SZDataset, self).__init__(root=root, num_nodes=num_nodes,
                                        threshold=threshold, time_length=time_length, Atlas_name=Atlas_name,
                                        transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data_sz_brain.dataset']

    def __create_data_object(self, person: str,
                             ts: np.ndarray,
                             ind: int,
                             adj: np.ndarray,
                             edge_index: torch.Tensor,
                             edge_weight: torch.Tensor,
                             H_k: np.ndarray,
                             H_lasso: np.ndarray,
                             H_elastic: np.ndarray,
                             ):
        time_series = ts.T

        x = torch.tensor(time_series, dtype=torch.float)

        y = torch.tensor([0], dtype=torch.float) if "SCZ" in person else torch.tensor([1], dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)
        H_k = torch.tensor(H_k, dtype=torch.float)
        H_lasso = torch.tensor(H_lasso, dtype=torch.float)
        H_elastic = torch.tensor(H_elastic, dtype=torch.float)

        data = Data(x=x, y=y, adj=adj, edge_index=edge_index, edge_weight=edge_weight,
                    H_lasso=H_lasso, H_elastic=H_elastic, H_k=H_k
                    )
        data.id = torch.tensor([ind])
        data.index = torch.tensor([ind])
        data.group = person

        return data

    def process(self):
        data_list: List[Data] = []
        dataset_name = 'SZ'

        with open(f'.\datasets\SZ_ID.txt', 'r') as file:
            file_content = file.read()
        # 将内容按换行符进行分割成列表
        content_list = file_content.split('\n')
        ind = 0
        for person in content_list:
            ID = person
            ts = loadmat(get_ts_path(dataset_name, Atalas_name=self.Atlas_name, person=person, roi_num=self.num_nodes))[
                'time_series']

            if ts.shape[0] > self.time_length:
                ts = ts[:self.time_length, :]
            adj = \
            loadmat(get_adj_path(dataset_name, Atalas_name=self.Atlas_name, person=person, roi_num=self.num_nodes))[
                'adj']

            num_nodes = ts.shape[1]
            adj_thre = threshold_adj_array(adj, self.threshold, num_nodes)
            G = nx.from_numpy_array(adj_thre, create_using=nx.DiGraph)
            edge_index = torch.tensor(np.array(G.edges()), dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()),
                                       dtype=torch.float).unsqueeze(1)

            H_k = loadmat(get_H_path(dataset_name, self.Atlas_name, person, 'H_k', self.num_nodes))['H']
            H_lasso = loadmat(get_H_path(dataset_name, self.Atlas_name, person, 'H_lasso', self.num_nodes))['H']
            H_elastic = loadmat(get_H_path(dataset_name, self.Atlas_name, person, 'H_elastic', self.num_nodes))['H']

            data = self.__create_data_object(person=ID, ts=ts, ind=ind, adj=adj,
                                             edge_index=edge_index, edge_weight=edge_weight,
                                             H_lasso=H_lasso, H_elastic=H_elastic, H_k=H_k,
                                             )

            data_list.append(data)
            ind += 1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
