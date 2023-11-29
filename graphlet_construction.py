
import numpy as np
import math
from torch_geometric.data import Data

import pandas as pd
from config_directory import config_directory

import torch
from typing import List, Union
from torch import Tensor

# K-hop Subgraph Extraction
def k_hop_subgraph(
    node_idx: int,
    num_hops: int,
    max_nodes: int,
    edge_index: Tensor,
    edge_type: Union[int, List[int], Tensor],
):
    num_nodes = edge_index.max().item() + 1
    head, tail = edge_index

    node_mask = head.new_empty(num_nodes, dtype=torch.bool)
    reduce_mask_ = head.new_empty(edge_index.size(1), dtype=torch.bool)
    reduce_mask_.fill_(False)
    subset = int(node_idx)
    limit = [int(math.ceil(max_nodes * 10 ** (-i))) for i in range(num_hops)]

    for i in range(num_hops):
        node_mask.fill_(False)
        node_mask[subset] = True
        idx_rm = node_mask[head].nonzero().view(-1)
        idx_rm = idx_rm[torch.randperm(idx_rm.size(0))[: limit[i]]]
        reduce_mask_[idx_rm] = True
        subset = tail[reduce_mask_].unique()

    edge_index = edge_index[:, reduce_mask_]
    edge_type = edge_type[reduce_mask_]

    subset = edge_index.unique()

    mapping = torch.reshape(torch.tensor((node_idx, 0)), (2, 1))
    mapping_temp = torch.vstack(
        (subset[subset != node_idx], torch.arange(1, subset.size()[0]))
    )
    mapping = torch.hstack((mapping, mapping_temp))

    head_ = edge_index[0, :]
    tail_ = edge_index[1, :]

    sort_idx = torch.argsort(mapping[0, :])
    idx_h = torch.searchsorted(mapping[0, :], head_, sorter=sort_idx)
    idx_t = torch.searchsorted(mapping[0, :], tail_, sorter=sort_idx)

    out_h = mapping[1, :][sort_idx][idx_h]
    out_t = mapping[1, :][sort_idx][idx_t]

    edge_index_new = torch.vstack((out_h, out_t))

    edge_index_new = torch.hstack(
        (
            edge_index_new,
            torch.tensor(
                (
                    [0, edge_index_new.max().item() + 1],
                    [edge_index_new.max().item() + 1, 0],
                )
            ),
        )
    )
    edge_type = torch.hstack((edge_type, torch.zeros(2, dtype=torch.long)))

    mapping = torch.hstack(
        (mapping, torch.tensor([[node_idx], [edge_index_new.max().item()]]))
    )

    return edge_index_new, edge_type, mapping

#Remove duplicate function
def remove_duplicates(
    edge_index: Tensor,
    edge_type: Tensor = None,
    edge_attr: Tensor = None,
    perturb_tensor: Tensor = None,
):
    nnz = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])

    if edge_type is not None:
        idx[1:].add_((edge_type + 1) * (10 ** (len(str(num_nodes)) + 3)))

    idx[1:], perm = torch.sort(
        idx[1:],
    )

    mask = idx[1:] > idx[:-1]

    edge_index = edge_index[:, perm]
    edge_index = edge_index[:, mask]

    if edge_type is not None:
        edge_type, edge_attr = edge_type[perm], edge_attr[perm, :]
        edge_type, edge_attr = edge_type[mask], edge_attr[mask, :]
        if perturb_tensor is not None:
            perturb_tensor = perturb_tensor[perm]
            perturb_tensor = perturb_tensor[mask]
            return edge_index, edge_type, edge_attr, perturb_tensor
        else:
            return edge_index, edge_type, edge_attr
    else:
        return edge_index
# To undirected function
def to_undirected(
    edge_index: Tensor,
    edge_type: Tensor = None,
    edge_attr: Tensor = None,
    idx_perturb=None,
):
    row = torch.cat([edge_index[0, :], edge_index[1, :]])
    col = torch.cat([edge_index[1, :], edge_index[0, :]])

    edge_index = torch.stack([row, col], dim=0)

    if edge_type is not None:
        edge_type = torch.cat([edge_type, edge_type])
        edge_attr = torch.vstack((edge_attr, edge_attr))
        if idx_perturb is not None:
            perturb_tensor = torch.zeros(edge_type.size(0))
            perturb_tensor[idx_perturb] = -1
            perturb_tensor = torch.cat([perturb_tensor, perturb_tensor])
            edge_index, edge_type, edge_attr, perturb_tensor = remove_duplicates(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
                perturb_tensor=perturb_tensor,
            )
            idx_perturb = (perturb_tensor < 0).nonzero().squeeze()
            return edge_index, edge_type, edge_attr, idx_perturb
        else:
            edge_index, edge_type, edge_attr = remove_duplicates(
                edge_index=edge_index,
                edge_type=edge_type,
                edge_attr=edge_attr,
            )
        idx_perturb = []
        return edge_index, edge_type, edge_attr, idx_perturb
    else:
        edge_index = remove_duplicates(edge_index=edge_index)
        return edge_index

# Graphlet class to construct a graphlet of a given entity
class Graphlet:
    def __init__(self, kg_data, num_hops: int = 1, max_nodes: int = 100):
        super(Graphlet, self).__init__()

        self.main_data = kg_data
        self.edge_index = kg_data["edge_index"]
        self.edge_type = kg_data["edge_type"]
        self.ent2idx = kg_data["ent2idx"]
        self.rel2idx = kg_data["rel2idx"]

        # KEN embeddings
        self.ken_emb = pd.read_parquet(config_directory["ken_embed_dir"])
        self.ken_ent = self.ken_emb["Entity"]
        self.ken_embed_ent2idx = {self.ken_ent[i]: i for i in range(len(self.ken_emb))}

        self.num_hops = num_hops
        self.max_nodes = max_nodes

    def extract_graph_data(
        self,
        cen_idx: Union[int, List[int], Tensor],
    ):

        if isinstance(cen_idx, Tensor):
            cen_idx = cen_idx.tolist()
        if isinstance(cen_idx, int):
            cen_idx = [cen_idx]

        # Obtain the of entities and edge_types in the batch (reduced set)
        head_ = self.edge_index[0, :]
        tail_ = self.edge_index[1, :]

        node_mask = head_.new_empty(self.edge_index.max().item() + 1, dtype=torch.bool)
        node_mask.fill_(False)

        subset = cen_idx

        for _ in range(self.num_hops):
            node_mask[subset] = True
            reduce_mask = node_mask[head_]
            subset = tail_[reduce_mask].unique()

        self.edge_index_reduced = self.edge_index[:, reduce_mask]
        self.edge_type_reduced = self.edge_type[reduce_mask]

        # Obtain the list of data with original and perturbed graphs
        data_total = []
        for g_idx in range(len(cen_idx)):
            # Obtain the original graph
            data_original_ = self._make_graphlet(node_idx=cen_idx[g_idx])

            # Obtain the perturbed graphs
            data_total = data_total + [data_original_]

        return data_total

    def _make_graphlet(
        self,
        node_idx: Union[int, List[int], Tensor],
    ):
        if isinstance(node_idx, Tensor):
            node_idx = int(node_idx)
        elif isinstance(node_idx, List):
            node_idx = node_idx[0]

        edge_index, edge_type, mapping = k_hop_subgraph(
            edge_index=self.edge_index_reduced,
            node_idx=node_idx,
            max_nodes=self.max_nodes,
            num_hops=self.num_hops,
            edge_type=self.edge_type_reduced,
        )

        ent_names = self.ent2idx[mapping[0, 1:], 0]

        x = self._extract_ken_embed(ent_names)
        x = torch.vstack((torch.zeros((1, x.size(1))), x))

        edge_index = to_undirected(edge_index)

        data_out = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            g_idx=node_idx,
            y=torch.tensor([1]),
            flag_perturb=torch.tensor([0]),
            mapping=torch.transpose(mapping, 0, 1),
        )

        return data_out

    def _extract_ken_embed(self, ent_names: list):
        ent_names_ = pd.Series(ent_names)

        # Mapping
        mapping = ent_names_.map(self.ken_embed_ent2idx)
        mapping = mapping.dropna()
        mapping = mapping.astype(np.int64)
        mapping = np.array(mapping)

        # KEN data
        data_ken = self.ken_emb.iloc[mapping]
        data_ken = np.array(data_ken.drop(columns="Entity"))
        return torch.tensor(data_ken)