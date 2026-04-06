import torch
import torch.nn as nn
import torch.utils.data
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import size_repr
# from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum
from tqdm import tqdm


class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys():
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class RevIndexedDataset(Dataset):
    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in tqdm(orig)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3, dropout_rate=0.1):
        super(DMPNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = hidden_size  # Output embedding dimension
        self.node_in_norm = nn.LayerNorm(node_fdim)
        self.edge_in_norm = nn.LayerNorm(edge_fdim)
        self.edge_gate = nn.Linear(edge_fdim, 1) 
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth
        self.msg_norm = nn.LayerNorm(hidden_size)
        self.node_norm = nn.LayerNorm(hidden_size)
        self.readout_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-10, 10)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-10, 10)
        
        x = self.node_in_norm(x)
        edge_attr = self.edge_in_norm(edge_attr)

        gate = torch.sigmoid(self.edge_gate(edge_attr))  # [E, 1]

        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg)) * gate

        h = h0
        for _ in range(self.depth - 1):
            # degree-normalized message aggregation
            # compute in-degree per node (target of edges)
            deg = torch.bincount(edge_index[1], minlength=num_nodes).clamp(min=1).float().unsqueeze(1)
            m_node = scatter_sum(h, edge_index[1], dim=0, dim_size=num_nodes) / deg
            m_all = m_node[edge_index[0]]
            m_rev = h[revedge_index]
            m = m_all - m_rev
            h = self.act_func(h0 + self.W2(m)) * gate #gate
            h = self.msg_norm(h)
            h = self.dropout(h)

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))
        node_attr = self.node_norm(node_attr)
        node_attr = self.dropout(node_attr)

        # readout: pyg global pooling
        readout = global_mean_pool(node_attr, batch)
        readout = self.readout_norm(readout)
        return readout