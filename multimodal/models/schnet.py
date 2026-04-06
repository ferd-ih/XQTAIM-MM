import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing, ShiftedSoftplus

# Schnet
class EdgeFiLM(torch.nn.Module):
    """
    FiLM conditioning for SchNet distance features.

    We keep SchNet's original message passing (CFConv/InteractionBlock) intact,
    but let QTAIM edge descriptors modulate the Gaussian-smeared distance
    expansion via learned (gamma, beta):

        rbf' = rbf * (1 + gamma(edge_desc)) + beta(edge_desc)

    The final linear layer is initialized to zeros so that conditioning starts
    as an identity transform (stable warm start).
    """

    def __init__(self, edge_desc_dim: int, num_gaussians: int, hidden_dim: int | None = None):
        super().__init__()
        if edge_desc_dim <= 0:
            raise ValueError("edge_desc_dim must be > 0 to use EdgeFiLM")
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be > 0")

        if hidden_dim is None:
            hidden_dim = max(64, num_gaussians)

        self.edge_desc_dim = edge_desc_dim
        self.num_gaussians = num_gaussians

        self.net = Sequential(
            Linear(edge_desc_dim, hidden_dim),
            ShiftedSoftplus(),
            Linear(hidden_dim, 2 * num_gaussians),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.net[0].weight)
        self.net[0].bias.data.fill_(0)
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)

    def forward(self, rbf: Tensor, edge_desc: Tensor) -> Tensor:
        """
        Args:
            rbf: (E, num_gaussians) Gaussian-smearing of distances
            edge_desc: (E, edge_desc_dim) QTAIM edge descriptors
        """
        if edge_desc is None or edge_desc.numel() == 0:
            return rbf

        params = self.net(edge_desc)  # (E, 2*num_gaussians)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return rbf * (1.0 + gamma) + beta


class SchNet(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=64,
        cutoff=8,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        num_gaussians=50,
        use_edge_descriptors="True",
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(SchNet, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.cutoff = float(cutoff)
        self.num_gaussians = int(num_gaussians)
        self.use_edge_descriptors = (use_edge_descriptors != "False")
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        edge_desc_dim = 0
        try:
            if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_desc_dim = int(data.edge_attr.shape[-1]) if data.edge_attr.dim() > 1 else 1
        except Exception:
            edge_desc_dim = 0
        self.edge_desc_dim = edge_desc_dim

        self.edge_film = None
        if self.use_edge_descriptors and self.edge_desc_dim > 0:
            self.edge_film = EdgeFiLM(edge_desc_dim=self.edge_desc_dim, num_gaussians=self.num_gaussians)
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if post_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        # if data[0].y.ndim == 0:
        #     output_dim = 1
        # else:
        #     output_dim = len(data[0].y[0])
        output_dim = data.y.numel()

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers       
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            # PyG's InteractionBlock takes num_gaussians (not raw edge descriptor dim).
            conv = InteractionBlock(gc_dim, self.num_gaussians, dim3, self.cutoff)
            self.conv_list.append(conv)
            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

        # Expose embedding dimension for downstream fusion models
        if self.pool_order == "early":
            if post_fc_count > 0:
                self.embedding_dim = dim2
            else:
                self.embedding_dim = post_fc_dim
        else:
            # For late pooling, embeddings are node-level before pooling
            if post_fc_count > 0:
                self.embedding_dim = dim2
            else:
                self.embedding_dim = gc_dim

    def forward(self, data, return_embedding: bool = False):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        # Prefer using `data.edge_weight` (raw interatomic distances) if present,
        # otherwise fall back to computing distances from `data.pos`.
        # NOTE: For SchNet, `edge_weight` must be in real distance units (not normalized).
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None or edge_weight.numel() == 0:
            edge_weight = None

        if edge_weight is None:
            if not hasattr(data, "pos") or data.pos is None:
                raise ValueError(
                    "SchNet requires either `data.edge_weight` (raw distances) "
                    "or `data.pos` to compute interatomic distances for Gaussian smearing."
                )
            row, col = data.edge_index
            edge_weight = (data.pos[row] - data.pos[col]).pow(2).sum(dim=-1).sqrt()
        else:
            edge_weight = edge_weight.view(-1)

        edge_weight = edge_weight.clamp(min=0.0, max=self.cutoff)

        edge_attr = self.distance_expansion(edge_weight)

        if self.edge_film is not None and hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
            edge_desc = data.edge_attr
            if edge_desc.dim() == 1:
                edge_desc = edge_desc.unsqueeze(-1)
            edge_attr = self.edge_film(edge_attr, edge_desc)

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = data.x + self.conv_list[i](data.x, data.edge_index, edge_weight, edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = data.x + self.conv_list[i](data.x, data.edge_index, edge_weight, edge_attr)
            else:
                if self.batch_norm == "True":
                    out = out + self.conv_list[i](out, data.edge_index, edge_weight, edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = out + self.conv_list[i](out, data.edge_index, edge_weight, edge_attr)
            #out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            if return_embedding:
                return out
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            if return_embedding:
                return out
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
