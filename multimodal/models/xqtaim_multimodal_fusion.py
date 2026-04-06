import torch
import torch.nn as nn
import torch.nn.functional as F
from .schnet import SchNet
from .mm_fusion_modules import GatedFusion
from .gatgnn import GATGNN


class Gated_DGL_MM_Fusion(nn.Module):
    def __init__(self, 
                 crystal_data,
                 qtaim_data,
                 hidden_size=300, fusion_size=512, output_size=1, dropout=0.1,
                 adaptive_temp: float = 1.0,
                 gradient_truncation: bool = True,
                 schnet_dim1=64,
                 schnet_dim2=64,
                 schnet_dim3=64,
                 schnet_cutoff=8.0,
                 schnet_pre_fc_count=1,
                 schnet_gc_count=3,
                 schnet_post_fc_count=1,
                 schnet_num_gaussians=50,
                 schnet_use_edge_descriptors="True",
                 schnet_pool="global_mean_pool",
                 schnet_pool_order="early",
                 schnet_batch_norm="True",
                 schnet_act="relu",
                 schnet_dropout_rate=0.0): 
        super().__init__()

        self.crystal_encoder = GATGNN(
            crystal_data, 
            heads=3, neurons=256, nl=5, 
            global_attention="composition", 
            unpooling_technique="learnable", 
            concat_comp=False)
        self.crystal_proj = nn.Linear(self.crystal_encoder.embedding_dim, hidden_size)
        
        self.qtaim_encoder = SchNet(
            data=qtaim_data,
            dim1=schnet_dim1,
            dim2=schnet_dim2,
            dim3=schnet_dim3,
            cutoff=schnet_cutoff,
            pre_fc_count=schnet_pre_fc_count,
            gc_count=schnet_gc_count,
            post_fc_count=schnet_post_fc_count,
            num_gaussians=schnet_num_gaussians,
            use_edge_descriptors=schnet_use_edge_descriptors,
            pool=schnet_pool,
            pool_order=schnet_pool_order,
            batch_norm=schnet_batch_norm,
            act=schnet_act,
            dropout_rate=schnet_dropout_rate,
        )
        self.qtaim_proj = nn.Linear(self.qtaim_encoder.embedding_dim, hidden_size)
        
        self.gated_fusion_c_gate = GatedFusion(input_dim=hidden_size, dim=hidden_size, output_dim=hidden_size, x_gate=True)
        self.gated_fusion_q_gate = GatedFusion(input_dim=hidden_size, dim=hidden_size, output_dim=hidden_size, x_gate=False)

        final_fusion_input_size = 2 * hidden_size

        self.fusion_initial = nn.Sequential(
            nn.Linear(final_fusion_input_size, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(fusion_size, dropout),
            ResidualBlock(fusion_size, dropout),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.LayerNorm(fusion_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fusion_final = nn.Linear(fusion_size // 2, output_size)
        
        self.qtaim_unimodal_lin_out = nn.Linear(self.qtaim_encoder.embedding_dim, output_size)
        self.crystal_unimodal_lin_out = nn.Linear(self.crystal_encoder.embedding_dim, output_size)

        self.output_size = output_size
        self.crystal_emb_norm = nn.LayerNorm(hidden_size)
        self.qtaim_emb_norm = nn.LayerNorm(hidden_size)
        self.gradient_truncation = gradient_truncation

    def forward(self, crystal_data, qtaim_data):
        crystal_emb_enc = self.crystal_encoder(crystal_data, return_embedding=True)
        qtaim_emb_enc = self.qtaim_encoder(qtaim_data, return_embedding=True)

        self._cached_crystal_emb = crystal_emb_enc
        self._cached_qtaim_emb = qtaim_emb_enc

        if self.gradient_truncation:
            crystal_emb_enc_fusion = crystal_emb_enc.detach()
            qtaim_emb_enc_fusion = qtaim_emb_enc.detach()
        else:
            crystal_emb_enc_fusion = crystal_emb_enc
            qtaim_emb_enc_fusion = qtaim_emb_enc

        crystal_emb = self.crystal_proj(crystal_emb_enc_fusion)
        crystal_emb = self.crystal_emb_norm(crystal_emb)

        qtaim_emb = self.qtaim_proj(qtaim_emb_enc_fusion)
        qtaim_emb = self.qtaim_emb_norm(qtaim_emb)

        c_f = crystal_emb
        q_f = qtaim_emb

        _, _, gated_c_gate = self.gated_fusion_c_gate(c_f, q_f)
        _, _, gated_q_gate = self.gated_fusion_q_gate(q_f, c_f)

        final_combined = torch.cat([gated_c_gate, gated_q_gate], dim=-1)
        final_combined = torch.tanh(final_combined)
        fusion_intermediate = self.fusion_initial(final_combined)
        output = self.fusion_final(fusion_intermediate)

        crystal_pred = self.crystal_unimodal_lin_out(crystal_emb_enc)
        qtaim_pred = self.qtaim_unimodal_lin_out(qtaim_emb_enc)
        crystal_pred = crystal_pred.view(-1, self.output_size)
        qtaim_pred = qtaim_pred.view(-1, self.output_size)
        
        self.crystal_prediction = crystal_pred
        self.qtaim_prediction = qtaim_pred

        self.last_activations = {
            'crystal_emb': crystal_emb,
            'crystal_emb_enc': crystal_emb_enc,
            'qtaim_emb': qtaim_emb,
            'qtaim_emb_enc': qtaim_emb_enc,
            'gated_c_gate': gated_c_gate,
            'gated_q_gate': gated_q_gate,
            'final_combined': final_combined,
            'output': output,
            'crystal_prediction': crystal_pred,
            'qtaim_prediction': qtaim_pred,
            'crystal_emb_norm_mean': crystal_emb.norm(dim=1).mean().detach().cpu().item(),
            'qtaim_emb_norm_mean': qtaim_emb.norm(dim=1).mean().detach().cpu().item(),
            'gated_c_gate_norm_mean': gated_c_gate.norm(dim=1).mean().detach().cpu().item(),
            'gated_q_gate_norm_mean': gated_q_gate.norm(dim=1).mean().detach().cpu().item(),
        }
        return output.reshape(output.size(0), self.output_size)
    
    def get_unimodal_predictions(self):
        if not hasattr(self, 'crystal_prediction') or self.crystal_prediction is None:
            raise RuntimeError(
                "get_unimodal_predictions() must be called after forward(). "
                "No crystal predictions found. Make sure forward() was called first."
            )
        if not hasattr(self, 'qtaim_prediction') or self.qtaim_prediction is None:
            raise RuntimeError(
                "get_unimodal_predictions() must be called after forward(). "
                "No QTAIM predictions found. Make sure forward() was called first."
            )
        
        return self.crystal_prediction, self.qtaim_prediction
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = out + residual
        out = F.relu(out)
        out = self.dropout(out)
        return out


