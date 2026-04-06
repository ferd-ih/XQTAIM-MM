import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output
        
#from https://github.com/AbhiroopBhattacharya/MatMMFuse 's cross_attention_fusion.py
class ImprovedAttentionCombiner(nn.Module):
        def __init__(self, dim, num_heads=8, dropout=0.2):
            super(ImprovedAttentionCombiner, self).__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.fc_out = nn.Linear(dim, dim)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, supervised_embedding, transformer_embedding):
            batch_size = supervised_embedding.size(0)

            # Layer normalization
            supervised_embedding = self.norm1(supervised_embedding)
            transformer_embedding = self.norm1(transformer_embedding)

            # Linear transformations
            query = self.query(transformer_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                                                    2)
            key = self.key(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(supervised_embedding).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
                raise ValueError("NaN or Inf detected in attention_scores")
            attention_weights = self.softmax(attention_scores)
            attention_weights = self.dropout(attention_weights)  # Apply dropout
            combined = torch.matmul(attention_weights, value)

            # Concatenate heads and apply final linear layer
            combined = combined.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            combined = self.fc_out(combined)

            # Add residual connection and apply layer normalization
            combined = self.norm2(combined + transformer_embedding)

            return combined.squeeze(1)
