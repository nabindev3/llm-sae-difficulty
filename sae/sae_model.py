import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, d_model=768, d_hidden=6144, k=32, aux_k=512):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.aux_k = aux_k
        
        self.W_enc = nn.Parameter(torch.empty(self.d_model, self.d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        
        self.W_dec = nn.Parameter(torch.empty(self.d_hidden, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder columns to unit norm
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)
        
    def forward(self, x, dead_mask=None):
        # Center the input
        x_centered = x - self.b_dec
        
        # Encode
        pre_acts = x_centered @ self.W_enc + self.b_enc
        
        # Top-K routing
        top_acts, top_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, top_idx, F.relu(top_acts))
        
        # Decode
        x_reconstruct = acts @ self.W_dec + self.b_dec
        
        # Aux loss for dead feature revival
        aux_loss = 0.0
        if dead_mask is not None and dead_mask.any():
            dead_pre_acts = pre_acts[:, dead_mask]
            k_aux = min(self.aux_k, dead_pre_acts.shape[-1])
            if k_aux > 0:
                aux_top_acts, aux_top_idx = torch.topk(dead_pre_acts, k_aux, dim=-1)
                aux_acts = torch.zeros_like(dead_pre_acts)
                aux_acts.scatter_(-1, aux_top_idx, F.relu(aux_top_acts))
                
                # Reconstruct residual using only dead features
                aux_reconstruct = aux_acts @ self.W_dec[dead_mask, :]
                residual = x - x_reconstruct.detach()
                aux_loss = F.mse_loss(aux_reconstruct, residual)
        
        return acts, x_reconstruct, aux_loss
    
    @torch.no_grad()
    def normalize_decoder(self):
        """Keep decoder weights normalized during training"""
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)
