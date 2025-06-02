import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(x.device)  # [batch_size, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        x = torch.matmul(attention, V)  # [batch_size, seq_len, hidden_dim]
        
        return x, attention

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        # For MultiheadAttention, key_padding_mask should be (batch_size, seq_len)
        # with False for valid positions and True for padding positions
        # This is the opposite of our mask, so we need to invert it
        if mask is not None:
            key_padding_mask = ~mask.bool()  # Invert mask: 0 -> True (padding), 1 -> False (valid)
        else:
            key_padding_mask = None
            
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class CADEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3, num_heads=8, dropout=0.1):
        super(CADEncoder, self).__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling to get fixed-size representation
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final projection to latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        # mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
        
        # Embed input
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
            
        # Global attention pooling
        attn_weights = self.pool(x).squeeze(-1)  # [batch_size, seq_len]
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
            
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        global_repr = torch.sum(x * attn_weights, dim=1)  # [batch_size, hidden_dim]
        
        # Project to latent space
        mu = self.fc_mu(global_repr)
        logvar = self.fc_logvar(global_repr)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class CADDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, max_seq_len=100, num_layers=3, num_heads=8, dropout=0.1):
        super(CADDecoder, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # Project latent vector to initial hidden state
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Operation type classifier
        self.op_type_proj = nn.Linear(hidden_dim, 10)  # Assuming 10 operation types
        
        # Parameter regressor
        self.param_proj = nn.Linear(hidden_dim, 10)  # Assuming 10 parameters per operation
        
        # Sketch feature generator
        self.sketch_proj = nn.Linear(hidden_dim, 128)  # 128-dimensional sketch representation
        
    def forward(self, z, teacher_forcing_ratio=0.5, target_seq=None, mask=None):
        batch_size = z.shape[0]
        
        # Initialize sequence with latent vector
        hidden = self.latent_proj(z).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Create initial sequence with positional embedding
        seq = hidden.repeat(1, self.max_seq_len, 1)  # [batch_size, max_seq_len, hidden_dim]
        seq = seq + self.pos_embedding[:, :self.max_seq_len, :]
        
        # Create a default mask if none is provided
        if mask is None:
            # If no mask is provided, assume all positions are valid
            mask = torch.ones(batch_size, self.max_seq_len, device=z.device)
        else:
            # If mask is provided but its sequence length doesn't match max_seq_len,
            # we need to pad or truncate it
            if mask.size(1) < self.max_seq_len:
                # Pad mask with zeros (padding positions)
                pad_mask = torch.zeros(batch_size, self.max_seq_len - mask.size(1), device=mask.device)
                mask = torch.cat([mask, pad_mask], dim=1)
            elif mask.size(1) > self.max_seq_len:
                # Truncate mask
                mask = mask[:, :self.max_seq_len]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            seq = layer(seq, mask)
            
        # Output projections
        outputs = self.output_proj(seq)  # [batch_size, max_seq_len, output_dim]
        op_types = self.op_type_proj(seq)  # [batch_size, max_seq_len, 10]
        params = self.param_proj(seq)  # [batch_size, max_seq_len, 10]
        sketch = self.sketch_proj(seq)  # [batch_size, max_seq_len, 128]
        
        return {
            'output': outputs,
            'op_types': op_types,
            'params': params,
            'sketch': sketch
        }

class DeepCAD(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=128, max_seq_len=100, num_layers=3, num_heads=8, dropout=0.1):
        super(DeepCAD, self).__init__()
        
        self.encoder = CADEncoder(input_dim, hidden_dim, latent_dim, num_layers, num_heads, dropout)
        self.decoder = CADDecoder(latent_dim, hidden_dim, input_dim, max_seq_len, num_layers, num_heads, dropout)
        self.max_seq_len = max_seq_len
        
    def forward(self, x, mask=None, teacher_forcing_ratio=0.5, target_seq=None):
        # Encode input
        mu, logvar = self.encoder(x, mask)
        
        # Sample latent vector
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode
        outputs = self.decoder(z, teacher_forcing_ratio, target_seq, mask)
        
        # Add KL divergence term
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        outputs['mu'] = mu
        outputs['logvar'] = logvar
        outputs['kl_div'] = kl_div
        
        return outputs
    
    def encode(self, x, mask=None):
        mu, logvar = self.encoder(x, mask)
        return mu
    
    def decode(self, z, mask=None):
        # Create a default mask if none is provided
        if mask is None:
            # If no mask is provided, assume all positions are valid
            mask = torch.ones(z.size(0), self.max_seq_len, device=z.device)
        
        return self.decoder(z, 0.0, None, mask)
    
    def generate(self, num_samples, device):
        # Sample from prior
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(device)
        return self.decode(z) 