# my_deepcad.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.attention = nn.Linear(out_dim * 2, 1)
        
    def forward(self, x, adj, mask=None):
        # x: [B, N, F], adj: [B, N, N], mask: [B, N] or None
        h = self.linear(x)  # [B, N, out_dim]
        B, N, D = h.shape
        # Prepare for attention: pairwise concat
        h_i = h.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        h_j = h.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        att_input = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2D]
        att_score = torch.sigmoid(self.attention(att_input)).squeeze(-1)  # [B, N, N]
        # Mask out non-edges
        att_score = att_score * adj
        # Mask out padded nodes (if mask provided)
        if mask is not None:
            # mask: [B, N], need to mask both sender and receiver
            mask_i = mask.unsqueeze(2)  # [B, N, 1]
            mask_j = mask.unsqueeze(1)  # [B, 1, N]
            att_score = att_score * mask_i * mask_j
        # Normalize attention weights
        att_sum = att_score.sum(dim=-1, keepdim=True) + 1e-8
        att_weights = att_score / att_sum  # [B, N, N]
        # Aggregate
        h_new = torch.bmm(att_weights, h)  # [B, N, D]
        # Zero out padded nodes
        if mask is not None:
            h_new = h_new * mask.unsqueeze(-1)
        return h_new

class DeepCAD(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_ops, num_params):
        super(DeepCAD, self).__init__()
        
        # GNN layers
        self.gnn1 = GNNLayer(node_dim, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)
        self.gnn3 = GNNLayer(hidden_dim, hidden_dim)
        
        # LSTM for operation sequence
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Operation type prediction
        self.op_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ops)
        )
        
        # Parameter prediction
        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_params)
        )
        
        # Sketch prediction
        self.sketch_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128)  # 128-dimensional sketch representation
        )
        
        # Node feature prediction
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, batch):
        # Extract inputs
        node_features = batch['node_features']  # [B, N, F]
        adjacency = batch['adjacency']  # [B, N, N]
        mask = batch['mask']  # [B, N]
        
        # GNN processing
        h = self.gnn1(node_features, adjacency, mask)
        h = F.relu(h)
        h = self.gnn2(h, adjacency, mask)
        h = F.relu(h)
        h = self.gnn3(h, adjacency, mask)
        
        # Pack sequence for LSTM
        lengths = mask.sum(dim=1).cpu().int()
        packed_h = nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM processing
        packed_out, _ = self.lstm(packed_h)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Ensure lstm_out has the same sequence length as op_types
        B, T, _ = batch['op_types'].shape
        if lstm_out.size(1) != T:
            # Pad or truncate lstm_out to match op_types length
            if lstm_out.size(1) < T:
                # Pad with zeros
                padding = torch.zeros(B, T - lstm_out.size(1), lstm_out.size(2), device=lstm_out.device)
                lstm_out = torch.cat([lstm_out, padding], dim=1)
            else:
                # Truncate
                lstm_out = lstm_out[:, :T, :]
        
        # Predictions
        op_pred = self.op_predictor(lstm_out)  # [B, T, num_ops]
        param_pred = self.param_predictor(lstm_out)  # [B, T, num_params]
        sketch_pred = self.sketch_predictor(lstm_out)  # [B, T, 128]
        node_pred = self.node_predictor(h)  # [B, N, node_dim]
        
        return {
            'op_pred': op_pred,
            'param_pred': param_pred,
            'sketch_pred': sketch_pred,
            'node_pred': node_pred
        }

class MyDeepCAD(nn.Module):
    """Complete DeepCAD model"""
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=128, max_ops=20, num_op_types=10):
        super(MyDeepCAD, self).__init__()
        
        self.encoder = DeepCAD(input_dim, hidden_dim, max_ops, 10)
        self.decoder = DeepCAD(hidden_dim, hidden_dim, max_ops, 10)
        
    def forward(self, node_features, adjacency, mask=None):
        # Encode CAD graph
        z = self.encoder(node_features, adjacency, mask)
        
        # Decode into operations
        output = self.decoder(z)
        
        return output
    
    def generate(self, node_features, adjacency, mask=None):
        """Generate CAD operations from input graph"""
        with torch.no_grad():
            output = self.forward(node_features, adjacency, mask)
            
            # Get predicted operation types (greedy decoding)
            op_types = torch.argmax(output['op_pred'], dim=2)
            op_params = output['param_pred']
            sketch_params = output['sketch_pred']
            
            # Convert to CAD program
            cad_program = []
            
            for i in range(self.decoder.max_ops):
                op_type = op_types[0, i].item()
                
                # Check if this is an end-of-sequence token
                if op_type == 0:
                    break
                
                # Get parameters for this operation
                params = op_params[0, i].cpu().numpy()
                
                # If this operation needs a sketch, get sketch parameters
                if op_type in [1, 2]:  # Assuming 1=extrude, 2=revolve require sketches
                    sketch = sketch_params[0, i].cpu().numpy()
                    
                    cad_program.append({
                        'type': op_type,
                        'params': params,
                        'sketch': sketch
                    })
                else:
                    cad_program.append({
                        'type': op_type,
                        'params': params
                    })
                    
            return cad_program