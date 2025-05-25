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
        
    def forward(self, x, adj):
        # Message passing
        h = self.linear(x)
        
        # Attention mechanism
        attention_input = torch.cat([h.unsqueeze(1).expand(-1, h.size(1), -1),
                                  h.unsqueeze(2).expand(-1, -1, h.size(1), -1)], dim=-1)
        attention_weights = torch.sigmoid(self.attention(attention_input)).squeeze(-1)
        attention_weights = attention_weights * adj
        
        # Aggregate messages
        h = torch.bmm(attention_weights, h)
        
        return h

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
            batch_first=True
        )
        
        # Operation type prediction
        self.op_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ops)
        )
        
        # Parameter prediction
        self.param_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params)
        )
        
        # Sketch prediction
        self.sketch_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # 128-dimensional sketch representation
        )
        
        # Node feature prediction
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, batch):
        # Extract inputs
        node_features = batch['node_features']
        adjacency = batch['adjacency']
        mask = batch['mask']
        
        # GNN processing
        h = self.gnn1(node_features, adjacency)
        h = F.relu(h)
        h = self.gnn2(h, adjacency)
        h = F.relu(h)
        h = self.gnn3(h, adjacency)
        
        # LSTM processing
        lstm_out, _ = self.lstm(h)
        
        # Predictions
        op_pred = self.op_predictor(lstm_out)
        param_pred = self.param_predictor(lstm_out)
        sketch_pred = self.sketch_predictor(lstm_out)
        node_pred = self.node_predictor(h)
        
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