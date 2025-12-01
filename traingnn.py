import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader
import os

# --- 1. Define the Model Architecture ---
class CircuitGNN(torch.nn.Module):
    def __init__(self, input_dim=5):
        super(CircuitGNN, self).__init__()
        
        # Layer 1: Input (5 features) -> Hidden (64)
        self.conv1 = GCNConv(input_dim, 64)
        self.bn1 = BatchNorm1d(64) 
        
        # Layer 2: Hidden (64) -> Hidden (128)
        self.conv2 = GCNConv(64, 128)
        self.bn2 = BatchNorm1d(128)
        
        # Layer 3: Hidden (128) -> Hidden (128)
        self.conv3 = GCNConv(128, 128)
        self.bn3 = BatchNorm1d(128)
        
        # --- Readout Layers (Regression) ---
        self.lin_area = Linear(128, 1)
        self.lin_delay = Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Message Passing (Convolution)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 2. Global Pooling
        graph_embedding = global_add_pool(x, batch)
        
        # 3. Final Prediction
        pred_area = self.lin_area(graph_embedding)
        pred_delay = self.lin_delay(graph_embedding)
        
        return pred_area, pred_delay

# --- 2. Training Setup ---
def train():
    # A. Load Data
    data_path = "processed_data.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run parserstep2.py first.")
        return

    # --- FIX IS HERE ---
    # We set weights_only=False because PyG Data objects are complex Python classes
    # This is safe because you generated the file locally.
    data = torch.load(data_path, weights_only=False)
    
    # Create a Loader
    dataset = [data]
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # B. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CircuitGNN(input_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Model initialized on {device}")
    print("Starting Training...")

    # C. Training Loop
    model.train()
    for epoch in range(200): # 200 Epochs
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_area, pred_delay = model(batch)
            
            # Get Targets
            target_area = batch.y[:, 0].view(-1, 1)
            target_delay = batch.y[:, 1].view(-1, 1)
            
            # Calculate Loss (MSE)
            loss_area = F.mse_loss(pred_area, target_area)
            loss_delay = F.mse_loss(pred_delay, target_delay)
            
            # Combine losses
            loss = loss_area + loss_delay 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")
            print(f"   Pred: Area={pred_area.item():.2f}, Delay={pred_delay.item():.2f}")
            print(f"   Real: Area={target_area.item():.2f}, Delay={target_delay.item():.2f}")
            print("-" * 30)

    # D. Final Evaluation
    print("Training Complete.")
    print(f"Final Prediction: Area={pred_area.item():.2f}, Delay={pred_delay.item():.2f}")

if __name__ == "__main__":
    train()
