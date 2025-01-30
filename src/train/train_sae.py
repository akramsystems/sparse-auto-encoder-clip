import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.clip_extractor import CLIPViTBaseExtractor
from src.models.sae_model import SparseAutoencoder
from src.data.dataloader import load_data
from src.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Configurations
layer_index = 11  # Adjusted to a valid index for the base model
batch_size = 64  # Adjust as needed
input_dim = 768 # clip base
expansion_factor = 64
lr = 1e-3
beta_sparsity = 1e-4
epochs = 10

# Initialize models and data
feature_extractor = CLIPViTBaseExtractor(layer_index=layer_index).to(device)
sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)
dataloader = load_data(batch_size=batch_size)

# Loss function
def sparse_loss(decoded, original, encoded, beta):
    mse_loss = torch.nn.MSELoss()(decoded, original)
    sparsity_loss = beta * torch.mean(torch.abs(encoded))
    return mse_loss + sparsity_loss

optimizer = optim.Adam(sae.parameters(), lr=lr)

# Training loop with progress bar
for epoch in range(epochs):
    total_loss = 0.0
    
    # Use tqdm to show percentage progress across batches
    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=True
    )

    for i, batch in progress_bar:
        print(f"Processing batch {i+1}/{len(dataloader)}...")
        batch = batch.to(device)
        
        with torch.no_grad():
            batch_features = feature_extractor(batch)
        
        # Ensure features are on the correct device
        batch_features = batch_features.to(device)
        
        encoded, decoded = sae(batch_features)
        loss = sparse_loss(decoded, batch_features, encoded, beta_sparsity)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Update progress bar description with running average loss
        progress_bar.set_postfix({'loss': f'{(total_loss / (i + 1)):.4f}'})

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Save the model after each epoch
    torch.save(sae.state_dict(), f"sae_epoch_{epoch+1}.pth")
