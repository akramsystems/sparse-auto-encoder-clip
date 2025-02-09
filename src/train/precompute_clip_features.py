import torch
from tqdm import tqdm
from src.data.dataloader import load_data
from src.models.clip_extractor import CLIPViTBaseExtractor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    # Use layer_index=11 (or another valid index for this model)
    layer_index = 11
    batch_size = 64
    
    # Calculate subset size (25% of the dataset)
    subset_size = int(100000 * 0.25)  # Assuming 100k images in total
    
    feature_extractor = CLIPViTBaseExtractor(
        model_name="openai/clip-vit-base-patch32",
        layer_index=layer_index
    ).to(device)
    
    # Add subset_size parameter to load_data
    dataloader = load_data(batch_size=batch_size, subset_size=subset_size)

    all_features = []
    
    for batch in tqdm(dataloader, desc="Precomputing CLIP Features"):
        batch = batch.to(device)
        with torch.no_grad():
            batch_features = feature_extractor(batch)
        # Move the batch features to CPU for storage
        all_features.append(batch_features.cpu())

    # Concatenate all features into a single tensor
    all_features = torch.cat(all_features, dim=0)

    # Save to disk for future use
    torch.save(all_features, "clip_features.pt")
    print("Saved CLIP features to clip_features.pt")

if __name__ == "__main__":
    main() 