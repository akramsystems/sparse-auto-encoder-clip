import torch
from tqdm import tqdm
import pickle
from pathlib import Path

from src.data.dataloader import load_data
from src.models.clip_extractor import CLIPViTBaseExtractor
from src.models.sae_model import SparseAutoencoder

def precompute_top_neurons(save_dir="data/top_neurons", subset_percentage=0.25):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Model parameters
    layer_index = 11
    input_dim = 768
    expansion_factor = 64
    model_file_path = "sae_epoch_10.pth"
    
    # Load models
    feature_extractor = CLIPViTBaseExtractor(layer_index=layer_index).to(device)
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)
    sae.load_state_dict(torch.load(model_file_path, map_location=device))
    sae.eval()
    
    # Calculate subset size (25% of 10000 validation images = 2500)
    subset_size = int(10000 * subset_percentage)
    
    # Load data with subset
    test_loader = load_data(batch_size=32, split="valid", subset_size=subset_size)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find top activating images
    print("Finding top activating images...")
    results = find_top_activating_images(
        test_loader=test_loader,
        sae=sae,
        feature_extractor=feature_extractor,
        n_top_neurons=10,
        n_top_images=10,
        device=device
    )
    
    # Save results
    for neuron_idx, data in results.items():
        neuron_data = {
            'images': [img.cpu() for img in data['images']],
            'activation_values': data['activation_values']
        }
        
        with open(save_dir / f"neuron_{neuron_idx}.pkl", 'wb') as f:
            pickle.dump(neuron_data, f)
    
    print(f"Saved neuron data to {save_dir}")

def find_top_activating_images(test_loader, sae, feature_extractor, n_top_neurons=10, n_top_images=10, device="cuda"):
    """
    Find the top n images that activate each of the top n neurons the most.
    
    Args:
        test_loader: DataLoader containing the test images
        sae: Trained Sparse Autoencoder model
        feature_extractor: CLIP feature extractor
        n_top_neurons: Number of top neurons to analyze
        n_top_images: Number of top activating images per neuron
        device: Device to run computations on
    
    Returns:
        dict: Dictionary containing top activating images and their activation values for each top neuron
    """
    sae.eval()
    all_activations = []
    all_images = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing images"):
            # Store original images
            all_images.extend(batch.cpu())
            
            # Get activations
            batch = batch.to(device)
            clip_features = feature_extractor(batch)
            encoded, _ = sae(clip_features)
            all_activations.append(encoded.cpu())
    
    # Convert to tensors
    all_activations = torch.cat(all_activations, dim=0)
    
    # Find the neurons with highest average activation
    mean_activations = torch.mean(all_activations, dim=0)
    top_neuron_indices = torch.topk(mean_activations, n_top_neurons).indices
    
    results = {}
    for neuron_idx in top_neuron_indices:
        # Get activations for this neuron across all images
        neuron_activations = all_activations[:, neuron_idx]
        
        # Find top activating images for this neuron
        top_image_indices = torch.topk(neuron_activations, n_top_images).indices
        
        results[neuron_idx.item()] = {
            'images': [all_images[i] for i in top_image_indices],
            'activation_values': neuron_activations[top_image_indices].tolist()
        }
    
    return results

if __name__ == "__main__":
    precompute_top_neurons()
