import torch
from tqdm import tqdm
import pickle
from pathlib import Path

from src.data.dataloader import load_data
from src.models.clip_extractor import CLIPViTBaseExtractor
from src.models.sae_model import SparseAutoencoder

# Data parameters
BATCH_SIZE = 32
DATA_SPLIT = 0.25
DATASET_SIZE = 10000
DATA_SPLIT_NAME = "valid"
N_TOP_NEURONS = 50
N_TOP_IMAGES = 10


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def precompute_top_neurons(save_dir="data/top_neurons", subset_percentage=DATA_SPLIT):
    # Device setup
    
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
    subset_size = int(DATASET_SIZE * subset_percentage)
    
    # Load data with subset
    test_loader = load_data(batch_size=BATCH_SIZE, split=DATA_SPLIT_NAME, subset_size=subset_size)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find top activating images
    print("Finding top activating images...")
    results = find_top_activating_images(
        test_loader=test_loader,
        sae=sae,
        feature_extractor=feature_extractor,
        n_top_neurons=N_TOP_NEURONS,
        n_top_images=N_TOP_IMAGES,
        device=device
    )
    
    # Save results
    for neuron_idx, data in results.items():
        neuron_data = {
            'images': [img.to(device) for img in data['images']],
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
            # Adjust unpacking to match the structure of the batch
            inputs = batch[0]  # Assuming the first element is the input data
            
            # Store original images
            all_images.extend(inputs.to(device))
            
            # Get activations
            inputs = inputs.to(device)
            clip_features = feature_extractor(inputs)
            encoded, _ = sae(clip_features)
            all_activations.append(encoded.to(device))
    
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

def find_top_activating_images_from_precomputed(
    precomputed_features, 
    sae, 
    n_top_neurons=10, 
    n_top_images=10, 
    device="cuda", 
    batch_size=64
):
    sae.eval()
    all_activations = []

    with torch.no_grad():
        for i in range(0, len(precomputed_features), batch_size):
            # Extract only the features from the (features, image) tuple
            batch_features = torch.stack([
                precomputed_features[j][0]
                for j in range(i, min(i + batch_size, len(precomputed_features)))
            ]).to(device)

            encoded, _ = sae(batch_features)
            all_activations.append(encoded.to(device))

    # Concatenate all batched activations
    all_activations = torch.cat(all_activations, dim=0)

    # Find the neurons with highest average activation
    mean_activations = torch.mean(all_activations, dim=0)
    
    top_neuron_indices = torch.topk(mean_activations, n_top_neurons).indices

    results = {}
    for neuron_idx in top_neuron_indices:
        neuron_activations = all_activations[:, neuron_idx]
        top_image_indices = torch.topk(neuron_activations, n_top_images).indices

        results[neuron_idx.item()] = {
            'indices': top_image_indices.tolist(),
            'activation_values': neuron_activations[top_image_indices].tolist()
        }

    return results

if __name__ == "__main__":
    precompute_top_neurons()
