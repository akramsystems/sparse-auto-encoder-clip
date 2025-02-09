import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_sparsity(test_loader, sae, feature_extractor, device="mps"):
    """
    Evaluates different sparsity metrics of the SAE's encoded representations:
    1. Average activation (L1 norm)
    2. Fraction of dead neurons (never activate)
    3. Activation frequency per neuron
    4. Distribution of activations
    """
    sae.eval()
    all_activations = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating sparsity")
        
        for batch in progress_bar:
            batch = batch.to(device)
            clip_features = feature_extractor(batch)
            encoded, _ = sae(clip_features)
            all_activations.append(encoded.cpu())
    
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)
    
    # 1. Average activation (L1 norm per sample)
    l1_norms = torch.mean(torch.abs(all_activations), dim=1)
    avg_l1 = torch.mean(l1_norms)
    
    # 2. Dead neurons (never activate)
    neuron_activations = torch.sum(all_activations > 0, dim=0)
    dead_neurons = torch.sum(neuron_activations == 0)
    total_neurons = all_activations.shape[1]
    dead_ratio = dead_neurons / total_neurons
    
    # 3. Activation frequency per neuron
    activation_frequencies = torch.mean((all_activations > 0).float(), dim=0)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Distribution of L1 norms
    plt.subplot(1, 3, 1)
    plt.hist(l1_norms.numpy(), bins=50, density=True)
    plt.title('Distribution of L1 Norms')
    plt.xlabel('L1 Norm')
    plt.ylabel('Density')
    
    # Plot 2: Distribution of activation values
    plt.subplot(1, 3, 2)
    plt.hist(all_activations.numpy().flatten(), bins=50, density=True)
    plt.title('Distribution of Activation Values')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    
    # Plot 3: Neuron activation frequencies
    plt.subplot(1, 3, 3)
    plt.hist(activation_frequencies.numpy(), bins=50, density=True)
    plt.title('Neuron Activation Frequencies')
    plt.xlabel('Activation Frequency')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nSparsity Statistics:")
    print(f"Average L1 norm: {avg_l1:.6f}")
    print(f"Dead neurons: {dead_neurons}/{total_neurons} ({dead_ratio*100:.2f}%)")
    print(f"Mean activation frequency: {torch.mean(activation_frequencies):.6f}")
    print(f"Std activation frequency: {torch.std(activation_frequencies):.6f}")
    
    return {
        'l1_norms': l1_norms,
        'activation_frequencies': activation_frequencies,
        'all_activations': all_activations,
        'dead_neurons': dead_neurons,
        'avg_l1': avg_l1
    }