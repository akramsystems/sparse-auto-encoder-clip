import streamlit as st

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import pickle
from pathlib import Path

from src.data.dataloader import load_data
from src.evaluation.neuron_activation import find_top_activating_images
from src.models.clip_extractor import CLIPViTBaseExtractor
from src.models.sae_model import SparseAutoencoder

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Model parameters
layer_index = 11  # As used in training
input_dim = 768  # CLIP base dimension
expansion_factor = 64  # Matches training setup
model_file_path = "sae_epoch_10.pth"

# Load models
@st.cache_resource
def load_models():
    feature_extractor = CLIPViTBaseExtractor(layer_index=layer_index).to(device)
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)
    sae.load_state_dict(torch.load(model_file_path, map_location=device))
    sae.eval()  # Set model to evaluation mode
    return feature_extractor, sae

feature_extractor, sae = load_models()

# Add sidebar for navigation
st.sidebar.title("Navigation")
visualization_type = st.sidebar.radio(
    "Select Visualization Type",
    ["Neuron Activation Distribution", "Top Activating Images"]
)

# Main title
st.title("Sparse Autoencoder Neuron Activation Visualization")

if visualization_type == "Neuron Activation Distribution":
    # File uploader for single image analysis
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        with torch.no_grad():
            image_tensor = feature_extractor.preprocess(image).unsqueeze(0).to(device)
            image_embedding = feature_extractor(image_tensor)
            encoded, _ = sae(image_embedding)
        
        # Convert activations to numpy array
        activations = encoded.squeeze().cpu().numpy()
        
        # Plot distribution of active neurons
        plt.figure(figsize=(10, 5))
        plt.hist(activations, bins=50, alpha=0.75, color='b')
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title("Distribution of Neuron Activations")
        st.pyplot(plt)
        
elif visualization_type == "Top Activating Images":
    neuron_data_dir = Path("data/top_neurons")
    
    if not neuron_data_dir.exists():
        st.error("Neuron data not found. Please run precompute_top_neurons.py first.")
    else:
        # Get all neuron files
        neuron_files = sorted(neuron_data_dir.glob("neuron_*.pkl"))
        
        for neuron_file in neuron_files:
            neuron_idx = int(neuron_file.stem.split('_')[1])
            
            with open(neuron_file, 'rb') as f:
                data = pickle.load(f)
            
            st.subheader(f"Neuron {neuron_idx}")
            
            # Create a grid of images
            image_tensor = torch.stack(data['images'])
            grid = make_grid(image_tensor, nrow=5, normalize=True, padding=2)
            
            # Convert to PIL image for display
            grid_image = transforms.ToPILImage()(grid)
            st.image(grid_image, caption=f"Top activating images for neuron {neuron_idx}")
            
            # Display activation values
            st.write("Activation values:", data['activation_values'])
