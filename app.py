import streamlit as st

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import pickle
from pathlib import Path

from src.data.dataloader import load_data
from src.evaluation.neuron_activation import find_top_activating_images_from_precomputed
from src.models.clip_extractor import CLIPViTBaseExtractor
from src.models.sae_model import SparseAutoencoder
from src.data.precomputed_features_dataset import PrecomputedFeaturesDataset

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
    
    # Load precomputed features
    precomputed_features = PrecomputedFeaturesDataset("clip_features.pt")
    
    # Find top activating images using precomputed features
    top_activations = find_top_activating_images_from_precomputed(
        precomputed_features=precomputed_features,
        sae=sae,
        n_top_neurons=10,
        n_top_images=10,
        device=device,
        batch_size=32  # Added smaller batch size
    )
    
    return feature_extractor, sae, precomputed_features, top_activations

feature_extractor, sae, precomputed_features, top_activations = load_models()

# Add sidebar for navigation
st.sidebar.title("Navigation")
visualization_type = st.sidebar.radio(
    "Select Visualization Type",
    ["Neuron Activation Distribution", "Top Activating Features"]
)

if st.sidebar.checkbox("Debug Info"):
    st.write("Number of precomputed features:", len(precomputed_features))
    st.write("Feature dimension:", precomputed_features[0].shape)

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
        
elif visualization_type == "Top Activating Features":
    # Add a neuron selector
    selected_neuron = st.selectbox(
        "Select Neuron",
        options=sorted(top_activations.keys())
    )
    
    st.subheader(f"Top Activating Features for Neuron {selected_neuron}")
    
    # Display activation values and feature indices in columns
    data = top_activations[selected_neuron]
    cols = st.columns(5)
    for idx, (feature_idx, activation) in enumerate(zip(data['indices'], data['activation_values'])):
        with cols[idx % 5]:
            st.write(f"Feature Index: {feature_idx}")
            st.write(f"Activation: {activation:.4f}")
            # You can add more visualizations here for each feature
