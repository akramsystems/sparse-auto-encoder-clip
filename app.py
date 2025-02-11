import pickle
import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as transforms

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
model_file_path = "/workspaces/sparse-auto-encoder-clip/sae_epoch.pth"
clip_features_path = "/workspaces/sparse-auto-encoder-clip/clip_features.pt"
top_activations_path = "/workspaces/sparse-auto-encoder-clip/top_activations.pkl"
N_TOP_NEURONS = 10
N_TOP_IMAGES = 10
BATCH_SIZE = 32

@st.cache_resource
def load_models():
    # If the pickle exists, load it
    if Path(top_activations_path).exists():
        with open(top_activations_path, "rb") as f:
            top_activations = pickle.load(f)
    else:
        # Otherwise, compute top_activations once
        feature_extractor = CLIPViTBaseExtractor(layer_index=layer_index).to(device)
        sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)
        sae.load_state_dict(torch.load(model_file_path, map_location=device))
        sae.eval()

        precomputed_features = PrecomputedFeaturesDataset(clip_features_path)

        top_activations = find_top_activating_images_from_precomputed(
            precomputed_features=precomputed_features,
            sae=sae,
            n_top_neurons=N_TOP_NEURONS,
            n_top_images=N_TOP_IMAGES,
            device=device,
            batch_size=BATCH_SIZE
        )

        # Save to pickle for future calls
        with open(top_activations_path, "wb") as f:
            pickle.dump(top_activations, f)

    # If your other objects (feature_extractor, sae, precomputed_features) 
    # don't change often, you could also cache them similarly
    feature_extractor = CLIPViTBaseExtractor(layer_index=layer_index).to(device)
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor).to(device)
    sae.load_state_dict(torch.load(model_file_path, map_location=device))
    sae.eval()

    precomputed_features = PrecomputedFeaturesDataset()

    return feature_extractor, sae, precomputed_features, top_activations

if "models_loaded" not in st.session_state:
    st.session_state["feature_extractor"], \
    st.session_state["sae"], \
    st.session_state["precomputed_features"], \
    st.session_state["top_activations"] = load_models()
    st.session_state["models_loaded"] = True

feature_extractor = st.session_state["feature_extractor"]
sae = st.session_state["sae"]
precomputed_features = st.session_state["precomputed_features"]
top_activations = st.session_state["top_activations"]

# Add sidebar for navigation
st.sidebar.title("Navigation")
visualization_type = st.sidebar.radio(
    "Select Visualization Type",
    ["Neuron Activation Distribution", "Top Activating Features"]
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
        
elif visualization_type == "Top Activating Features":
    # Add a neuron selector
    selected_neuron = st.selectbox(
        "Select Neuron",
        options=sorted(top_activations.keys())
    )
    
    st.subheader(f"Top Activating Images for Neuron {selected_neuron}")
    
    # Display activation values and images in columns
    data = top_activations[selected_neuron]
    cols = st.columns(5)
    for idx, (feature_idx, activation) in enumerate(zip(data['indices'], data['activation_values'])):
        with cols[idx % 5]:
            # Get the image tensor and convert to displayable forma
            # this is a tensor not a path
            # so we need to convert it to a PIL image
            tensor = precomputed_features.original_images[feature_idx]
            image = transforms.ToPILImage()(tensor)
            st.image(image, caption=f"Activation: {activation:.4f}")
