import torch
from torch.utils.data import Dataset

class PrecomputedFeaturesDataset(Dataset):
    def __init__(self, features_path: str):
        super().__init__()
        # Load the features into memory (if you have enough RAM).
        # Alternatively, consider using something like an HDF5 file loader.
        breakpoint()
        data = torch.load(features_path)
        self.features = data['features']
        self.images = data['images_224']
        self.original_images = data['images_original']

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.images[idx], self.original_images[idx] 