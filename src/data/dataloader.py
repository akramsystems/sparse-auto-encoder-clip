import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset, Dataset
import os

from src.config import Config   

BATCH_SIZE = Config.batch_size
MODEL_NAME = Config.model_name
DATASET_NAME = Config.dataset_name
NUMBER_OF_WORKERS = 10
def load_data(
        batch_size=BATCH_SIZE,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        subset_size=None):
    """
    Loads and preprocesses a subset of the specified dataset for use with a CLIP model.

    Args:
        batch_size (int): The number of samples per batch to load.
        dataset_name (str): The name of the dataset to load from the Hugging Face datasets library.
        model_name (str): The name of the pre-trained CLIP model to use for processing images.
        subset_size (int): The number of samples to load for testing.

    Returns:
        DataLoader: A PyTorch DataLoader object that yields batches of preprocessed image data.
    """

    dataset = load_dataset(
        dataset_name,
        split="train",
        download_mode="reuse_cache_if_exists",
    )
    
    # Select a smaller subset for testing
    if subset_size:
        dataset = dataset.select(range(subset_size))
    
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    
    def preprocess_fn(examples):
        return {
            "pixel_values": clip_processor(images=examples["image"], return_tensors="pt")["pixel_values"],
            "labels": examples["label"]
        }
    
    dataset = dataset.map(preprocess_fn, batched=True, cache_file_name=f"{dataset_name}_processed.arrow")

    def collate_fn(batch):
        # Convert list of lists to a tensor
        pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch], dim=0)

        assert pixel_values.shape == (len(batch), 3, 224, 224)
        
        return pixel_values

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=NUMBER_OF_WORKERS)

if __name__ == "__main__":
    # Example usage with a smaller subset
    data_loader = load_data(dataset_name="zh-plus/tiny-imagenet")
    for batch in data_loader:
        print(batch.shape)
        break
