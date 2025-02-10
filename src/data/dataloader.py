import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset, Dataset
import os

from src.config import Config   

BATCH_SIZE = Config.batch_size
MODEL_NAME = Config.model_name
DATASET_NAME = Config.dataset_name
NUMBER_OF_WORKERS = Config.number_of_workers

def collate_fn(batch):
    # Convert list of lists to a tensor
    pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch], dim=0)
    filepaths = [item.get("file_path", str(i)) for i, item in enumerate(batch)]
    
    # Pull out the *raw* PIL images so we can save them later
    raw_images = [item["image"] for item in batch]  # item["image"] is the PIL image from HF

    # Make sure shapes match
    assert pixel_values.shape[0] == len(filepaths) == len(raw_images)
    
    return pixel_values, filepaths, raw_images

def load_data(
        batch_size=BATCH_SIZE,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        subset_size=None,
        split="train"):
    """
    Loads and preprocesses a subset of the specified dataset for use with a CLIP model.
    Also returns the *raw* images.

    Args:
        batch_size (int): The number of samples per batch to load.
        dataset_name (str): The name of the dataset to load from the Hugging Face datasets library.
        model_name (str): The name of the pre-trained CLIP model to use for processing images.
        subset_size (int): The number of samples to load for testing.

    Returns:
        DataLoader: A PyTorch DataLoader object that yields batches of preprocessed image data.
    """

    # Load the dataset
    dataset = load_dataset(
        dataset_name,
        split=split,
        download_mode="reuse_cache_if_exists",
    )
    
    # Select a smaller subset for testing
    if subset_size:
        # Select subset BEFORE preprocessing to avoid processing the entire dataset
        dataset = dataset.select(range(subset_size))
    
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    
    def preprocess_fn(examples):
        processed = clip_processor(images=examples["image"], return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": processed,
            "labels": examples["label"],
            "file_path": [str(i) for i in range(len(examples["image"]))],
            # Keep the original PIL images under "image" 
            # so we can retrieve them in collate_fn
        }
    
    # Turn off HF caching to ensure we get fresh data each run
    dataset = dataset.map(preprocess_fn, batched=True, load_from_cache_file=False)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=NUMBER_OF_WORKERS
    )

if __name__ == "__main__":
    # Example usage with a smaller subset
    data_loader = load_data(dataset_name="zh-plus/tiny-imagenet")
    for batch in data_loader:
        print(batch.shape)
        break
