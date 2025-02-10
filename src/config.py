import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class Config:
    # Huggingface
    hf_token: str = os.getenv("HF_TOKEN")
    # Model
    model_name: str = "openai/clip-vit-base-patch32"
    # Dataset
    cwd: str = os.getcwd()
    dataset_folder: str = os.path.join(cwd, "datasets")
    dataset_file_path: str = os.path.join(dataset_folder, "imagenet_test.tar.gz")
    dataset_name: str = "zh-plus/tiny-imagenet"
    # Training
    number_of_workers: int = 2
    layer_index: int = 22
    batch_size: int = 1024
    input_dim: int = 1024
    expansion_factor: int = 64
    lr: float = 1e-3
    beta_sparsity: float = 1e-4
    epochs: int = 4
