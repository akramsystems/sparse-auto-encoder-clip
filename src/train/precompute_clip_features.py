import torch
from tqdm import tqdm
from src.data.dataloader import load_data
from src.models.clip_extractor import CLIPViTBaseExtractor
import torchvision.transforms as transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    # Use layer_index=11 (or another valid index for CLIP base)
    layer_index = 11
    batch_size = 64
    subset_size = 10000  # Just an example for debugging

    feature_extractor = CLIPViTBaseExtractor(
        model_name="openai/clip-vit-base-patch32",
        layer_index=layer_index
    ).to(device)

    print(f"Loading dataset with subset_size={subset_size}")
    dataloader = load_data(batch_size=batch_size, subset_size=subset_size)

    dataset_size = len(dataloader.dataset)
    print(f"Dataset size: {dataset_size} images")

    all_features = []
    all_images_for_clip = []
    all_original_images = []

    # We'll use a simple transform to convert raw PIL to tensors
    pil_to_tensor = transforms.ToTensor()

    for (transformed_batch, transformed_images, raw_images) in tqdm(dataloader, desc="Precomputing CLIP Features"):
        # transformed_batch is the preprocessed [B,3,224,224]
        # raw_images is a list of the original PIL images (or however HF loaded them)

        print("Transformed Batch Shape:", transformed_batch.shape)

        # Convert each raw PIL image to a tensor
        # (If your originals vary wildly in size, you have to store them as a list, not a single stack.)
        raw_image_tensors = [pil_to_tensor(img) for img in raw_images]

        # Append them
        all_original_images.extend(raw_image_tensors)

        # CLIP feature extraction
        transformed_batch = transformed_batch.to(device)
        with torch.no_grad():
            batch_features = feature_extractor(transformed_batch)
        all_features.append(batch_features.cpu())

        # Keep a copy of the (already) 224x224 images
        all_images_for_clip.extend(transformed_batch.cpu())

    # Stack up all features and images
    all_features = torch.cat(all_features, dim=0)
    all_images_for_clip = torch.stack(all_images_for_clip, dim=0)

    # If all your raw images have the *same shape*, you could do:
    #   all_original_images = torch.stack(all_original_images, dim=0)
    # Otherwise, you'll have to store them as a list in the .pt file:
    #   all_original_images = all_original_images
    # For demonstration, we'll assume they're consistent size (like 64x64).
    try:
        all_original_images = torch.stack(all_original_images, dim=0)
    except RuntimeError:
        # fallback if shape mismatch
        pass

    print(f"Final feature shape: {all_features.shape}")
    print(f"Final images shape (transformed): {all_images_for_clip.shape}")
    if isinstance(all_original_images, torch.Tensor):
        print(f"Final images shape (original images as Tensors): {all_original_images.shape}")
    else:
        print(f"Final images shape (original images) is a list of length: {len(all_original_images)}")

    torch.save({
        'features': all_features,
        'images_224': all_images_for_clip,
        'images_original': all_original_images,  # can be a list or a stacked tensor
    }, "clip_features.pt")
    print("Saved CLIP features and images to clip_features.pt")

if __name__ == "__main__":
    main() 