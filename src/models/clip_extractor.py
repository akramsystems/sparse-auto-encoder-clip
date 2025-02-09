import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPViTBaseExtractor(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", layer_index=11):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_model = self.clip_model.vision_model
        self.layer_index = layer_index
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def preprocess(self, image):
        # Process the image using CLIP's processor
        inputs = self.processor(images=image, return_tensors="pt")
        # This will return a tensor of shape (batch_size, channels, height, width)
        return inputs.pixel_values.squeeze(0)  # Remove the batch dimension since we add it later

    def forward(self, pixel_values):
        # Make sure pixel_values has the right shape (batch_size, channels, height, width)
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
            
        # Enable output of all hidden states
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        # Retrieve the hidden state at self.layer_index
        hidden_states = outputs.hidden_states[self.layer_index]
        # Return the [CLS] token embedding
        return hidden_states[:, 0, :]
