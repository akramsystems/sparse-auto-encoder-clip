import torch
import torch.nn as nn
from transformers import CLIPModel

class CLIPViTBaseExtractor(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", layer_index=11):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_model = self.clip_model.vision_model
        self.layer_index = layer_index

    def forward(self, pixel_values):
        # Enable output of all hidden states
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        # Retrieve the hidden state at self.layer_index
        hidden_states = outputs.hidden_states[self.layer_index]
        # Return the [CLS] token embedding
        return hidden_states[:, 0, :]
