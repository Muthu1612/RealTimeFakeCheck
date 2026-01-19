import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioEmbeddingModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        if hasattr(self.encoder, "freeze_feature_extractor"):
            self.encoder.freeze_feature_extractor()

    def forward(self, input_values, attention_mask):
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state   # (B, T, H)
        embeddings = hidden_states.mean(dim=1)      # (B, H)

        return embeddings
