import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionClassifier(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        fusion_dim = embed_dim * 3   # img_emb + vid_emb = 1024

        self.fc1 = nn.Linear(fusion_dim, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)

        self.dropout = nn.Dropout(0.3)

        self.fc_out = nn.Linear(256, 2)  # For CrossEntropy (real/fake)

    def forward(self, img_emb, vid_emb, aud_emb):
        x = torch.cat([img_emb, vid_emb, aud_emb], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc_out(x)