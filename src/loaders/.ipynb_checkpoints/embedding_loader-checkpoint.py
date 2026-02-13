import torch
from torch.utils.data import Dataset
import torch.nn as nn
class EmbeddingDataset(Dataset):
    def __init__(self, img, vid, aud):
        self.img_emb = img["embeddings"]
        self.vid_emb = vid["embeddings"]
        self.aud_emb = aud["embeddings"]

        assert len(self.img_emb) == len(self.vid_emb) == len(self.aud_emb)

        self.labels = (
            img["labels"] + vid["labels"] + aud["labels"] >= 2
        ).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.img_emb[idx],
            self.vid_emb[idx],
            self.aud_emb[idx],
            self.labels[idx]
        )
