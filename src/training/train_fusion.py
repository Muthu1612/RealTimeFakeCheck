import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.models.fusion_classifier import FusionClassifier
from src.loaders.embedding_loader import EmbeddingDataset
from pathlib import Path


img_path = "embeddings/image_embeddings.pt"
vid_path = "embeddings/video_embeddings.pt"
aud_path = "embeddings/audio_embeddings.pt"
img = torch.load(img_path)
vid = torch.load(vid_path)
aud = torch.load(aud_path)

aud_projection = nn.Linear(768, 512)
aud_embed_512 = aud_projection(aud["embeddings"])
dim_reduced_aud = {}
dim_reduced_aud["embeddings"] = aud_embed_512
dim_reduced_aud["labels"] = aud["labels"]
dim_reduced_aud

emb = dim_reduced_aud["embeddings"]  
lbl = dim_reduced_aud["labels"] 

A, C = vid["embeddings"].shape
print(A, C)
B, D = emb.shape
k = B // A
B_new = A * k

perm = torch.randperm(B)
emb_shuffled = emb[perm]
lbl_shuffled = lbl[perm]

emb_trim = emb_shuffled[:B_new]
lbl_trim = lbl_shuffled[:B_new]

emb_reshaped = emb_trim.view(A, k, C)      # [A, k, 512]
emb_pooled = emb_reshaped.mean(dim=1)           # [A, 512]

lbl_reshaped = lbl_trim.view(A, k)           # [A, k]
lbl_final = lbl_reshaped.mode(dim=1).values     # majority vote, shape [6529]

# 5. Pack results
aud_pooled = {
    "embeddings": emb_pooled,
    "labels": lbl_final
}

dataset = EmbeddingDataset(img, vid, aud_pooled)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FusionClassifier(embed_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for img_emb, vid_emb, aud_emb, labels in loader:
        
        img_emb = img_emb.detach().to(device)
        vid_emb = vid_emb.detach().to(device)
        aud_emb = aud_emb.detach().to(device)
        
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img_emb, vid_emb, aud_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={total_loss:.4f} | Acc={acc:.4f}")


save_path = Path("artifacts/fusion_classifier/model.pth")

save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), save_path)