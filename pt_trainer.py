import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from collections import Counter

# Konfiguracija
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4

writer = SummaryWriter(log_dir='runs/fakenews_from_embeddings')

# Dataset iz shranjenih embeddingov
class EmbeddingDataset(Dataset):
    def __init__(self, embedding_path):
        self.data = torch.load(embedding_path)
        self.labels = [int(d['label']) for d in self.data]
        print(f"Loaded {len(self.data)} samples from {embedding_path}.")
        print("Label distribution:", Counter(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['image_embed'], item['text_embed'], torch.tensor(item['label'], dtype=torch.float32)

# Klasifikator
class FakeNewsClassifier(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, hidden_dim=256):
        super(FakeNewsClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_embed, text_embed):
        x = torch.cat((image_embed, text_embed), dim=1)
        return self.fc(x)

# Eval funkcija z izpisom razreda porazdelitve
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for img_emb, txt_emb, labels in loader:
            img_emb, txt_emb = img_emb.to(DEVICE), txt_emb.to(DEVICE)
            outputs = model(img_emb, txt_emb)
            predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5
            preds.extend(predictions.flatten())
            targets.extend(labels.numpy().flatten())

    print("Predicted label distribution:", Counter(preds))
    print("True label distribution:", Counter(targets))
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=0)
    return acc, f1

# Naloži podatke
train_dataset = EmbeddingDataset("train_embeddings.pt")
val_dataset = EmbeddingDataset("val_embeddings.pt")
test_dataset = EmbeddingDataset("test_embeddings.pt")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model + trening
model = FakeNewsClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, val_accuracies, val_f1s = [], [], []
global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for img_emb, txt_emb, labels in train_loader:
        img_emb, txt_emb, labels = img_emb.to(DEVICE), txt_emb.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(img_emb, txt_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    train_losses.append(avg_loss)

    val_acc, val_f1 = evaluate(model, val_loader)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("F1/val", val_f1, epoch)
    print(f"Epoch {epoch+1} - Val Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}\n")

# Končna evalvacija
print("Final Evaluation on Test Set:")
test_acc, test_f1 = evaluate(model, test_loader)
print("Test Accuracy:", test_acc)
print("Test F1 Score:", test_f1)

# Grafi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(val_f1s, label='Validation F1')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics')
plt.legend()

plt.tight_layout()
plt.savefig("training_results_from_embeddings.png")
plt.show()
