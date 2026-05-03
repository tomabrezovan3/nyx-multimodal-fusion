import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------
# 1. Settings
# -----------

TRAIN_DIR = 'data_synth_rd_radar/train'
VAL_DIR = 'data_synth_rd_radar/val'
TEST_DIR = 'data_synth_rd_radar/test'

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------
# 2. Radar dataset
# ----------------

class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label_str in ['0', '1']:
            class_dir =  os.path.join(root_dir, label_str)

            if not os.path.exists(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.endswith('.npy'):
                    path = os.path.join(class_dir, fname)
                    label = int(label_str)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        radar = np.load(path).astype(np.float32)
        radar = np.expand_dims(radar, axis=0) # [1, 64, 64]
        radar_tensor = torch.tensor(radar, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return radar_tensor, label_tensor

# --------
# 3. Model
# --------

class SmallRadarCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), # 64x64 -> 32x32 -> 16x16
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------
# 4. Data
# -------

train_dataset = RadarDataset(TRAIN_DIR)
val_dataset = RadarDataset(VAL_DIR)
test_dataset = RadarDataset(TEST_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Train samples:', len(train_dataset))
print('Val samples:', len(val_dataset))
print('Test samples:', len(test_dataset))

# -------------------------
# 5. Model, loss, optimizer
# -------------------------

model = SmallRadarCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------
# 6. Train one epoch
# ------------------

def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for radar_maps, labels in train_loader:
        radar_maps = radar_maps.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(radar_maps)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# -----------
# 7. Evaluate
# -----------

def evaluate(loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for radar_maps, labels in loader:
            radar_maps = radar_maps.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(radar_maps)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# ----------------
# 8. Training loop
# ----------------

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} | '
        f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | '
        f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}'
    )

# -------------
# 9. Final test
# -------------

test_loss, test_acc = evaluate(test_loader)
print(f'RD Radar Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# --------
# 10. Save
# --------

torch.save(model.state_dict(), 'rd_radar_classifier.pth')
print('Model saved to rd_radar_classifier.pth')


