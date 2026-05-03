import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------
# 1. Settings
# -----------

TRAIN_DIR = 'data/radar_train'
VAL_DIR = 'data/radar_val'
TEST_DIR = 'data/radar_test'

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------
# 2. Custom dataset
# -----------------

class RadarDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for label_str in ['0', '1']:
            class_dir = os.path.join(root_dir, label_str)
            label = int(label_str)

            for fname in os.listdir(class_dir):
                if fname.endswith('.npy'):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        radar = np.load(path).astype(np.float32) # shape: H x W
        radar = np.expand_dims(radar, axis=0) # shape: 1 x H x W
        radar = torch.tensor(radar, dtype=torch.float32)
        return radar, label

# ------------
# 3. Small CNN
# ------------

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
            nn.Linear(32 * 8 * 8, 64), # assumes input radar map is 32x32
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------
# 4. Data loaders
# ---------------

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

    for radar, labels in train_loader:
        radar, labels = radar.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(radar)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * radar.size(0)
        preds = outputs.argmax(dim=1)
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
        for radar, labels in loader:
            radar, labels = radar.to(DEVICE), labels.to(DEVICE)

            outputs = model(radar)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * radar.size(0)
            preds = outputs.argmax(dim=1)
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
        f'Epoch {epoch+1}/{EPOCHS} | '
        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} '
    )

# -------------
# 9. Final test
# -------------

test_loss, test_acc = evaluate(test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# --------------
# 10. Save model
# --------------

torch.save(model.state_dict(), 'radar_classifier.pth')
print('Model saved to radar_classifier.pth')