import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -----------
# 1. Settings
# -----------

TRAIN_IR_DIR = 'data_synth_ir/train'
VAL_IR_DIR = 'data_synth_ir/val'
TEST_IR_DIR = 'data_synth_ir/test'

TRAIN_RADAR_DIR = 'data_synth_rd_radar/train'
VAL_RADAR_DIR = 'data_synth_rd_radar/val'
TEST_RADAR_DIR = 'data_synth_rd_radar/test'

BATCH_SIZE = 8
IMAGE_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IR_EMBED_DIM = 128
RADAR_EMBED_DIM = 64
FUSION_HIDDEN_DIM = 64

# ----------------
# 2. IR transforms
# ----------------

ir_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------
# 3. Paired dataset
# -----------------

class PairedFusionDataset(Dataset):
    def __init__(self, ir_root, radar_root, ir_transform=None):
        self.samples = []
        self.ir_transform = ir_transform

        for label_str in ['0', '1']:
            ir_class_dir = os.path.join(ir_root, label_str)
            radar_class_dir = os.path.join(radar_root, label_str)

            if not os.path.exists(ir_class_dir) or not os.path.exists(radar_class_dir):
                continue

            radar_files = {
                os.path.splitext(fname)[0]: os.path.join(radar_class_dir, fname)
                for fname in os.listdir(radar_class_dir)
                if fname.endswith('.npy')
            }

            for fname in os.listdir(ir_class_dir):
                if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')):
                    continue

                base_name = os.path.splitext(fname)[0]
                if base_name not in radar_files:
                    continue

                ir_path = os.path.join(ir_class_dir, fname)
                radar_path = radar_files[base_name]
                label = int(label_str)

                self.samples.append((ir_path, radar_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ir_path, radar_path, label = self.samples[idx]

        ir_image = Image.open(ir_path).convert('L')

        if self.ir_transform is not None:
            ir_tensor = self.ir_transform(ir_image)
        else:
            raise ValueError('ir_transform must be provided')

        radar = np.load(radar_path).astype(np.float32)
        radar = np.expand_dims(radar, axis=0) # [1, H, W]
        radar_tensor = torch.tensor(radar, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return ir_tensor, radar_tensor, label_tensor

# -------------
# 4. IR encoder
# -------------

class IREncoder(nn.Module): # Define a neural network module (inherits from PyTorch base class)
    def __init__(self, embed_dim=128): # Constructor, embed_dim = size of output embedding
        super().__init__() # Initialize parent nn.Module

        # Load a ResNet18 model (without pretrained weights)
        backbone = models.resnet18(weights=None)

        # Get number of input features to the final fully connected layer
        # For ResNet18, this is typically 512
        in_features = backbone.fc.in_features

        # Replace the final classification layer with identity (does nothing)
        # This removes the classifier and keeps only feature extraction
        backbone.fc = nn.Identity()

        # Save the modified ResNet as the backbone (feature extractor)
        self.backbone = backbone

        # Define embedding layer:
        # Takes 512-dim feature vector and reduces it to embed_dim (e.g. 128)
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embed_dim), # Linear projection: 512 -> 128
            nn.ReLU() # Non-linearity to improve representation
        )

    def forward(self, x): # Defines how data flows through the model
        x = self.backbone(x) # [B, 512]
        # After backbone:
        # shape = [B, 512]
        # This is global feature vector (NOT a feature map anymore)

        emb = self.embedding(x) # [B, embed_dim]
        # After embedding layer:
        # shape = [B, embed_dim] (e.g. [B, 128])
        # This is the final embedding representation

        return emb # Return embedding (no classification here)

# ----------------
# 5. Radar encoder
# ----------------

class RadarEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, embed_dim), # assumes 64x64 radar maps
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x) # [B, 32, 16, 16]
        emb = self.embedding(x) # [B, embed_dim]
        return emb

# ----------------------------
# 6. Intermediate fusion model
# ----------------------------

class IntermediateFusionModel(nn.Module):
    def __init__(self, ir_embed_dim=128, radar_embed_dim=64, fusion_hidden_dim=64):
        super().__init__()

        self.ir_encoder = IREncoder(embed_dim=ir_embed_dim)
        self.radar_encoder = RadarEncoder(embed_dim=radar_embed_dim)

        fused_dim = ir_embed_dim + radar_embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 2)
        )

    def forward(self, ir_x, radar_x):
        ir_emb = self.ir_encoder(ir_x)
        radar_emb = self.radar_encoder(radar_x)

        fused = torch.cat([ir_emb, radar_emb], dim=1)
        logits = self.classifier(fused)

        return logits, ir_emb, radar_emb, fused

# -------
# 7. Data
# -------

train_dataset = PairedFusionDataset(TRAIN_IR_DIR, TRAIN_RADAR_DIR, ir_transform=ir_transform)
val_dataset = PairedFusionDataset(VAL_IR_DIR, VAL_RADAR_DIR, ir_transform=ir_transform)
test_dataset = PairedFusionDataset(TEST_IR_DIR, TEST_RADAR_DIR, ir_transform=ir_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Train paired samples:', len(train_dataset))
print('Val paired samples:', len(val_dataset))
print('Test paired samples:', len(test_dataset))

# -------------------------
# 8. Model, loss, optimizer
# -------------------------

model = IntermediateFusionModel(
    ir_embed_dim=IR_EMBED_DIM,
    radar_embed_dim=RADAR_EMBED_DIM,
    fusion_hidden_dim=FUSION_HIDDEN_DIM
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------
# 9. Train one epoch
# ------------------

def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for ir_images, radar_maps, labels in train_loader:
        ir_images = ir_images.to(DEVICE)
        radar_maps = radar_maps.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, _, _, _ = model(ir_images, radar_maps)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ------------
# 10. Evaluate
# ------------

def evaluate(loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for ir_images, radar_maps, labels in loader:
            ir_images = ir_images.to(DEVICE)
            radar_maps = radar_maps.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _, _, _ = model(ir_images, radar_maps)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

# ---------------------------------
# 11. Sanity-check embedding shapes
# ---------------------------------

with torch.no_grad():
    ir_sample, radar_sample, _ = train_dataset[0]
    ir_sample = ir_sample.unsqueeze(0).to(DEVICE)
    radar_sample = radar_sample.unsqueeze(0).to(DEVICE)

    _, ir_emb, radar_emb, fused = model(ir_sample, radar_sample)
    print('IR embedding shape:', ir_emb.shape)
    print('Radar embedding shape:', radar_emb.shape)
    print('Fused embedding shape:', fused.shape)

# -----------------
# 12. Training loop
# -----------------

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} | '
        f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | '
        f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}'
    )

# --------------
# 13. Final test
# --------------

test_loss, test_acc = evaluate(test_loader)
print(f'Intermediate Fusion Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# --------------
# 14. Save model
# --------------

torch.save(model.state_dict(), 'intermediate_fusion_model.pth')
print('Model saved to intermediate_fusion_model.pth')




