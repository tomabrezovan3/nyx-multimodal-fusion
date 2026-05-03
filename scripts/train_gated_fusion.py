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

# Synthetic IR dataset paths
TRAIN_IR_DIR = 'data_synth_ir/train'
VAL_IR_DIR = 'data_synth_ir/val'
TEST_IR_DIR = 'data_synth_ir/test'

# Synthetic Range-Doppler radar dataset paths
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
SHARED_DIM = 128
FUSION_HIDDEN_DIM = 64

# ---------------
# 2. IR transform
# ---------------

# ResNet18 expects 3-channel images
# Our IR image is grayscale, so we convert grayscale -> 3 channels
ir_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# ----------------------------
# 3. Paired IR + radar dataset
# ----------------------------

class PairedFusionDataset(Dataset):
    def __init__(self, ir_root, radar_root, ir_transform=None):
        self.samples = []
        self.ir_transform = ir_transform

        # Class folders: 0 = no human, 1 = human target
        for label_str in ['0', '1']:
            ir_class_dir = os.path.join(ir_root, label_str)
            radar_class_dir = os.path.join(radar_root, label_str)

            if not os.path.exists(ir_class_dir) or not os.path.exists(radar_class_dir):
                continue

            # Store radar files by base filename
            # Example: test_1_005.npy -> key = test_1_005
            radar_files = {
                os.path.splitext(fname)[0]: os.path.join(radar_class_dir, fname)
                for fname in os.listdir(radar_class_dir)
                if fname.endswith('.npy')
            }

            # Match every IR image with radar files of same base name
            for fname in os.listdir(ir_class_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
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

        # Load IR image
        ir_image = Image.open(ir_path).convert('L')

        if self.ir_transform is None:
            raise ValueError('ir_transform must be provided')

        ir_tensor = self.ir_transform(ir_image)

        # Load radar RD map
        radar = np.load(radar_path).astype(np.float32)

        # Shape: [H, W] -> [1, H, W]
        radar = np.expand_dims(radar, axis=0)
        radar_tensor = torch.tensor(radar, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return ir_tensor, radar_tensor, label_tensor

# -------------
# 4. IR encoder
# -------------

class IREncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        # ResNet18 extracts visual features from the IR image
        backbone = models.resnet18(weights=None)

        # Original ResNet classifier input size, usually 512
        in_features = backbone.fc.in_features

        # Remove ResNet's classifier
        # Now ResNet outputs a feature vector instead of class logits
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Convert ResNet feature vector into IR embedding
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x) # [B, 512]
        emb = self.embedding(x) # [B, 128]
        return emb

# ----------------
# 5. Radar encoder
# ----------------

class RadarEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()

        # CNN extracts patterns from RD maps:
        # blobs, clutter, Doppler offsets, false alarms
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
        )

        # After two pools:
        # [B, 1, 64, 64] -> [B, 32, 16, 16]
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x) # [B, 32, 16, 16]
        emb = self.embedding(x) # [B, 64]
        return emb

# ---------------------
# 6. Gated fusion model
# ---------------------

class GatedFusionModel(nn.Module):
    def __init__(
            self,
            ir_embed_dim=128,
            radar_embed_dim=64,
            shared_dim=128,
            fusion_hidden_dim=64
    ):
        super().__init__()

        # Separate modality encoders
        self.ir_encoder = IREncoder(embed_dim=ir_embed_dim)
        self.radar_encoder = RadarEncoder(embed_dim=radar_embed_dim)

        # IR and radar embeddings have different sizes:
        # IR: 128
        # Radar: 64
        #
        # To combine them with a weighted sum, we first project both
        # into the same shared latent dimension
        self.ir_proj = nn.Linear(ir_embed_dim, shared_dim)
        self.radar_proj = nn.Linear(radar_embed_dim, shared_dim)

        # Gate network:
        # It looks at both projected embeddings and decides how much
        # to trust IR vs radar for each feature dimension
        #
        # Output values are between 0 and 1 because of Sigmoid
        self.gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.Sigmoid()
        )

        # Final classifier after gated fusion
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 2)
        )

    def forward(self, ir_x, radar_x):
        # Encode both modalities separately
        ir_emb = self.ir_encoder(ir_x) # [B, 128]
        radar_emb = self.radar_encoder(radar_x) # [B, 64]

        # Project both embeddings into the same dimensional space
        ir_z = self.ir_proj(ir_emb) # [B, shared_dim]
        radar_z = self.radar_proj(radar_emb) # [B, shared_dim]

        # Concatenates projected embeddings so gate can inspect both
        gate_input = torch.cat([ir_z, radar_z], dim=1) # [B, shared_dim * 2]

        # Gate values:
        # close to 1 -> trust IR more
        # close to 0 -> trust radar more
        gate = self.gate(gate_input) # [B, shared_dim]

        # Weighted fusion:
        # each feature dimension gets its own IR/radar trust weight
        fused = gate * ir_z + (1.0 - gate) * radar_z # [B, shared_dim]

        # Final classification
        logits = self.classifier(fused) # [B, 2]

        return logits, ir_emb, radar_emb, fused, gate

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

model = GatedFusionModel(
    ir_embed_dim=IR_EMBED_DIM,
    radar_embed_dim=RADAR_EMBED_DIM,
    shared_dim=SHARED_DIM,
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

        logits, _, _, _, _ = model(ir_images, radar_maps)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# -----------
# 10.Evaluate
# -----------

def evaluate(loader):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    gate_sum = 0.0
    gate_batches = 0

    with torch.no_grad():
        for ir_images, radar_maps, labels in loader:
            ir_images = ir_images.to(DEVICE)
            radar_maps = radar_maps.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _, _, _, gate = model(ir_images, radar_maps)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Track average gate value
            # Around 0.5 = balanced
            # Higher = more IR influence
            # Lower = more radar influence
            gate_sum += gate.mean().item()
            gate_batches += 1

    avg_gate = gate_sum / gate_batches

    return running_loss / total, correct / total, avg_gate

# -----------------------
# 11. Sanity-check shapes
# -----------------------

with torch.no_grad():
    ir_sample, radar_sample, _ = train_dataset[0]

    ir_sample = ir_sample.unsqueeze(0).to(DEVICE)
    radar_sample = radar_sample.unsqueeze(0).to(DEVICE)

    _, ir_emb, radar_emb, fused, gate = model(ir_sample, radar_sample)

    print('IR embedding shape:', ir_emb.shape)
    print('Radar embedding shape:', radar_emb.shape)
    print('Fused embedding shape:', fused.shape)
    print('Gate shape:', gate.shape)
    print('Initial average gate value:', gate.mean().item())

# -----------------
# 12. Training loop
# -----------------

best_val_acc = 0.0
best_model_path = 'best_gated_fusion_model.pth'

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc, val_gate = evaluate(val_loader)

    print(
        f'Epoch {epoch + 1}/{EPOCHS} | '
        f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | '
        f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} | '
        f'Avg gate: {val_gate:.4f}'
    )

    # --- KEY PART ---
    # Save model ONLY if validation improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved (val acc = {val_acc:.4f})')

# --------------
# 13. Final test
# --------------

# Load best model before testing
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

test_loss, test_acc, test_gate = evaluate(test_loader)

print(
    f'Gated Fusion Test Loss: {test_loss:.4f}, '
    f'Test Acc: {test_acc:.4f}, '
    f'Avg gate: {test_gate:.4f}'
)

print(f'Best model kept at {best_model_path}')







