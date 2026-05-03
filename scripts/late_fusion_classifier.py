import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -----------
# 1. Settings
# -----------

IR_TEST_DIR = 'data_synth_ir/test'
RADAR_TEST_DIR = 'data_synth_rd_radar/test'

IR_MODEL_PATH = 'ir_classifier.pth'
RADAR_MODEL_PATH = 'rd_radar_classifier.pth'

BATCH_SIZE = 8
IMAGE_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------
# 2. IR transform
# ---------------

ir_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --------------------------------
# 3. Dataset for paired IR + radar
# --------------------------------

class PairedLateFusionDataset(Dataset):
    def __init__(self, ir_root, radar_root, ir_transform=None):
        self.samples = []
        self.ir_transform = ir_transform

        for label_str in ['0', '1']:
            ir_class_dir = os.path.join(ir_root, label_str)
            radar_class_dir = os.path.join(radar_root, label_str)

            if not os.path.exists(ir_class_dir):
                continue
            if not os.path.exists(radar_class_dir):
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

        # Load IR image
        ir_image = Image.open(ir_path).convert('L')
        if self.ir_transform is not None:
            ir_tensor = self.ir_transform(ir_image)
        else:
            raise ValueError('IR transform must be provided')

        # Load radar map
        radar = np.load(radar_path).astype(np.float32) # H x W
        radar = np.expand_dims(radar, axis=0) # 1 x H x W
        radar_tensor = torch.tensor(radar, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return ir_tensor, radar_tensor, label_tensor

# -------------------------
# 4. Radar model definition
# -------------------------

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
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------
# 5. Load IR model
# ----------------

ir_model = models.resnet18(weights=None)
ir_model.fc = nn.Linear(ir_model.fc.in_features, 2)
ir_model.load_state_dict(torch.load(IR_MODEL_PATH, map_location=DEVICE))
ir_model = ir_model.to(DEVICE)
ir_model.eval()

# -------------------
# 6. Load radar model
# -------------------

radar_model = SmallRadarCNN()
radar_model.load_state_dict(torch.load(RADAR_MODEL_PATH, map_location=DEVICE))
radar_model = radar_model.to(DEVICE)
radar_model.eval()

# ---------------------------
# 7. Build dataset and loader
# ---------------------------

test_dataset = PairedLateFusionDataset(
    ir_root=IR_TEST_DIR,
    radar_root=RADAR_TEST_DIR,
    ir_transform=ir_transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Paired test samples:', len(test_dataset))

# -----------------------
# 8. Evaluate late fusion
# -----------------------

correct = 0
total = 0

with torch.no_grad():
    for ir_images, radar_maps, labels in test_loader:
        ir_images = ir_images.to(DEVICE)
        radar_maps = radar_maps.to(DEVICE)
        labels = labels.to(DEVICE)

        ir_logits = ir_model(ir_images)
        radar_logits = radar_model(radar_maps)

        fused_logits = (ir_logits + radar_logits) / 2.0
        preds = fused_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

fusion_acc = correct / total if total > 0 else 0.0
print(f'Late Fusion Test Accuracy: {fusion_acc:.4f}')