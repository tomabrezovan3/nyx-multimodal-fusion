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

# These are the test folders for the two modalities
# We analyze test samples because we want to understand generalization behavior, not training memorization
IR_TEST_DIR = 'data_synth_ir/test'
RADAR_TEST_DIR = 'data_synth_rd_radar/test'

# These are the trained single-modality models
# The IR model was trained only on IR images
# The radar model was trained only on RD radar maps
IR_MODEL_PATH = 'ir_classifier.pth'
RADAR_MODEL_PATH = 'rd_radar_classifier.pth'

BATCH_SIZE = 1 # batch size 1 makes it easy to inspect each sample individually
IMAGE_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------
# 2. IR transform
# ---------------

# The model expects 3-channel input because ResNet18 expects RGB-like images
# Even though our IR image is grayscale, we convert it to 3 channels
ir_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------
# 3. Paired dataset
# -----------------

class PairedDataset(Dataset):
    def __init__(self, ir_root, radar_root, ir_transform=None):
        self.samples = []
        self.ir_transform = ir_transform

        # We loop through both class folders: 0 = no target, 1 = human target
        for label_str in ['0', '1']:
            ir_class_dir = os.path.join(ir_root, label_str)
            radar_class_dir = os.path.join(radar_root, label_str)

            # If either modality folder is missing, skip it
            if not os.path.exists(ir_class_dir) or not os.path.exists(radar_class_dir):
                continue

            # Build a dictionary of radar files:
            # key = filename without extension
            # value = full radar file path
            #
            # Example:
            # train_1_004.npy -> key 'train_1_004'
            radar_files = {
                os.path.splitext(fname)[0]: os.path.join(radar_class_dir, fname)
                for fname in os.listdir(radar_class_dir)
                if fname.endswith('.npy')
            }

            # Now loop through IR images and find the matching radar file
            for fname in os.listdir(ir_class_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Remove file extension so:
                # test_1_004.png -> test_1_004
                base_name = os.path.splitext(fname)[0]

                # Only keep samples where IR and radar have the same base filename
                if base_name not in radar_files:
                    continue

                ir_path = os.path.join(ir_class_dir, fname)
                radar_path = radar_files[base_name]
                label = int(label_str)

                # Store filename too, so later we can print which sample failed
                self.samples.append((base_name, ir_path, radar_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_name, ir_path, radar_path, label = self.samples[idx]

        # Load IR image as grayscale
        ir_image = Image.open(ir_path).convert('L')

        # Apply transform: grayscale -> 3 channels -> resize -> tensor
        if self.ir_transform is None:
            raise ValueError('ir_transform must be provided')

        ir_tensor = self.ir_transform(ir_image)

        # Load radar RD map from .npy
        # Original shape is [H, W]
        radar = np.load(radar_path).astype(np.float32)

        # Add channel dimension so shape becomes [1, H, W]
        # CNNs expect [channels, height, width]
        radar = np.expand_dims(radar, axis=0)

        radar_tensor = torch.tensor(radar, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return base_name, ir_tensor, radar_tensor, label_tensor

# -------------------------
# 4. Radar model definition
# -------------------------

class SmallRadarCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extractor:
        # input: [B, 1, 64, 64]
        # output: [B, 32, 16, 16]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
        )

        # Classifier:
        # flatten [32, 16, 16] into one vector, then classify into 2 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

# ------------------------
# 5. Load trained IR model
# ------------------------

# Recreate the exact IR model architecture used during training
ir_model = models.resnet18(weights=None)
ir_model.fc = nn.Linear(ir_model.fc.in_features, 2)

# Load learned weights
ir_model.load_state_dict(torch.load(IR_MODEL_PATH, map_location=DEVICE))

# Move model to CPU/GPU and set evaluation mode
ir_model = ir_model.to(DEVICE)
ir_model.eval()

# ---------------------------
# 6. Load trained radar model
# ---------------------------

# Recreate radar model architecture
radar_model = SmallRadarCNN()

# Load learned weights
radar_model.load_state_dict(torch.load(RADAR_MODEL_PATH, map_location=DEVICE))

# Move model to CPU/GPU and set evaluation mode
radar_model = radar_model.to(DEVICE)
radar_model.eval()

# ------------------------------
# 7. Build paired dataset loader
# ------------------------------

dataset = PairedDataset(
    ir_root=IR_TEST_DIR,
    radar_root=RADAR_TEST_DIR,
    ir_transform=ir_transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Paired test samples:', len(dataset))
print()

# ---------------------------
# 8. Complementarity counters
# ---------------------------

# These counters answer the key question:
# Do IR and radar fail on the same samples or different samples?
both_correct = 0
ir_only_correct = 0
radar_only_correct = 0
both_wrong = 0

# Store detailed per-sample rows for printing
rows = []

# ----------------------------
# 9. Run per-sample evaluation
# ----------------------------

with torch.no_grad():
    for base_names, ir_images, radar_maps, labels in loader:
        # Move tensors to device
        ir_images = ir_images.to(DEVICE)
        radar_maps = radar_maps.to(DEVICE)
        labels = labels.to(DEVICE)

        # Get logits from  each single-modality model
        ir_logits = ir_model(ir_images)
        radar_logits = radar_model(radar_maps)

        # Convert logits to predicted class index
        ir_pred = ir_logits.argmax(dim=1)
        radar_pred = radar_logits.argmax(dim=1)

        # Since batch size is 1, extract scalar values
        true_label = labels.item()
        ir_p = ir_pred.item()
        radar_p = radar_pred.item()

        # Check whether each modality was correct
        ir_correct = (ir_p == true_label)
        radar_correct = (radar_p == true_label)

        # Categorize this sample
        if ir_correct and radar_correct:
            category = 'both_correct'
            both_correct += 1
        elif ir_correct and not radar_correct:
            category = 'ir_only_correct'
            ir_only_correct += 1
        elif radar_correct and not ir_correct:
            category = 'radar_only_correct'
            radar_only_correct += 1
        else:
            category = 'both_wrong'
            both_wrong += 1

        # Store result for detailed inspection
        rows.append((base_names[0], true_label, ir_p, radar_p, category))

# -----------------------------------
# 10. Print detailed per-sample table
# -----------------------------------

print('Per-sample predictions:')
print('filename | true | ir_pred | radar_pred | category')
print('-' * 70)

for filename, true_label, ir_p, radar_p, category in rows:
    print(f'{filename} | {true_label} | {ir_p} | {radar_p} | {category}')

# -----------------
# 11. Print summary
# -----------------

total = len(dataset)

print()
print('Summary:')
print(f'Both correct:   {both_correct}')
print(f'IR only correct:  {ir_only_correct}')
print(f'Radar only correct:  {radar_only_correct}')
print(f'Both wrong:  {both_wrong}')

print()
print('Percentages:')
print(f'Both correct:   {both_correct/total:.2f}')
print(f'IR only correct:  {ir_only_correct/total:.2f}')
print(f'Radar only correct:  {radar_only_correct/total:.2f}')
print(f'Both wrong:  {both_wrong/total:.2f}')

# -------------------------
# 12. Interpretation helper
# -------------------------

print()
print('Interpretation guide:')

print('- If IR only correct and radar only correct are both high, the modalities are complementary.')
print('- If both wrong is high, fusion cannot recover these cases yet.')
print('- If radar only correct is much higher than IR only correct, radar carries more useful signal.')
print('- If IR only correct is much higher than radar only correct, IR carries more useful signal.')
