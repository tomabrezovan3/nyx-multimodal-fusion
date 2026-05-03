import os
import numpy as np

# --------
# Settings
# --------
IR_BASE = 'data' # your existing IR dataset
RADAR_BASE = 'data' # we will create radar folders here

SPLITS = ['train', 'val', 'test']

RADAR_SIZE = 32

# ----------------
# Radar simulation
# ----------------

def gaussian_blob(size=32, cx=16, cy=16, sigma=3.0):
    y, x = np.mgrid[0:size, 0:size]
    blob = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return blob

def make_radar_map(label, size=32):
    radar = np.random.normal(0.0, 0.05, (size, size))

    if label == 1:
        cx = np.random.randint(8, size - 8)
        cy = np.random.randint(8, size - 8)
        blob = gaussian_blob(size=size, cx=cx, cy=cy, sigma=np.random.uniform(2.0, 4.0))
        radar += blob

    radar = np.clip(radar, 0.0, 1.0)
    return radar.astype(np.float32)

# --------------------
# Main generation loop
# --------------------

for split in SPLITS:
    for label in ['0', '1']:
        ir_dir = os.path.join(IR_BASE, split, label)
        radar_dir = os.path.join(RADAR_BASE, f'radar_{split}', label)

        os.makedirs(radar_dir, exist_ok=True)

        if not os.path.exists(ir_dir):
            print(f'Skipping missing folder: {ir_dir}')
            continue

        for fname in os.listdir(ir_dir):
            if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')):
                continue

            label_int = int(label)

            radar = make_radar_map(label_int, size=RADAR_SIZE)

            base_name = os.path.splitext(fname)[0]
            out_path = os.path.join(radar_dir, base_name + '.npy')

            np.save(out_path, radar)

print('Radar dataset generated!')
