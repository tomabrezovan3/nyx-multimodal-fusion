import os
import numpy as np

# --------
# Settings
# --------

IR_BASE = 'data_synth_ir'
RADAR_ROOT = 'data_synth_rd_radar'

SPLITS = ['train', 'val', 'test']

RANGE_BINS = 64
DOPPLER_BINS = 64

# -------
# Helpers
# -------

def gaussian_2d(h, w, cy, cx, sigma=2.0, amplitude=1.0):
    y, x = np.mgrid[0:h, 0:w]
    return amplitude * np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))

def make_rd_map(label, h=64, w=64):
    # Base noise
    rd = np.random.normal(0.0, 0.03, (h, w)).astype(np.float32)

    # Zero-Doppler clutter (center column)
    center = w // 2
    rd[:, center-1:center+2] += np.random.uniform(0.0, 0.05, size=(h, 3))

    # Add target (for label=1)
    if label == 1:
        if np.random.rand() > 0.15: # 85% chance detection exists
            range_bin = np.random.randint(6, h - 6)

            doppler_offset = np.random.choice([-15, -10, -6, 6, 10, 15])
            doppler_bin = np.clip(center + doppler_offset, 4, w - 5)

            rd += gaussian_2d(
                h, w,
                cy=range_bin,
                cx=doppler_bin,
                sigma=np.random.uniform(1.5, 3.0),
                amplitude = np.random.uniform(0.7, 1.2)
            )

    # False alarms
    for _ in range(np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])):
        ry = np.random.randint(4, h - 4)
        rx = np.random.randint(4, w - 4)

        rd += gaussian_2d(
            h, w,
            cy=ry,
            cx=rx,
            sigma=np.random.uniform(1.0, 2.5),
            amplitude=np.random.uniform(0.1, 0.4)
        )

    return np.clip(rd, 0.0, 1.0).astype(np.float32)

# ---------------
# Generation loop
# ---------------

for split in SPLITS:
    for label in ['0', '1']:
        ir_dir = os.path.join(IR_BASE, split, label)
        radar_dir = os.path.join(RADAR_ROOT, split, label)

        os.makedirs(radar_dir, exist_ok=True)

        if not os.path.exists(ir_dir):
            continue

        for fname in os.listdir(ir_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base = os.path.splitext(fname)[0]
            rd = make_rd_map(int(label), RANGE_BINS, DOPPLER_BINS)

            np.save(os.path.join(radar_dir, base + '.npy'), rd)

print('RD radar dataset generated.')

