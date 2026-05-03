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
    rd = np.random.normal(0.0, 0.04, (h, w)).astype(np.float32)

    # Stronger zero-Doppler clutter
    center = w // 2
    rd[:, center -2:center + 3] += np.random.uniform(0.0, 0.10, size=(h, 5))

    # Positive class target with lower reliability
    if label == 1:
        if np.random.rand() > 0.40: # 60% detection probability
            range_bin = np.random.randint(8, h - 8)

            doppler_offset = np.random.choice([-18, -12, -8, -5, 5, 8, 12, 18])
            doppler_bin = np.clip(center + doppler_offset, 4, w - 5)

            rd += gaussian_2d(
                h, w,
                cy=range_bin,
                cx=doppler_bin,
                sigma=np.random.uniform(1.5, 3.5),
                amplitude=np.random.uniform(0.4, 0.8)
            )

    # More false alarms
    num_false = np.random.choice([1, 2, 3,], p=[0.3, 0.4, 0.3])
    for _ in range(num_false):
        ry = np.random.randint(4, h - 4)
        rx = np.random.randint(4, w - 4)

        rd += gaussian_2d(
            h, w,
            cy=ry,
            cx=rx,
            sigma=np.random.uniform(1.0, 3.0),
            amplitude=np.random.uniform(0.08, 0.35)
        )

    return np.clip(rd, 0.0, 1.0).astype(np.float32)

# --------------------
# Main generation loop
# --------------------

for split in SPLITS:
    for label in ['0', '1']:
        ir_dir = os.path.join(IR_BASE, split, label)
        radar_dir = os.path.join(RADAR_ROOT, split, label)

        os.makedirs(radar_dir, exist_ok=True)

        if not os.path.exists(ir_dir):
            print(f'Skipping missing folder: {ir_dir}')
            continue

        for fname in os.listdir(ir_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base_name = os.path.splitext(fname)[0]
            rd = make_rd_map(int(label), RANGE_BINS, DOPPLER_BINS)

            out_path = os.path.join(radar_dir, base_name + '.npy')
            np.save(out_path, rd)

print('Stage 3 synthetic RD radar dataset generated.')
