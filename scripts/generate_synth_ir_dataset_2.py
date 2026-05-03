import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# --------
# Settings
# --------

OUTPUT_ROOT = 'data_synth_ir'

SPLITS = {
    'train': 100,
    'val': 20,
    'test': 20,
}

IMAGE_SIZE = 128

# -------
# Helpers
# -------

def add_gaussian_noise(img, std=0.04):
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    img = img + noise
    return np.clip(img, 0.0, 1.0)

def draw_circle(img, cy, cx, radius, value):
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (y - cy)**2 + (x - cx)**2 <= radius**2
    img[mask] = np.maximum(img[mask], value)

def draw_rectangle(img, y1, x1, y2, x2, value):
    h, w = img.shape
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))

    if y2 > y1 and x2 > x1:
        img[y1:y2, x1:x2] = np.maximum(img[y1:y2, x1:x2], value)

def add_background_hotspots(img, num_spots=3):
    h, w = img.shape
    for _ in range(num_spots):
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        radius = np.random.randint(3, 12)
        value = np.random.uniform(0.18, 0.55)
        draw_circle(img, cy, cx, radius, value)

def draw_human_shape(img, weak=False, small=False, partial=False):
    """
    Draw a thermal human-like silhouette:
    - head
    - torso
    - legs
    - optional arms
    """
    h, w = img.shape

    center_x = np.random.randint(w // 5, 4 * w // 5)
    top_y = np.random.randint(h // 6, h // 2)

    scale = 0.65 if small else 1.0

    head_radius = int(np.random.randint(4, 8) * scale)
    torso_height = int(np.random.randint(16, 30) * scale)
    torso_width = int(np.random.randint(8, 16) * scale)
    leg_height = int(np.random.randint(14, 26) * scale)
    leg_width = max(2, int(np.random.randint(3, 6) * scale))

    if weak:
        heat_head = np.random.uniform(0.35, 0.60)
        heat_body = np.random.uniform(0.28, 0.55)
    else:
        heat_head = np.random.uniform(0.55, 0.85)
        heat_body = np.random.uniform(0.45, 0.75)

    # Head
    head_cy = top_y
    head_cx = center_x

    if not partial or np.random.rand() > 0.35:
        draw_circle(img, head_cy, head_cx, head_radius, heat_head)

    #Torso
    torso_y1 = head_cy + head_radius
    torso_y2 = torso_y1 + torso_height
    torso_x1 = center_x - torso_width // 2
    torso_x2 = center_x + torso_width // 2

    if not partial or np.random.rand() > 0.25:
        draw_rectangle(img, torso_y1, torso_x1, torso_y2, torso_x2, heat_body)

    # Legs
    leg_gap = np.random.randint(1, 4)
    left_leg_x1 = center_x - leg_gap - leg_width
    left_leg_x2 = center_x - leg_gap
    right_leg_x1 = center_x + leg_gap
    right_leg_x2 = center_x + leg_gap + leg_width

    leg_y1 = torso_y2
    leg_y2 = torso_y2 + leg_height

    if not partial or np.random.rand() > 0.45:
        draw_rectangle(img, leg_y1, left_leg_x1, leg_y2, left_leg_x2, heat_body)

    if not partial or np.random.rand() > 0.45:
        draw_rectangle(img, leg_y1, right_leg_x1, leg_y2, right_leg_x2, heat_body)

    # Optional arms
    if np.random.rand() < 0.5:
        arm_y1 = torso_y1 + np.random.randint(2, 7)
        arm_y2 = arm_y1 + np.random.randint(3, 6)
        arm_len = np.random.randint(5, 12)

        if not partial or np.random.rand() > 0.5:
            draw_rectangle(img, arm_y1, torso_x1 - arm_len, arm_y2, torso_x1, heat_body * 0.9)

        if not partial or np.random.rand() > 0.5:
            draw_rectangle(img, arm_y1, torso_x2, arm_y2, torso_x2 + arm_len, heat_body * 0.9)

def add_human_like_distractor(img):
    h, w = img.shape

    cx = np.random.randint(w // 5, 4 * w // 5)
    cy = np.random.randint(h // 5, 4 * h // 5)

    value = np.random.uniform(0.25, 0.55)

    if np.random.rand() < 0.5:
        draw_rectangle(
            img,
            cy,
            cx - np.random.randint(4, 10),
            cy + np.random.randint(15, 30),
            cx + np.random.randint(4, 10),
            value
        )
    else:
        draw_circle(img, cy, cx, np.random.randint(5, 12), value)

def add_partial_occlusion(img):
    h, w = img.shape
    occ_h = np.random.randint(20, 45)
    occ_w = np.random.randint(20, 45)
    oy = np.random.randint(0, h - occ_h)
    ox = np.random.randint(0, w - occ_w)

    # darken a block
    img[oy:oy + occ_h, ox:ox + occ_w] *= np.random.uniform(0.0, 0.20)
    return img

def make_ir_image(label, size=128):
    img = np.zeros((size, size), dtype=np.float32)

    # Dark thermal background
    base_level = np.random.uniform(0.03, 0.14)
    img[:] = base_level

    # Background clutter in both classes
    add_background_hotspots(img, num_spots=np.random.randint(5, 12))

    # Add target only in positive class
    if label == 1:
        weak = np.random.rand() < 0.65
        small = np.random.rand() < 0.45
        partial = np.random.rand() < 0.55

        draw_human_shape(
            img,
            weak=weak,
            small=small,
            partial=partial
        )

        # Partial occlusion
        if np.random.rand() < 0.65:
            img = add_partial_occlusion(img)

        # Sometimes reduce target visibility
        if np.random.rand() < 0.60:
            img *= np.random.uniform(0.55, 0.85)
    else:
        if np.random.rand() < 0.70:
            add_human_like_distractor(img)

    # Blur sometimes, in both classes
    if np.random.rand() < 0.75:
        img = gaussian_filter(img, sigma=np.random.uniform(0.8, 2.2))

    # Add sensor noise
    img = add_gaussian_noise(img, std=0.04)

    img = np.clip(img, 0.0, 1.0)
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8

# --------------------
# Main generation loop
# --------------------

for split, num_per_class in SPLITS.items():
    for label in ['0', '1']:
        out_dir = os.path.join(OUTPUT_ROOT, split, label)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(num_per_class):
            img = make_ir_image(int(label), size=IMAGE_SIZE)

            filename = f'{split}_{label}_{i:03d}.png'
            out_path = os.path.join(out_dir, filename)

            Image.fromarray(img, mode='L').save(out_path)

print('Stage 3 synthetic IR dataset generated')




