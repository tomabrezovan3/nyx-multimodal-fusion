import os
import numpy as np
from PIL import Image

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

def add_gaussian_noise(img, std=0.03):
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
        radius = np.random.randint(2, 6)
        value = np.random.uniform(0.15, 0.35)
        draw_circle(img, cy, cx, radius, value)

def draw_human_shape(img):
    """
    Draw a simple thermal human-like silhouette:
    - head (circle)
    - torso (rectangle)
    - legs (rectangles)
    """
    h, w = img.shape

    # Choose a center position with margins
    center_x = np.random.randint(w // 3, 2 * w // 3)
    top_y = np.random.randint(h // 5, h // 2)

    # Random size variation
    head_radius = np.random.randint(5, 8)
    torso_height = np.random.randint(18, 28)
    torso_width = np.random.randint(10, 16)
    leg_height = np.random.randint(16, 24)
    leg_width = np.random.randint(4, 6)

    heat_head = np.random.uniform(0.75, 0.95)
    heat_body = np.random.uniform(0.6, 0.85)

    # Head
    head_cy = top_y
    head_cx = center_x
    draw_circle(img, head_cy, head_cx, head_radius, heat_head)

    # Torso
    torso_y1 = head_cy + head_radius
    torso_y2 = torso_y1 + torso_height
    torso_x1 = center_x - torso_width // 2
    torso_x2 = center_x + torso_width // 2
    draw_rectangle(img, torso_y1, torso_x1, torso_y2, torso_x2, heat_body)

    # Legs
    leg_gap = 2
    left_leg_x1 = center_x - leg_gap - leg_width
    left_leg_x2 = center_x - leg_gap
    right_leg_x1 = center_x + leg_gap
    right_leg_x2 = center_x + leg_gap + leg_width

    leg_y1 = torso_y2
    leg_y2 = torso_y2 + leg_height

    draw_rectangle(img, leg_y1, left_leg_x1, leg_y2, left_leg_x2, heat_body)
    draw_rectangle(img, leg_y1, right_leg_x1, leg_y2, right_leg_x2, heat_body)

    # Optional arms
    if np.random.rand() < 0.7:
        arm_y1 = torso_y1 + 4
        arm_y2 = arm_y1 + 4
        left_arm_x1 = torso_x1 - 10
        left_arm_x2 = torso_x1
        right_arm_x1 = torso_x2
        right_arm_x2 = torso_x2 + 10

        draw_rectangle(img, arm_y1, left_arm_x1, arm_y2, left_arm_x2, heat_body * 0.9)
        draw_rectangle(img, arm_y1, right_arm_x1, arm_y2, right_arm_x2, heat_body * 0.9)

def make_ir_image(label, size=128):
    """
    Make one synthetic thermal image
    label=0 -> no human
    label=1 -> human present
    """
    img = np.zeros((size, size), dtype=np.float32)

    # Slight background gradient / ambient variation
    base_level = np.random.uniform(0.02, 0.08)
    img[:] = base_level

    # Add small random hotspots in both classes
    add_background_hotspots(img, num_spots=np.random.randint(1, 4))

    # Add human target only for class 1
    if label == 1:
        draw_human_shape(img)

    # Add sensor noise
    img = add_gaussian_noise(img, std=0.025)

    # Clip to valid thermal-like intensity range
    img = np.clip(img, 0.0, 1.0)

    # Convert to 8-bit grayscale image
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

print('Synthetic IR dataset generated.')