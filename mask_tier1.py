import os
import numpy as np
from PIL import Image

OLD_MASK_DIR = "Dataset/masks"
NEW_MASK_DIR = "Dataset/masks_tier1"

os.makedirs(NEW_MASK_DIR, exist_ok=True)

for fname in os.listdir(OLD_MASK_DIR):
    if not fname.endswith(".png"):
        continue

    mask = np.array(Image.open(os.path.join(OLD_MASK_DIR, fname)))

    new_mask = np.zeros_like(mask)

    # Built-up
    new_mask[(mask == 1) | (mask == 2)] = 1

    # Green
    new_mask[(mask == 3) | (mask == 4)] = 2

    Image.fromarray(new_mask, mode="L").save(
        os.path.join(NEW_MASK_DIR, fname)
    )

print("Tier 1 masks created successfully.\n")

print(np.unique(new_mask))

