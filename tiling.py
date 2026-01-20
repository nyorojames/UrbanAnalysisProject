import os
from PIL import Image
import numpy as np

IMAGE_DIR = "Dataset/train"
MASK_DIR  = "Dataset/masks_tier1"

OUT_IMG_DIR  = "/content/drive/MyDrive/Veg__detection/Dataset_tiled/images"
OUT_MASK_DIR = "/content/drive/MyDrive/Veg__detection/Dataset_tiled/masks"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

TILE_SIZE = 384 # 256
STRIDE = 192 # 256

# -------------------------------------------------
# Build mask lookup using shared prefix
# -------------------------------------------------
def get_prefix(name):
    return name.split(".rf.")[0]

mask_lookup = {}
for m in os.listdir(MASK_DIR):
    mask_lookup[get_prefix(m)] = m

print("Masks indexed:", len(mask_lookup))

# -------------------------------------------------
# Tiling
# -------------------------------------------------
tile_count = 0

for img_name in os.listdir(IMAGE_DIR):
    prefix = get_prefix(img_name)

    if prefix not in mask_lookup:
        print("❌ No mask for:", img_name)
        continue

    img_path  = os.path.join(IMAGE_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, mask_lookup[prefix])

    image = Image.open(img_path).convert("RGB")
    mask  = Image.open(mask_path)

    w, h = image.size

    for y in range(0, h - TILE_SIZE + 1, STRIDE):
        for x in range(0, w - TILE_SIZE + 1, STRIDE):
            img_tile  = image.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            mask_tile = mask.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

            tile_name = f"{prefix}_{y}_{x}.png"

            img_tile.save(os.path.join(OUT_IMG_DIR, tile_name))
            mask_tile.save(os.path.join(OUT_MASK_DIR, tile_name))

            tile_count += 1

print("✅ Total tiles created:", tile_count)
