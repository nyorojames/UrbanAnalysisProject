import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from skimage.draw import polygon

# =====================================================
# PATHS
# =====================================================

COCO_ANNOTATION_FILE = "Dataset/train/_annotations.coco.json"
IMAGE_DIR = "Dataset/train"
MASK_DIR = "Dataset/masks"

os.makedirs(MASK_DIR, exist_ok=True)

# =====================================================
# CLASS DEFINITIONS (FIXED ORDER)
# =====================================================
# NOTE:
# 0 = background (mandatory for semantic segmentation)

CLASS_NAMES = [
    "background",     # 0
    "building",       # 1
    "road",           # 2
    "tree",           # 3
    "grass",          # 4
]

NUM_CLASSES = len(CLASS_NAMES)

# Explicit name → index mapping (NO guessing, NO enumeration)
NAME_TO_CLASS_IDX = {
    "building": 1,
    "road": 2, # 1
    "tree": 3, # 2
    "grass": 4 # 2
}

# =====================================================
# LOAD COCO DATASET
# =====================================================

coco = COCO(COCO_ANNOTATION_FILE)

cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)

# Map COCO category IDs → fixed class indices
cat_id_to_class_idx = {}

for cat in categories:
    cat_name = cat["name"]
    if cat_name in NAME_TO_CLASS_IDX:
        cat_id_to_class_idx[cat["id"]] = NAME_TO_CLASS_IDX[cat_name]
    else:
        # Any unknown category goes to background
        cat_id_to_class_idx[cat["id"]] = 0

print("\nCOCO → CLASS INDEX MAPPING")
for k, v in cat_id_to_class_idx.items():
    print(f"COCO category ID {k} → class index {v}")

# =====================================================
# PROCESS IMAGES
# =====================================================

image_ids = coco.getImgIds()

for image_id in image_ids:

    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(IMAGE_DIR, image_info["file_name"])

    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path)
    width, height = image.size

    # Initialize empty mask (background = 0)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Load annotations for this image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)

    # =================================================
    # DRAW EACH INSTANCE ON MASK
    # =================================================

    for ann in annotations:

        if "segmentation" not in ann:
            continue

        coco_cat_id = ann["category_id"]
        class_idx = cat_id_to_class_idx.get(coco_cat_id, 0)

        # --- Polygon segmentation ---
        if isinstance(ann["segmentation"], list):
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((-1, 2))
                rr, cc = polygon(
                    poly[:, 1],
                    poly[:, 0],
                    shape=(height, width)
                )
                mask[rr, cc] = class_idx

        # --- RLE segmentation ---
        else:
            rle = maskUtils.frPyObjects(
                ann["segmentation"], height, width
            )
            m = maskUtils.decode(rle)
            mask[m == 1] = class_idx

    # =================================================
    # SAVE MASK
    # =================================================

    mask_image = Image.fromarray(mask, mode="L")

    base_name = os.path.splitext(image_info["file_name"])[0]
    output_path = os.path.join(MASK_DIR, f"{base_name}.png")

    mask_image.save(output_path)
    print(f"Saved mask: {output_path}")

print("\nMask generation completed successfully.")


# Class Verification
values = set()

for f in os.listdir("Dataset/masks"):
    m = np.array(Image.open(os.path.join("Dataset/masks", f)))
    values.update(np.unique(m))

print("\nAll values in dataset:", sorted(values))
