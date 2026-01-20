# =====================================================
# FRAMEWORK CONFIGURATION (IMPORTANT)
# =====================================================
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

# =====================================================
# IMPORTS
# =====================================================
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split

import segmentation_models as sm

# =====================================================
# MODEL CONFIGURATION
# =====================================================
ARCH     = "unet" # "unet" | "unetpp" | "deeplabv3"
BACKBONE = "resnet34" # "resnet34" | "resnet152" | "efficientnetb3"

# =====================================================
# DATASET SETTINGS
# =====================================================
IMAGE_DIR = "Dataset/train"
#MASK_DIR  = "Dataset/masks"
MASK_DIR = "Dataset/masks_tier1"

IMG_SIZE   = 448        
BATCH_SIZE = 2
EPOCHS     = 100

SEED      = 42
VAL_SPLIT = 0.20

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =====================================================
# CLASS DEFINITIONS (MUST MATCH MASK VALUES)
# =====================================================
# 0 = background
# 1 = building
# 2 = road
# 3 = tree
# 4 = grass

# CLASS_NAMES = [
#   "background",
#   "building",
#   "road",
#   "tree",
#   "grass"
# ]

CLASS_NAMES = [
    "background",
    "built_up",
    "green"
]

NUM_CLASSES = len(CLASS_NAMES)

# =====================================================
# DATA GENERATOR WITH LIGHT AUGMENTATION
# =====================================================
class DataGenerator(Sequence):
    def __init__(self, files, batch_size, img_size, num_classes, augment=False):
        self.files = files
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []

        for f in batch:
            base = os.path.splitext(f)[0]

            # Load image
            img = load_img(
                os.path.join(IMAGE_DIR, f),
                target_size=(self.img_size, self.img_size)
            )
            img = img_to_array(img).astype(np.float32) / 255.0

            # Load mask
            mask = load_img(
                os.path.join(MASK_DIR, base + ".png"),
                color_mode="grayscale",
                target_size=(self.img_size, self.img_size)
            )
            mask = img_to_array(mask).squeeze().astype(np.uint8)

            # Data augmentation (same transform for image + mask)
            # if self.augment:
            #     if np.random.rand() < 0.5:
            #         img  = np.fliplr(img)
            #         mask = np.fliplr(mask)
            #     if np.random.rand() < 0.5:
            #         img  = np.flipud(img)
            #         mask = np.flipud(mask)
            if self.augment:
                k = np.random.randint(0, 4)
                img  = np.rot90(img, k)
                mask = np.rot90(mask, k)

            # One-hot encode mask
            mask = np.clip(mask, 0, NUM_CLASSES - 1)
            mask = tf.keras.utils.to_categorical(mask, self.num_classes)

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)

# =====================================================
# LOSS FUNCTIONS (FOCAL DICE + CE)
# =====================================================
def focal_dice_loss(y_true, y_pred, gamma=2.0, smooth=1e-6):
    y_true = K.reshape(y_true, (-1, NUM_CLASSES))
    y_pred = K.reshape(y_pred, (-1, NUM_CLASSES))

    intersection = K.sum(y_true * y_pred, axis=0)
    union = K.sum(y_true + y_pred, axis=0)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    focal = K.pow(1.0 - dice, gamma)

    return K.mean(focal[1:])  # exclude background


def final_loss(y_true, y_pred):
    ce = tf.keras.losses.categorical_crossentropy(
    y_true, y_pred,
    label_smoothing=0.05
    )
    return tf.reduce_mean(ce) + focal_dice_loss(y_true, y_pred)

# =====================================================
# METRICS (NO BACKGROUND)
# =====================================================
class MeanIoU_NoBG(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="miou_no_bg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(
            name="conf_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        cm = tf.math.confusion_matrix(
            tf.reshape(y_true, [-1]),
            tf.reshape(y_pred, [-1]),
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        self.cm.assign_add(cm)

    def result(self):
        diag = tf.linalg.diag_part(self.cm)
        denom = tf.reduce_sum(self.cm, 0) + tf.reduce_sum(self.cm, 1) - diag
        iou = tf.math.divide_no_nan(diag, denom)
        return tf.reduce_mean(iou[1:])

    def reset_states(self):
        self.cm.assign(tf.zeros_like(self.cm))

# =====================================================
# MODEL CREATION
# =====================================================
def build_model():
    model = sm.Unet(
        BACKBONE,
        classes=NUM_CLASSES,
        activation="softmax",
        encoder_weights="imagenet"
    )
    return model

# =====================================================
# DATASET SPLIT
# =====================================================
files = [f for f in os.listdir(IMAGE_DIR)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))]

train_files, val_files = train_test_split(
    files, test_size=VAL_SPLIT, random_state=SEED
)

train_gen = DataGenerator(train_files, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, augment=True)
val_gen   = DataGenerator(val_files, BATCH_SIZE, IMG_SIZE, NUM_CLASSES, augment=False)

print(f"Train images: {len(train_files)} | Validation images: {len(val_files)}")

# =====================================================
# BUILD + FREEZE ENCODER
# =====================================================
model = build_model()

# for layer in model.layers:
#     if "encoder" in layer.name:
#         layer.trainable = False
for layer in model.layers:
    if "stage4" in layer.name or "stage5" in layer.name:
        layer.trainable = True
    elif "encoder" in layer.name:
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss=final_loss,
    metrics=[MeanIoU_NoBG(NUM_CLASSES)]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================
callbacks = [
    ModelCheckpoint(
        "best_unet_resnet34.h5",
        monitor="val_miou_no_bg",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_miou_no_bg",
        mode="max",
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_miou_no_bg",
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

# =====================================================
# TRAIN
# =====================================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================================
# CLASS-WISE IoU EVALUATION
# =====================================================
def evaluate_classwise_iou(model, data_gen, class_names):
    num_classes = len(class_names)
    total_inter = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    for i in range(len(data_gen)):
        imgs, masks = data_gen[i]
        preds = model.predict(imgs, verbose=0)

        pred_lbl = np.argmax(preds, axis=-1)
        true_lbl = np.argmax(masks, axis=-1)

        for c in range(num_classes):
            total_inter[c] += np.logical_and(pred_lbl == c, true_lbl == c).sum()
            total_union[c] += np.logical_or(pred_lbl == c, true_lbl == c).sum()

    ious = total_inter / (total_union + 1e-7)

    print("\n=== CLASS-WISE IoU ===")
    for i, name in enumerate(class_names):
        print(f"{name:15s}: {ious[i]:.4f}")

    print(f"\nMean IoU (no background): {np.mean(ious[1:]):.4f}")
    return ious


evaluate_classwise_iou(model, val_gen, CLASS_NAMES)

print("\nTRAINING AND EVALUATION COMPLETED")
