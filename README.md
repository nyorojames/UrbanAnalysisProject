# 🏙️ Quantifying Built-up and Green Space Ratios in Urban Areas Using Semantic Segmentation

## 📄 Project Overview

This project focuses on automating the analysis of urban environments by quantifying the proportions of **Built-up Areas** versus **Green Spaces**. Motivated by the rapid urban development in Samsun, Turkey, the study investigates how much land is occupied by constructed surfaces compared to vegetation.

Using deep learning-based semantic segmentation, the system processes satellite imagery to distinguish between these land cover types, offering a feasibility demonstration for automated urban planning analysis.

## 🧠 Model Architecture & Methodology

- **Architecture:** U-Net 
- **Backbone:** ResNet-34 (pretrained on ImageNet) 
- **Framework:** TensorFlow (tf.keras) 
- **Data Source:** Google Earth satellite screenshots (approx. 100 images of Samsun).
- **Annotation:** Manually annotated using Roboflow.
- **Class Definitions:** \* **Built-up:** Merged class containing buildings and roads.
 \* **Green:** Merged class containing trees and grass

## 📂 Dataset

The dataset was manually annotated and processed using Roboflow.

### 🔧 Key Techniques

- **Dataset Tiling:** Large screenshots were split into smaller tiles using a custom Python script to increase sample count and preserve spatial detail.
- **Stride Optimization:** A stride of **192 pixels** was found to offer the best balance between numerical IoU and visual coherence.

## 📊 Results and Evaluation

The model was evaluated using both the **Mean Intersection over Union (IoU)** metric and visual overlay inspection.

### Performance Metrics

- **Built-up IoU:** 0.45 
- **Green IoU:** 0.60 
- **Mean IoU:** 0.52 (excluding background) 

### Visual Validation

Visual overlays confirmed that major built-up areas and road networks were detected reliably. While a smaller stride of 128 produced a higher numerical IoU (0.66), it resulted in noisier predictions, leading to the selection of the 192-stride model.

### 🌍 Urban Analysis (Ratio Estimation)

The final model successfully estimated land-cover ratios from test images. For example:

- **Image 1:** 85.27% Built-up | 14.73% Green 
- **Image 2:** 99.97% Built-up | 0.03% Green 
- **Image 3:** 0.00% Built-up | 100.00% Green 

<img width="1927" height="1790" alt="image" src="https://github.com/user-attachments/assets/9a642067-8997-4c87-9964-3a55b601a3a1" />

## 🚀 Innovation

- **Ratio-Based Analysis:** Moving beyond simple segmentation masks to actionable urban metrics.
- **Empirical Refinement:** Iteratively merging fine-grained classes (e.g., combining "road" and "building") to improve segmentation stability.
- **Visual-First Evaluation:** Prioritizing visual coherence over raw IoU scores to ensure practical usability.

## 🔮 Future Work

- **Dataset Expansion:** Incorporating additional satellite imagery to improve robustness.
- **Multi-City Testing:** quantitatively evaluating the model on cities beyond Samsun.
