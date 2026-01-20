# 🏙️ Quantifying Built-up and Green Space Ratios in Urban Areas Using Semantic Segmentation

## 📄 Project Overview

[cite_start]This project focuses on automating the analysis of urban environments by quantifying the proportions of **Built-up Areas** versus **Green Spaces**[cite: 9]. [cite_start]Motivated by the rapid urban development in Samsun, Turkey, the study investigates how much land is occupied by constructed surfaces compared to vegetation[cite: 10].

[cite_start]Using deep learning-based semantic segmentation, the system processes satellite imagery to distinguish between these land cover types, offering a feasibility demonstration for automated urban planning analysis[cite: 19].

## 👥 Project Team

- [cite_start]**Name:** James Nyoro [cite: 4]
- [cite_start]**Role:** Data collection, annotation, model development, evaluation, and reporting [cite: 7]

## 🧠 Model Architecture & Methodology

- [cite_start]**Architecture:** U-Net [cite: 35]
- [cite_start]**Backbone:** ResNet-34 (pretrained on ImageNet) [cite: 36]
- [cite_start]**Framework:** TensorFlow (tf.keras) [cite: 37]
- [cite_start]**Data Source:** Google Earth satellite screenshots (approx. 100 images of Samsun)[cite: 25, 30].
- [cite_start]**Annotation:** Manually annotated using Roboflow[cite: 30].
- [cite_start]**Class Definitions:** \* **Built-up:** Merged class containing buildings and roads[cite: 32].
  - [cite_start]**Green:** Merged class containing trees and grass[cite: 32].

## 📂 Dataset

The dataset was manually annotated and processed using Roboflow.
[View the Full Dataset on Roboflow Universe](urban-vegetation-project-2)

### 🔧 Key Techniques

- [cite_start]**Dataset Tiling:** Large screenshots were split into smaller tiles using a custom Python script to increase sample count and preserve spatial detail[cite: 40, 41].
- [cite_start]**Stride Optimization:** A stride of **192 pixels** was found to offer the best balance between numerical IoU and visual coherence[cite: 47, 53].

## 📊 Results and Evaluation

[cite_start]The model was evaluated using both the **Mean Intersection over Union (IoU)** metric and visual overlay inspection[cite: 38, 54].

### Performance Metrics

- [cite_start]**Built-up IoU:** 0.45 [cite: 48]
- [cite_start]**Green IoU:** 0.60 [cite: 49]
- [cite_start]**Mean IoU:** 0.52 (excluding background) [cite: 50]

### Visual Validation

[cite_start]Visual overlays confirmed that major built-up areas and road networks were detected reliably[cite: 56]. [cite_start]While a smaller stride of 128 produced a higher numerical IoU (0.66), it resulted in noisier predictions, leading to the selection of the 192-stride model[cite: 51, 52, 53].

### 🌍 Urban Analysis (Ratio Estimation)

[cite_start]The final model successfully estimated land-cover ratios from test images[cite: 58]. For example:

- **Image 1:** 85.27% Built-up | [cite_start]14.73% Green [cite: 60, 67]
- **Image 2:** 99.97% Built-up | [cite_start]0.03% Green [cite: 67]
- **Image 3:** 0.00% Built-up | [cite_start]100.00% Green [cite: 67]

![Segmentation Results](https://raw.githubusercontent.com/YOUR-USERNAME/Urban-Analysis-Project/main/path/to/your/image.png)
_(Note: Upload the segmentation image from your report here to show the overlay visualization)_

## 🚀 Innovation

- [cite_start]**Ratio-Based Analysis:** Moving beyond simple segmentation masks to actionable urban metrics[cite: 70].
- [cite_start]**Empirical Refinement:** Iteratively merging fine-grained classes (e.g., combining "road" and "building") to improve segmentation stability[cite: 32, 71].
- [cite_start]**Visual-First Evaluation:** Prioritizing visual coherence over raw IoU scores to ensure practical usability[cite: 73, 74].

## 🔮 Future Work

- [cite_start]**Dataset Expansion:** Incorporating additional satellite imagery to improve robustness[cite: 77].
- [cite_start]**Multi-City Testing:** quantitatively evaluating the model on cities beyond Samsun[cite: 78].
