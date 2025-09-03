# Optimizing Traffic Sign Recognition: EfficientNet vs DenseNet

## Overview
This project compares the performance of **EfficientNet-B0** and **DenseNet-121** models for **traffic sign recognition**, a critical component of autonomous driving systems.  
Both models are evaluated on a curated traffic sign dataset with **58 distinct classes**, using **Convolutional Neural Networks (CNNs)**, transfer learning, and data augmentation.

## Objectives
- Develop robust deep learning models for **traffic sign classification**.  
- Compare **EfficientNet-B0** and **DenseNet-121** in terms of accuracy, loss, and generalization ability.  
- Assess performance using metrics like **accuracy, precision, recall, F1-score, and confusion matrix**.  

## Dataset
- **Classes:** 58 traffic sign categories.  
- **Samples:** ~120 images per class.  
- **Split:** Training (~80%), Testing (~20%).  
- **Format:** Images with labels.csv for mapping.  

## Technologies & Tools
- **Languages:** Python  
- **Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-learn  
- **Models:** DenseNet-121, EfficientNet-B0  
- **Optimizer:** Adam  

## Methodology
1. **Preprocessing**: Image resizing, normalization, and augmentation (rotation, flipping, scaling).  
2. **Transfer Learning**: Pre-trained ImageNet weights for both models.  
3. **Training**: Adam optimizer with tuned hyperparameters.  
4. **Evaluation**: Accuracy, loss curves, confusion matrices, ROC analysis.  

##  Results
- **DenseNet-121**  
  - Accuracy: **93.43%**  
  - Test Loss: **0.1931**  
  - Strong generalization and consistent validation accuracy (>90%).  

- **EfficientNet-B0**  
  - Validation Accuracy: ~**36.4%**  
  - Showed limitations in generalizing to unseen data.  

**Conclusion:** DenseNet-121 outperformed EfficientNet-B0 in accuracy, precision, recall, and F1-score, making it better suited for traffic sign recognition tasks.  

## Future Work
- Explore deeper variants (DenseNet-169, DenseNet-201).  
- Experiment with hyperparameter tuning and advanced augmentation.  
- Integrate real-time detection for autonomous vehicles.  

## Authors
- **Abhinay Babu M** – VIT-AP University
- **Guide :**  Kalyani S

---
🔗 This project demonstrates the effectiveness of **CNN architectures** in optimizing traffic sign recognition and contributes towards safer **autonomous driving systems**.
