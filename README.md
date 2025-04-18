# ML-Based Activity Prediction for *Salmonella Typhi* Compounds

This repository contains scripts and resources for building machine learning models to predict biological activity against *Salmonella Typhi* using molecular descriptors.

## 🧪 Objective

The goal of this project was to develop predictive models for compound activity against *Salmonella Typhi*, leveraging cheminformatics tools and machine learning algorithms. This was achieved through feature selection, model training, and evaluation using curated descriptors.

---

## 📁 Workflow Overview

### 1. **Dataset Acquisition**
- The bioactivity dataset was downloaded from the [ChEMBL database](https://www.ebi.ac.uk/chembl/).
- Data included experimental MIC values and compound structures (SMILES).

### 2. **Data Preparation**
- The raw ChEMBL dataset was processed using the `S_typhi_dataset.py` script.
- This script cleaned and formatted the dataset, generated classification labels based on MIC thresholds, and exported the final dataset for descriptor generation.

### 3. **Descriptor Generation**
- Molecular structures were converted to `.mol` format.
- Descriptors and fingerprints were generated using **PaDEL-Descriptor**, a Java-based software.
    - Both 1D/2D descriptors and PubChem fingerprints were calculated.

### 4. **Merging Descriptor Files**
- Output from PaDEL was combined into a single dataset using the `dataset_merge.py` script.
- This script also linked the descriptors to activity labels and SMILES for each compound.

### 5. **Train-Test Split**
- The merged dataset was split into training and testing sets (commonly using an 80:20 ratio).
- The splits were stratified to preserve class balance.

### 6. **Feature Selection**
- Various feature selection methods were applied to the training dataset using the `feature_selection.py` script.
- Techniques included:
    - Fisher Score
    - Random Forest Feature Importance
    - Weighted Random Forest
    - Principal Component Analysis (PCA)
    - Weighted PCA (wPCA)
- The selected features were saved for use in modeling.

### 7. **Model Training and Evaluation**
- Machine learning models were trained using the `ml_model_training.py` script.
- The following classifiers were used:
    - k-Nearest Neighbors (kNN)
    - Support Vector Machines (SVM)
    - Decision Trees
    - Random Forest
    - Naive Bayes
    - Multi-layer Perceptron (MLP)
- Evaluation metrics included:
    - Accuracy
    - F1-score
    - Matthews Correlation Coefficient (MCC)
    - Area Under the Curve (AUC)
    - Confusion Matrix Visualizations

---

---

## 📌 Notes

- MIC cutoffs were initially used to classify compounds as active/inactive.
- Multiple feature sets were compared to evaluate their effect on model performance.

---

## 🔧 Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- RDKit (for structure handling)
- Java (for running PaDEL-Descriptor)

---

## 📬 Contact

For questions or collaborations, feel free to reach out or open an issue!

---

