# Credit Card Fraud Detection (Balanced Dataset)

A binary classification project detecting fraudulent credit card transactions using multiple ML models, including Logistic Regression, Random Forest, SVM, KNN, and a Neural Network.  
Dataset is balanced, cleaned, and scaled. All experiments compare performance with and without key features (`Is_International`, `Location_Distance`).

---

## 📊 Dataset
- **Type:** Balanced (Fraud = Non-Fraud)
- **Rows:** ~2000 samples  
- **Key features:** Transaction amount, Merchant type, Device type, V1–V5 (PCA-derived), Is_International, Location_Distance  
- **Target:** Fraud / Non-Fraud (binary)

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Cleaned columns, encoded categorical features
   - Split: 80% train / 20% test (stratified)
   - StandardScaler applied to numeric data

2. **Models Trained**
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Random Forest  
   - Neural Network (TensorFlow Sequential Model)

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Confusion Matrix for each model

---

## 🧠 Results Summary

| Model               | With Features (%) | Without Features (%) |
|----------------------|------------------:|----------------------:|
| SVM                 | 99.95             | 99.70 |
| KNN                 | 99.65             | 99.50 |
| Random Forest       | 99.70             | 99.70 |
| Logistic Regression | 99.85             | 99.85 |
| Neural Network      | 99.85             | 99.80 |

**Insight:**  
`Is_International` and `Location_Distance` strongly influence fraud detection.  
Removing them slightly reduces performance across all models.

---

## 🧩 Tools & Libraries
- Python 3.x  
- Pandas, NumPy  
- scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## 🚀 Conclusion
All models achieved near-perfect accuracy due to strong feature separability.  
The neural network matched classical models, confirming the dataset’s clarity.  
Future work: test on an **imbalanced, real-world dataset** to assess true performance robustness.

---

## 📁 Structure
├─ Fraud_Detection.ipynb # Main notebook (EDA → Models → Evaluation)
├─ data/ # Dataset (processed sample)
├─ reports/figures/ # Plots and accuracy comparison
└─ README.md
