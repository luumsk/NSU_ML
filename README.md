# NSU Machine Learning Course

Welcome to the **Machine Learning Course** at Novosibirsk State University. This repository includes all course materials: lecture notes, code examples, labs, assignments, and project guidelines.

## 📚 Course Overview

This hands-on course covers essential machine learning techniques, including key algorithms like linear regression, decision trees, KNN, SVMs, and clustering. Through coding exercises and assignments, students will apply their skills to real-world data, culminating in a final project.

**Learning Outcomes:**

- Understand and implement core ML algorithms.
- Apply techniques to real-world data.
- Complete a final project showcasing practical ML skills.

## 🗂 Repository Contents

- **Lectures**: Slides and notes for each topic covered in the course.
- **Labs**: Jupyter notebooks and exercises for hands-on practice.
- **Assignments**: Weekly assignments for reinforcing the learned concepts.
- **Projects**: Guidelines and examples for the final project.
- **Resources**: Additional reading materials, research papers, and references.

## 🚀 Getting Started

To get started with the course materials:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/luumsk/NSU_ML.git
   ```
2. **Install the required dependencies**:
   Ensure you have Python 3.x and necessary libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Tools and Libraries


- Python 3.x
- Jupyter Notebook
- NumPy, pandas, scikit-learn, matplotlib

Alternatively, you can upload this notebook to Google Colab or Kaggle for easier access to powerful computational resources and a collaborative environment.

## 📅 Course Schedule

Each lecture = **2-hour seminar + 1-hour lab**

| **Lecture** | **Topic** | **Seminar Content (2h)** | **Lab (1h)** |
|--------------|------------|---------------------------|---------------|
| **1** | **Introduction to Machine Learning** | What is ML? History and categories (supervised, unsupervised, reinforcement, self-supervised). Difference between AI/ML/DL. ML pipeline: data → model → evaluation → deployment. Types of data (tabular, image, text). Real-world applications. | Use scikit-learn toy datasets; load and inspect data (Iris, Boston). Plot basic features. |
| **2** | **Data Preprocessing & Visualization** | Data quality issues: missing values, duplicates, outliers. Encoding (label, one-hot). Normalization vs. standardization. Train-test split. Visualization: histograms, pairplots, correlation heatmaps. | Hands-on data cleaning, encoding, and visualization using Pandas, Matplotlib, Seaborn. |
| **3** | **Linear & Nonlinear Regression** | Simple & multiple linear regression, matrix form, least squares derivation, gradient descent. Polynomial regression, overfitting, regularization (L1, L2). Model evaluation metrics (MSE, RMSE, R²). | Implement linear regression from scratch + scikit-learn regression example. |
| **4** | **Logistic Regression & Evaluation Metrics** | Sigmoid function, log-loss, decision boundary, odds and log-odds interpretation. Multi-class (one-vs-rest). Metrics: precision, recall, F1, ROC-AUC, confusion matrix. | Implement logistic regression + evaluate with scikit-learn metrics. |
| **5** | **Decision Trees & Random Forests** | Splitting criteria (Gini, entropy), depth control, pruning, bias vs. variance. Ensemble concept, bagging, random subspace method, OOB score, feature importance. | Build and visualize trees, compare single tree vs. random forest accuracy. |
| **6** | **k-NN & SVM** | Distance metrics (Euclidean, Manhattan). k value and curse of dimensionality. SVM: margins, support vectors, kernel trick, soft vs. hard margin, regularization parameter (C). | Classify data with k-NN and SVM; visualize decision boundaries. |
| **7** | **Naive Bayes & Probabilistic Models** | Bayes theorem refresher, Gaussian NB, Multinomial NB, independence assumption. Compare NB with logistic regression. Intro to probabilistic reasoning and prior/posterior. | Spam classification with Naive Bayes. Compare with logistic regression. |
| **8** | **Clustering Techniques** | Distance-based clustering (k-means, k-medoids), hierarchical clustering, DBSCAN. Cluster validity metrics (silhouette score, Davies-Bouldin index). | Apply clustering on real dataset (e.g., customers or health data). Visualize clusters. |
| **9** | **Dimensionality Reduction & Feature Extraction** | PCA: covariance matrix, eigenvalues/eigenvectors, explained variance. SVD. Feature projection, whitening. Intro to manifold learning (t-SNE, UMAP conceptually). | Implement PCA, visualize 2D/3D projections. |
| **10** | **Ensemble & Boosting Methods** | Bagging vs. boosting, AdaBoost mechanism, Gradient Boosting, XGBoost, LightGBM, CatBoost. Comparison of speed, interpretability, and accuracy. | Train and tune gradient boosting on classification dataset. |
| **11** | **Model Validation & Optimization** | Cross-validation (k-fold, stratified), grid search, random search, early stopping. Regularization (L1/L2), learning curves. Bias–variance tradeoff visualization. Feature selection methods. | Practice hyperparameter tuning and model comparison. |
| **12** | **Advanced Topics & Project Preparation** | Model interpretability (SHAP, LIME), feature importance, fairness, explainability. Concept of pipelines and deployment. Ethics in ML and dataset bias. Capstone project briefing. | SHAP analysis on trained model + mini project setup. |                                        |



## 🎓 Grading Policy

The final grade for the course is based on continuous assessment, a midterm examination, a final project, and a final exam.  
Grades follow the standard Russian 5-point system, where **5** = Excellent, **4** = Good, **3** = Satisfactory, and **2** = Fail.

### **Grade Components**

| **Component** | **Weight (%)** | **Description** |
|----------------|----------------|-----------------|
| **Assignments (10 total)** | **40%** | Each lecture is followed by an assignment that includes practical exercises or short analytical questions. Assignments assess understanding of key concepts and implementation skills. |
| **Midterm Exam** | **15%** | Covers the first six lectures. Includes theoretical questions and one applied problem to assess both conceptual and mathematical understanding. |
| **Final Project** | **25%** | Students select a dataset and apply the complete machine learning pipeline: data preprocessing, model training, and evaluation. A short written report (5–10 pages) and code submission are required. |
| **Final Exam** | **20%** | A comprehensive examination covering all twelve lectures. Tests both theoretical knowledge and the ability to apply ML algorithms to practical problems. |

### **Grade Conversion**

| **Total Score (%)** | **Final Grade (1–5)** | **Meaning** |
|----------------------|------------------------|--------------|
| **90–100** | **5 (Excellent)** | Deep understanding, complete and correct solutions, and high-quality project work. |
| **75–89** | **4 (Good)** | Good grasp of theory and practice, with minor inaccuracies. |
| **60–74** | **3 (Satisfactory)** | Basic understanding with partial or incomplete reasoning. |
| **Below 60** | **2 (Fail)** | Insufficient understanding or incomplete submission of required work. |

### **Notes**
- All assignments must be submitted before the respective deadlines to receive credit.  
- Late submissions may result in a deduction of up to 20% of the assignment grade.  
- Active participation and consistent engagement throughout the course are expected.  
- The final project must demonstrate an independent application of machine learning techniques and proper interpretation of results.


**Total:** 100 points → converted to the 5-point grading scale.


## 📜 License

This repository is licensed under the [MIT License](LICENSE).

## 📧 Contact

For any questions or additional information, please contact [Khue Luu] at [khue.luu@g.nsu.ru].

## Course Versions
- [2024 Edition (archived)](https://github.com/luumsk/NSU_ML/tree/24155)
- [2025 Edition (current)](https://github.com/luumsk/NSU_ML/tree/main)

Happy Learning! 🚀
