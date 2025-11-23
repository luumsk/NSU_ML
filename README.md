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

Each lecture = **2-hour seminar + 2-hour lab** (1 academic hour = 45 minutes)

| **Lecture** | **Topic** | **Seminar** | **Lab ** |
|--------------|------------|---------------------------|---------------|
| **1** | **Introduction to Machine Learning** | What is ML? History and categories (supervised, unsupervised, reinforcement, self-supervised). Difference between AI/ML/DL. ML pipeline: data → model → evaluation → deployment. Types of data (tabular, image, text). Real-world applications. | Use scikit-learn toy datasets; load and inspect data (Iris, Boston). Plot basic features. |
| **2** | **Linear & Nonlinear Regression** | Simple & multiple linear regression, ordinary least squares, gradient descent. Polynomial regression, overfitting, regularization (L1, L2). Model evaluation metrics (MSE, RMSE, R²). | Implement linear regression from scratch + scikit-learn regression example. |
| **3** | **Logistic Regression and Neural Network** | Sigmoid function, log-loss, perceptron, multilayer perceptron (MLP), neural network. Metrics: precision, recall, F1, ROC-AUC, confusion matrix. | Implement logistic regression + evaluate with scikit-learn metrics. |
| **4** | **Decision Trees** | Splitting criteria (Gini, entropy), information gain, recursive partitioning, overfitting and pruning, interpreting trees. | Build and visualize decision trees; inspect structure and decision boundaries. |
| **5** | **Random Forests & Bagging** | Bootstrap sampling, random subspace method, OOB score, variance reduction, feature importance. | Train Random Forests, compare with single trees, analyze feature importance.|
| **6** | **Boosting Methods** | AdaBoost mechanism, Gradient Boosting, XGBoost/LightGBM/CatBoost concepts, bias–variance behavior, boosting vs. bagging. | Train boosting models and compare performance across algorithms. |
| **7** | **k-NN & Distance-Based Learning** | Distance metrics, weighted k-NN, feature scaling, curse of dimensionality, KD/Ball trees (conceptual). | Apply k-NN, visualize decision boundaries, show effects of scaling and k. |
| **8** | **Support Vector Machines** | Margins, support vectors, soft vs. hard margin, kernel trick, RBF and polynomial kernels, role of C and gamma. | Train SVM models, visualize margins and kernel effects. |
| **9** | **Naive Bayes & Probabilistic Models** | Bayes theorem, Gaussian NB, Multinomial NB, likelihood/prior/posterior, comparison with logistic regression and SVM. | Implement NB for text or email classification and compare models. |
| **10** | **Clustering Methods** | k-Means, k-Medoids, hierarchical clustering, DBSCAN, cluster validity metrics (silhouette, Davies–Bouldin), strengths and limitations. | Apply clustering to real datasets and visualize clusters in 2D/3D. |
| **11** | **Dimensionality Reduction** | PCA (covariance matrix, eigenvalues, explained variance), SVD, t-SNE/UMAP concepts, interpreting low-dimensional embeddings. | Implement PCA and visualize 2D/3D projections. |
| **12** | **Final Presentations & Course Summary** | Review of all ML algorithms, exam preparation, Q&A, and student project presentations. | S(No lab activity). |                                        |



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
