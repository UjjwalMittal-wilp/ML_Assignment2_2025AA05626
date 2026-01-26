Breast Cancer Classification using Machine Learning
--
a. Problem Statement
--
The objective of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is malignant or benign based on diagnostic features. The project also demonstrates deployment of these models using an interactive Streamlit web application.

----------
b. Dataset Description
--

The Breast Cancer Wisconsin (Diagnostic) dataset is obtained from the UCI Machine Learning Repository and is available through scikit-learn. The dataset contains 569 patient records with 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses.

Each instance is labeled as either:

 - Malignant (1)

 - Benign (0)

There are no missing values in the dataset. All features are continuous, making the dataset well-suited for training a wide range of machine learning classification algorithms.

------------------
c. Models Used and Evaluation Metrics
--
The following six classification models were implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

For each model, the following evaluation metrics were computed:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
-----
d. Model Comparison Table
--

| Model Name          | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------------|--|--|--|--|--|--|
| Logistic Regression |1.0|1.0|1.0|1.0|1.0|1.0|
| Decision Tree       |1.0|1.0|1.0|1.0|1.0|1.0|
| KNN                 |1.0|1.0|1.0|1.0|1.0|1.0|
| Naive Bayes         |1.0|1.0|1.0|1.0|1.0|1.0|
| Random Forest       |1.0|1.0|1.0|1.0|1.0|1.0|
|  XGBoost            |0.63|1.0|0.63|1.0|0.77|0.0|
-----
e. Observation
--
| ML Model            | Observation about model performance                                                                                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved perfect performance across all metrics, indicating that the dataset is highly linearly separable and that Logistic Regression is well-suited for this problem.                                          |
| Decision Tree       | Also obtained perfect scores, suggesting that the tree was able to capture decision boundaries very effectively on this dataset. However, such performance may indicate possible overfitting.                    |
| KNN                 | Delivered perfect results after feature scaling, showing that neighborhood-based classification works extremely well for this dataset.                                                                           |
| Naive Bayes         | Achieved perfect performance, implying that feature distributions align well with Gaussian assumptions for this dataset.                                                                                         |
| Random Forest       | Obtained perfect scores, demonstrating strong ensemble learning capability and excellent generalization performance.                                                                                             |
| XGBoost             | Achieved perfect AUC and recall but lower accuracy and precision, indicating that while it correctly identifies most malignant cases, it misclassifies some benign cases, resulting in reduced overall accuracy. |

