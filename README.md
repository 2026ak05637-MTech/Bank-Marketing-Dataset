# Bank Marketing Dataset - Classification Project

## a. Problem Statement
The objective of this project is to predict whether a client will subscribe to a term deposit (indicated by the target variable `y` - 'yes' or 'no') based on demographic and marketing campaign data. This is a classic binary classification problem characterized by highly imbalanced class distributions, which requires training robust machine learning models and evaluating them using comprehensive metrics beyond simple accuracy (such as ROC AUC, F1-Score, and Matthew's Correlation Coefficient).

## b. Dataset Description
The dataset used is the **Bank Marketing** dataset from the UCI Machine Learning Repository. It contains information about direct marketing campaigns (phone calls) conducted by a Portuguese banking institution.

- **Number of instances:** 4,521
- **Number of features:** 16 input features + 1 target variable (`y`)
- **Features include:**
  - **Demographic data:** Age, Job, Marital Status, Education.
  - **Financial history:** Default status, Average annual balance, Housing loan status, Personal loan status.
  - **Campaign details:** Contact communication type, last contact day of the month, last contact month, last contact duration, number of contacts performed during this campaign.
  - **Target Variable (`y`):** Has the client subscribed to a term deposit? (binary: yes/no, mapped to 1/0).

## c. GitHub Repository Link
- [GitHub Repository](https://github.com/2026ak05637-MTech/Bank-Marketing-Dataset)

## d. Models Used & Performance Comparison
To solve this classification problem, five different Machine Learning algorithms were trained, tuned, and compared on the test dataset:
1. **Logistic Regression (Balanced)**
2. **Decision Tree (Balanced)**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes**
5. **Support Vector Machine (SVM, Balanced)**

### Comparison Table
Below is the summary of the performance of each model on the test dataset (905 samples):

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8066 | 0.8669 | 0.3348 | 0.7959 | 0.4713 | 0.4292 |
| **Decision Tree** | 0.8619 | 0.6357 | 0.3579 | 0.3469 | 0.3523 | 0.2751 |
| **KNN** | 0.8906 | 0.7452 | 0.4865 | 0.1837 | 0.2667 | 0.2513 |
| **Naive Bayes** | 0.8265 | 0.8001 | 0.2993 | 0.4490 | 0.3592 | 0.2707 |
| **Support Vector Machine (SVM)** | 0.8409 | 0.8761 | 0.3802 | 0.7449 | 0.5034 | 0.4541 |
