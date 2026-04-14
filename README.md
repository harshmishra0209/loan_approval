# 🧠 loan_approval

## 📌 Overview

This project demonstrates the implementation and evaluation of classification models using:

* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes (GaussianNB)
* Logistic Regression

The goal is to compare model performance using multiple evaluation metrics.

---

## ⚙️ Technologies Used

* Python 🐍
* Scikit-learn
* NumPy
* Pandas (optional for dataset handling)

---

## 📊 Models Used

### 🔹 1. K-Nearest Neighbors (KNN)

* Works based on distance between data points
* Parameter used: `n_neighbors = 5`

### 🔹 2. Gaussian Naive Bayes (GaussianNB)

* Based on probability and Bayes theorem
* Assumes data follows normal distribution

### 🔹 3. Logistic Regression

* Used for binary classification problems
* Works well when data is linearly separable
* Outputs probability between 0 and 1

---

## 🛠️ Workflow

1. Data Preprocessing

   * Feature scaling applied (`x_train_scaled`, `x_test_scaled`)
2. Train-Test Split
3. Model Training
4. Prediction
5. Evaluation

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate model performance:

* **Accuracy Score** → Overall correctness
* **Precision** → Correct positive predictions
* **Recall** → Coverage of actual positives
* **F1 Score** → Balance of precision & recall
* **Confusion Matrix** → Detailed performance breakdown

---

## 💻 Example Code (Gaussian Naive Bayes)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

gnb_model = GaussianNB()

gnb_model.fit(x_train_scaled, y_train)

y_pred = gnb_model.predict(x_test_scaled)

print("GaussianNB Model")
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred)) 
print("f1_score:", f1_score(y_test, y_pred)) 
print("accuracy_score:", accuracy_score(y_test, y_pred))
print("confusion_matrix:", confusion_matrix(y_test, y_pred))



## 💻 Example Code (Logistic Regression)

python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

log_model = LogisticRegression()

log_model.fit(x_train_scaled, y_train)

y_pred = log_model.predict(x_test_scaled)

print("Logistic Regression Model")
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred)) 
print("f1_score:", f1_score(y_test, y_pred)) 
print("accuracy_score:", accuracy_score(y_test, y_pred))
print("confusion_matrix:", confusion_matrix(y_test, y_pred))




## 🔍 Model Comparison

| Metric   | KNN             | GaussianNB        | Logistic Regression |
| -------- | --------------- | ----------------- | ------------------- |
| Accuracy | Depends on data | Depends on data   | Often stable        |
| Speed    | Slower          | Faster            | Fast                |
| Type     | Distance-based  | Probability-based | Linear model        |

---

## ⚠️ Important Notes

* Scaling is important for KNN and Logistic Regression
* GaussianNB may work well even without scaling
* Always check multiple metrics, not just accuracy

---

## 🚀 Conclusion

* All three models are useful for classification
* Logistic Regression is simple and powerful baseline
* Best model depends on dataset

---

## 📁 Future Improvements

* Add SVM and Decision Tree
* Use cross-validation
* Hyperparameter tuning

---

