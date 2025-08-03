# 📘 Machine Learning Assignment 4
## Naive Bayes, Decision Trees, and Ensemble Learning

This repository contains solutions for **Assignment 4** of the Machine Learning Online course, covering theory and practical implementation of **Naive Bayes**, **Decision Trees**, and **Ensemble Learning** techniques such as **Random Forest**, **AdaBoost**, and **Gradient Boosting**.

---

## 📚 Part I: Naive Bayes Classifier

### 🎯 Objective:
Understand the **probabilistic foundation** of Naive Bayes classifiers and apply them to both **text** and **numerical** datasets.

---

### ✅ Task 1: Theory Questions
Answer briefly:
1. **What is the core assumption of Naive Bayes?**
2. **Differentiate between GaussianNB, MultinomialNB, and BernoulliNB.**
3. **Why is Naive Bayes considered suitable for high-dimensional data?**

---

### ✅ Task 2: Spam Detection using MultinomialNB
- Used the **SMS Spam Collection** dataset.
- Preprocessing using **CountVectorizer** / **TfidfVectorizer**.
- Trained a **MultinomialNB** classifier.
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

### ✅ Task 3: GaussianNB with Iris or Wine Dataset
- Trained a **GaussianNB** classifier on the **Iris** or **Wine** dataset.
- Splitted dataset into train/test sets.
- Evaluated performance using common classification metrics.
- Compared briefly with **Logistic Regression** or **Decision Tree**.

---

## 🌳 Part II: Decision Trees

### 🎯 Objective:
Implement **Decision Tree classifiers**, understand tree structure, feature splits, and risks of **overfitting**.

---

### ✅ Task 4: Conceptual Questions
Answer briefly:
1. What is **entropy** and **information gain**?
2. Difference between **Gini Index** and **Entropy**?
3. How can a decision tree **overfit**? How to avoid it?

---

### ✅ Task 5: Decision Tree on Titanic Dataset
- Loaded the **Titanic** dataset.
- Handled **missing values**, **encoded categorical variables**.
- Trained a `DecisionTreeClassifier`.
- Visualized the tree using `plot_tree`.
- Evaluated model using:
  - Accuracy
  - Confusion Matrix

---

### ✅ Task 6: Model Tuning
- Tuned model using:
  - `max_depth`
  - `min_samples_split`
- Evaluated impact on model performance.
- Plotted **Training vs Testing Accuracy** to visualize **overfitting**.

---

## 🤝 Part III: Ensemble Learning – Bagging, Boosting, Random Forest

### 🎯 Objective:
Apply **ensemble techniques** to improve classification performance:
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**

---

### ✅ Task 7: Conceptual Questions
1. Difference between **Bagging** and **Boosting**?
2. How does **Random Forest reduce variance**?
3. What is the **weakness** of Boosting-based methods?

---

### ✅ Task 8: Random Forest vs Decision Tree
- Trained a `RandomForestClassifier` on the same dataset from **Task 5**.
- Compared with Decision Tree on:
  - Accuracy
  - Precision
  - Recall
- Plotted **Feature Importances**.

---

### ✅ Task 9: AdaBoost or Gradient Boosting
- Trained either an `AdaBoostClassifier` or `GradientBoostingClassifier`.
- Used an appropriate dataset (e.g., Titanic, Breast Cancer, etc.).
- Compared with Random Forest and Decision Tree using:
  - Accuracy
  - F1-score
  - Training Time (optional)

---

## 📈 Visualizations
- Feature Importance plots for Random Forest.
- Confusion Matrices.
- Overfitting graphs (Training vs Testing Accuracy).
- Decision Tree structure.

---

## 🛠️ Technologies Used
- Python
- Jupyter Notebook
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- CountVectorizer / TfidfVectorizer

---

## 💾 Dataset Sources
- [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Titanic Dataset (Kaggle)](https://www.kaggle.com/c/titanic/data)
- [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [Wine Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

---



