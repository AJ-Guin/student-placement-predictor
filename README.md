# 🎓 Student Placement Prediction using Logistic Regression

## 📌 Project Overview

This project uses **Machine Learning to predict whether a student will get placed or not** based on two features:

* **CGPA (Cumulative Grade Point Average)**
* **IQ (Intelligence Quotient)**

A **Logistic Regression** classification model is trained on the student placement dataset to predict the placement outcome.

The project also includes data visualization and decision-boundary visualization to understand how CGPA and IQ influence placement predictions.

---

## 🎯 Objective

The main objective of this project is to build a simple binary classification model that can predict:

* `1` → Student is **Placed**
* `0` → Student is **Not Placed**

The model uses:

```text
CGPA + IQ → Logistic Regression → Placement Prediction
```

---

## 🛠️ Technologies & Libraries Used

* 🐍 Python
* 📊 NumPy
* 🐼 Pandas
* 📈 Matplotlib
* 🤖 Scikit-learn
* 📉 MLxtend

### Python Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlxtend.plotting import plot_decision_regions
```

---

## 📂 Dataset

The project uses a dataset named:

```text
placement.csv
```

The dataset contains the following important columns:

| Feature     | Description               |
| ----------- | ------------------------- |
| `cgpa`      | Student's CGPA            |
| `iq`        | Student's IQ              |
| `placement` | Placement result (0 or 1) |

Example:

| CGPA |  IQ | Placement |
| ---: | --: | --------: |
|  6.8 | 123 |         1 |
|  5.9 | 106 |         0 |
|  5.3 | 121 |         0 |
|  7.4 | 132 |         1 |
|  5.8 | 142 |         0 |

The dataset contains **100 records** with CGPA, IQ, and placement information.

---

## 🔎 Project Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Visualization
   ↓
Feature & Target Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Prediction
   ↓
Accuracy Evaluation
   ↓
Decision Boundary Visualization
```

---

## 1️⃣ Import Required Libraries

NumPy and Pandas are imported for numerical operations and data manipulation.

```python
import numpy as np
import pandas as pd
```

---

## 2️⃣ Load the Dataset

The placement dataset is loaded using Pandas.

```python
df = pd.read_csv('/content/placement.csv')
df = df.iloc[:,1:]

df.head()
```

The first unnecessary column is removed before working with the data.

---

## 3️⃣ Visualize the Dataset

A scatter plot is created using **CGPA** and **IQ**.

```python
import matplotlib.pyplot as plt

plt.scatter(
    df['cgpa'],
    df['iq'],
    c=df['placement']
)
```

This visualization helps understand the relationship between:

* CGPA
* IQ
* Placement status

Different colors represent different placement outcomes.

---

## 4️⃣ Select Features and Target

The independent variables are:

```text
CGPA
IQ
```

The target variable is:

```text
Placement
```

Code:

```python
x = df.iloc[:,0:2]
y = df.iloc[:,-1]
```

Where:

```text
X → Input features
Y → Target/output
```

---

## 5️⃣ Split the Dataset

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.1
)
```

Here:

* **90%** of the data is used for training.
* **10%** of the data is used for testing.

---

## 6️⃣ Feature Scaling

Standardization is performed using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

The training and testing features are then transformed:

```python
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
```

Feature scaling helps put CGPA and IQ on comparable scales before training the model.

---

## 7️⃣ Create the Logistic Regression Model

The project uses **Logistic Regression** for binary classification.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
```

Logistic Regression is suitable for this problem because the output has two possible classes:

```text
0 → Not Placed
1 → Placed
```

---

## 8️⃣ Train the Model

The model is trained using the training dataset.

```python
clf.fit(x_train, y_train)
```

The model learns the relationship between:

```text
CGPA + IQ → Placement
```

---

## 9️⃣ Make Predictions

Predictions are generated using the test dataset.

```python
y_pred = clf.predict(x_test)
```

The predicted values can then be compared with the actual values:

```python
y_test
```

---

## 🔟 Evaluate Model Accuracy

The project's model performance is evaluated using **accuracy score**.

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

The accuracy represents the proportion of test predictions that were classified correctly.

> **Note:** The exact accuracy can vary because `train_test_split()` is used without specifying a `random_state`.

---

## 📊 Decision Boundary Visualization

The project uses MLxtend to visualize the regions classified by the Logistic Regression model.

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(
    x_train,
    y_train.values,
    clf=clf,
    legend=2
)
```

This visualization shows how the trained Logistic Regression model separates students into:

```text
Placed
   vs
Not Placed
```

based on CGPA and IQ.

---

## 📈 Model

### Logistic Regression

Logistic Regression is a supervised machine-learning algorithm commonly used for classification problems.

In this project:

```text
Input:
    CGPA
    IQ

Output:
    Placement
```

The model predicts the probability of belonging to one of the two placement classes.

---

## 💡 Key Learnings

Through this project, I practiced:

* Loading datasets using Pandas
* Data preprocessing
* Feature and target selection
* Data visualization using Matplotlib
* Train-test splitting
* Feature standardization
* Logistic Regression
* Model training
* Making predictions
* Evaluating classification accuracy
* Visualizing decision boundaries

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd student-placement-prediction
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend
```

### 4. Add the Dataset

Make sure `placement.csv` is available in the expected project directory.

### 5. Run the Notebook

Open the Jupyter Notebook or upload it to Google Colab.

```bash
jupyter notebook
```

---

## 📁 Project Structure

```text
Student-Placement-Prediction/
│
├── placement.csv
├── placement_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 🔮 Future Improvements

Some possible improvements for this project are:

* Add more student-related features such as:

  * Communication skills
  * Academic performance
  * Internship experience
  * Projects
  * Work experience
* Use `random_state` for reproducible results.
* Use a proper train/test preprocessing pipeline.
* Add a confusion matrix.
* Calculate precision, recall, and F1-score.
* Compare Logistic Regression with other classification algorithms.
* Deploy the model as a web application.

---

## 👨‍💻 Author

**AJ**

Engineering Student | Machine Learning & AI Enthusiast

---

## ⭐ Conclusion

This project demonstrates a basic **binary classification workflow using Logistic Regression** to predict student placement based on CGPA and IQ.

It serves as a practical introduction to the complete Machine Learning workflow:

```text
Data → Visualization → Preprocessing → Training → Prediction → Evaluation
```

⭐ If you found this project useful, consider giving the repository a star!
