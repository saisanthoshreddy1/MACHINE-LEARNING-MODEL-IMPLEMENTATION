# MACHINE-LEARNING-MODEL-IMPLEMENTATION

## **Overview of the Script**

This Python script implements a basic spam email classification system using **Natural Language Processing (NLP)** and **machine learning** techniques. It leverages the **Naive Bayes algorithm**—specifically, the `MultinomialNB` model from `scikit-learn`—which is effective for text classification tasks such as spam detection.

The script loads email data, processes it into a numerical format, trains a classifier, evaluates its accuracy, visualizes performance using a confusion matrix, and allows real-time testing through user input. Finally, it saves the trained model and vectorizer for future use.

## **1. Importing Required Libraries**

```python
import pandas as pd
```

* Used to handle structured data (like CSV files).
* Essential for reading and manipulating the dataset.

```python
from sklearn.model_selection import train_test_split
```

* Helps in dividing the dataset into training and testing sets.

```python
from sklearn.feature_extraction.text import CountVectorizer
```

* Converts textual data into a matrix of token counts for model input.

```python
from sklearn.naive_bayes import MultinomialNB
```

* The core algorithm used to classify text as spam or not spam.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

* These are used to evaluate the model’s performance.

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

* For visualizing the confusion matrix using a heatmap.

```python
import joblib
```

* Used to save and load machine learning models or preprocessing steps.

## **2. Loading and Preparing the Dataset**

```python
df = pd.read_csv("email.csv")
```

* Reads the CSV file containing emails.
* Assumes the CSV has two columns: `text` (email content) and `label` (spam = 1, not spam = 0).

```python
X = df['text']
y = df['label']
```

* `X`: features (text of the email).
* `y`: labels (spam or not).

## **3. Text Vectorization**

```python
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
```

* `CountVectorizer` tokenizes the text and counts word frequencies.
* The result is a **sparse matrix** that can be used by machine learning models.

## **4. Splitting the Dataset**

```python
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
```

* Splits the data into:

  * **80% training**
  * **20% testing**
* `random_state` ensures reproducibility.

## **5. Model Creation and Training**

```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

* A Naive Bayes model is initialized and trained using the training data.
* It’s called "Multinomial" because it's designed for text or count data.

## **6. Making Predictions**

```python
y_pred = model.predict(X_test)
```

* The model predicts labels for the test dataset.

## **7. Evaluating the Model**

```python
print("accuracy:", accuracy_score(y_test, y_pred))
```

* Measures how many predictions were correct.

```python
print("\nclassification report:\n", classification_report(y_test, y_pred))
```

* Gives precision, recall, f1-score, and support for both classes.

```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix")
plt.show()
```

* Creates a visual confusion matrix to understand model performance:

  * True Positives
  * True Negatives
  * False Positives (Type I errors)
  * False Negatives (Type II errors)

## **8. Saving the Model**

```python
joblib.dump(model, 'spam_classifier.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
```

* Saves the trained model and the vectorizer to disk.
* Allows future predictions without retraining.

## **9. Real-Time Prediction**

```python
user_input = input("enter the email text: ")
input_vector = vectorizer.transform([user_input])
prediction = model.predict(input_vector)
```

* Accepts email content from the user.
* Transforms it using the same vectorizer.
* Predicts if the email is spam.

```python
if prediction[0] == 1:
    print("this email is classified as spam.")
else:
    print("this email is not spam.")
```

* Displays the classification result to the user.

## **Conclusion**

This script provides a full pipeline for training and using a machine learning model for spam classification. It uses standard practices:

* **Text preprocessing** with `CountVectorizer`.
* **Model training** with `MultinomialNB`.
* **Evaluation** with accuracy and confusion matrix.
* **Model persistence** using `joblib`.

You can extend it by:

* Using `TfidfVectorizer` for better weighting.
* Adding more data cleaning (e.g., removing URLs).
* Using other models like Logistic Regression or SVM.
