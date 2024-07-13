import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

iris=load_iris()
dataset=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(dataset.head())
dataset['target']=iris.target
dataset['species']=dataset['target'].map({0: iris.target_names[0], 
                                  1: iris.target_names[1], 
                                  2: iris.target_names[2]})

# Define features (X) and target (y)
X = dataset.drop(['target', 'species'], axis=1)
y = dataset['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Print classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=iris.target_names))
